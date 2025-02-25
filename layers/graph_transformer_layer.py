import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np
from TCFMM import f3s_1tb1rw_scheduled_permuteV_scaleQK
from DFGNN.operators.fused_gtconv import GTConvFuse_inference_tiling, GTConvFuse_inference_hyper
import FS_SDDMM
import FS_SpMM
"""
    Graph Transformer Layer
    
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}
    return func

"""
    Single Attention Head
"""
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, seed=False):
        super().__init__()
        if seed:
            torch.manual_seed(13)
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        self.Q = nn.Linear(in_dim, out_dim * num_heads, use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, use_bias)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, h):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)
        head_out = g.ndata['wV']/g.ndata['z']
        return head_out

# Assumes 1 head for now.
class MultiHeadAttentionLayerF3S(MultiHeadAttentionLayer):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, seed=False):
        super().__init__(in_dim, out_dim, num_heads, use_bias, seed)
        # Add any additional initialization specific to F3S version here
    
    def forward(self, f3s_input, h):
        # Override the forward method to use F3S-specific implementation
        Q_h = self.Q(h).to(torch.float16)
        K_h = self.K(h).to(torch.float16)
        V_h = self.V(h).to(torch.float16)

        n_warp_per_block = 8
        # out is [time, O, S]
        out = f3s_1tb1rw_scheduled_permuteV_scaleQK(
                f3s_input.RowWindowOffset, 
                f3s_input.sortedRowWindows, 
                f3s_input.SparseAToXindex, 
                f3s_input.TCblockBitMap, 
                f3s_input.num_nodes, 
                Q_h, K_h, V_h, 
                np.sqrt(self.out_dim), n_warp_per_block)
        # return out
        return out[1]

class MultiHeadAttentionLayerFlashSparse(MultiHeadAttentionLayer):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, seed=False):
        super().__init__(in_dim, out_dim, num_heads, use_bias, seed)
    
    def forward(self, inputInfo, h):
        Q_h = self.Q(h).to(torch.float16)
        K_h = self.K(h).to(torch.float16)
        V_h = self.V(h).to(torch.float16)
        sddmm_time, att = FS_SDDMM.forward_gen_fp16_gnn(   
                            Q_h.size(1),                                      
                            inputInfo.row_pointers, 
                            inputInfo.column_index, 
                            inputInfo.degrees, 
                            inputInfo.t_window_rowTensor,
                            Q_h,K_h,inputInfo.max)
        spmm_ones_time, rows_sum = FS_SpMM.forward_fp16_gnn_ones(   
                                    inputInfo.row_pointers, 
                                    inputInfo.column_index, 
                                    att, 
                                    inputInfo.t_window_rowTensor,
                                    inputInfo.t_atomicTensor,
                                    inputInfo.ones, 
                                    inputInfo.num_nodes, 
                                    inputInfo.ones.size(1), 
                                    inputInfo.num_nodes_ori)
        spmm_time, h_prime = FS_SpMM.forward_fp16_gnn(   
                                inputInfo.row_pointers, 
                                inputInfo.column_index, 
                                att, 
                                inputInfo.t_window_rowTensor,
                                inputInfo.t_atomicTensor,
                                V_h, 
                                inputInfo.num_nodes, 
                                V_h.size(1), 
                                inputInfo.num_nodes_ori)
        h_prime = h_prime.div(rows_sum) 
        return h_prime

class MultiHeadAttentionLayerDfgnnHyper(MultiHeadAttentionLayer):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, seed=False):
        super().__init__(in_dim, out_dim, num_heads, use_bias, seed)
    
    def forward(self, dfgnn_input, h):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        # Reshape into [num_nodes, num_heads, feat_dim]
        Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        K_h = K_h.view(-1, self.num_heads, self.out_dim)
        V_h = V_h.view(-1, self.num_heads, self.out_dim)
        out = GTConvFuse_inference_hyper(dfgnn_input.row_pointers, dfgnn_input.column_index, dfgnn_input.rows, dfgnn_input.val, dfgnn_input.smem_consume, Q_h, K_h, V_h)
        return out
    
class MultiHeadAttentionLayerDfgnnTiling(MultiHeadAttentionLayer):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, seed=False):
        super().__init__(in_dim, out_dim, num_heads, use_bias, seed)
    
    def forward(self, dfgnn_input, h):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        # Reshape into [num_nodes, num_heads, feat_dim]
        Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        K_h = K_h.view(-1, self.num_heads, self.out_dim)
        V_h = V_h.view(-1, self.num_heads, self.out_dim)
        out = GTConvFuse_inference_tiling(dfgnn_input.row_pointers, dfgnn_input.column_index, dfgnn_input.val, dfgnn_input.smem_consume, Q_h, K_h, V_h)
        return out

class MultiHeadAttentionLayerDense(MultiHeadAttentionLayer):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, seed=False):
        super().__init__(in_dim, out_dim, num_heads, use_bias, seed)
    
    def propagate_attention(self, g):
        rowInd, colInd = g.adj_tensors('coo')
        S = torch.squeeze(g.ndata['Q_h']) @ torch.squeeze(g.ndata['K_h']).T
        E = torch.zeros_like(S) 
        # E[rowInd, colInd] = torch.exp(S[rowInd, colInd] / np.sqrt(self.out_dim)).clamp(-5, 5)
        E[rowInd, colInd] = S[rowInd, colInd]/ np.sqrt(self.out_dim)
        E_max, _ = torch.max(E, dim=1)
        E[rowInd, colInd] = torch.exp(E[rowInd, colInd] - E_max[rowInd])
        O = E @ torch.squeeze(g.ndata['V_h'])
        z = torch.sum(E, dim=1, keepdim=True)
        return O/z
    
    def forward(self, g, h):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        return self.propagate_attention(g)
    
class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, MLA_type='dgl', layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm        
        self.batch_norm = batch_norm
        
        if MLA_type == 'dgl':
            self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        elif MLA_type == 'f3s':
            self.attention = MultiHeadAttentionLayerF3S(in_dim, out_dim//num_heads, num_heads, use_bias)
        elif MLA_type == 'dense':
            self.attention = MultiHeadAttentionLayerDense(in_dim, out_dim//num_heads, num_heads, use_bias)
        elif MLA_type == 'flashSparse':
            self.attention = MultiHeadAttentionLayerFlashSparse(in_dim, out_dim//num_heads, num_heads, use_bias)
        elif MLA_type == 'dfgnn_hyper':
            self.attention = MultiHeadAttentionLayerDfgnnHyper(in_dim, out_dim//num_heads, num_heads, use_bias)
        elif MLA_type == 'dfgnn_tiling':
            self.attention = MultiHeadAttentionLayerDfgnnTiling(in_dim, out_dim//num_heads, num_heads, use_bias)
        else:
            raise ValueError(f"Invalid MLA type: {MLA_type}")
        
        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        
        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, input, h):
        h_in1 = h # for first residual connection
        
        # multi-head attention out
        attn_out = self.attention(input, h)
        h = attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        h = self.O(h)
        
        if self.residual:
            h = h_in1 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h # for second residual connection
        
        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm2(h)
            
        if self.batch_norm:
            h = self.batch_norm2(h)       

        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
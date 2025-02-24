import torch
import torch.nn as nn
import torch.nn.functional as F

# import dgl

"""
    Graph Transformer
    
"""
from layers.graph_transformer_layer import GraphTransformerLayer, \
MultiHeadAttentionLayerF3S, MultiHeadAttentionLayer, MultiHeadAttentionLayerDense, \
MultiHeadAttentionLayerFlashSparse
from layers.mlp_readout_layer import MLPReadout
from TCFMM import preprocess_gpu
import FS_Block

class f3sInput:
    def __init__(self, RowWindowOffset, sortedRowWindows,\
                 SparseAToXindex, TCblockBitMap, num_nodes):
        self.RowWindowOffset = RowWindowOffset
        self.sortedRowWindows = sortedRowWindows
        self.SparseAToXindex = SparseAToXindex
        self.TCblockBitMap = TCblockBitMap
        self.num_nodes = num_nodes

class fsInput:
    def __init__(self, row_pointers, column_index, degrees, num_nodes,\
                 num_nodes_ori, max, t_window_rowTensor, t_atomicTensor, ones):
        self.row_pointers = row_pointers
        self.column_index = column_index
        self.degrees = degrees
        self.num_nodes = num_nodes
        self.num_nodes_ori = num_nodes_ori
        self.max = max
        self.t_window_rowTensor = t_window_rowTensor
        self.t_atomicTensor = t_atomicTensor
        self.ones = ones

class GraphTransformerNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.out_dim = net_params['out_dim']
        self.hidden_dim = net_params['hidden_dim']
        self.num_heads = net_params['n_heads']
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = net_params['n_classes']
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        self.MLA_type = net_params['MLA_type']
        max_wl_role_index = 100 
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, self.hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, self.hidden_dim)
        
        self.embedding_h = nn.Embedding(in_dim_node, self.hidden_dim) # node feat is an integer
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(self.hidden_dim, self.hidden_dim, self.num_heads,
                                   self.dropout, self.MLA_type, 
                                   self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GraphTransformerLayer(self.hidden_dim, self.out_dim, self.num_heads, 
                                                 self.dropout, self.MLA_type, 
                                                 self.layer_norm, self.batch_norm,  
                                                 self.residual))
        self.MLP_layer = MLPReadout(self.out_dim, self.n_classes)

    def f3s_preprocess_dataset(self, g):
        BLK_H = 16
        BLK_W = 16
        indptr, indices, _ = g.adj_tensors('csr')
        num_nodes = indptr.shape[0] - 1
        num_edges = indices.shape[0]
        # Set up tensors for preprocessing
        num_row_windows = (num_nodes + BLK_H - 1) // BLK_H
        # Move tensors to GPU
        blockPartition_cuda = torch.zeros(num_row_windows, dtype=torch.int).cuda()
        edgeToColumn_cuda = torch.zeros(num_edges, dtype=torch.int).cuda()
        edgeToRow_cuda = torch.zeros(num_edges, dtype=torch.int).cuda()
        row_pointers = indptr.to(torch.int32)
        column_index = indices.to(torch.int32)
        RowWindowOffset, sortedRowWindows, TCblockRowid,\
        TCblocktileId, TCblockoffset, SparseAToXindex,\
        TBBoundaries, TCblockBitMap, block_count =  preprocess_gpu(column_index, row_pointers, num_nodes, 
                                BLK_H, BLK_W, blockPartition_cuda, 
                                edgeToColumn_cuda, edgeToRow_cuda)
        f3s_input = f3sInput(RowWindowOffset, sortedRowWindows, SparseAToXindex, TCblockBitMap, num_nodes)
        return f3s_input

    def flashSparse_preprocess_dataset(self, g):
        partSize = 32
        window = 8
        wide = 16
        indptr, indices, _ = g.adj_tensors('csr')
        column_index = indices.to(torch.int32)
        row_pointers = indptr.to(torch.int32)
        row_pointers, column_index,\
        degrees, t_window_rowTensor,\
        t_atomicTensor = FS_Block.blockProcess_sddmm_balance_gnn(row_pointers.cpu(), column_index.cpu(), 
                                                                 window, wide, partSize)
        print("Number of nonzero elements in degrees: ", torch.count_nonzero(degrees))
        row_pointers = row_pointers.cuda()
        column_index = column_index.cuda()
        degrees = degrees.cuda()
        t_window_rowTensor = t_window_rowTensor.cuda()
        t_atomicTensor = t_atomicTensor.cuda()
        num_nodes_ori = g.num_nodes()
        if g.num_nodes()%16 !=0 :
            num_nodes = g.num_nodes() + 16 - g.num_nodes()%16
        else:
            num_nodes = g.num_nodes()
        ones = torch.ones((num_nodes_ori,1), dtype=torch.float16, device=self.device)
        max_vectors = torch.max(row_pointers[1:]- row_pointers[:-1])
        if max_vectors%wide > 0 :
            max_vectors += (wide - (max_vectors%wide))
        max = max_vectors / wide
        if max % 4 > 0 :
            max += 4 - max%4
        inputInfo = fsInput(row_pointers, column_index,\
                            degrees, num_nodes, num_nodes_ori, max,\
                            t_window_rowTensor, t_atomicTensor, ones)
        return inputInfo

    def check_accuracy(self, g):
        h = g.ndata['feat']
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        layer_f3s = MultiHeadAttentionLayerF3S(self.hidden_dim, self.hidden_dim, self.num_heads, False, seed=True).to(self.device)
        layer_dgl = MultiHeadAttentionLayer(self.hidden_dim, self.hidden_dim, self.num_heads, False, seed=True).to(self.device)
        layer_dense = MultiHeadAttentionLayerDense(self.hidden_dim, self.hidden_dim, self.num_heads, False, seed=True).to(self.device)
        layer_flashSparse = MultiHeadAttentionLayerFlashSparse(self.hidden_dim, self.hidden_dim, self.num_heads, False, seed=True).to(self.device)
        head_out_dgl = layer_dgl(g, h)
        head_out_dgl = torch.squeeze(head_out_dgl)
        print("head_out_dgl.shape: ", head_out_dgl.shape)

        f3s_input = self.f3s_preprocess_dataset(g)
        head_out_f3s = layer_f3s(f3s_input, h)

        head_out_dense = layer_dense(g, h)
        diff_dense = head_out_dgl - head_out_dense

        fs_input = self.flashSparse_preprocess_dataset(g)
        head_out_flashSparse = layer_flashSparse(fs_input, h)
        # print("head_out[10:10, 10:10]: ", head_out_dgl[:10, :10])
        # print("head_out_f3s[10:10, 10:10]: ", head_out_f3s[1][:10, :10])
        print("relative error dense: ", torch.norm(diff_dense)/torch.norm(head_out_dgl))
        diff_f3s = head_out_dgl - head_out_f3s[1]
        print("relative error f3s: ", torch.norm(diff_f3s)/torch.norm(head_out_dgl))
        diff_flashSparse = head_out_dgl - head_out_flashSparse
        print("relative error flashSparse: ", torch.norm(diff_flashSparse)/torch.norm(head_out_dgl))

    def forward(self, g, h_lap_pos_enc=None, h_wl_pos_enc=None):
        if self.MLA_type == 'f3s':
            input = self.f3s_preprocess_dataset(g)
        elif self.MLA_type == 'dgl':
            input = g
        elif self.MLA_type == 'dense':
            input = g
        elif self.MLA_type == 'flashSparse':
            input = self.flashSparse_preprocess_dataset(g)
        else:
            raise ValueError("Invalid MLA type, choose from 'f3s', 'dgl', 'dense'")
        
        # input embedding
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        h = g.ndata['feat']
        h = self.embedding_h(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
        h = self.in_feat_dropout(h)
        
        # GraphTransformer Layers
        for i, conv in enumerate(self.layers):
            h = conv(input, h)
            
        # output
        h_out = self.MLP_layer(h)
        end.record()
        torch.cuda.synchronize()
        time_taken = start.elapsed_time(end)
        return h_out, time_taken
    
    
    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss



        

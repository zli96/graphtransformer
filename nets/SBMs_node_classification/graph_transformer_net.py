import torch
import torch.nn as nn
import numpy as np
# import dgl
from layers.graph_transformer_layer import GraphTransformerLayer
"""
    Graph Transformer
    
"""
from layers.mlp_readout_layer import MLPReadout
from F3S import preprocess_gpu

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

class dfgnnInput:
    def __init__(self, row_pointers, column_index, rows, val, smem_consume):
        self.row_pointers = row_pointers
        self.column_index = column_index
        self.rows = rows
        self.val = val
        self.smem_consume = smem_consume

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

        self.linear_h = nn.Linear(in_dim_node, self.hidden_dim)
        
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
        num_row_windows = (num_nodes + BLK_H - 1) // BLK_H
        blockPartition_cuda = torch.zeros(num_row_windows, dtype=torch.int).cuda()
        edgeToColumn_cuda = torch.zeros(num_edges, dtype=torch.int).cuda()
        edgeToRow_cuda = torch.zeros(num_edges, dtype=torch.int).cuda()
        row_pointers = indptr.to(torch.int32).cuda()
        column_index = indices.to(torch.int32).cuda()
        RowWindowOffset, sortedRowWindows, _, _, _, \
        SparseAToXindex, TCblockBitMap, _ = preprocess_gpu(column_index, row_pointers, num_nodes, 
                                BLK_H, BLK_W, blockPartition_cuda, 
                                edgeToColumn_cuda, edgeToRow_cuda)
        f3s_input = f3sInput(RowWindowOffset, sortedRowWindows, SparseAToXindex, TCblockBitMap, num_nodes)
        return f3s_input

    def flashSparse_preprocess_dataset(self, g):
        import FS_Block
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

    def dfgnn_preprocess_dataset(self, g, alg):
        max_neigh = 128 # according to DF-GNN/DFGNN/layers/util.py
        WARP_SIZE = 32
        row_pointers, column_index, _ = g.adj_tensors('csr')
        row_pointers = row_pointers.to(torch.int32)
        column_index = column_index.to(torch.int32)
        val = torch.ones(g.num_edges(), dtype=torch.float32, device=self.device)/np.sqrt(self.hidden_dim)
        row_nnz = np.diff(row_pointers.cpu().numpy())
        row_indices = np.repeat(np.arange(g.num_nodes()), row_nnz)
        rows = torch.IntTensor(row_indices).cuda()
        if alg == "dfgnn_tiling":
            smem_consume = (max_neigh + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
        elif alg == "dfgnn_hyper":
            smem_consume = (max_neigh * 8 + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
        else:
            raise ValueError(f"Invalid algorithm: {alg}")
        dfgnn_input = dfgnnInput(row_pointers, column_index, rows, val, smem_consume)
        return dfgnn_input

    def check_accuracy(self, g):
        from layers.graph_transformer_layer import MultiHeadAttentionLayerF3S, \
            MultiHeadAttentionLayer, MultiHeadAttentionLayerDense, \
            MultiHeadAttentionLayerFlashSparse, MultiHeadAttentionLayerDfgnnHyper, \
            MultiHeadAttentionLayerDfgnnTiling
        h = g.ndata['feat']
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        layer_dgl = MultiHeadAttentionLayer(self.hidden_dim, self.hidden_dim, self.num_heads, False, seed=True).to(self.device)
        layer_f3s = MultiHeadAttentionLayerF3S(self.hidden_dim, self.hidden_dim, self.num_heads, False, seed=True).to(self.device)
        layer_dense = MultiHeadAttentionLayerDense(self.hidden_dim, self.hidden_dim, self.num_heads, False, seed=True).to(self.device)
        layer_flashSparse = MultiHeadAttentionLayerFlashSparse(self.hidden_dim, self.hidden_dim, self.num_heads, False, seed=True).to(self.device)
        layer_dfgnn_hyper = MultiHeadAttentionLayerDfgnnHyper(self.hidden_dim, self.hidden_dim, self.num_heads, False, seed=True).to(self.device)
        layer_dfgnn_tiling = MultiHeadAttentionLayerDfgnnTiling(self.hidden_dim, self.hidden_dim, self.num_heads, False, seed=True).to(self.device)

        #######
        # run the things
        #######
        f3s_input = self.f3s_preprocess_dataset(g)
        head_out_f3s, _ = layer_f3s(f3s_input, h)

        h = g.ndata['feat']
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        head_out_dgl, _ = layer_dgl(g, h)
        head_out_dgl = torch.squeeze(head_out_dgl)

        head_out_dense, _ = layer_dense(g, h)

        fs_input = self.flashSparse_preprocess_dataset(g)
        head_out_flashSparse, _ = layer_flashSparse(fs_input, h)

        dfgnn_input = self.dfgnn_preprocess_dataset(g, "dfgnn_hyper")
        head_out_dfgnn_hyper, _ = layer_dfgnn_hyper(dfgnn_input, h)
        head_out_dfgnn_hyper = torch.squeeze(head_out_dfgnn_hyper)
        dfgnn_input = self.dfgnn_preprocess_dataset(g, "dfgnn_tiling")
        head_out_dfgnn_tiling, _ = layer_dfgnn_tiling(dfgnn_input, h)
        head_out_dfgnn_tiling = torch.squeeze(head_out_dfgnn_tiling)
        print(f"head_out_dfgnn_hyper.shape: {head_out_dfgnn_hyper.shape} \nhead_out_dfgnn_tiling.shape: {head_out_dfgnn_tiling.shape}")
        

        #######
        # compare the things
        #######
        diff_f3s = head_out_dgl - head_out_f3s
        diff_dense = head_out_dgl - head_out_dense
        diff_flashSparse = head_out_dgl - head_out_flashSparse
        diff_dfgnn_hyper = head_out_dgl - head_out_dfgnn_hyper
        diff_dfgnn_tiling = head_out_dgl - head_out_dfgnn_tiling
        print(f" relative error dfgnn_tiling: {torch.norm(diff_dfgnn_tiling)/torch.norm(head_out_dgl)} \n relative error dfgnn_hyper: {torch.norm(diff_dfgnn_hyper)/torch.norm(head_out_dgl)} \n relative error flashSparse: {torch.norm(diff_flashSparse)/torch.norm(head_out_dgl)} \n relative error dense: {torch.norm(diff_dense)/torch.norm(head_out_dgl)} \n relative error f3s: {torch.norm(diff_f3s)/torch.norm(head_out_dgl)}")

    def forward(self, g, h_lap_pos_enc=None, h_wl_pos_enc=None):
        if self.MLA_type == 'f3s':
            input = self.f3s_preprocess_dataset(g)
        elif self.MLA_type == 'dgl':
            input = g
        elif self.MLA_type == 'dense':
            input = g
        elif self.MLA_type == 'flashSparse':
            input = self.flashSparse_preprocess_dataset(g)
        elif self.MLA_type == 'dfgnn_hyper':
            input = self.dfgnn_preprocess_dataset(g, "dfgnn_hyper")
        elif self.MLA_type == 'dfgnn_tiling':
            input = self.dfgnn_preprocess_dataset(g, "dfgnn_tiling")
        else:
            raise ValueError("Invalid MLA type, choose from 'f3s', 'dgl', 'dense'")
        
        # input embedding
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        h = g.ndata['feat']
        start.record()
        if h.dtype == torch.int32 or h.dtype == torch.int64:
            h = self.embedding_h(h)
            if self.lap_pos_enc:
                h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
                h = h + h_lap_pos_enc
            if self.wl_pos_enc:
                h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
                h = h + h_wl_pos_enc
            h = self.in_feat_dropout(h)
        else:
            h = self.linear_h(h)
        # GraphTransformer Layers
        kernel_time_total = 0
        for i, conv in enumerate(self.layers):
            h, kernel_time = conv(input, h)
            kernel_time_total += kernel_time
        # output
        h_out = self.MLP_layer(h)
        end.record()
        end.synchronize()
        total_time = start.elapsed_time(end)
        return h_out, total_time, kernel_time_total
    
    
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



        

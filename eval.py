import torch
from dgl.data import CLUSTERDataset, RedditDataset, PubmedGraphDataset, CiteseerGraphDataset, CoraGraphDataset, YelpDataset, FlickrDataset, QuestionsDataset
from nets.SBMs_node_classification.graph_transformer_net import GraphTransformerNet
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


hidden_dim = 64
niter = 10
raw_dir = '/share/crsp/lab/amowli/share/Fused3S/dataLoader'

def time_model(net_params, test_graph, device):
  model_times_total = np.empty((1, niter))
  model_times_kernel = np.empty((1, niter))
  try:
    net_params['hidden_dim'] = hidden_dim
    net_params['out_dim'] = hidden_dim
    model = GraphTransformerNet(net_params)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
      for k in range(niter): 
          model(test_graph)
      for k in range(niter):
          pred, total_time, kernel_time = model(test_graph)
          model_times_total[0, k] = total_time
          model_times_kernel[0, k] = kernel_time
  except Exception as e:
    print(f"Error in layer {net_params['MLA_type']}, hidden_dim {hidden_dim}: {e}")
    for k in range(niter):
      model_times_total[0, k] = np.nan
      model_times_kernel[0, k] = np.nan
  return model_times_total, model_times_kernel

def test_accuracy(net_params, test_graph, device):
  net_params['MLA_type'] = 'dgl' # this doesn't matter
  model = GraphTransformerNet(net_params)
  model = model.to(device)
  model.eval()
  pred, _ = model.check_accuracy(test_graph)
  return pred

def data_router(dataset_name, device):
    print(f"===========loading dataset: {dataset_name}===========")
    if dataset_name == 'cluster':
        dataset = CLUSTERDataset(mode='test', raw_dir=raw_dir)
        in_dim = dataset[0].ndata['feat'].shape[0]
        test_graph = dataset[0].to(device)
        n_classes = dataset.num_classes
    elif dataset_name == 'reddit':
        dataset = RedditDataset(raw_dir=raw_dir)
        in_dim = dataset[0].ndata['feat'].shape[1]
        test_graph = dataset[0].to(device)
        n_classes = dataset.num_classes
    elif dataset_name == 'pubmed':
        dataset = PubmedGraphDataset(raw_dir=raw_dir)
        in_dim = dataset[0].ndata['feat'].shape[1]
        test_graph = dataset[0].to(device)
        n_classes = dataset.num_classes
    elif dataset_name == 'citeseer':
        dataset = CiteseerGraphDataset(raw_dir=raw_dir)
        in_dim = dataset[0].ndata['feat'].shape[1]
        test_graph = dataset[0].to(device)
        n_classes = dataset.num_classes
    elif dataset_name == 'cora':
        dataset = CoraGraphDataset(raw_dir=raw_dir)
        in_dim = dataset[0].ndata['feat'].shape[1]
        test_graph = dataset[0].to(device)
        n_classes = dataset.num_classes
    elif dataset_name == 'yelp':
        dataset = YelpDataset(raw_dir=raw_dir)
        in_dim = dataset[0].ndata['feat'].shape[1]
        test_graph = dataset[0].to(device)
        n_classes = dataset.num_classes
    elif dataset_name == 'flickr':
        dataset = FlickrDataset(raw_dir=raw_dir)
        in_dim = dataset[0].ndata['feat'].shape[1]
        test_graph = dataset[0].to(device)
        n_classes = dataset.num_classes
    elif dataset_name == 'Questions':
        dataset = QuestionsDataset(raw_dir=raw_dir)
        in_dim = dataset[0].ndata['feat'].shape[1]
        test_graph = dataset[0].to(device)
        n_classes = dataset.num_classes
    elif dataset_name in ['ogbn-products', 'ogbn-arxiv', 'Ell', 'github', 'AmazonProducts']:
        test_graph, n_classes = load_dataset(dataset_name)
        in_dim = test_graph.ndata['feat'].shape[1]
        test_graph = test_graph.to(device)
    elif dataset_name == 'igb_small':
        test_graph, n_classes = load_igb_dataset('small', device)
        in_dim = test_graph.ndata['feat'].shape[1]
    elif dataset_name == 'igb_medium':
        test_graph, n_classes = load_igb_dataset('medium', device)
        in_dim = test_graph.ndata['feat'].shape[1]
    elif dataset_name == 'igb_large':
        test_graph, n_classes = load_igb_dataset('large', device)
        in_dim = test_graph.ndata['feat'].shape[1]
    elif dataset_name == 'ZINC':
        test_graph, n_classes = load_dataset(dataset_name)
        in_dim = torch.max(test_graph.ndata['feat']).item() + 1
        test_graph = test_graph.to(device)
    elif dataset_name in ['Peptides-struct', 'Peptides-func', 'PascalVOC-SP', 'COCO-SP']:
        test_graph, n_classes = load_dataset(dataset_name)
        print(test_graph)
        test_graph.ndata['feat'] = test_graph.ndata['feat'].to(torch.float32)
        in_dim = test_graph.ndata['feat'].shape[1]
        test_graph = test_graph.to(device)
    return test_graph, in_dim, n_classes

def convert_pyg_to_dgl(pyg_graph):
    import dgl
    x = pyg_graph.x
    edge_index = pyg_graph.edge_index
    g = dgl.graph((edge_index[0], edge_index[1]))
    g.ndata['feat'] = x
    return g

def load_dataset(dataset_name, batch_size=1024):
    if dataset_name == 'ogbn-products':
        from ogb.nodeproppred import DglNodePropPredDataset
        dataset = DglNodePropPredDataset(name=dataset_name, root=raw_dir+'/ogbn-products')
        dgl_graph = dataset[0][0]
        return dgl_graph, dataset.num_classes
    elif dataset_name == 'ogbn-arxiv':
        from ogb.nodeproppred import DglNodePropPredDataset
        dataset = DglNodePropPredDataset(name=dataset_name, root=raw_dir+'/ogbn-arxiv')
        dgl_graph = dataset[0][0]
        return dgl_graph, dataset.num_classes
    elif dataset_name == 'ZINC':
        print("loading ZINC")
        from dgl.data import ZINCDataset
        import dgl
        dataset = ZINCDataset(mode='train', raw_dir=raw_dir+'/ZINC')
        batched_g = dgl.batch([dataset[i][0] for i in range(batch_size)])
        return batched_g, 2
    elif dataset_name in ['Peptides-struct', 'Peptides-func', 'PascalVOC-SP', 'COCO-SP']:
        print("loading Peptides-struct")
        from torch_geometric.datasets import LRGBDataset
        from torch_geometric.data import Batch
        dataset = LRGBDataset(root=raw_dir+'/PeptidesStruct', name=dataset_name)
        batched_g = Batch.from_data_list([dataset[i] for i in range(batch_size)])
        batched_g = convert_pyg_to_dgl(batched_g)
        return batched_g, 2
    elif dataset_name == 'Ell':
        print("loading elliptic bitcoin")
        from torch_geometric.datasets import EllipticBitcoinDataset
        dataset = EllipticBitcoinDataset(root=raw_dir+'/EllipticBitcoin')
        pyg_graph = dataset[0]
        dgl_graph = convert_pyg_to_dgl(pyg_graph)
        return dgl_graph, dataset.num_classes
    elif dataset_name == 'github':
       print("loading github")
       from torch_geometric.datasets import GitHub
       dataset = GitHub(root=raw_dir+'/GitHub')
       pyg_graph = dataset[0]
       dgl_graph = convert_pyg_to_dgl(pyg_graph)
       return dgl_graph, dataset.num_classes
    elif dataset_name == 'AmazonProducts':
        print("loading AmazonProducts")
        from torch_geometric.datasets import AmazonProducts
        dataset = AmazonProducts(root=raw_dir+'/AmazonProducts')
        pyg_graph = dataset[0]
        dgl_graph = convert_pyg_to_dgl(pyg_graph)
        return dgl_graph, dataset.num_classes
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

class igbArgs:
    def __init__(self, size):
        self.path = raw_dir+'/igb'
        self.in_memory = True
        self.num_classes = 19
        self.in_memory = 0
        self.synthetic = 0
        self.dataset_size = size

def load_igb_dataset(dataset_name, device):
   from igb.dataloader import IGB260MDGLDataset
   args = igbArgs(dataset_name)
   dataset = IGB260MDGLDataset(args)
   test_graph = dataset[0].to(device)
   return test_graph, args.num_classes

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_params = {
        'n_heads': 1,
        'in_feat_dropout': 0.0,
        'dropout': 0.0,
        'L': 10,  # Number of layers
        'residual': True,
        'readout': 'mean',
        'layer_norm': True,
        'batch_norm': False,
        'self_loop': False,
        'lap_pos_enc': False,
        'wl_pos_enc': False,
        'full_graph': False,
        'device': device
    }

    test_graph, in_dim, n_classes = data_router(args.dataset, device)
    net_params['in_dim'] = in_dim
    net_params['n_classes'] = n_classes
    net_params['MLA_type'] = args.alg
    # Get a single test grap

    print("test_graph.n_nodes: ", test_graph.num_nodes())
    print("test_graph.n_edges: ", test_graph.num_edges())
    print("in_dim: ", in_dim)
    print("n_classes: ", n_classes)
    times_total, times_kernel = time_model(net_params, test_graph, device)
    df = pd.DataFrame(times_kernel, index=[args.alg], columns=range(niter))
    df.index.name = 'layer'
    df.to_csv(f'gt_times_kernel_{net_params["n_heads"]}heads_{args.alg}_{args.dataset}.csv')
    df = pd.DataFrame(times_total, index=[args.alg], columns=range(niter))
    df.index.name = 'layer'
    df.to_csv(f'gt_times_total_{net_params["n_heads"]}heads_{args.alg}_{args.dataset}.csv')

if __name__ == '__main__':
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='f3s', 
                        choices=['dgl', 'f3s', 'dfgnn_hyper', 
                                 'dfgnn_tiling', 'flashSparse'])
    parser.add_argument('--dataset', type=str, default='Ell',
                        choices=['Ell', 'github', 'igb_small', 
                                 'reddit', 'ogbn-products', 
                                 'AmazonProducts', 'ZINC', 
                                 'Peptides-struct', 'Peptides-func',
                                 'PascalVOC-SP', 'COCO-SP'])
    args = parser.parse_args()
    main(args)

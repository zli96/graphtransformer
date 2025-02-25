import torch
from dgl.data import CLUSTERDataset
from nets.SBMs_node_classification.graph_transformer_net import GraphTransformerNet

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = CLUSTERDataset(mode='test', raw_dir='/share/crsp/lab/amowli/share/Fused3S/dataLoader')
    
    # Get a single test graph
    test_graph = dataset[0]  # Get first graph
    print("test_graph.n_nodes: ", test_graph.num_nodes())
    print("test_graph.n_edges: ", test_graph.num_edges())
    
    # Define model parameters (using same params as in config files)
    net_params = {
        'in_dim': test_graph.ndata['feat'].shape[0],  # Node feature dimension from data
        'hidden_dim': 64,
        'out_dim': 64,
        'n_classes': dataset.num_classes,  # Number of classes from dataset
        'n_heads': 1,
        'in_feat_dropout': 0.0,
        'dropout': 0.0,
        'MLA_type': 'dgl',
        'L': 1,  # Number of layers
        'residual': True,
        'readout': 'mean',
        'layer_norm': False,
        'batch_norm': True,
        'self_loop': False,
        'lap_pos_enc': False,
        'wl_pos_enc': False,
        'full_graph': False,
        'device': device
    }
    
    # Initialize model
    model = GraphTransformerNet(net_params)
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Prepare input
    test_graph = test_graph.to(device)
    test_labels = test_graph.ndata['label']
    
    # Forward pass
    with torch.no_grad():
        model.check_accuracy(test_graph)
        # for i in range(3):
        #     model(test_graph)
        # times = []
        # for i in range(50):
        #     pred, time_taken = model(test_graph)
        #     times.append(time_taken)
        # print(times)
        # print(f"avg time_taken: {sum(times)/len(times)} ms")
        
    # Print predictions shape and a few values
    print(f"Graph has {test_graph.number_of_nodes()} nodes")
    print(f"Node features shape: {test_graph.ndata['feat'].shape}")
    # print(f"Predictions shape: {pred.shape}")
    # print(f"First few predictions:\n{pred[:5]}")
    print(f"True labels shape: {test_labels.shape}")
    print(f"First few true labels:\n{test_labels[:5]}")
    print(f"Number of classes: {net_params['n_classes']}")

if __name__ == '__main__':
    main()

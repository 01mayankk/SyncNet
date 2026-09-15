import sys
import torch

try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, SAGEConv
except ImportError as e:
    print(f"[!] CRITICAL ERROR: PyTorch Geometric (torch_geometric) is not installed: {e}")
    sys.exit(1)

def run_pyg_verification():
    print("=" * 60)
    print("SYNCNET — PYTORCH GEOMETRIC (PyG) GPU DIAGNOSTIC")
    print("=" * 60)
    
    pyg_ver = torch_geometric.__version__
    print(f"PyG Version       : {pyg_ver}")
    
    if not torch.cuda.is_available():
        print("[!] CRITICAL ERROR: CUDA is NOT available for PyG verification.")
        sys.exit(1)
        
    device = torch.device("cuda:0")
    print(f"Target Device     : {device} ({torch.cuda.get_device_name(0)})")
    
    # 1. Construct PyG Graph Data Object
    num_nodes = 100
    in_channels = 64
    out_channels = 32
    num_edges = 500
    
    x = torch.randn(num_nodes, in_channels, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    
    graph_data = Data(x=x, edge_index=edge_index)
    print(f"Graph Nodes       : {graph_data.num_nodes}")
    print(f"Graph Edges       : {graph_data.num_edges}")
    print(f"Feature Dim (x)   : {graph_data.x.shape}")
    print(f"Data Device       : {graph_data.x.device}")
    
    # 2. Test GCNConv Forward Pass on GPU
    print("\n--- Testing GCNConv Layer Execution ---")
    gcn = GCNConv(in_channels, out_channels).to(device)
    gcn_out = gcn(graph_data.x, graph_data.edge_index)
    
    print(f"GCN Output Shape  : {gcn_out.shape}")
    print(f"GCN Output Device : {gcn_out.device}")
    assert gcn_out.shape == (num_nodes, out_channels)
    assert gcn_out.is_cuda
    print("[x] GCNConv GPU Forward Pass: SUCCESS")
    
    # 3. Test SAGEConv Forward Pass on GPU
    print("\n--- Testing SAGEConv Layer Execution ---")
    sage = SAGEConv(in_channels, out_channels).to(device)
    sage_out = sage(graph_data.x, graph_data.edge_index)
    
    print(f"SAGE Output Shape : {sage_out.shape}")
    print(f"SAGE Output Device: {sage_out.device}")
    assert sage_out.shape == (num_nodes, out_channels)
    assert sage_out.is_cuda
    print("[x] SAGEConv GPU Forward Pass: SUCCESS")
    
    print("\n[x] SUCCESS: PyTorch Geometric GCN & GraphSAGE GPU layer execution verified!")
    print("=" * 60)

if __name__ == "__main__":
    run_pyg_verification()

"""
SyncNet PyTorch Geometric Graph Construction Script
===================================================
Constructs a PyTorch Geometric (PyG) Data object representing the social network interaction graph G = (V, E, X).

Graph Specifications:
- Nodes (V): 1,000 user accounts.
- Edges (E): Retweet, mention, and reply interaction links.
- Node Features (X): 12 numerical behavioral features.
- Labels (y): Binary (0: Human, 1: Bot).
- Node Split Masks: train_mask (800), val_mask (100), test_mask (100).

Output:
- data/features/graph_data.pt (Serialized PyG Data Object)
"""

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

DATA_PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DATA_FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")

def build_pyg_interaction_graph():
    print("=" * 60)
    print("SYNCNET — PYTORCH GEOMETRIC GRAPH CONSTRUCTION")
    print("=" * 60)
    
    users_path = os.path.join(DATA_PROC_DIR, "clean_users.parquet")
    edges_path = os.path.join(DATA_PROC_DIR, "clean_edges.parquet")
    beh_path = os.path.join(DATA_FEAT_DIR, "behavioral_features.parquet")
    
    if not os.path.exists(users_path) or not os.path.exists(edges_path) or not os.path.exists(beh_path):
        raise FileNotFoundError("Processed users, edges, or behavioral feature files not found.")
        
    df_users = pd.read_parquet(users_path)
    df_edges = pd.read_parquet(edges_path)
    df_beh = pd.read_parquet(beh_path)
    
    # 1. Map String User IDs to Integer Indices 0..N-1
    user_id_list = df_users["user_id"].tolist()
    user2idx = {u_id: i for i, u_id in enumerate(user_id_list)}
    num_nodes = len(user_id_list)
    
    print(f"Total Graph Nodes (V) : {num_nodes}")
    
    # 2. Build Edge Index COO Tensor
    src_indices = []
    dst_indices = []
    
    for _, row in df_edges.iterrows():
        src = row["source"]
        tgt = row["target"]
        if src in user2idx and tgt in user2idx:
            src_indices.append(user2idx[src])
            dst_indices.append(user2idx[tgt])
            
    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    num_edges = edge_index.shape[1]
    print(f"Total Graph Edges (E) : {num_edges}")
    
    # 3. Build Node Feature Matrix (X)
    feature_cols = [c for c in df_beh.columns if c not in ["user_id", "split", "label"]]
    
    # Align features with user_id index mapping
    df_beh_indexed = df_beh.set_index("user_id").reindex(user_id_list)
    X_mat = df_beh_indexed[feature_cols].values.astype(np.float32)
    
    # Apply standard normalization to node features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Scale based ONLY on training nodes to avoid feature leakage
    train_mask_arr = (df_beh_indexed["split"] == "train").values
    scaler.fit(X_mat[train_mask_arr])
    X_mat_scaled = scaler.transform(X_mat)
    
    x = torch.tensor(X_mat_scaled, dtype=torch.float)
    print(f"Node Feature Dim (X)  : {x.shape}")
    
    # 4. Build Label Tensor (y)
    y = torch.tensor(df_beh_indexed["label"].values, dtype=torch.long)
    
    # 5. Build Split Masks
    train_mask = torch.tensor((df_beh_indexed["split"] == "train").values, dtype=torch.bool)
    val_mask = torch.tensor((df_beh_indexed["split"] == "val").values, dtype=torch.bool)
    test_mask = torch.tensor((df_beh_indexed["split"] == "test").values, dtype=torch.bool)
    
    print(f"Train Nodes Count     : {train_mask.sum().item()}")
    print(f"Val Nodes Count       : {val_mask.sum().item()}")
    print(f"Test Nodes Count      : {test_mask.sum().item()}")
    
    # Construct PyG Data Object
    graph_data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        user_ids=user_id_list
    )
    
    output_path = os.path.join(DATA_FEAT_DIR, "graph_data.pt")
    torch.save(graph_data, output_path)
    
    print(f"\n[x] SUCCESS: PyG Interaction Graph saved to {output_path}")
    print("=" * 60)
    return graph_data

if __name__ == "__main__":
    build_pyg_interaction_graph()

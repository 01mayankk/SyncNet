"""
SyncNet Multimodal Feature Fusion & GraphSAGE Encoder
=====================================================
Fuses Transformer content text embeddings (384-dim) + Behavioral profile features (12-dim)
into a joint 396-dimensional node feature tensor, and trains a GraphSAGE model on GPU (cuda:0).

Outputs:
- Models: models/feature_fusion_sage.pt
- Embeddings: data/features/fused_node_embeddings.parquet (.csv)
"""

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

DATA_PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DATA_FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RANDOM_SEED = 42

class FusionGraphSAGENet(nn.Module):
    def __init__(self, in_channels=396, hidden_channels=64, out_channels=2, dropout=0.3):
        super(FusionGraphSAGENet, self).__init__()
        torch.manual_seed(RANDOM_SEED)
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def get_node_embeddings(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return F.relu(h)

    def forward(self, x, edge_index):
        h = self.get_node_embeddings(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.classifier(h)
        return F.log_softmax(logits, dim=1)

def train_feature_fusion_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    beh_path = os.path.join(DATA_FEAT_DIR, "behavioral_features.parquet")
    content_path = os.path.join(DATA_FEAT_DIR, "content_embeddings.parquet")
    graph_path = os.path.join(DATA_FEAT_DIR, "graph_data.pt")
    
    if not os.path.exists(beh_path) or not os.path.exists(content_path) or not os.path.exists(graph_path):
        raise FileNotFoundError("Behavioral, content, or graph files not found. Run previous phase scripts first.")
        
    print("=" * 70)
    print("SYNCNET — MULTIMODAL FEATURE FUSION & GRAPHSAGE ENCODER")
    print("=" * 70)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training Device : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    df_beh = pd.read_parquet(beh_path)
    df_content = pd.read_parquet(content_path)
    graph_raw = torch.load(graph_path, weights_only=False)
    
    # 1. Feature Fusion Matrix Construction
    beh_cols = [c for c in df_beh.columns if c not in ["user_id", "split", "label"]]
    cnt_cols = [c for c in df_content.columns if c not in ["user_id", "split", "label"]]
    
    user_id_list = graph_raw.user_ids
    df_beh_idx = df_beh.set_index("user_id").reindex(user_id_list)
    df_cnt_idx = df_content.set_index("user_id").reindex(user_id_list)
    
    X_beh = df_beh_idx[beh_cols].values.astype(np.float32)
    X_cnt = df_cnt_idx[cnt_cols].values.astype(np.float32)
    
    # Scale behavioral features based on Train mask
    scaler = StandardScaler()
    train_mask_np = graph_raw.train_mask.cpu().numpy()
    scaler.fit(X_beh[train_mask_np])
    X_beh_scaled = scaler.transform(X_beh)
    
    # Concatenate [Behavioral (12) || Content (384)] = 396-dim fused feature vector
    X_fusion = np.hstack([X_beh_scaled, X_cnt])
    in_channels = X_fusion.shape[1]
    
    print(f"Behavioral Feature Dim : {X_beh_scaled.shape[1]}")
    print(f"Content Feature Dim    : {X_cnt.shape[1]}")
    print(f"Fused Node Feature Dim : {in_channels}")
    
    x = torch.tensor(X_fusion, dtype=torch.float).to(device)
    edge_index = graph_raw.edge_index.to(device)
    y = graph_raw.y.to(device)
    train_mask = graph_raw.train_mask.to(device)
    val_mask = graph_raw.val_mask.to(device)
    test_mask = graph_raw.test_mask.to(device)
    
    # 2. Train Fusion GraphSAGE Model
    model = FusionGraphSAGENet(in_channels=in_channels, hidden_channels=64, out_channels=2, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.NLLLoss()
    
    epochs = 100
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_out = model(x, edge_index)
                val_loss = criterion(val_out[val_mask], y[val_mask])
                val_acc = accuracy_score(y[val_mask].cpu(), val_out[val_mask].argmax(dim=1).cpu())
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f}")
            
    training_time = time.time() - start_time
    
    # 3. Held-Out Test Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        probs = torch.exp(logits[:, 1]).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        node_embeddings = model.get_node_embeddings(x, edge_index).cpu().numpy()
        
    test_mask_np = test_mask.cpu().numpy()
    y_test = y.cpu().numpy()[test_mask_np]
    preds_test = preds[test_mask_np]
    probs_test = probs[test_mask_np]
    
    acc = accuracy_score(y_test, preds_test)
    prec = precision_score(y_test, preds_test, zero_division=0)
    rec = recall_score(y_test, preds_test, zero_division=0)
    f1 = f1_score(y_test, preds_test, zero_division=0)
    roc_auc = roc_auc_score(y_test, probs_test)
    pr_auc = average_precision_score(y_test, probs_test)
    cm = confusion_matrix(y_test, preds_test)
    
    print("\n--- FEATURE FUSION GRAPHSAGE TEST RESULTS ---")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    
    # 4. Save Model & Structural Node Embeddings
    ckpt_path = os.path.join(MODELS_DIR, "feature_fusion_sage.pt")
    torch.save(model.state_dict(), ckpt_path)
    
    # Export fused 64-dim structural node embeddings DataFrame
    df_fused_emb = pd.DataFrame(user_id_list, columns=["user_id"])
    df_fused_emb["split"] = df_beh_idx["split"].values
    df_fused_emb["label"] = y.cpu().numpy()
    df_fused_emb["pred_label"] = preds
    df_fused_emb["pred_prob"] = probs
    
    for d in range(node_embeddings.shape[1]):
        df_fused_emb[f"z_{d:02d}"] = node_embeddings[:, d]
        
    out_emb_parquet = os.path.join(DATA_FEAT_DIR, "fused_node_embeddings.parquet")
    out_emb_csv = os.path.join(DATA_FEAT_DIR, "fused_node_embeddings.csv")
    df_fused_emb.to_parquet(out_emb_parquet, index=False)
    df_fused_emb.to_csv(out_emb_csv, index=False)
    
    print(f"\n[x] SUCCESS: Model weights saved to {ckpt_path}")
    print(f"[x] SUCCESS: Fused node embeddings saved to {out_emb_parquet}")
    print("=" * 70)
    
    return {
        "Name": "Feature Fusion GraphSAGE",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Confusion_Matrix": cm.tolist(),
    }

if __name__ == "__main__":
    train_feature_fusion_model()

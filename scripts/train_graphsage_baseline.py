"""
SyncNet GraphSAGE Model Training Engine
=======================================
Implements a 2-layer GraphSAGE (SAGEConv) model using PyTorch Geometric with mean neighborhood
aggregation and trains on GPU (cuda:0).

Outputs:
- Test metrics (Accuracy, F1, ROC-AUC)
- Checkpoint: models/graphsage_baseline.pt
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

DATA_FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RANDOM_SEED = 42

class GraphSAGENet(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=2, dropout=0.3):
        super(GraphSAGENet, self).__init__()
        torch.manual_seed(RANDOM_SEED)
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train_graphsage():
    os.makedirs(MODELS_DIR, exist_ok=True)
    graph_path = os.path.join(DATA_FEAT_DIR, "graph_data.pt")
    
    if not os.path.exists(graph_path):
        raise FileNotFoundError("graph_data.pt not found. Run scripts/build_graph.py first.")
        
    print("=" * 60)
    print("SYNCNET — GRAPHSAGE MODEL TRAINING (PyG GPU)")
    print("=" * 60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training Device : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    data = torch.load(graph_path, weights_only=False).to(device)
    
    in_channels = data.x.shape[1]
    model = GraphSAGENet(in_channels=in_channels, hidden_channels=32, out_channels=2, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.NLLLoss()
    
    epochs = 100
    start_time = time.time()
    
    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])
                val_pred = val_out[data.val_mask].argmax(dim=1)
                val_acc = accuracy_score(data.y[data.val_mask].cpu(), val_pred.cpu())
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f}")
            
    training_time = time.time() - start_time
    
    # Final Held-Out Test Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.exp(logits[:, 1]).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        
    test_mask = data.test_mask.cpu().numpy()
    y_test = data.y.cpu().numpy()[test_mask]
    preds_test = preds[test_mask]
    probs_test = probs[test_mask]
    
    acc = accuracy_score(y_test, preds_test)
    prec = precision_score(y_test, preds_test, zero_division=0)
    rec = recall_score(y_test, preds_test, zero_division=0)
    f1 = f1_score(y_test, preds_test, zero_division=0)
    roc_auc = roc_auc_score(y_test, probs_test)
    pr_auc = average_precision_score(y_test, probs_test)
    cm = confusion_matrix(y_test, preds_test)
    
    print("\n--- GRAPHSAGE HELD-OUT TEST RESULTS ---")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Confusion Matrix:\n  TN: {cm[0][0]} | FP: {cm[0][1]}\n  FN: {cm[1][0]} | TP: {cm[1][1]}")
    
    ckpt_path = os.path.join(MODELS_DIR, "graphsage_baseline.pt")
    torch.save(model.state_dict(), ckpt_path)
    
    print(f"\n[x] SUCCESS: Saved GraphSAGE model weights to {ckpt_path}")
    print("=" * 60)
    
    return {
        "Name": "PyG GraphSAGE",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Confusion_Matrix": cm.tolist(),
    }

if __name__ == "__main__":
    train_graphsage()

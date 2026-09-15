"""
SyncNet Comprehensive Training, Evaluation & Visualization Pipeline
===================================================================
Executes PyG Multimodal GraphSAGE model training on GPU (cuda:0) with epoch-by-epoch metric logging,
calculates loss functions, validation/test accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC,
and saves high-resolution experiment evaluation charts to `docs/plots/`.

Outputs:
- docs/plots/loss_curves.png
- docs/plots/accuracy_curves.png
- docs/plots/confusion_matrix.png
- docs/plots/roc_pr_curves.png
- docs/plots/experiment_metrics.json
"""

import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    roc_curve,
    precision_recall_curve,
)

DATA_PROC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
DATA_FEAT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "features"))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "plots"))

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

def run_experiment_and_plot():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 80)
    print("SYNCNET — EXPERIMENTAL MODEL TRAINING & PLOTTING PIPELINE")
    print("=" * 80)

    beh_path = os.path.join(DATA_FEAT_DIR, "behavioral_features.parquet")
    content_path = os.path.join(DATA_FEAT_DIR, "content_embeddings.parquet")
    graph_path = os.path.join(DATA_FEAT_DIR, "graph_data.pt")

    if not os.path.exists(beh_path) or not os.path.exists(content_path) or not os.path.exists(graph_path):
        raise FileNotFoundError("Feature parquet or PyG graph data not found. Run previous pipeline stages first.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device               : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    df_beh = pd.read_parquet(beh_path)
    df_content = pd.read_parquet(content_path)
    graph_raw = torch.load(graph_path, weights_only=False)

    beh_cols = [c for c in df_beh.columns if c not in ["user_id", "split", "label"]]
    cnt_cols = [c for c in df_content.columns if c not in ["user_id", "split", "label"]]

    user_id_list = graph_raw.user_ids
    df_beh_idx = df_beh.set_index("user_id").reindex(user_id_list)
    df_cnt_idx = df_content.set_index("user_id").reindex(user_id_list)

    X_beh = df_beh_idx[beh_cols].values.astype(np.float32)
    X_cnt = df_cnt_idx[cnt_cols].values.astype(np.float32)

    # Scale behavioral features ONLY based on Train set to prevent data leakage
    scaler = StandardScaler()
    train_mask_np = graph_raw.train_mask.cpu().numpy()
    scaler.fit(X_beh[train_mask_np])
    X_beh_scaled = scaler.transform(X_beh)

    # Concatenate 12-dim scaled behavior + 384-dim text embedding = 396-dim fused feature vector
    X_fusion = np.hstack([X_beh_scaled, X_cnt])
    in_channels = X_fusion.shape[1]

    print(f"Input Feature Dimension: {in_channels} (12 Profile + 384 MiniLM Text Embeddings)")

    x = torch.tensor(X_fusion, dtype=torch.float).to(device)
    edge_index = graph_raw.edge_index.to(device)
    y = graph_raw.y.to(device)
    train_mask = graph_raw.train_mask.to(device)
    val_mask = graph_raw.val_mask.to(device)
    test_mask = graph_raw.test_mask.to(device)

    # Model Hyperparameters
    learning_rate = 0.005
    weight_decay = 1e-4
    dropout_rate = 0.3
    epochs = 100

    model = FusionGraphSAGENet(in_channels=in_channels, hidden_channels=64, out_channels=2, dropout=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.NLLLoss()

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
        "val_f1": [],
        "test_f1": [],
        "test_precision": [],
        "test_recall": [],
        "test_roc_auc": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    print("\nStarting Epoch-by-Epoch Training & Validation...")
    print("-" * 80)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val Acc':<9} | {'Val F1':<8} | {'Test Acc':<9} | {'Test F1':<8}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate on Train, Val, Test masks
        model.eval()
        with torch.no_grad():
            eval_out = model(x, edge_index)
            val_loss = criterion(eval_out[val_mask], y[val_mask]).item()
            test_loss = criterion(eval_out[test_mask], y[test_mask]).item()

            preds_all = eval_out.argmax(dim=1).cpu().numpy()
            probs_all = torch.exp(eval_out[:, 1]).cpu().numpy()
            y_np = y.cpu().numpy()

            train_acc = accuracy_score(y_np[train_mask_np], preds_all[train_mask_np])
            val_acc = accuracy_score(y_np[val_mask.cpu().numpy()], preds_all[val_mask.cpu().numpy()])
            test_acc = accuracy_score(y_np[test_mask.cpu().numpy()], preds_all[test_mask.cpu().numpy()])

            val_f1 = f1_score(y_np[val_mask.cpu().numpy()], preds_all[val_mask.cpu().numpy()], zero_division=0)
            test_f1 = f1_score(y_np[test_mask.cpu().numpy()], preds_all[test_mask.cpu().numpy()], zero_division=0)

            test_prec = precision_score(y_np[test_mask.cpu().numpy()], preds_all[test_mask.cpu().numpy()], zero_division=0)
            test_rec = recall_score(y_np[test_mask.cpu().numpy()], preds_all[test_mask.cpu().numpy()], zero_division=0)
            test_auc = roc_auc_score(y_np[test_mask.cpu().numpy()], probs_all[test_mask.cpu().numpy()])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "feature_fusion_sage.pt"))

        history["epoch"].append(epoch)
        history["train_loss"].append(round(loss.item(), 4))
        history["val_loss"].append(round(val_loss, 4))
        history["test_loss"].append(round(test_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["test_acc"].append(round(test_acc, 4))
        history["val_f1"].append(round(val_f1, 4))
        history["test_f1"].append(round(test_f1, 4))
        history["test_precision"].append(round(test_prec, 4))
        history["test_recall"].append(round(test_rec, 4))
        history["test_roc_auc"].append(round(test_auc, 4))

        if epoch % 10 == 0 or epoch == epochs:
            print(f"{epoch:<6d} | {loss.item():<10.4f} | {val_loss:<10.4f} | {val_acc:<9.4f} | {val_f1:<8.4f} | {test_acc:<9.4f} | {test_f1:<8.4f}")


    total_time = time.time() - start_time
    print("-" * 80)
    print(f"Training Complete in {total_time:.2f} seconds. Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")

    # Held-out Final Test Evaluation
    test_mask_np = test_mask.cpu().numpy()
    y_test = y_np[test_mask_np]
    preds_test = preds_all[test_mask_np]
    probs_test = probs_all[test_mask_np]

    acc = accuracy_score(y_test, preds_test)
    prec = precision_score(y_test, preds_test, zero_division=0)
    rec = recall_score(y_test, preds_test, zero_division=0)
    f1 = f1_score(y_test, preds_test, zero_division=0)
    roc_auc = roc_auc_score(y_test, probs_test)
    pr_auc = average_precision_score(y_test, probs_test)
    cm = confusion_matrix(y_test, preds_test)

    print("\n" + "=" * 80)
    print("FINAL HELD-OUT TEST METRICS SUMMARY (100 Test Accounts)")
    print("=" * 80)
    print(f"Loss Function          : Negative Log-Likelihood (nn.NLLLoss on LogSoftmax)")
    print(f"Optimizer              : Adam (lr={learning_rate}, weight_decay={weight_decay})")
    print(f"Dropout Rate           : {dropout_rate}")
    print(f"Total Epochs           : {epochs}")
    print(f"Test Accuracy          : {acc * 100:.2f}%")
    print(f"Test Precision         : {prec:.4f}")
    print(f"Test Recall            : {rec:.4f}")
    print(f"Test F1 Score          : {f1:.4f}")
    print(f"Test ROC-AUC           : {roc_auc:.4f}")
    print(f"Test PR-AUC            : {pr_auc:.4f}")
    print(f"Confusion Matrix (TN, FP, FN, TP): [{cm[0][0]}, {cm[0][1]}, {cm[1][0]}, {cm[1][1]}]")
    print("=" * 80)

    # Save metrics JSON
    with open(os.path.join(PLOTS_DIR, "experiment_metrics.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Set dark cyber plotting style
    plt.style.use('dark_background')
    sns.set_theme(style="darkgrid", palette="dark")

    # 1. Plot Loss Curves (Training vs Validation Loss)
    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", color="#06b6d4", linewidth=2.5)
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss", color="#f59e0b", linewidth=2.5, linestyle="--")
    plt.plot(history["epoch"], history["test_loss"], label="Test Loss", color="#8b5cf6", linewidth=2, linestyle=":")
    plt.title("SyncNet GraphSAGE Training & Validation Loss Curves", fontsize=14, fontweight='bold', color='#f1f5f9', pad=12)
    plt.xlabel("Epochs", fontsize=11, color='#cbd5e1')
    plt.ylabel("Negative Log-Likelihood Loss", fontsize=11, color='#cbd5e1')
    plt.legend(frameon=True, facecolor='#0f172a', edgecolor='#334155')
    plt.tight_layout()
    loss_fig_path = os.path.join(PLOTS_DIR, "loss_curves.png")
    plt.savefig(loss_fig_path, dpi=300)
    plt.close()

    # 2. Plot Accuracy Curves (Train vs Validation vs Test Accuracy)
    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], [a * 100 for a in history["train_acc"]], label="Train Accuracy", color="#10b981", linewidth=2.5)
    plt.plot(history["epoch"], [a * 100 for a in history["val_acc"]], label="Validation Accuracy", color="#3b82f6", linewidth=2.5, linestyle="--")
    plt.plot(history["epoch"], [a * 100 for a in history["test_acc"]], label="Test Accuracy", color="#ec4899", linewidth=2, linestyle=":")
    plt.title("SyncNet Accuracy Curves across Epochs (%)", fontsize=14, fontweight='bold', color='#f1f5f9', pad=12)
    plt.xlabel("Epochs", fontsize=11, color='#cbd5e1')
    plt.ylabel("Accuracy (%)", fontsize=11, color='#cbd5e1')
    plt.ylim(50, 105)
    plt.legend(frameon=True, facecolor='#0f172a', edgecolor='#334155')
    plt.tight_layout()
    acc_fig_path = os.path.join(PLOTS_DIR, "accuracy_curves.png")
    plt.savefig(acc_fig_path, dpi=300)
    plt.close()

    # 3. Plot Confusion Matrix Heatmap
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Human (0)', 'Bot (1)'],
                yticklabels=['Human (0)', 'Bot (1)'],
                annot_kws={"size": 16, "weight": "bold"})
    plt.title("Confusion Matrix — Test Set (100 Accounts)", fontsize=13, fontweight='bold', color='#f1f5f9', pad=12)
    plt.xlabel("Predicted Label", fontsize=11, color='#cbd5e1')
    plt.ylabel("Ground Truth Label", fontsize=11, color='#cbd5e1')
    plt.tight_layout()
    cm_fig_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()

    # 4. Plot ROC-AUC and Precision-Recall Curves
    fpr, tpr, _ = roc_curve(y_test, probs_test)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, probs_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(fpr, tpr, color='#06b6d4', linewidth=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='#64748b', linestyle='--')
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold', color='#f1f5f9')
    ax1.set_xlabel('False Positive Rate', fontsize=10, color='#cbd5e1')
    ax1.set_ylabel('True Positive Rate', fontsize=10, color='#cbd5e1')
    ax1.legend(facecolor='#0f172a', edgecolor='#334155')

    ax2.plot(rec_curve, prec_curve, color='#10b981', linewidth=2.5, label=f'PR Curve (AUC = {pr_auc:.4f})')
    ax2.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold', color='#f1f5f9')
    ax2.set_xlabel('Recall', fontsize=10, color='#cbd5e1')
    ax2.set_ylabel('Precision', fontsize=10, color='#cbd5e1')
    ax2.legend(facecolor='#0f172a', edgecolor='#334155')

    plt.tight_layout()
    roc_fig_path = os.path.join(PLOTS_DIR, "roc_pr_curves.png")
    plt.savefig(roc_fig_path, dpi=300)
    plt.close()

    print(f"[x] SUCCESS: Saved loss curves to {loss_fig_path}")
    print(f"[x] SUCCESS: Saved accuracy curves to {acc_fig_path}")
    print(f"[x] SUCCESS: Saved confusion matrix heatmap to {cm_fig_path}")
    print(f"[x] SUCCESS: Saved ROC/PR curves to {roc_fig_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_experiment_and_plot()

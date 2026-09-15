# Experimentation Tracking Log — SyncNet

Master tracking log recording model architecture choices, dataset splits, hyperparameters, training logs, hardware resource metrics, and test performance metrics.

---

## Experiment Index

| Exp ID | Date | Model Type | Feature Input | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Hardware Device |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-01** | 2026-09-15 | Logistic Regression | Behavioral (12 metrics) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | CPU |
| **EXP-01** | 2026-09-15 | Random Forest | Behavioral (12 metrics) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | CPU |
| **EXP-02** | 2026-09-15 | Logistic Regression | Content Embeddings (384-dim) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | RTX 5050 GPU |
| **EXP-02** | 2026-09-15 | Random Forest | Content Embeddings (384-dim) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | RTX 5050 GPU |
| **EXP-03** | 2026-09-15 | PyG GCN Baseline | Interaction Graph (4,332 edges) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | RTX 5050 GPU |
| **EXP-04** | 2026-09-15 | PyG GraphSAGE | Interaction Graph (4,332 edges) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | RTX 5050 GPU |
| **EXP-05** | 2026-09-15 | Feature Fusion + GraphSAGE | 396-dim (MiniLM + Behavior + Graph) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | RTX 5050 GPU |

---

## Detailed Experiment Logs

### Experiment 01 — Behavioral Profile Feature Baselines
- **Date**: 2026-09-15
- **Dataset Version**: Synthetic TwiBot-22 Benchmark (1,000 accounts: 800 Train, 100 Val, 100 Test)
- **Features**: 12 tabular behavioral metrics (`account_age_days`, `follower_following_ratio`, `tweet_frequency`, `screen_name_digit_ratio`, `url_density`, `mention_density`, etc.)
- **Preprocessing**: `StandardScaler` fitted on Train set only.
- **Results**: 
  - Logistic Regression: Test Acc = 100%, F1 = 1.000, ROC-AUC = 1.000 (70 TN, 30 TP, 0 FP, 0 FN).
  - Random Forest: Test Acc = 100%, F1 = 1.000, ROC-AUC = 1.000 (70 TN, 30 TP, 0 FP, 0 FN).

### Experiment 02 — Transformer Content Embedding Baselines
- **Date**: 2026-09-15
- **Dataset Version**: Synthetic TwiBot-22 Benchmark (3,938 posts aggregated across 1,000 accounts)
- **Transformer Encoder**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim) on GPU (`cuda:0`)
- **Extraction Time**: 1.32 seconds on RTX 5050 GPU.
- **Results**: 
  - Logistic Regression: Test Acc = 100%, F1 = 1.000, ROC-AUC = 1.000 (70 TN, 30 TP, 0 FP, 0 FN).
  - Random Forest: Test Acc = 100%, F1 = 1.000, ROC-AUC = 1.000 (70 TN, 30 TP, 0 FP, 0 FN).

### Experiment 03 — PyG GCN Baseline Model
- **Date**: 2026-09-15
- **Architecture**: 2-layer `GCNConv` (in_channels=13, hidden=32, out=2)
- **Execution Device**: `cuda:0` (NVIDIA GeForce RTX 5050 Laptop GPU)
- **Training Time**: 0.92 seconds (100 epochs)
- **Results**: Test Acc = 100%, F1 = 1.000, ROC-AUC = 1.000 (70 TN, 30 TP, 0 FP, 0 FN).

### Experiment 04 — PyG GraphSAGE Baseline Model
- **Date**: 2026-09-15
- **Architecture**: 2-layer `SAGEConv` (in_channels=13, hidden=32, out=2, aggr='mean')
- **Execution Device**: `cuda:0` (NVIDIA GeForce RTX 5050 Laptop GPU)
- **Training Time**: 0.79 seconds (100 epochs)
- **Results**: Test Acc = 100%, F1 = 1.000, ROC-AUC = 1.000 (70 TN, 30 TP, 0 FP, 0 FN).

### Experiment 05 — Multimodal Feature Fusion GraphSAGE & Coordination Scoring
- **Date**: 2026-09-15
- **Feature Vector**: 396 dimensions (384-dim MiniLM Text Vectors + 12-dim Normalized Profile Metrics)
- **Execution Device**: `cuda:0` (NVIDIA GeForce RTX 5050 Laptop GPU)
- **Training Time**: 0.77 seconds (100 epochs)
- **Results**: Test Acc = 100%, F1 = 1.000, ROC-AUC = 1.000
- **Cluster Coordination Output**: Identified 8 interaction clusters; detected high-risk coordinated cluster `cluster_00` (298 members, 100% bot ratio, density=0.289, text_sim=0.7623, coordination_score=0.6324 $\rightarrow$ `THROTTLED`).

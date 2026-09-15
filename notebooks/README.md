# Experimentation Notebooks — SyncNet

All machine learning experimentation, feature extraction, graph construction, and ablation studies are organized sequentially in this directory.

## Notebook Roadmap
1. `01_data_exploration.ipynb`: Dataset statistics, label distributions, missing value checks.
2. `02_data_preprocessing.ipynb`: Text cleaning, user profiling, split creation.
3. `03_transformer_embeddings.ipynb`: MiniLM / DistilBERT text embeddings generation.
4. `04_behavioral_features.ipynb`: Tabular profile feature engineering & normalization.
5. `05_graph_construction.ipynb`: User-user interaction graph (mentions/retweets) & PyG data structure build.
6. `06_gnn_baseline.ipynb`: PyTorch Geometric `GCNConv` baseline training & evaluation.
7. `07_graphsage_experiment.ipynb`: PyTorch Geometric `SAGEConv` training & hyperparameter tuning.
8. `08_feature_fusion.ipynb`: Joint multimodal feature fusion model evaluation.
9. `09_cluster_detection.ipynb`: Graph clustering (Louvain / HDBSCAN) & cluster embeddings.
10. `10_model_evaluation.ipynb`: Comparative metrics, ROC-AUC, PR-AUC, ablation study.
11. `11_reactive_simulation.ipynb`: Reactive state machine simulation, decay math, and threshold evaluation.

## Naming & Style Rules
- Use meaningful variable names (`node_embeddings`, `interaction_graph`, `coordination_score`). Avoid `foo`, `bar`, `temp`.
- Always set random seed `RANDOM_SEED = 42` for reproducibility.
- Log GPU VRAM and System RAM usage in experiment markdown cells.

# Project Status — SyncNet

**Full Project Title**: SyncNet: Coordinated Bot-Network Detection with Reactive Throttling  
**Repository**: [https://github.com/01mayankk/SyncNet.git](https://github.com/01mayankk/SyncNet.git)  
**Last Major Update**: 2026-09-15  

---

## Current Phase
**Phase 3: Transformer Content Embeddings & Behavioral Baseline Models**

## Current Objective
Extract 384-dimensional semantic text representations using Hugging Face Transformer (`all-MiniLM-L6-v2`) on GPU (`cuda:0`), train traditional machine learning baselines (Logistic Regression & Random Forest) on behavioral profile metrics and content embeddings, evaluate test metrics, and store model checkpoints.

## Completed
- Initialized Git repository on `main` branch connected to `https://github.com/01mayankk/SyncNet.git`.
- Created project directory structure (`docs/`, `data/raw/`, `data/processed/`, `data/features/`, `notebooks/`, `backend/`, `frontend/`, `scripts/`, `models/`).
- Configured `.venv` local virtual environment with PyTorch CUDA 12.8 wheel (`2.12.0.dev20260408+cu128`), PyTorch Geometric (`2.8.0.post1`), Hugging Face Transformers (`5.17.0`), FastAPI, Pydantic, Scikit-learn, Pandas, PyArrow, Pytest, and Psutil.
- Verified NVIDIA GeForce RTX 5050 GPU tensor computation and PyG `GCNConv` / `SAGEConv` GPU forward passes on `cuda:0`.
- Built benchmark dataset ingestion pipeline in [`scripts/ingest_dataset.py`](file:///c:/Projects/Syncnet/scripts/ingest_dataset.py) generating 1,000 accounts (300 Bots, 700 Humans), 3,938 posts, and 4,332 interaction edges.
- Built preprocessing and behavioral feature extraction engine in [`scripts/preprocess_data.py`](file:///c:/Projects/Syncnet/scripts/preprocess_data.py).
- Built Transformer content embedding extraction pipeline in [`scripts/extract_transformer_embeddings.py`](file:///c:/Projects/Syncnet/scripts/extract_transformer_embeddings.py) generating 384-dimensional content vectors per user on `cuda:0` in 1.32s.
- Built baseline model training engine in [`scripts/train_baselines.py`](file:///c:/Projects/Syncnet/scripts/train_baselines.py) training Logistic Regression & Random Forest classifiers on behavioral features and content embeddings.
- Evaluated test performance metrics on held-out 100-account test set and logged detailed metrics to [`docs/experiments.md`](file:///c:/Projects/Syncnet/docs/experiments.md).
- Saved trained model checkpoints to `models/` (`behavioral_logistic_regression.joblib`, `behavioral_random_forest.joblib`, `content_logistic_regression.joblib`, `content_random_forest.joblib`, `behavioral_scaler.joblib`).
- Authored notebooks: [`notebooks/01_data_exploration.ipynb`](file:///c:/Projects/Syncnet/notebooks/01_data_exploration.ipynb), [`notebooks/02_data_preprocessing.ipynb`](file:///c:/Projects/Syncnet/notebooks/02_data_preprocessing.ipynb), [`notebooks/03_transformer_embeddings.ipynb`](file:///c:/Projects/Syncnet/notebooks/03_transformer_embeddings.ipynb), [`notebooks/04_behavioral_features.ipynb`](file:///c:/Projects/Syncnet/notebooks/04_behavioral_features.ipynb), and [`notebooks/10_model_evaluation.ipynb`](file:///c:/Projects/Syncnet/notebooks/10_model_evaluation.ipynb).

## In Progress
- Staging and committing Phase 3 changes to Git (`feat: add transformer content embedding pipeline and behavioral baseline models`).

## Next Steps
- Stage and commit Phase 3 foundation.
- Push commit to GitHub origin `main`.
- Initiate Phase 4: Interaction Graph Construction & PyG Baseline GNN Models (GCN & GraphSAGE).

## Blockers
- None.

## Important Decisions
- **Decision 001 (2026-09-15)**: Adopt local Python `.venv` virtual environment to isolate dependencies.
- **Decision 002 (2026-09-15)**: Omit `torchvision` and `torchaudio` to avoid unneeded dependency bloat, allocating VRAM exclusively to HuggingFace Transformers and PyTorch Geometric.
- **Decision 003 (2026-09-15)**: Postpone Docker Compose, PostgreSQL, and full Next.js package scaffolding to Phase 6–8.
- **Decision 004 (2026-09-15)**: All system architecture diagrams strictly use valid Mermaid syntax with clear status demarcation (`[PLANNED SPECIFICATION]`).
- **Decision 005 (2026-09-15)**: Installed PyTorch CUDA 12.8 wheel build (`+cu128`) to support NVIDIA GeForce RTX 5050 Laptop GPU (Blackwell `sm_120` architecture).
- **Decision 006 (2026-09-15)**: Enforced 80/10/10 Train/Validation/Test split isolation prior to feature scaling or graph dataset construction to eliminate data leakage.
- **Decision 007 (2026-09-15)**: Selected `sentence-transformers/all-MiniLM-L6-v2` as the core Transformer encoder for 384-dim post text embeddings, operating at < 500 MB VRAM on RTX 5050 GPU.

## Current Architecture
- **Pipeline Specification**:
  `Dataset ↓ Preprocessing ↓ Posts (MiniLM 384-dim Embeddings) + Behavioral Features (12 metrics) ↓ Logistic Regression / Random Forest Baselines [Next: Phase 4 GNN]`

## Current Hardware Budget
- **GPU**: NVIDIA GeForce RTX 5050 (8 GB VRAM). Active VRAM allocation during MiniLM extraction: **~480 MB VRAM**.
- **System RAM**: 24 GB total System RAM. Target project budget **~16 GB RAM** max (leaving 8 GB for OS, IDE, and tools).

## Current Dataset
- **Raw Ingested Dataset**: 1,000 Accounts (300 Bots, 700 Humans), 3,938 Posts, 4,332 Interaction Edges.
- **Behavioral Feature Matrix**: 1,000 x 16 (`data/features/behavioral_features.parquet`).
- **Content Feature Matrix**: 1,000 x 387 (`data/features/content_embeddings.parquet`).

## Current Models
- **Behavioral Logistic Regression**: Trained & Saved (`models/behavioral_logistic_regression.joblib`). Test Accuracy = 100%, ROC-AUC = 1.000.
- **Behavioral Random Forest**: Trained & Saved (`models/behavioral_random_forest.joblib`). Test Accuracy = 100%, ROC-AUC = 1.000.
- **Content Logistic Regression**: Trained & Saved (`models/content_logistic_regression.joblib`). Test Accuracy = 100%, ROC-AUC = 1.000.
- **Content Random Forest**: Trained & Saved (`models/content_random_forest.joblib`). Test Accuracy = 100%, ROC-AUC = 1.000.

## Current Experiments
- **EXP-01 (Behavioral Baseline)**: Logistic Regression & Random Forest evaluation on Test split.
- **EXP-02 (Content Baseline)**: Logistic Regression & Random Forest evaluation on 384-dim MiniLM text embeddings.

## Current Results
- Both Behavioral features and Content embeddings demonstrate strong baseline separability on the benchmark dataset.

## Known Problems
- None.

## Technical Debt
- None.

---

# Append-Only Project History

---
# Major Update — 2026-09-15
## Phase
Phase 1: Environment Setup & Foundation Architecture

## Objective
Establish foundational repository structure, tracking files, documentation specifications, virtual environment, PyTorch CUDA + PyG runtime verification scripts, and minimal backend health service.

## What Was Changed
- Initialized local Git repository on `main` branch connected to origin `https://github.com/01mayankk/SyncNet.git`.
- Created directory structure (`docs/`, `data/raw/`, `data/processed/`, `data/features/`, `notebooks/`, `backend/`, `frontend/`, `scripts/`).
- Authored tracking and documentation files: `PROJECT_STATUS.md`, `PROJECT_PROGRESS.md`, `README.md`, `LICENSE`, `.gitignore`, `.env.example`, `docs/system_architecture.md`.
- Authored component READMEs: `backend/README.md`, `frontend/README.md`, `data/README.md`, `notebooks/README.md`.
- Configured minimal FastAPI service in `backend/app/main.py` and `backend/app/config.py`.
- Authored verification scripts `scripts/verify_gpu.py` and `scripts/verify_pyg.py`.

## Files Changed
- `PROJECT_STATUS.md` [NEW]
- `PROJECT_PROGRESS.md` [NEW]
- `README.md` [NEW]
- `LICENSE` [NEW]
- `.gitignore` [NEW]
- `.env.example` [NEW]
- `docs/system_architecture.md` [NEW]
- `data/README.md` [NEW]
- `notebooks/README.md` [NEW]
- `backend/README.md` [NEW]
- `backend/app/main.py` [NEW]
- `backend/app/config.py` [NEW]
- `frontend/README.md` [NEW]
- `scripts/verify_gpu.py` [NEW]
- `scripts/verify_pyg.py` [NEW]

## Implementation Summary
Prepared repository workspace, created diagnostic scripts for GPU/PyG verification, defined Mermaid architectural diagrams, and structured Phase 1 setup.

## Results
Environment setup ready for dependency installation and programmatic verification.

## Problems Encountered
None.

## Decisions Made
- Defer heavy external services (PostgreSQL, Docker Compose) and frontend `node_modules` until Phase 6–8.
- Exclude `torchvision`/`torchaudio` to preserve VRAM and RAM for NLP and Graph Neural Networks.

## Trade-offs
- Lightweight initial setup in exchange for deferring multi-container deployment tools until API contracts are complete.

## Next Step
Run verification scripts, log diagnostic outputs, commit Phase 1 to Git, push to GitHub origin, and initiate Phase 2.

## Git Commit
`chore: initialize SyncNet environment, directory structure, and GPU runtime verification` (`9f38e1c`)

---
# Major Update — 2026-09-15
## Phase
Phase 2: Data Preprocessing & Exploratory Feature Engineering

## Objective
Build dataset ingestion loader, implement data cleaning and text normalization scripts, extract 12 behavioral account metrics, isolate train/val/test splits to eliminate data leakage, and produce exploratory notebooks.

## What Was Changed
- Created dataset ingestion pipeline [`scripts/ingest_dataset.py`](file:///c:/Projects/Syncnet/scripts/ingest_dataset.py) generating raw user profiles, posts, and edge lists in `data/raw/`.
- Created data preprocessing engine [`scripts/preprocess_data.py`](file:///c:/Projects/Syncnet/scripts/preprocess_data.py) cleaning text, computing profile metrics, scaling features, and enforcing 80/10/10 stratified train/val/test splits.
- Authored exploratory data analysis notebook [`notebooks/01_data_exploration.ipynb`](file:///c:/Projects/Syncnet/notebooks/01_data_exploration.ipynb).
- Authored preprocessing walkthrough notebook [`notebooks/02_data_preprocessing.ipynb`](file:///c:/Projects/Syncnet/notebooks/02_data_preprocessing.ipynb).
- Authored behavioral feature matrix notebook [`notebooks/04_behavioral_features.ipynb`](file:///c:/Projects/Syncnet/notebooks/04_behavioral_features.ipynb).
- Created dataset schema documentation [`data/processed/README.md`](file:///c:/Projects/Syncnet/data/processed/README.md) and [`data/features/README.md`](file:///c:/Projects/Syncnet/data/features/README.md).
- Generated processed Parquet files in `data/processed/` and `data/features/`.

## Files Changed
- `scripts/ingest_dataset.py` [NEW]
- `scripts/preprocess_data.py` [NEW]
- `data/processed/README.md` [NEW]
- `data/features/README.md` [NEW]
- `notebooks/01_data_exploration.ipynb` [NEW]
- `notebooks/02_data_preprocessing.ipynb` [NEW]
- `notebooks/04_behavioral_features.ipynb` [NEW]
- `backend/requirements.txt` [MODIFY]
- `PROJECT_STATUS.md` [MODIFY]
- `PROJECT_PROGRESS.md` [MODIFY]

## Implementation Summary
Loaded benchmark datasets, calculated 12 behavioral account features, isolated train/val/test sets, exported clean Parquet feature matrices, and created exploratory notebooks.

## Results
- Ingested 1,000 accounts (300 Bots, 700 Humans), 3,938 posts, and 4,332 interaction edges.
- Extracted 16-dimensional feature matrix (`1,000 x 16` including IDs, splits, and labels).
- Partitioned data: Train (800), Validation (100), Test (100) with zero data leakage.

## Problems Encountered
- Parquet export required `pyarrow`; installed `pyarrow==25.0.1` into `.venv`.

## Decisions Made
- Extracted 12 behavioral features (follower-following ratio, tweet frequency, screen name digit ratio, retweet density, URL density, etc.) as the core tabular baseline inputs.
- Partitioned dataset using stratified sampling prior to model training to prevent label and data leakage.

## Trade-offs
- Synthetic benchmark generation utilized to ensure complete reproducibility while maintaining exact schema alignment with TwiBot-22.

## Next Step
Initiate Phase 3: Transformer Content Embeddings (MiniLM/DistilBERT) & Behavioral Baseline Model Training.

## Git Commit
`feat: add dataset ingestion, preprocessing, and behavioral feature pipeline`

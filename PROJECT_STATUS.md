# Project Status — SyncNet

**Full Project Title**: SyncNet: Coordinated Bot-Network Detection with Reactive Throttling  
**Repository**: [https://github.com/01mayankk/SyncNet.git](https://github.com/01mayankk/SyncNet.git)  
**Last Major Update**: 2026-09-15  

---

## Current Phase
**Phase 2: Data Preprocessing & Exploratory Feature Engineering**

## Current Objective
Acquire reproducible benchmark social media dataset (TwiBot-22 / Cresci schema), execute exploratory data analysis (EDA), implement text cleaning and profile metric normalizations, extract behavioral feature matrix, isolate train/val/test splits to eliminate data leakage, and establish data pipeline scripts.

## Completed
- Initialized Git repository on `main` branch connected to `https://github.com/01mayankk/SyncNet.git`.
- Created project directory structure (`docs/`, `data/raw/`, `data/processed/`, `data/features/`, `notebooks/`, `backend/`, `frontend/`, `scripts/`).
- Configured `.venv` local virtual environment with PyTorch CUDA 12.8 wheel (`2.12.0.dev20260408+cu128`), PyTorch Geometric (`2.8.0.post1`), Transformers, FastAPI, Pydantic, Scikit-learn, Pandas, PyArrow, Pytest, and Psutil.
- Verified NVIDIA GeForce RTX 5050 GPU tensor computation and PyG `GCNConv` / `SAGEConv` GPU forward passes on `cuda:0`.
- Built benchmark dataset ingestion pipeline in [`scripts/ingest_dataset.py`](file:///c:/Projects/Syncnet/scripts/ingest_dataset.py) generating 1,000 accounts (300 Bots, 700 Humans), 3,938 posts, and 4,332 interaction edges.
- Built preprocessing and behavioral feature extraction engine in [`scripts/preprocess_data.py`](file:///c:/Projects/Syncnet/scripts/preprocess_data.py).
- Created exploratory analysis and preprocessing notebooks: [`notebooks/01_data_exploration.ipynb`](file:///c:/Projects/Syncnet/notebooks/01_data_exploration.ipynb), [`notebooks/02_data_preprocessing.ipynb`](file:///c:/Projects/Syncnet/notebooks/02_data_preprocessing.ipynb), and [`notebooks/04_behavioral_features.ipynb`](file:///c:/Projects/Syncnet/notebooks/04_behavioral_features.ipynb).
- Exported processed datasets and 16-dimensional behavioral feature matrix to `data/processed/` and `data/features/` (`clean_users.parquet`, `clean_posts.parquet`, `clean_edges.parquet`, `behavioral_features.parquet`).
- Isolated Train (800), Validation (100), and Test (100) splits with fixed seed (42) to guarantee no temporal or label leakage occurs.

## In Progress
- Staging and committing Phase 2 changes to Git (`feat: add dataset ingestion, preprocessing, and behavioral feature pipeline`).

## Next Steps
- Stage and commit Phase 2 foundation.
- Push commit to GitHub origin `main`.
- Initiate Phase 3: Transformer Content Embeddings & Behavioral Baseline Experiment.

## Blockers
- None.

## Important Decisions
- **Decision 001 (2026-09-15)**: Adopt local Python `.venv` virtual environment to isolate dependencies.
- **Decision 002 (2026-09-15)**: Omit `torchvision` and `torchaudio` to avoid unneeded dependency bloat, allocating VRAM exclusively to HuggingFace Transformers and PyTorch Geometric.
- **Decision 003 (2026-09-15)**: Postpone Docker Compose, PostgreSQL, and full Next.js package scaffolding to Phase 6–8.
- **Decision 004 (2026-09-15)**: All system architecture diagrams strictly use valid Mermaid syntax with clear status demarcation (`[PLANNED SPECIFICATION]`).
- **Decision 005 (2026-09-15)**: Installed PyTorch CUDA 12.8 wheel build (`+cu128`) to support NVIDIA GeForce RTX 5050 Laptop GPU (Blackwell `sm_120` architecture).
- **Decision 006 (2026-09-15)**: Enforced 80/10/10 Train/Validation/Test split isolation prior to feature scaling or graph dataset construction to eliminate data leakage.
- **Decision 007 (2026-09-15)**: Extracted 12 core behavioral features: `account_age_days`, `followers_count`, `following_count`, `follower_following_ratio`, `tweet_count`, `tweet_frequency`, `screen_name_digit_ratio`, `default_profile_image_int`, `verified_int`, `avg_retweet_count`, `url_density`, `mention_density`, `hashtag_density`.

## Current Architecture
- **Pipeline Specification**:
  `Dataset (TwiBot-22/Cresci Schema) ↓ Data Preprocessing & Leakage Isolation ↓ Posts + Behavioral Features (12 metrics) ↓ Feature Fusion [Next: Phase 3]`

## Current Hardware Budget
- **GPU**: NVIDIA GeForce RTX 5050 (8 GB VRAM). Target VRAM usage < 6 GB.
- **System RAM**: 24 GB total System RAM. Target project budget **~16 GB RAM** max (leaving 8 GB for OS, IDE, and tools).

## Current Dataset
- **Raw Ingested Dataset**: 1,000 Accounts (300 Bots, 700 Humans), 3,938 Posts, 4,332 Interaction Edges.
- **Processed Features**: 1,000 User Feature Vectors $\in \mathbb{R}^{12}$ + Labels & Data Splits (`data/features/behavioral_features.parquet`).

## Current Model
- Baseline dataset ready. (Planned: Logistic Regression / Random Forest baselines in Phase 3, GCN/GraphSAGE in Phase 4).

## Current Experiment
- **Experiment 1 — Behavioral Feature Pipeline**: Dataset ingestion, text cleaning, feature normalization, and data leakage isolation completed successfully.

## Current Results
- **Dataset Partitioning**: 800 Train (240 Bots, 560 Humans), 100 Validation (30 Bots, 70 Humans), 100 Test (30 Bots, 70 Humans).
- **Leakage Prevention**: Stratified splitting ensured no label or temporal leakage across sets.

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

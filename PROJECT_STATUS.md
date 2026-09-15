# Project Status — SyncNet

**Full Project Title**: SyncNet: Coordinated Bot-Network Detection with Reactive Throttling  
**Repository**: [https://github.com/01mayankk/SyncNet.git](https://github.com/01mayankk/SyncNet.git)  
**Last Major Update**: 2026-09-15  

---

## Current Phase
**Phase 1: Environment Setup & Foundation Architecture**

## Current Objective
Initialize clean project structure, establish `.venv` with PyTorch CUDA (RTX 5050) and PyTorch Geometric (PyG) runtime verification, maintain project progress tracking, document system architecture specifications, build minimal FastAPI health service, and set up Git remote origin.

## Completed
- Initialized Git repository on `main` branch connected to `https://github.com/01mayankk/SyncNet.git`.
- Created project directory structure (`docs/`, `data/raw/`, `data/processed/`, `data/features/`, `notebooks/`, `backend/`, `frontend/`, `scripts/`).
- Created baseline security and environment configuration templates (`.gitignore`, `.env.example`, `LICENSE`).
- Created project status (`PROJECT_STATUS.md`) and project progress tracking (`PROJECT_PROGRESS.md`).
- Designed initial system architecture specification using Mermaid (`docs/system_architecture.md`).
- Configured `.venv` local virtual environment with PyTorch (`2.12.0.dev20260408+cu128`), PyTorch Geometric (`2.8.0.post1`), Hugging Face Transformers, FastAPI, Pydantic, Scikit-learn, Pandas, Pytest, and Psutil.
- Verified NVIDIA GeForce RTX 5050 GPU tensor computation and VRAM memory bounds via `scripts/verify_gpu.py`.
- Verified PyTorch Geometric (PyG) `GCNConv` and `SAGEConv` GPU forward passes on `cuda:0` via `scripts/verify_pyg.py`.
- Recorded environment setup diagnostic log in `docs/environment_setup.log`.
- Configured minimal FastAPI health service (`backend/app/main.py`, `backend/app/config.py`).
- Exported pinned environment dependencies into `backend/requirements.txt`.

## In Progress
- Git staging, conventional commit (`chore: initialize SyncNet environment, directory structure, and GPU runtime verification`), and pushing Phase 1 foundation to GitHub `main` branch.

## Next Steps
- Stage and commit Phase 1 foundation (`chore: initialize SyncNet environment, directory structure, and GPU runtime verification`).
- Push commit to GitHub origin `main`.
- Initiate Phase 2: Data Preprocessing & Exploratory Feature Engineering.

## Blockers
- None.

## Important Decisions
- **Decision 001 (2026-09-15)**: Adopt local Python `.venv` virtual environment to isolate dependencies.
- **Decision 002 (2026-09-15)**: Omit `torchvision` and `torchaudio` to avoid unneeded dependency bloat, allocating VRAM exclusively to HuggingFace Transformers and PyTorch Geometric.
- **Decision 003 (2026-09-15)**: Postpone Docker Compose, PostgreSQL, and full Next.js package scaffolding to their respective implementation phases (Phase 6–8) to keep Phase 1 lean and reproducible.
- **Decision 004 (2026-09-15)**: All system architecture diagrams strictly use valid Mermaid syntax with clear status demarcation (`[PLANNED SPECIFICATION]`).
- **Decision 005 (2026-09-15)**: Installed PyTorch CUDA 12.8 wheel build (`+cu128`) to support NVIDIA GeForce RTX 5050 Laptop GPU (Blackwell `sm_120` architecture).
- **Decision 006 (2026-09-15)**: Reactive enforcement actions (throttling, escalation, decay) are strictly simulated within decision support state machine logic and will never execute actions against real social media platforms.

## Current Architecture
- **Pipeline Specification (Planned)**:
  `Dataset ↓ Data Preprocessing ↓ Posts (Transformer Embeddings) + Behavioral Features ↓ Feature Fusion ↓ Interaction Graph ↓ GraphSAGE / GCN ↓ Node Representations ↓ Cluster Detection ↓ Coordination Score ↓ Reactive Decision Layer (NORMAL, FLAGGED, THROTTLED, ESCALATED, DECAYING, APPEALED) ↓ FastAPI Backend ↓ Next.js Frontend`
- **Serving Stack**: Minimal FastAPI backend (`backend/app/main.py`) with `/health` endpoint verified.

## Current Hardware Budget
- **GPU**: NVIDIA GeForce RTX 5050 (8 GB VRAM). Verified operating at 80 MB VRAM baseline allocation.
- **System RAM**: 24 GB total System RAM. Target project budget **~16 GB RAM** max (leaving 8 GB for OS, IDE, and tools).

## Current Dataset
- None loaded yet (Planned: TwiBot-22 / Cresci-2017 reproducible subset for Phase 2).

## Current Model
- None trained yet (Planned: Logistic Regression / Random Forest baselines in Phase 3, GCN/GraphSAGE in Phase 4).

## Current Experiment
- **Experiment 0 — Environment Verification**: Verified PyTorch CUDA tensor execution (236.12 ms) and PyTorch Geometric `GCNConv` & `SAGEConv` GPU forward passes on `cuda:0`.

## Current Results
- **PyTorch CUDA**: 2000x2000 matrix multiplication executed on `cuda:0` in 236.12 ms.
- **PyG Conv Execution**: `GCNConv` `[100, 64] -> [100, 32]` and `SAGEConv` `[100, 64] -> [100, 32]` executed successfully on `cuda:0`.

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
`chore: initialize SyncNet environment, directory structure, and GPU runtime verification`

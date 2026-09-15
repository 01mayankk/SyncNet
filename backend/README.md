# Backend API Service — SyncNet

FastAPI microservice serving GNN inference, cluster coordination scores, and simulated reactive decision support APIs.

## Architecture & Structure
```text
backend/
├── README.md
├── requirements.txt
└── app/
    ├── main.py      # FastAPI application instance & health endpoint
    └── config.py    # Environment configuration & hardware settings
```

## Running locally (Phase 1 Baseline)
From the project root:
```bash
uvicorn backend.app.main:app --reload --port 8000
```
Health Check Endpoint: `http://127.0.0.1:8000/health`

## Planned API Endpoints (Phases 5–6)
- `GET /health`: Health check and GPU diagnostic status.
- `GET /api/v1/clusters`: List detected coordinated bot clusters.
- `GET /api/v1/clusters/{cluster_id}`: Detailed cluster metrics and account members.
- `GET /api/v1/graph`: Subgraph topology nodes and edges payload.
- `POST /api/v1/simulate/throttle`: Trigger simulated reactive throttling action.

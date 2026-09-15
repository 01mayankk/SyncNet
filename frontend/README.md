# Frontend Dashboard — SyncNet

Interactive decision-support web application built with Next.js App Router, Tailwind CSS, and network graph visualization components.

> [!NOTE]
> **Implementation Status**: `[PLANNED SPECIFICATION]`  
> Package initialization (`package.json`, Next.js scaffolding) will occur in **Phase 7** following backend API stabilization.

## Planned Features
- **Cluster Summary Dashboard**: Total accounts, flagged clusters, high-risk networks, average coordination scores.
- **Interactive Graph Canvas**: User-user interaction graph visualization rendering clusters and coordination edges (`react-force-graph` / D3).
- **Evidence Panel**: Detailed post timing metrics, content similarity scores, and behavioral anomaly indicators.
- **Reactive Control Simulation**: Interactive trigger interface for simulated throttling, escalation, decay interval testing, and appeal review.

## Environment Integration
- `NEXT_PUBLIC_API_URL`: Points to FastAPI backend (e.g. `http://127.0.0.1:8000`).
- Deployable on **Vercel** with zero code modifications.

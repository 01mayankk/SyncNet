# SyncNet Production Deployment Guide

## Overview
This document provides instructions for deploying **SyncNet: Coordinated Bot-Network Detection with Reactive Throttling** across local development, multi-container Docker Compose environments, and Vercel cloud hosting.

---

## 1. Local Development Execution

### Backend Microservice (FastAPI)
1. Activate virtual environment:
   ```powershell
   .venv\Scripts\activate
   ```
2. Launch Uvicorn development server:
   ```powershell
   uvicorn backend.app.main:app --reload --port 8000
   ```
3. Access API Interactive Docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Frontend Operations Dashboard (Next.js)
1. Navigate to frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Launch development server:
   ```bash
   npm run dev
   ```
4. Access dashboard at [http://localhost:3000](http://localhost:3000).

---

## 2. Multi-Container Docker Deployment

### Prerequisites
- Docker Engine & Docker Compose installed.

### Execution
1. Build and launch all services:
   ```bash
   docker-compose up --build -d
   ```
2. Verify container status:
   ```bash
   docker-compose ps
   ```
3. Inspect backend health:
   ```bash
   curl http://localhost:8000/health
   ```
4. Stop container stack:
   ```bash
   docker-compose down
   ```

---

## 3. Vercel Cloud Deployment

### Overview
The frontend Next.js App Router application can be deployed directly to Vercel.

### Steps
1. Connect repository `https://github.com/01mayankk/SyncNet.git` to Vercel.
2. Root Directory: `./` (Vercel automatically detects `vercel.json`).
3. Build Command: `cd frontend && npm run build`.
4. Deploy.

---

## 4. Verification & Testing

Run unit test suites prior to deployment:
```powershell
.venv\Scripts\python.exe -m pytest backend/tests/test_api_endpoints.py tests/test_reactive_engine.py
```

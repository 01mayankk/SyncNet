import time
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.api.v1 import api_v1_router
from backend.app.inference.gnn_inference import GNNInferenceService
from backend.app.reactive.reactive_engine import ReactiveEngine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm GNN inference and Reactive engine resources during startup
    GNNInferenceService.get_instance()
    ReactiveEngine.get_instance()
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="SyncNet Backend Service — Coordinated Bot-Network Detection with Reactive Throttling",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1_router, prefix="/api/v1")

@app.get("/")
def root():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "timestamp": time.time(),
    }

@app.get("/health")
def health_check():
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    vram_total_mb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2)
        if cuda_available
        else 0
    )
    
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "vram_total_mb": vram_total_mb,
        "system_ram_budget_gb": settings.MAX_SYSTEM_RAM_GB,
    }


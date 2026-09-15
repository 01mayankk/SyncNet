import time
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="SyncNet Backend Service — Coordinated Bot-Network Detection with Reactive Throttling",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

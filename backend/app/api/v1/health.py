import time
import torch
from fastapi import APIRouter
from backend.app.config import settings
from backend.app.inference.gnn_inference import GNNInferenceService
from backend.app.reactive.reactive_engine import ReactiveEngine

router = APIRouter()

@router.get("/health")
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
        "timestamp": time.time(),
    }

@router.get("/status")
def detailed_status():
    gnn_service = GNNInferenceService.get_instance()
    reactive_engine = ReactiveEngine.get_instance()
    
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": time.time(),
        "gnn_service": gnn_service.get_status(),
        "reactive_engine": {
            "loaded": reactive_engine.is_loaded,
            "cluster_count": len(reactive_engine.get_all_clusters()),
        },
        "device": str(gnn_service.device),
    }

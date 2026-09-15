from fastapi import APIRouter
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.clusters import router as clusters_router
from backend.app.api.v1.graph import router as graph_router
from backend.app.api.v1.reactive import router as reactive_router

api_v1_router = APIRouter()

api_v1_router.include_router(health_router, prefix="", tags=["Health & Status"])
api_v1_router.include_router(clusters_router, prefix="/clusters", tags=["Clusters"])
api_v1_router.include_router(graph_router, prefix="/graph", tags=["Graph & Nodes"])
api_v1_router.include_router(reactive_router, prefix="/reactive", tags=["Reactive Simulation"])

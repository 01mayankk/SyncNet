from fastapi import APIRouter, HTTPException
from backend.app.schemas.reactive import (
    ThrottleRequest,
    DecayRequest,
    AppealRequest,
    SimulationActionResponse,
    EntityStatusResponse,
)
from backend.app.reactive.reactive_engine import ReactiveEngine

router = APIRouter()

@router.post("/simulate/throttle", response_model=SimulationActionResponse)
def simulate_throttle(req: ThrottleRequest):
    engine = ReactiveEngine.get_instance()
    result = engine.simulate_throttle(
        entity_id=req.entity_id,
        throttle_rate_pct=req.throttle_rate_pct,
        reason=req.reason or "",
    )
    return SimulationActionResponse(**result)

@router.post("/simulate/decay", response_model=SimulationActionResponse)
def simulate_decay(req: DecayRequest):
    engine = ReactiveEngine.get_instance()
    result = engine.simulate_decay(
        entity_id=req.entity_id,
        time_delta=req.time_delta,
        lambda_decay=req.lambda_decay,
    )
    return SimulationActionResponse(**result)

@router.post("/simulate/appeal", response_model=SimulationActionResponse)
def simulate_appeal(req: AppealRequest):
    engine = ReactiveEngine.get_instance()
    result = engine.simulate_appeal(
        entity_id=req.entity_id,
        reason=req.reason,
    )
    return SimulationActionResponse(**result)

@router.get("/states/{entity_id}", response_model=EntityStatusResponse)
def get_entity_state(entity_id: str):
    engine = ReactiveEngine.get_instance()
    status = engine.get_entity_status(entity_id)
    return EntityStatusResponse(**status)

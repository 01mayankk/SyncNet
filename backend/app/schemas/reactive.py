from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class ThrottleRequest(BaseModel):
    entity_id: str
    entity_type: str = Field(default="cluster", description="cluster or account")
    throttle_rate_pct: float = Field(default=80.0, ge=0.0, le=100.0)
    reason: Optional[str] = "Manual simulation throttle trigger"

class DecayRequest(BaseModel):
    entity_id: str
    time_delta: float = Field(default=1.0, ge=0.1, description="Simulated time elapsed in hours")
    lambda_decay: float = Field(default=0.1, ge=0.01, le=1.0)

class AppealRequest(BaseModel):
    entity_id: str
    reason: str = Field(..., min_length=5, description="Appeal submission justification")
    evidence_url: Optional[str] = None

class SimulationActionResponse(BaseModel):
    entity_id: str
    entity_type: str
    previous_state: str
    current_state: str
    coordination_score: float
    active_throttle_pct: float
    timestamp: float
    message: str

class EntityStatusResponse(BaseModel):
    entity_id: str
    entity_type: str
    current_state: str
    coordination_score: float
    active_throttle_pct: float
    is_throttled: bool
    history_count: int

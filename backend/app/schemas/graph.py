from typing import List, Optional
from pydantic import BaseModel, Field

class NodeItem(BaseModel):
    id: str
    label: int
    predicted_label: int
    bot_probability: float
    cluster_id: str
    embedding_preview: List[float] = Field(default_factory=list, max_length=4)

class EdgeItem(BaseModel):
    source: str
    target: str
    weight: float = 1.0

class GraphTopologyResponse(BaseModel):
    total_nodes: int
    total_edges: int
    timestamp: float
    nodes: List[NodeItem]
    edges: List[EdgeItem]

class NodeDetailResponse(BaseModel):
    id: str
    label: int
    predicted_label: int
    bot_probability: float
    cluster_id: str
    degree: int
    embedding_64d: List[float]

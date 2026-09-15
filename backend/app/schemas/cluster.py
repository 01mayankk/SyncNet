from typing import List, Optional
from pydantic import BaseModel, Field

class ClusterMember(BaseModel):
    account_id: str
    ground_truth: int
    predicted_label: int
    bot_probability: float

class ClusterSummary(BaseModel):
    cluster_id: str
    size: int
    num_bots: int
    bot_ratio: float
    edge_density: float
    content_similarity: float
    coordination_score: float
    risk_tier: str
    status: str

class ClusterDetail(ClusterSummary):
    members: List[ClusterMember] = Field(default_factory=list)

class ClusterListResponse(BaseModel):
    total_clusters: int
    timestamp: float
    clusters: List[ClusterSummary]

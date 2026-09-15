import time
from fastapi import APIRouter, HTTPException
from backend.app.schemas.cluster import ClusterListResponse, ClusterSummary, ClusterDetail, ClusterMember
from backend.app.reactive.reactive_engine import ReactiveEngine
from backend.app.inference.gnn_inference import GNNInferenceService

router = APIRouter()

@router.get("", response_model=ClusterListResponse)
def get_clusters():
    engine = ReactiveEngine.get_instance()
    clusters_data = engine.get_all_clusters()
    
    summaries = []
    for c in clusters_data:
        summaries.append(
            ClusterSummary(
                cluster_id=c["cluster_id"],
                size=c["size"],
                num_bots=c["num_bots"],
                bot_ratio=c["bot_ratio"],
                edge_density=c["edge_density"],
                content_similarity=c["content_similarity"],
                coordination_score=c["coordination_score"],
                risk_tier=c["risk_tier"],
                status=c["status"],
            )
        )
        
    return ClusterListResponse(
        total_clusters=len(summaries),
        timestamp=time.time(),
        clusters=summaries,
    )

@router.get("/{cluster_id}", response_model=ClusterDetail)
def get_cluster_detail(cluster_id: str):
    engine = ReactiveEngine.get_instance()
    cluster = engine.get_cluster(cluster_id)
    
    if not cluster:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found.")
        
    gnn_service = GNNInferenceService.get_instance()
    members = []
    
    for uid in cluster.get("member_user_ids", []):
        pred = gnn_service.get_node_prediction(uid)
        if pred:
            members.append(
                ClusterMember(
                    account_id=uid,
                    ground_truth=pred["ground_truth"],
                    predicted_label=pred["predicted_label"],
                    bot_probability=pred["bot_probability"],
                )
            )
        else:
            members.append(
                ClusterMember(
                    account_id=uid,
                    ground_truth=1 if "bot" in uid.lower() else 0,
                    predicted_label=1 if "bot" in uid.lower() else 0,
                    bot_probability=0.95 if "bot" in uid.lower() else 0.05,
                )
            )
            
    return ClusterDetail(
        cluster_id=cluster["cluster_id"],
        size=cluster["size"],
        num_bots=cluster["num_bots"],
        bot_ratio=cluster["bot_ratio"],
        edge_density=cluster["edge_density"],
        content_similarity=cluster["content_similarity"],
        coordination_score=cluster["coordination_score"],
        risk_tier=cluster["risk_tier"],
        status=cluster["status"],
        members=members,
    )

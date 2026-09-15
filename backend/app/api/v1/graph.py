import os
import time
import pandas as pd
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from backend.app.schemas.graph import GraphTopologyResponse, NodeItem, EdgeItem, NodeDetailResponse
from backend.app.inference.gnn_inference import GNNInferenceService
from backend.app.reactive.reactive_engine import ReactiveEngine

router = APIRouter()

DATA_PROC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "processed"))

@router.get("", response_model=GraphTopologyResponse)
def get_graph_topology(limit_edges: int = Query(default=500, ge=1, le=5000)):
    gnn_service = GNNInferenceService.get_instance()
    reactive_engine = ReactiveEngine.get_instance()
    
    # 1. Fetch Nodes
    raw_nodes = gnn_service.get_all_node_predictions()
    clusters = reactive_engine.get_all_clusters()
    
    # Map user_id -> cluster_id
    user_to_cluster = {}
    for c in clusters:
        cid = c["cluster_id"]
        for uid in c.get("member_user_ids", []):
            user_to_cluster[uid] = cid
            
    nodes: List[NodeItem] = []
    for n in raw_nodes:
        uid = n["user_id"]
        nodes.append(
            NodeItem(
                id=uid,
                label=n["ground_truth"],
                predicted_label=n["predicted_label"],
                bot_probability=n["bot_probability"],
                cluster_id=user_to_cluster.get(uid, "cluster_00"),
                embedding_preview=n["embedding_preview"],
            )
        )
        
    # 2. Fetch Edges
    edges_path = os.path.join(DATA_PROC_DIR, "clean_edges.parquet")
    edges: List[EdgeItem] = []
    
    if os.path.exists(edges_path):
        df_edges = pd.read_parquet(edges_path)
        if limit_edges < len(df_edges):
            df_edges = df_edges.head(limit_edges)
            
        for _, row in df_edges.iterrows():
            edges.append(
                EdgeItem(
                    source=str(row["source"]),
                    target=str(row["target"]),
                    weight=1.0,
                )
            )
            
    return GraphTopologyResponse(
        total_nodes=len(nodes),
        total_edges=len(edges),
        timestamp=time.time(),
        nodes=nodes,
        edges=edges,
    )

@router.get("/nodes/{account_id}", response_model=NodeDetailResponse)
def get_node_detail(account_id: str):
    gnn_service = GNNInferenceService.get_instance()
    reactive_engine = ReactiveEngine.get_instance()
    
    pred = gnn_service.get_node_prediction(account_id)
    if not pred:
        raise HTTPException(status_code=404, detail=f"Node/account {account_id} not found.")
        
    # Calculate degree
    edges_path = os.path.join(DATA_PROC_DIR, "clean_edges.parquet")
    degree = 0
    if os.path.exists(edges_path):
        df_edges = pd.read_parquet(edges_path)
        degree = int(((df_edges["source"] == account_id) | (df_edges["target"] == account_id)).sum())
        
    # Cluster ID lookup
    cluster_id = "cluster_00"
    for c in reactive_engine.get_all_clusters():
        if account_id in c.get("member_user_ids", []):
            cluster_id = c["cluster_id"]
            break
            
    return NodeDetailResponse(
        id=account_id,
        label=pred["ground_truth"],
        predicted_label=pred["predicted_label"],
        bot_probability=pred["bot_probability"],
        cluster_id=cluster_id,
        degree=degree,
        embedding_64d=pred["embedding_64d"],
    )

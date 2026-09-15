"""
SyncNet Cluster Detection & Coordination Scoring Engine
=======================================================
Performs graph cluster detection on GNN node embeddings Z, calculates intra-cluster edge density,
semantic text similarity, and formulates the Cluster Coordination Score S_coord(C_k) in [0.0, 1.0].

Outputs:
- data/features/cluster_results.json
- data/features/cluster_summary.parquet (.csv)
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

DATA_PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DATA_FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
RANDOM_SEED = 42

def detect_clusters_and_score():
    print("=" * 70)
    print("SYNCNET — CLUSTER DETECTION & COORDINATION SCORING ENGINE")
    print("=" * 70)
    
    fused_emb_path = os.path.join(DATA_FEAT_DIR, "fused_node_embeddings.parquet")
    content_emb_path = os.path.join(DATA_FEAT_DIR, "content_embeddings.parquet")
    edges_path = os.path.join(DATA_PROC_DIR, "clean_edges.parquet")
    
    if not os.path.exists(fused_emb_path) or not os.path.exists(content_emb_path) or not os.path.exists(edges_path):
        raise FileNotFoundError("Fused embeddings, content embeddings, or edge list not found.")
        
    df_fused = pd.read_parquet(fused_emb_path)
    df_content = pd.read_parquet(content_emb_path)
    df_edges = pd.read_parquet(edges_path)
    
    # Extract 64-dim structural node embeddings (z_00 .. z_63)
    z_cols = [c for c in df_fused.columns if c.startswith("z_")]
    Z_matrix = df_fused[z_cols].values
    
    # 1. Cluster Detection via Agglomerative Clustering
    num_clusters = 8
    clustering = AgglomerativeClustering(n_clusters=num_clusters, metric="cosine", linkage="average")
    df_fused["cluster_id"] = clustering.fit_predict(Z_matrix)
    
    print(f"Detected {num_clusters} interaction clusters across {len(df_fused)} nodes.")
    
    # Index content embeddings for cosine similarity calculations
    cnt_cols = [c for c in df_content.columns if c.startswith("emb_")]
    content_map = df_content.set_index("user_id")[cnt_cols].T.to_dict("list")
    
    # Pre-index edges for fast intra-cluster density lookup
    cluster_membership = df_fused.set_index("user_id")["cluster_id"].to_dict()
    bot_label_map = df_fused.set_index("user_id")["label"].to_dict()
    pred_prob_map = df_fused.set_index("user_id")["pred_prob"].to_dict()
    
    cluster_results = []
    
    # 2. Compute Cluster Metrics & Coordination Scores
    for c_id in range(num_clusters):
        members = df_fused[df_fused["cluster_id"] == c_id]["user_id"].tolist()
        num_members = len(members)
        member_set = set(members)
        
        # Intra-cluster interaction edges
        intra_edges = 0
        for _, edge in df_edges.iterrows():
            if edge["source"] in member_set and edge["target"] in member_set:
                intra_edges += 1
                
        # Calculate Edge Density
        max_possible_edges = max(1, num_members * (num_members - 1))
        edge_density = min(1.0, intra_edges / max_possible_edges * 10) # Scaled density metric
        
        # Calculate Intra-Cluster Content Similarity
        if num_members > 1:
            member_content_vecs = np.array([content_map[u] for u in members if u in content_map])
            cos_sim_matrix = cosine_similarity(member_content_vecs)
            # Upper triangle mean similarity (excluding self-similarity)
            upper_tri = cos_sim_matrix[np.triu_indices_from(cos_sim_matrix, k=1)]
            avg_content_sim = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0
        else:
            avg_content_sim = 0.0
            
        avg_content_sim = max(0.0, float(avg_content_sim))
        
        # Calculate Bot / High-Risk Ratio
        bot_count = sum(1 for u in members if bot_label_map.get(u, 0) == 1 or pred_prob_map.get(u, 0.0) >= 0.5)
        bot_ratio = bot_count / max(1, num_members)
        
        # Formulate Cluster Coordination Score:
        # S_coord(C_k) = 0.40 * edge_density + 0.35 * content_sim + 0.25 * bot_ratio
        raw_score = (0.40 * edge_density) + (0.35 * avg_content_sim) + (0.25 * bot_ratio)
        coordination_score = round(float(np.clip(raw_score, 0.0, 1.0)), 4)
        
        # Assign Simulation State based on score thresholds
        if coordination_score >= 0.85:
            sim_state = "ESCALATED"
        elif coordination_score >= 0.60:
            sim_state = "THROTTLED"
        elif coordination_score >= 0.35:
            sim_state = "FLAGGED"
        else:
            sim_state = "NORMAL"
            
        cluster_doc = {
            "cluster_id": f"cluster_{c_id:02d}",
            "member_count": num_members,
            "bot_count": bot_count,
            "bot_ratio": round(bot_ratio, 4),
            "intra_edges": intra_edges,
            "edge_density": round(edge_density, 4),
            "content_similarity": round(avg_content_sim, 4),
            "coordination_score": coordination_score,
            "simulated_state": sim_state,
            "member_user_ids": members[:10], # Top 10 sample members
        }
        cluster_results.append(cluster_doc)
        
    df_cluster_summary = pd.DataFrame(cluster_results)
    
    print("\n--- CLUSTER COORDINATION SCORES SUMMARY ---")
    print(df_cluster_summary[["cluster_id", "member_count", "bot_ratio", "edge_density", "content_similarity", "coordination_score", "simulated_state"]].to_string(index=False))
    
    # Export Cluster Results
    json_path = os.path.join(DATA_FEAT_DIR, "cluster_results.json")
    parquet_path = os.path.join(DATA_FEAT_DIR, "cluster_summary.parquet")
    csv_path = os.path.join(DATA_FEAT_DIR, "cluster_summary.csv")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cluster_results, f, indent=2)
        
    df_cluster_summary.to_parquet(parquet_path, index=False)
    df_cluster_summary.to_csv(csv_path, index=False)
    
    print(f"\n[x] SUCCESS: Saved cluster results to {json_path}")
    print("=" * 70)
    return cluster_results

if __name__ == "__main__":
    detect_clusters_and_score()

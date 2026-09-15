import os
import json
import time
import math
from typing import Dict, Any, List, Optional

DATA_FEAT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "features"))

class ReactiveEngine:
    _instance: Optional["ReactiveEngine"] = None

    def __init__(self):
        self.clusters: Dict[str, Dict[str, Any]] = {}
        self.action_logs: List[Dict[str, Any]] = []
        self.is_loaded = False
        self.load_clusters()

    @classmethod
    def get_instance(cls) -> "ReactiveEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_clusters(self):
        json_path = os.path.join(DATA_FEAT_DIR, "cluster_results.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    cluster_list = json.load(f)
                    for item in cluster_list:
                        cid = item["cluster_id"]
                        self.clusters[cid] = {
                            "cluster_id": cid,
                            "size": item.get("member_count", 0),
                            "num_bots": item.get("bot_count", 0),
                            "bot_ratio": item.get("bot_ratio", 0.0),
                            "edge_density": item.get("edge_density", 0.0),
                            "content_similarity": item.get("content_similarity", 0.0),
                            "coordination_score": item.get("coordination_score", 0.0),
                            "base_score": item.get("coordination_score", 0.0),
                            "risk_tier": "HIGH" if item.get("coordination_score", 0.0) >= 0.6 else ("MEDIUM" if item.get("coordination_score", 0.0) >= 0.35 else "LOW"),
                            "status": item.get("simulated_state", "NORMAL"),
                            "active_throttle_pct": 80.0 if item.get("simulated_state") in ["THROTTLED", "ESCALATED"] else 0.0,
                            "member_user_ids": item.get("member_user_ids", []),
                            "updated_at": time.time(),
                        }
                self.is_loaded = True
            except Exception as e:
                print(f"[!] ReactiveEngine load error: {e}")
                self.is_loaded = False
        else:
            self.is_loaded = False

    def get_all_clusters(self) -> List[Dict[str, Any]]:
        return list(self.clusters.values())

    def get_cluster(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        return self.clusters.get(cluster_id)

    def simulate_throttle(self, entity_id: str, throttle_rate_pct: float = 80.0, reason: str = "") -> Dict[str, Any]:
        cluster = self.clusters.get(entity_id)
        prev_state = cluster["status"] if cluster else "UNKNOWN"
        prev_score = cluster["coordination_score"] if cluster else 0.0

        if cluster:
            cluster["status"] = "THROTTLED" if throttle_rate_pct < 100.0 else "ESCALATED"
            cluster["active_throttle_pct"] = float(throttle_rate_pct)
            cluster["updated_at"] = time.time()
            new_state = cluster["status"]
            current_score = cluster["coordination_score"]
        else:
            prev_state = "NORMAL"
            new_state = "THROTTLED"
            current_score = 0.65

        log_entry = {
            "entity_id": entity_id,
            "entity_type": "cluster" if entity_id.startswith("cluster_") else "account",
            "previous_state": prev_state,
            "current_state": new_state,
            "coordination_score": current_score,
            "active_throttle_pct": throttle_rate_pct,
            "timestamp": time.time(),
            "message": f"Successfully simulated rate throttling ({throttle_rate_pct}%). Reason: {reason}",
        }
        self.action_logs.append(log_entry)
        return log_entry

    def simulate_decay(self, entity_id: str, time_delta: float = 1.0, lambda_decay: float = 0.1) -> Dict[str, Any]:
        cluster = self.clusters.get(entity_id)
        if not cluster:
            return {
                "entity_id": entity_id,
                "entity_type": "cluster",
                "previous_state": "UNKNOWN",
                "current_state": "DECAYING",
                "coordination_score": 0.20,
                "active_throttle_pct": 0.0,
                "timestamp": time.time(),
                "message": f"Entity {entity_id} not found in state store.",
            }

        prev_state = cluster["status"]
        prev_score = cluster["coordination_score"]
        
        # Exponential decay: S(t) = max(S_min, S(0) * e^(-lambda * t))
        s_min = 0.10
        new_score = round(max(s_min, prev_score * math.exp(-lambda_decay * time_delta)), 4)
        cluster["coordination_score"] = new_score

        if new_score < 0.35:
            cluster["status"] = "NORMAL"
            cluster["active_throttle_pct"] = 0.0
        elif new_score < 0.60:
            cluster["status"] = "DECAYING"
            cluster["active_throttle_pct"] = round(cluster["active_throttle_pct"] * 0.5, 1)
        else:
            cluster["status"] = "DECAYING"

        cluster["updated_at"] = time.time()

        log_entry = {
            "entity_id": entity_id,
            "entity_type": "cluster",
            "previous_state": prev_state,
            "current_state": cluster["status"],
            "coordination_score": new_score,
            "active_throttle_pct": cluster["active_throttle_pct"],
            "timestamp": time.time(),
            "message": f"Simulated risk score decay over {time_delta}h (lambda={lambda_decay}). Score reduced from {prev_score} to {new_score}.",
        }
        self.action_logs.append(log_entry)
        return log_entry

    def simulate_appeal(self, entity_id: str, reason: str = "") -> Dict[str, Any]:
        cluster = self.clusters.get(entity_id)
        prev_state = cluster["status"] if cluster else "THROTTLED"

        if cluster:
            cluster["status"] = "APPEALED"
            cluster["active_throttle_pct"] = 0.0
            cluster["updated_at"] = time.time()
            curr_state = "APPEALED"
            curr_score = cluster["coordination_score"]
        else:
            curr_state = "APPEALED"
            curr_score = 0.30

        log_entry = {
            "entity_id": entity_id,
            "entity_type": "cluster" if entity_id.startswith("cluster_") else "account",
            "previous_state": prev_state,
            "current_state": curr_state,
            "coordination_score": curr_score,
            "active_throttle_pct": 0.0,
            "timestamp": time.time(),
            "message": f"Appeal submitted successfully. Simulated throttle lifted pending review. Reason: {reason}",
        }
        self.action_logs.append(log_entry)
        return log_entry

    def get_entity_status(self, entity_id: str) -> Dict[str, Any]:
        cluster = self.clusters.get(entity_id)
        if cluster:
            history_count = sum(1 for log in self.action_logs if log["entity_id"] == entity_id)
            return {
                "entity_id": entity_id,
                "entity_type": "cluster",
                "current_state": cluster["status"],
                "coordination_score": cluster["coordination_score"],
                "active_throttle_pct": cluster["active_throttle_pct"],
                "is_throttled": cluster["active_throttle_pct"] > 0,
                "history_count": history_count,
            }
        return {
            "entity_id": entity_id,
            "entity_type": "account" if not entity_id.startswith("cluster_") else "cluster",
            "current_state": "NORMAL",
            "coordination_score": 0.15,
            "active_throttle_pct": 0.0,
            "is_throttled": False,
            "history_count": 0,
        }

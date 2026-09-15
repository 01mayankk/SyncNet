import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))
DATA_FEAT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "features"))

class FusionGraphSAGENet(nn.Module):
    def __init__(self, in_channels: int = 396, hidden_channels: int = 64, out_channels: int = 2, dropout: float = 0.3):
        super(FusionGraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def get_node_embeddings(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return F.relu(h)

    def forward(self, x, edge_index):
        h = self.get_node_embeddings(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.classifier(h)
        return F.log_softmax(logits, dim=1)

class GNNInferenceService:
    _instance: Optional["GNNInferenceService"] = None

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model: Optional[FusionGraphSAGENet] = None
        self.embeddings_df: Optional[pd.DataFrame] = None
        self.graph_data: Optional[Any] = None
        self.user_id_to_idx: Dict[str, int] = {}
        self.is_loaded = False

    @classmethod
    def get_instance(cls) -> "GNNInferenceService":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load_resources()
        return cls._instance

    def load_resources(self):
        try:
            ckpt_path = os.path.join(MODELS_DIR, "feature_fusion_sage.pt")
            emb_path = os.path.join(DATA_FEAT_DIR, "fused_node_embeddings.parquet")
            graph_path = os.path.join(DATA_FEAT_DIR, "graph_data.pt")

            if os.path.exists(emb_path):
                self.embeddings_df = pd.read_parquet(emb_path)
                for idx, uid in enumerate(self.embeddings_df["user_id"]):
                    self.user_id_to_idx[str(uid)] = idx

            if os.path.exists(graph_path):
                self.graph_data = torch.load(graph_path, weights_only=False)

            if os.path.exists(ckpt_path):
                self.model = FusionGraphSAGENet(in_channels=396, hidden_channels=64, out_channels=2)
                self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()

            self.is_loaded = True
        except Exception as e:
            print(f"[!] Warning: GNNInferenceService load_resources failed: {e}")
            self.is_loaded = False

    def get_status(self) -> Dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "device": str(self.device),
            "total_nodes": len(self.embeddings_df) if self.embeddings_df is not None else 0,
            "model_path_exists": os.path.exists(os.path.join(MODELS_DIR, "feature_fusion_sage.pt")),
        }

    def get_node_prediction(self, user_id: str) -> Optional[Dict[str, Any]]:
        if self.embeddings_df is None or user_id not in self.user_id_to_idx:
            return None
        
        idx = self.user_id_to_idx[user_id]
        row = self.embeddings_df.iloc[idx]
        
        z_cols = [c for c in self.embeddings_df.columns if c.startswith("z_")]
        emb_64d = row[z_cols].values.astype(float).tolist()
        
        return {
            "user_id": user_id,
            "ground_truth": int(row["label"]),
            "predicted_label": int(row["pred_label"]),
            "bot_probability": float(row["pred_prob"]),
            "embedding_preview": emb_64d[:4],
            "embedding_64d": emb_64d,
        }

    def get_all_node_predictions(self) -> List[Dict[str, Any]]:
        if self.embeddings_df is None:
            return []
        
        results = []
        z_cols = [c for c in self.embeddings_df.columns if c.startswith("z_")]
        
        for _, row in self.embeddings_df.iterrows():
            emb_64d = row[z_cols].values.astype(float).tolist()
            results.append({
                "user_id": str(row["user_id"]),
                "ground_truth": int(row["label"]),
                "predicted_label": int(row["pred_label"]),
                "bot_probability": float(row["pred_prob"]),
                "embedding_preview": emb_64d[:4],
            })
        return results

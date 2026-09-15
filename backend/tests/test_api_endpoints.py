import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "SyncNet"
    assert data["status"] == "online"

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "cuda_available" in data

def test_v1_status_endpoint():
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "SyncNet"
    assert "gnn_service" in data
    assert "reactive_engine" in data

def test_clusters_endpoint():
    response = client.get("/api/v1/clusters")
    assert response.status_code == 200
    data = response.json()
    assert data["total_clusters"] >= 1
    assert len(data["clusters"]) >= 1

def test_cluster_detail_endpoint():
    response = client.get("/api/v1/clusters/cluster_00")
    assert response.status_code == 200
    data = response.json()
    assert data["cluster_id"] == "cluster_00"
    assert "members" in data

def test_graph_topology_endpoint():
    response = client.get("/api/v1/graph?limit_edges=100")
    assert response.status_code == 200
    data = response.json()
    assert data["total_nodes"] >= 1
    assert data["total_edges"] >= 1
    assert len(data["nodes"]) >= 1
    assert len(data["edges"]) <= 100

def test_node_detail_endpoint():
    response = client.get("/api/v1/graph/nodes/usr_0001")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "usr_0001"
    assert len(data["embedding_64d"]) == 64


def test_simulate_throttle_endpoint():
    payload = {
        "entity_id": "cluster_00",
        "entity_type": "cluster",
        "throttle_rate_pct": 85.0,
        "reason": "Test throttle simulation",
    }
    response = client.post("/api/v1/reactive/simulate/throttle", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["entity_id"] == "cluster_00"
    assert data["current_state"] in ["THROTTLED", "ESCALATED"]
    assert data["active_throttle_pct"] == 85.0

def test_simulate_decay_endpoint():
    payload = {
        "entity_id": "cluster_00",
        "time_delta": 2.0,
        "lambda_decay": 0.2,
    }
    response = client.post("/api/v1/reactive/simulate/decay", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["entity_id"] == "cluster_00"
    assert "coordination_score" in data

def test_simulate_appeal_endpoint():
    payload = {
        "entity_id": "cluster_00",
        "reason": "False positive verification evidence submitted.",
    }
    response = client.post("/api/v1/reactive/simulate/appeal", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["entity_id"] == "cluster_00"
    assert data["current_state"] == "APPEALED"
    assert data["active_throttle_pct"] == 0.0

def test_entity_state_endpoint():
    response = client.get("/api/v1/reactive/states/cluster_00")
    assert response.status_code == 200
    data = response.json()
    assert data["entity_id"] == "cluster_00"
    assert "is_throttled" in data

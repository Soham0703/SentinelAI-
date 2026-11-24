import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_presets_endpoint():
    response = client.get("/api/v1/presets")
    assert response.status_code == 200
    presets = response.json()
    assert len(presets) >= 4
    assert any(p["name"] == "Normal Coffee & Retail" for p in presets)


def test_predict_single_endpoint():
    payload = {
        "step": 14,
        "type": "PAYMENT",
        "amount": 35.0,
        "nameOrig": "C12345",
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 965.0,
        "nameDest": "M_HUB_01",
        "oldbalanceDest": 5000.0,
        "newbalanceDest": 5035.0
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "risk_tier" in data
    assert "explainability" in data
    assert "model_breakdown" in data
    assert "id" in data


def test_predict_batch_endpoint():
    payload = {
        "transactions": [
            {
                "step": 14,
                "type": "PAYMENT",
                "amount": 50.0,
                "nameOrig": "C1",
                "oldbalanceOrg": 500.0,
                "newbalanceOrig": 450.0,
                "nameDest": "M1",
                "oldbalanceDest": 1000.0,
                "newbalanceDest": 1050.0
            },
            {
                "step": 3,
                "type": "TRANSFER",
                "amount": 90000.0,
                "nameOrig": "C2",
                "oldbalanceOrg": 90000.0,
                "newbalanceOrig": 0.0,
                "nameDest": "MULE_001",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0
            }
        ]
    }
    response = client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert "flagged_fraud_count" in data


def test_metrics_endpoint():
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "curves" in data
    assert "feature_importance" in data
    assert "XGBoost" in data["models"]
    assert "Random Forest" in data["models"]
    assert "PyTorch GNN" in data["models"]


def test_transactions_endpoint():
    payload = {
        "step": 12,
        "type": "TRANSFER",
        "amount": 1200.0,
        "nameOrig": "TEST_USER_1",
        "oldbalanceOrg": 2000.0,
        "newbalanceOrig": 800.0,
        "nameDest": "TEST_MERCHANT_1",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 1200.0
    }
    client.post("/api/v1/predict", json=payload)

    response = client.get("/api/v1/transactions?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "transactions" in data
    assert len(data["transactions"]) > 0


def test_graph_endpoint():
    response = client.get("/api/v1/graph/subgraph?limit=20")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert "stats" in data


def test_stats_endpoint():
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_transactions" in data
    assert "fraud_detected_count" in data
    assert "prevented_loss_volume" in data
    assert "risk_distribution" in data

import pytest
import os
from app.ml.ensemble import SentinelEnsembleEngine
from app.core.config import settings


def test_ensemble_initialization():
    ensemble = SentinelEnsembleEngine(settings.ARTIFACTS_DIR)
    assert ensemble.is_loaded is True
    assert ensemble.xgb_model is not None
    assert ensemble.rf_model is not None
    assert ensemble.lr_model is not None
    assert ensemble.iso_model is not None
    assert ensemble.gnn_detector is not None


def test_predict_benign_transaction():
    ensemble = SentinelEnsembleEngine(settings.ARTIFACTS_DIR)
    benign_tx = {
        "step": 14,
        "type": "PAYMENT",
        "amount": 25.50,
        "nameOrig": "C9999",
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 4974.50,
        "nameDest": "M_HUB_01",
        "oldbalanceDest": 10000.0,
        "newbalanceDest": 10025.50
    }

    result = ensemble.predict_transaction(benign_tx)

    assert "risk_score" in result
    assert "risk_tier" in result
    assert "model_breakdown" in result
    assert result["risk_score"] < 50.0
    assert result["recommendation"] in ["APPROVE", "STEP_UP_AUTH"]


def test_predict_critical_fraud():
    ensemble = SentinelEnsembleEngine(settings.ARTIFACTS_DIR)
    fraud_tx = {
        "step": 3,
        "type": "TRANSFER",
        "amount": 150000.0,
        "nameOrig": "VICTIM_001",
        "oldbalanceOrg": 150000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "MULE_001",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
    }

    result = ensemble.predict_transaction(fraud_tx)

    assert result["risk_score"] >= 60.0
    assert result["is_fraud_flagged"] is True
    assert result["recommendation"] in ["MANUAL_REVIEW", "DECLINE"]
    assert len(result["rules_triggered"]) > 0

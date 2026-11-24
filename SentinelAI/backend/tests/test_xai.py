import pytest
from app.ml.rules_engine import RulesEngine
from app.ml.explainer import FraudExplainer
from app.core.config import settings


def test_rules_engine_drain_trigger():
    tx = {
        "amount": 45000.0,
        "oldbalanceOrg": 45000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C999",
        "type": "TRANSFER",
        "step": 12
    }
    rules = RulesEngine.evaluate(tx)
    rule_ids = [r["rule_id"] for r in rules]
    assert "RULE_001_TOTAL_DRAIN" in rule_ids


def test_rules_engine_mule_trigger():
    tx = {
        "amount": 5000.0,
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 5000.0,
        "nameDest": "MULE_005",
        "type": "CASH_OUT",
        "step": 12
    }
    rules = RulesEngine.evaluate(tx)
    rule_ids = [r["rule_id"] for r in rules]
    assert "RULE_002_MULE_HUB_MATCH" in rule_ids


def test_explainer_initialization():
    explainer = FraudExplainer(settings.ARTIFACTS_DIR)
    assert explainer.explainer is not None
    assert len(explainer.feature_names) > 0

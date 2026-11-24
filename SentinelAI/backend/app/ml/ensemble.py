import os
import joblib
import numpy as np
from typing import Dict, Any, List

from app.ml.data_pipeline import FinancialFraudDataPipeline
from app.ml.gnn_model import GraphFraudDetector
from app.ml.explainer import FraudExplainer
from app.ml.rules_engine import RulesEngine


class SentinelEnsembleEngine:
    def __init__(self, artifacts_dir: str = "app/ml/artifacts"):
        self.artifacts_dir = artifacts_dir
        self.pipeline = FinancialFraudDataPipeline()
        self.lr_model = None
        self.rf_model = None
        self.xgb_model = None
        self.iso_model = None
        self.gnn_detector = None
        self.explainer = FraudExplainer(artifacts_dir)
        self.is_loaded = False
        self.load_models()

    def load_models(self):
        try:
            preproc_path = os.path.join(self.artifacts_dir, "preprocessor.joblib")
            lr_path = os.path.join(self.artifacts_dir, "logistic_regression.joblib")
            rf_path = os.path.join(self.artifacts_dir, "random_forest.joblib")
            xgb_path = os.path.join(self.artifacts_dir, "xgboost.joblib")
            iso_path = os.path.join(self.artifacts_dir, "isolation_forest.joblib")
            gnn_path = os.path.join(self.artifacts_dir, "gnn_model.pt")

            if os.path.exists(preproc_path):
                self.pipeline.load(preproc_path)
            if os.path.exists(lr_path):
                self.lr_model = joblib.load(lr_path)
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
            if os.path.exists(xgb_path):
                self.xgb_model = joblib.load(xgb_path)
            if os.path.exists(iso_path):
                self.iso_model = joblib.load(iso_path)
            if os.path.exists(gnn_path):
                self.gnn_detector = GraphFraudDetector()
                feature_dim = len(self.pipeline.feature_names) if self.pipeline.feature_names else 18
                self.gnn_detector.load(gnn_path, edge_feature_dim=feature_dim)

            self.explainer = FraudExplainer(self.artifacts_dir)
            self.is_loaded = True
        except Exception:
            self.explainer = FraudExplainer(self.artifacts_dir)

    def predict_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load_models()

        x_trans = self.pipeline.transform_single(transaction)

        lr_prob = float(self.lr_model.predict_proba(x_trans)[0, 1]) if self.lr_model else 0.1
        rf_prob = float(self.rf_model.predict_proba(x_trans)[0, 1]) if self.rf_model else 0.1
        xgb_prob = float(self.xgb_model.predict_proba(x_trans)[0, 1]) if self.xgb_model else 0.1

        if self.iso_model:
            raw_score = float(self.iso_model.decision_function(x_trans)[0])
            anomaly_score = float(np.clip(1.0 - (raw_score + 0.25) / 0.5, 0.0, 1.0))
        else:
            anomaly_score = 0.05

        gnn_prob = self.gnn_detector.predict_single(transaction, x_trans) if self.gnn_detector else 0.1

        rule_hits = RulesEngine.evaluate(transaction)
        rule_impact = sum(r.get("score_impact", 0.0) for r in rule_hits)

        base_ensemble_score = (
            0.40 * xgb_prob +
            0.25 * rf_prob +
            0.15 * gnn_prob +
            0.10 * lr_prob +
            0.10 * anomaly_score
        )
        final_risk_score = min(1.0, base_ensemble_score + (rule_impact * 0.35))
        risk_percentage = round(final_risk_score * 100.0, 2)

        if risk_percentage >= 85.0:
            risk_tier = "CRITICAL"
            recommendation = "DECLINE"
        elif risk_percentage >= 60.0:
            risk_tier = "HIGH"
            recommendation = "MANUAL_REVIEW"
        elif risk_percentage >= 25.0:
            risk_tier = "MEDIUM"
            recommendation = "STEP_UP_AUTH"
        else:
            risk_tier = "LOW"
            recommendation = "APPROVE"

        xai_explanation = self.explainer.explain_transaction(transaction, x_trans)

        return {
            "transaction": transaction,
            "risk_score": risk_percentage,
            "risk_tier": risk_tier,
            "recommendation": recommendation,
            "is_fraud_flagged": risk_percentage >= 50.0,
            "model_breakdown": {
                "xgboost_probability": round(xgb_prob, 4),
                "random_forest_probability": round(rf_prob, 4),
                "pytorch_gnn_probability": round(gnn_prob, 4),
                "logistic_regression_probability": round(lr_prob, 4),
                "isolation_forest_anomaly_score": round(anomaly_score, 4)
            },
            "rules_triggered": rule_hits,
            "explainability": xai_explanation
        }

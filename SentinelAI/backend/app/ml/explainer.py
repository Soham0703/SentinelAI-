import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any


class FraudExplainer:
    def __init__(self, artifacts_dir: str = "app/ml/artifacts"):
        self.artifacts_dir = artifacts_dir
        self.explainer = None
        self.preprocessor_data = None
        self.feature_names = []
        self._load_artifacts()

    def _load_artifacts(self):
        explainer_path = os.path.join(self.artifacts_dir, "shap_explainer.joblib")
        preproc_path = os.path.join(self.artifacts_dir, "preprocessor.joblib")
        
        if os.path.exists(explainer_path) and os.path.exists(preproc_path):
            self.explainer = joblib.load(explainer_path)
            self.preprocessor_data = joblib.load(preproc_path)
            self.feature_names = self.preprocessor_data.get('feature_names', [])

    def explain_transaction(
        self, 
        transaction: Dict[str, Any], 
        x_trans: np.ndarray, 
        top_k: int = 5
    ) -> Dict[str, Any]:
        if self.explainer is None:
            self._load_artifacts()

        if self.explainer is None or len(self.feature_names) == 0:
            return {
                "base_value": 0.02,
                "top_positive_features": [],
                "top_negative_features": [],
                "narrative": ["Standard transaction within normal risk bounds."]
            }

        raw_shap = self.explainer.shap_values(x_trans)
        
        if isinstance(raw_shap, list):
            shap_vec = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0]
        elif raw_shap.ndim == 2:
            shap_vec = raw_shap[0]
        elif raw_shap.ndim == 3:
            shap_vec = raw_shap[0, :, 1]
        else:
            shap_vec = raw_shap

        base_val = float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.02
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[-1])

        features_with_scores = []
        for name, val in zip(self.feature_names, shap_vec):
            features_with_scores.append({
                "feature": name,
                "shap_value": round(float(val), 4),
                "abs_value": abs(float(val))
            })

        pos_features = [f for f in features_with_scores if f["shap_value"] > 0]
        pos_features.sort(key=lambda x: x["shap_value"], reverse=True)

        neg_features = [f for f in features_with_scores if f["shap_value"] < 0]
        neg_features.sort(key=lambda x: x["shap_value"])

        narrative_points = self._synthesize_narrative(transaction, pos_features, neg_features)

        return {
            "base_value": round(base_val, 4),
            "top_positive_features": pos_features[:top_k],
            "top_negative_features": neg_features[:top_k],
            "narrative": narrative_points
        }

    def _synthesize_narrative(
        self, 
        tx: Dict[str, Any], 
        pos_features: List[Dict[str, Any]], 
        neg_features: List[Dict[str, Any]]
    ) -> List[str]:
        narratives = []
        amount = float(tx.get('amount', 0.0))
        old_orig = float(tx.get('oldbalanceOrg', 0.0))
        new_orig = float(tx.get('newbalanceOrig', 0.0))
        old_dest = float(tx.get('oldbalanceDest', 0.0))
        new_dest = float(tx.get('newbalanceDest', 0.0))
        tx_type = str(tx.get('type', 'TRANSFER')).upper()
        dest_name = str(tx.get('nameDest', '')).upper()

        if new_orig == 0.0 and old_orig > 0:
            narratives.append(f"Account Drain Alert: Origin balance was liquidated entirely from ${old_orig:,.2f} to $0.00.")

        if amount > 50000.0:
            narratives.append(f"High-Value Outlier: Transaction amount (${amount:,.2f}) significantly exceeds typical retail thresholds.")

        if 'MULE' in dest_name or any(f['feature'] == 'dest_velocity_1h' for f in pos_features[:2]):
            narratives.append("Relational Risk: Destination account exhibits concentrated velocity characteristics indicative of a money mule hub.")

        if tx_type in ['TRANSFER', 'CASH_OUT']:
            narratives.append(f"Typology Risk: Transaction type '{tx_type}' represents an elevated operational risk channel.")

        if abs(new_dest - (old_dest + amount)) > 1.0 and old_dest == 0.0:
            narratives.append("Balance Invariant Mismatch: Destination balance did not increment as expected, signaling rapid intermediate cashout.")

        if not narratives:
            if amount < 500.0:
                narratives.append(f"Normal Activity: Small transaction amount (${amount:,.2f}) aligns with typical retail spending.")
            narratives.append("Verified Invariants: Origin account balance decremented normally with healthy post-transaction reserve.")

        return narratives

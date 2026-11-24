import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, roc_curve
)
import shap

from app.ml.data_pipeline import FinancialFraudDataPipeline
from app.ml.gnn_model import GraphFraudDetector


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall_curve, precision_curve))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }
    }


def train_and_evaluate_all(
    n_samples: int = 30000, 
    artifacts_dir: str = "app/ml/artifacts"
) -> Dict[str, Any]:
    os.makedirs(artifacts_dir, exist_ok=True)
    
    pipeline = FinancialFraudDataPipeline(random_state=42)
    raw_df = pipeline.generate_benchmark_dataset(n_samples=n_samples, fraud_ratio=0.02)
    X_train_trans, X_test_trans, y_train, y_test, df_train, df_test = pipeline.fit_transform_split(raw_df, test_size=0.2)

    pipeline.save(os.path.join(artifacts_dir, "preprocessor.joblib"))
    df_test.to_csv(os.path.join(artifacts_dir, "test_transactions_sample.csv"), index=False)

    results = {}
    curves_data = {}

    lr_model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        C=0.1,
        random_state=42
    )
    lr_model.fit(X_train_trans, y_train)
    lr_prob = lr_model.predict_proba(X_test_trans)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)
    results["Logistic Regression"] = calculate_metrics(y_test, lr_pred, lr_prob)
    joblib.dump(lr_model, os.path.join(artifacts_dir, "logistic_regression.joblib"))

    rf_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_trans, y_train)
    rf_prob = rf_model.predict_proba(X_test_trans)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)
    results["Random Forest"] = calculate_metrics(y_test, rf_pred, rf_prob)
    joblib.dump(rf_model, os.path.join(artifacts_dir, "random_forest.joblib"))

    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = n_neg / (n_pos + 1e-5)

    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        eval_metric='aucpr',
        n_jobs=-1
    )
    xgb_model.fit(X_train_trans, y_train)
    xgb_prob = xgb_model.predict_proba(X_test_trans)[:, 1]
    xgb_pred = (xgb_prob >= 0.5).astype(int)
    results["XGBoost"] = calculate_metrics(y_test, xgb_pred, xgb_prob)
    joblib.dump(xgb_model, os.path.join(artifacts_dir, "xgboost.joblib"))

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.02,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train_trans)
    iso_scores_raw = iso_forest.decision_function(X_test_trans)
    iso_prob = 1.0 - (iso_scores_raw - iso_scores_raw.min()) / (iso_scores_raw.max() - iso_scores_raw.min() + 1e-8)
    iso_pred = (iso_forest.predict(X_test_trans) == -1).astype(int)
    results["Isolation Forest"] = calculate_metrics(y_test, iso_pred, iso_prob)
    joblib.dump(iso_forest, os.path.join(artifacts_dir, "isolation_forest.joblib"))

    gnn_detector = GraphFraudDetector(hidden_dim=32, lr=0.008, epochs=25)
    gnn_detector.fit(df_train, X_train_trans, y_train)
    gnn_prob = gnn_detector.predict_proba(df_test, X_test_trans)
    gnn_pred = (gnn_prob >= 0.5).astype(int)
    results["PyTorch GNN"] = calculate_metrics(y_test, gnn_pred, gnn_prob)
    gnn_detector.save(os.path.join(artifacts_dir, "gnn_model.pt"))

    for model_name, prob_arr in [
        ("Logistic Regression", lr_prob),
        ("Random Forest", rf_prob),
        ("XGBoost", xgb_prob),
        ("Isolation Forest", iso_prob),
        ("PyTorch GNN", gnn_prob)
    ]:
        p_curve, r_curve, _ = precision_recall_curve(y_test, prob_arr)
        fpr_curve, tpr_curve, _ = roc_curve(y_test, prob_arr)
        
        idx_pr = np.linspace(0, len(p_curve) - 1, 25, dtype=int)
        idx_roc = np.linspace(0, len(fpr_curve) - 1, 25, dtype=int)
        
        curves_data[model_name] = {
            "pr_curve": [{"recall": round(float(r_curve[i]), 3), "precision": round(float(p_curve[i]), 3)} for i in idx_pr],
            "roc_curve": [{"fpr": round(float(fpr_curve[i]), 3), "tpr": round(float(tpr_curve[i]), 3)} for i in idx_roc]
        }

    explainer = shap.TreeExplainer(xgb_model)
    joblib.dump(explainer, os.path.join(artifacts_dir, "shap_explainer.joblib"))

    sample_background = X_test_trans[:300]
    shap_values = explainer.shap_values(sample_background)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    feature_importances = []
    for fname, importance in zip(pipeline.feature_names, mean_abs_shap):
        feature_importances.append({
            "feature": fname,
            "importance": round(float(importance), 4)
        })
    feature_importances.sort(key=lambda x: x["importance"], reverse=True)

    with open(os.path.join(artifacts_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(artifacts_dir, "evaluation_curves.json"), "w") as f:
        json.dump(curves_data, f, indent=2)

    with open(os.path.join(artifacts_dir, "feature_importance.json"), "w") as f:
        json.dump(feature_importances, f, indent=2)

    return results


if __name__ == "__main__":
    train_and_evaluate_all()

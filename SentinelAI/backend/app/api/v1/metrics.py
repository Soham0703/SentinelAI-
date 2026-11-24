from fastapi import APIRouter
import os
import json
from app.core.config import settings
from app.db.schemas import MetricsResponse

router = APIRouter(prefix="", tags=["Metrics"])


@router.get("/metrics", response_model=MetricsResponse)
def get_model_metrics():
    artifacts_dir = settings.ARTIFACTS_DIR
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    curves_path = os.path.join(artifacts_dir, "evaluation_curves.json")
    feat_path = os.path.join(artifacts_dir, "feature_importance.json")

    models_data = {}
    curves_data = {}
    feat_data = []

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            models_data = json.load(f)

    if os.path.exists(curves_path):
        with open(curves_path, "r") as f:
            curves_data = json.load(f)

    if os.path.exists(feat_path):
        with open(feat_path, "r") as f:
            feat_data = json.load(f)

    return {
        "models": models_data,
        "curves": curves_data,
        "feature_importance": feat_data
    }

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class TransactionInput(BaseModel):
    step: int = Field(default=12)
    type: str = Field(default="TRANSFER")
    amount: float = Field(..., gt=0)
    nameOrig: str = Field(default="C10042")
    oldbalanceOrg: float = Field(default=0.0, ge=0)
    newbalanceOrig: float = Field(default=0.0, ge=0)
    nameDest: str = Field(default="MULE_001")
    oldbalanceDest: float = Field(default=0.0, ge=0)
    newbalanceDest: float = Field(default=0.0, ge=0)


class RuleTriggerSchema(BaseModel):
    rule_id: str
    severity: str
    score_impact: float
    description: str


class FeatureImpactSchema(BaseModel):
    feature: str
    shap_value: float
    abs_value: float


class ExplainabilitySchema(BaseModel):
    base_value: float
    top_positive_features: List[FeatureImpactSchema]
    top_negative_features: List[FeatureImpactSchema]
    narrative: List[str]


class ModelScoresSchema(BaseModel):
    xgboost_probability: float
    random_forest_probability: float
    pytorch_gnn_probability: float
    logistic_regression_probability: float
    isolation_forest_anomaly_score: float


class PredictionResponse(BaseModel):
    id: Optional[int] = None
    transaction: Dict[str, Any]
    risk_score: float
    risk_tier: str
    recommendation: str
    is_fraud_flagged: bool
    model_breakdown: ModelScoresSchema
    rules_triggered: List[RuleTriggerSchema]
    explainability: ExplainabilitySchema


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionInput]


class BatchPredictionResponse(BaseModel):
    total_processed: int
    flagged_fraud_count: int
    fraud_rate_pct: float
    predictions: List[PredictionResponse]


class TransactionRecordSchema(BaseModel):
    id: int
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float
    isFraud: Optional[int] = None
    predictedFraud: int
    riskScore: float
    riskTier: str
    recommendation: str
    modelScores: Optional[Dict[str, Any]] = None
    shapValues: Optional[List[Any]] = None
    narrative: Optional[List[str]] = None
    rulesTriggered: Optional[List[Any]] = None
    createdAt: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class TransactionListResponse(BaseModel):
    total: int
    page: int
    limit: int
    transactions: List[TransactionRecordSchema]


class ConfusionMatrixSchema(BaseModel):
    tp: int
    fp: int
    tn: int
    fn: int


class ModelMetricSchema(BaseModel):
    precision: float
    recall: float
    f1_score: float
    pr_auc: float
    roc_auc: float
    specificity: float
    confusion_matrix: ConfusionMatrixSchema


class CurvePointPR(BaseModel):
    recall: float
    precision: float


class CurvePointROC(BaseModel):
    fpr: float
    tpr: float


class ModelCurveData(BaseModel):
    pr_curve: List[CurvePointPR]
    roc_curve: List[CurvePointROC]


class MetricsResponse(BaseModel):
    models: Dict[str, ModelMetricSchema]
    curves: Dict[str, ModelCurveData]
    feature_importance: List[Dict[str, Any]]


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    risk_score: float
    degree: int


class GraphEdge(BaseModel):
    source: str
    target: str
    amount: float
    type: str
    is_fraud: bool
    risk_score: float


class GraphNetworkResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    stats: Dict[str, Any]


class DashboardStatsResponse(BaseModel):
    total_transactions: int
    fraud_detected_count: int
    fraud_rate_pct: float
    total_volume_analyzed: float
    prevented_loss_volume: float
    risk_distribution: Dict[str, int]
    recommendation_distribution: Dict[str, int]
    model_count: int

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.database import get_db
from app.db.models import TransactionRecord
from app.db.schemas import DashboardStatsResponse

router = APIRouter(prefix="", tags=["Stats"])


@router.get("/stats", response_model=DashboardStatsResponse)
def get_dashboard_stats(db: Session = Depends(get_db)):
    total_tx = db.query(func.count(TransactionRecord.id)).scalar() or 0
    fraud_count = db.query(func.count(TransactionRecord.id)).filter(TransactionRecord.predictedFraud == 1).scalar() or 0
    
    total_vol = db.query(func.sum(TransactionRecord.amount)).scalar() or 0.0
    prevented_loss = db.query(func.sum(TransactionRecord.amount)).filter(TransactionRecord.predictedFraud == 1).scalar() or 0.0

    fraud_rate = round((fraud_count / total_tx * 100.0) if total_tx > 0 else 0.0, 2)

    tiers_query = db.query(TransactionRecord.riskTier, func.count(TransactionRecord.id)).group_by(TransactionRecord.riskTier).all()
    risk_dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for tier, count in tiers_query:
        if tier:
            risk_dist[tier.upper()] = count

    recs_query = db.query(TransactionRecord.recommendation, func.count(TransactionRecord.id)).group_by(TransactionRecord.recommendation).all()
    rec_dist = {"APPROVE": 0, "STEP_UP_AUTH": 0, "MANUAL_REVIEW": 0, "DECLINE": 0}
    for rec, count in recs_query:
        if rec:
            rec_dist[rec.upper()] = count

    return {
        "total_transactions": total_tx,
        "fraud_detected_count": fraud_count,
        "fraud_rate_pct": fraud_rate,
        "total_volume_analyzed": round(float(total_vol), 2),
        "prevented_loss_volume": round(float(prevented_loss), 2),
        "risk_distribution": risk_dist,
        "recommendation_distribution": rec_dist,
        "model_count": 5
    }

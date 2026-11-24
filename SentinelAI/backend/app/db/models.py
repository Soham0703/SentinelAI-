from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime, timezone
from app.db.database import Base


class TransactionRecord(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    step = Column(Integer, default=12)
    type = Column(String(32), index=True, nullable=False)
    amount = Column(Float, nullable=False)
    nameOrig = Column(String(64), index=True, nullable=False)
    oldbalanceOrg = Column(Float, default=0.0)
    newbalanceOrig = Column(Float, default=0.0)
    nameDest = Column(String(64), index=True, nullable=False)
    oldbalanceDest = Column(Float, default=0.0)
    newbalanceDest = Column(Float, default=0.0)
    
    isFraud = Column(Integer, nullable=True)
    predictedFraud = Column(Integer, default=0)
    riskScore = Column(Float, default=0.0)
    riskTier = Column(String(16), index=True, default="LOW")
    recommendation = Column(String(32), default="APPROVE")

    modelScores = Column(JSON, default=dict)
    shapValues = Column(JSON, default=list)
    narrative = Column(JSON, default=list)
    rulesTriggered = Column(JSON, default=list)
    createdAt = Column(DateTime, default=lambda: datetime.now(timezone.utc))

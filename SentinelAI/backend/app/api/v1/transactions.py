from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from app.db.database import get_db
from app.db.models import TransactionRecord
from app.db.schemas import TransactionListResponse, TransactionRecordSchema

router = APIRouter(prefix="", tags=["Transactions"])


@router.get("/transactions", response_model=TransactionListResponse)
def get_transactions(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    risk_tier: Optional[str] = Query(None),
    tx_type: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    is_fraud_only: bool = Query(False),
    db: Session = Depends(get_db)
):
    query = db.query(TransactionRecord)

    if risk_tier:
        query = query.filter(TransactionRecord.riskTier == risk_tier.upper())

    if tx_type:
        query = query.filter(TransactionRecord.type == tx_type.upper())

    if is_fraud_only:
        query = query.filter(TransactionRecord.predictedFraud == 1)

    if search:
        search_fmt = f"%{search.strip()}%"
        query = query.filter(
            (TransactionRecord.nameOrig.ilike(search_fmt)) |
            (TransactionRecord.nameDest.ilike(search_fmt))
        )

    total = query.count()
    records = query.order_by(TransactionRecord.id.desc()).offset((page - 1) * limit).limit(limit).all()

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "transactions": records
    }


@router.get("/transactions/{tx_id}", response_model=TransactionRecordSchema)
def get_transaction_by_id(tx_id: int, db: Session = Depends(get_db)):
    record = db.query(TransactionRecord).filter(TransactionRecord.id == tx_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Transaction with ID {tx_id} not found.")
    return record

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
import io
import pandas as pd
from typing import List, Dict, Any

from app.db.database import get_db
from app.db.models import TransactionRecord
from app.db.schemas import (
    TransactionInput, PredictionResponse, 
    BatchPredictionRequest, BatchPredictionResponse
)
from app.ml.ensemble import SentinelEnsembleEngine

router = APIRouter(prefix="", tags=["Inference"])
ensemble_engine = SentinelEnsembleEngine()


@router.get("/presets", response_model=List[Dict[str, Any]])
def get_attack_presets():
    return [
        {
            "name": "Normal Coffee & Retail",
            "category": "BENIGN",
            "description": "Routine small-dollar purchase with normal balance decrement and recognized merchant.",
            "data": {
                "step": 14,
                "type": "PAYMENT",
                "amount": 42.50,
                "nameOrig": "C10482",
                "oldbalanceOrg": 3450.00,
                "newbalanceOrig": 3407.50,
                "nameDest": "M_HUB_04",
                "oldbalanceDest": 45000.00,
                "newbalanceDest": 45042.50
            }
        },
        {
            "name": "Full Account Drain (Bust-Out)",
            "category": "CRITICAL_FRAUD",
            "description": "Total balance liquidation ($185,000) wiped to zero in a single unauthorized TRANSFER.",
            "data": {
                "step": 3,
                "type": "TRANSFER",
                "amount": 185000.00,
                "nameOrig": "VICTIM_4021",
                "oldbalanceOrg": 185000.00,
                "newbalanceOrig": 0.00,
                "nameDest": "C99812",
                "oldbalanceDest": 0.00,
                "newbalanceDest": 0.00
            }
        },
        {
            "name": "Coordinated Mule Hub Cash-Out",
            "category": "CRITICAL_FRAUD",
            "description": "High-velocity transfer funneled into a known money mule ring account.",
            "data": {
                "step": 2,
                "type": "CASH_OUT",
                "amount": 76500.00,
                "nameOrig": "VICTIM_8829",
                "oldbalanceOrg": 76500.00,
                "newbalanceOrig": 0.00,
                "nameDest": "MULE_004",
                "oldbalanceDest": 12000.00,
                "newbalanceDest": 0.00
            }
        },
        {
            "name": "Structuring / Smurfing Threshold",
            "category": "SUSPICIOUS",
            "description": "Transaction strategically sized just under the $10,000 regulatory reporting threshold.",
            "data": {
                "step": 18,
                "type": "TRANSFER",
                "amount": 9850.00,
                "nameOrig": "C55120",
                "oldbalanceOrg": 15000.00,
                "newbalanceOrig": 5150.00,
                "nameDest": "C44910",
                "oldbalanceDest": 500.00,
                "newbalanceDest": 10350.00
            }
        }
    ]


@router.post("/predict", response_model=PredictionResponse)
def predict_single(transaction: TransactionInput, db: Session = Depends(get_db)):
    tx_dict = transaction.model_dump()
    result = ensemble_engine.predict_transaction(tx_dict)

    db_record = TransactionRecord(
        step=tx_dict.get("step", 12),
        type=tx_dict.get("type", "TRANSFER"),
        amount=tx_dict.get("amount", 0.0),
        nameOrig=tx_dict.get("nameOrig", "UNKNOWN"),
        oldbalanceOrg=tx_dict.get("oldbalanceOrg", 0.0),
        newbalanceOrig=tx_dict.get("newbalanceOrig", 0.0),
        nameDest=tx_dict.get("nameDest", "UNKNOWN"),
        oldbalanceDest=tx_dict.get("oldbalanceDest", 0.0),
        newbalanceDest=tx_dict.get("newbalanceDest", 0.0),
        predictedFraud=1 if result["is_fraud_flagged"] else 0,
        riskScore=result["risk_score"],
        riskTier=result["risk_tier"],
        recommendation=result["recommendation"],
        modelScores=result["model_breakdown"],
        shapValues=result["explainability"]["top_positive_features"] + result["explainability"]["top_negative_features"],
        narrative=result["explainability"]["narrative"],
        rulesTriggered=result["rules_triggered"]
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    result["id"] = db_record.id
    return result


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest, db: Session = Depends(get_db)):
    predictions = []
    fraud_count = 0

    for item in payload.transactions:
        tx_dict = item.model_dump()
        res = ensemble_engine.predict_transaction(tx_dict)
        if res["is_fraud_flagged"]:
            fraud_count += 1
        predictions.append(res)

    total = len(payload.transactions)
    fraud_pct = round((fraud_count / total * 100.0) if total > 0 else 0.0, 2)

    return {
        "total_processed": total,
        "flagged_fraud_count": fraud_count,
        "fraud_rate_pct": fraud_pct,
        "predictions": predictions
    }


@router.post("/predict/upload-csv", response_model=BatchPredictionResponse)
async def upload_csv_batch(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {e}")

    required_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'nameOrig', 'nameDest', 'type']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns in CSV: {missing}")

    transactions = []
    for _, row in df.head(200).iterrows():
        tx = {
            "step": int(row.get('step', 12)),
            "type": str(row.get('type', 'TRANSFER')),
            "amount": float(row.get('amount', 0.0)),
            "nameOrig": str(row.get('nameOrig', 'C_CSV')),
            "oldbalanceOrg": float(row.get('oldbalanceOrg', 0.0)),
            "newbalanceOrig": float(row.get('newbalanceOrig', 0.0)),
            "nameDest": str(row.get('nameDest', 'M_CSV')),
            "oldbalanceDest": float(row.get('oldbalanceDest', 0.0)),
            "newbalanceDest": float(row.get('newbalanceDest', 0.0))
        }
        transactions.append(TransactionInput(**tx))

    return predict_batch(BatchPredictionRequest(transactions=transactions), db=db)

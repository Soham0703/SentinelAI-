import os
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from app.core.config import settings
from app.db.database import engine, Base, SessionLocal
from app.db.models import TransactionRecord
from app.api.router import api_router
from app.ml.ensemble import SentinelEnsembleEngine

Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = SessionLocal()
    try:
        count = db.query(TransactionRecord).count()
        if count == 0:
            sample_path = os.path.join(settings.ARTIFACTS_DIR, "test_transactions_sample.csv")
            if os.path.exists(sample_path):
                sample_df = pd.read_csv(sample_path).head(120)
                ensemble = SentinelEnsembleEngine(settings.ARTIFACTS_DIR)
                
                records = []
                for _, row in sample_df.iterrows():
                    tx = {
                        "step": int(row.get('step', 12)),
                        "type": str(row.get('type', 'TRANSFER')),
                        "amount": float(row.get('amount', 100.0)),
                        "nameOrig": str(row.get('nameOrig', 'C1000')),
                        "oldbalanceOrg": float(row.get('oldbalanceOrg', 0.0)),
                        "newbalanceOrig": float(row.get('newbalanceOrig', 0.0)),
                        "nameDest": str(row.get('nameDest', 'M1000')),
                        "oldbalanceDest": float(row.get('oldbalanceDest', 0.0)),
                        "newbalanceDest": float(row.get('newbalanceDest', 0.0))
                    }
                    res = ensemble.predict_transaction(tx)
                    
                    record = TransactionRecord(
                        step=tx["step"],
                        type=tx["type"],
                        amount=tx["amount"],
                        nameOrig=tx["nameOrig"],
                        oldbalanceOrg=tx["oldbalanceOrg"],
                        newbalanceOrig=tx["newbalanceOrig"],
                        nameDest=tx["nameDest"],
                        oldbalanceDest=tx["oldbalanceDest"],
                        newbalanceDest=tx["newbalanceDest"],
                        isFraud=int(row.get('isFraud', 0)),
                        predictedFraud=1 if res["is_fraud_flagged"] else 0,
                        riskScore=res["risk_score"],
                        riskTier=res["risk_tier"],
                        recommendation=res["recommendation"],
                        modelScores=res["model_breakdown"],
                        shapValues=res["explainability"]["top_positive_features"] + res["explainability"]["top_negative_features"],
                        narrative=res["explainability"]["narrative"],
                        rulesTriggered=res["rules_triggered"]
                    )
                    records.append(record)
                
                db.bulk_save_objects(records)
                db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_dashboard():
        index_file = os.path.join(static_dir, "index.html")
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>SentinelAI API Running</h1>"


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "SentinelAI Risk Intelligence Engine",
        "version": settings.VERSION,
        "database": "connected"
    }

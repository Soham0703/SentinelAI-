from fastapi import APIRouter
from app.api.v1.predict import router as predict_router
from app.api.v1.transactions import router as transactions_router
from app.api.v1.metrics import router as metrics_router
from app.api.v1.graph import router as graph_router
from app.api.v1.stats import router as stats_router

api_router = APIRouter()
api_router.include_router(predict_router)
api_router.include_router(transactions_router)
api_router.include_router(metrics_router)
api_router.include_router(graph_router)
api_router.include_router(stats_router)

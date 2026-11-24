import os
from typing import List


class Settings:
    PROJECT_NAME: str = "SentinelAI – Intelligent Fraud Detection & Risk Intelligence System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "sentinel")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "sentinelpass")
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "sentinel_db")
    
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "sqlite:///./sentinelai.db"
    )
    
    ARTIFACTS_DIR: str = os.getenv("ARTIFACTS_DIR", "app/ml/artifacts")
    CORS_ORIGINS: List[str] = ["*"]


settings = Settings()

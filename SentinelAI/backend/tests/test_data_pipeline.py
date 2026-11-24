import pytest
import numpy as np
import pandas as pd
from app.ml.data_pipeline import FinancialFraudDataPipeline


def test_dataset_generation():
    pipeline = FinancialFraudDataPipeline(random_state=42)
    df = pipeline.generate_benchmark_dataset(n_samples=1000, fraud_ratio=0.03)

    assert len(df) == 1000
    assert "isFraud" in df.columns
    assert "nameOrig" in df.columns
    assert "nameDest" in df.columns
    assert df["isFraud"].sum() == 30
    assert set(df["type"].unique()).issubset({"PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"})


def test_feature_engineering_invariants():
    pipeline = FinancialFraudDataPipeline()
    sample = pd.DataFrame([{
        "step": 12,
        "type": "TRANSFER",
        "amount": 50000.0,
        "nameOrig": "C1001",
        "oldbalanceOrg": 50000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "MULE_001",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "isFraud": 1
    }])

    featured = pipeline.engineer_features(sample)

    assert "errorBalanceOrig" in featured.columns
    assert "errorBalanceDest" in featured.columns
    assert "is_zero_balance_drain" in featured.columns
    assert featured["is_zero_balance_drain"].iloc[0] == 1.0
    assert featured["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-4)


def test_zero_leakage_split():
    pipeline = FinancialFraudDataPipeline(random_state=42)
    df = pipeline.generate_benchmark_dataset(n_samples=500, fraud_ratio=0.04)

    X_train, X_test, y_train, y_test, df_train, df_test = pipeline.fit_transform_split(df, test_size=0.2)

    assert X_train.shape[0] == 400
    assert X_test.shape[0] == 100
    assert len(y_train) == 400
    assert len(y_test) == 100
    assert pipeline.preprocessor is not None
    assert len(pipeline.feature_names) > 10

"""Shared fixtures for the test suite."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_PATH = ROOT_DIR / "data" / "Telco-Customer-Churn.csv"


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """The real source dataset, loaded once per test session."""
    return pd.read_csv(DATA_PATH)


@pytest.fixture
def sample_customer_dict() -> dict:
    """A single valid customer record matching the raw CSV schema
    (used for both feature-engineering unit tests and API request
    bodies -- kept in one place so the two stay in sync)."""
    return {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5, "TotalCharges": 427.5,
    }


@pytest.fixture
def sample_customer_df(sample_customer_dict) -> pd.DataFrame:
    row = {**sample_customer_dict, "customerID": "test-customer", "Churn": "No"}
    return pd.DataFrame([row])


@pytest.fixture(scope="session")
def trained_artifacts():
    """Loads the artifacts produced by `python -m src.train`. Skips
    (rather than fails) dependent tests if training hasn't been run
    yet, since these are integration tests against real artifacts, not
    something that should silently train a model as a side effect of
    running the test suite."""
    import joblib
    from src.preprocessing import ChurnPreprocessor

    models_dir = ROOT_DIR / "models"
    model_path = models_dir / "churn_model.pkl"
    preprocessor_path = models_dir / "preprocessor.pkl"
    if not model_path.exists() or not preprocessor_path.exists():
        pytest.skip("Model artifacts not found — run `python -m src.train` first.")

    model = joblib.load(model_path)
    preprocessor = ChurnPreprocessor.load(str(preprocessor_path))
    return {"model": model, "preprocessor": preprocessor}

"""
FastAPI backend for the Telecom Customer Retention Intelligence Platform.

Serves the trained model (Gradient Boosting, ADASYN-balanced — see
notebooks/Analysis.ipynb for the full methodology and model comparison)
behind a REST API with request validation, SHAP-based explanations, and
grounded retention recommendations, reusing the same src/ modules the
Streamlit app and training script use.

Run locally:
    uvicorn backend.main:app --reload

Interactive docs (Swagger UI): http://localhost:8000/docs
"""

import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.schemas import (
    BatchPredictionItem, BatchPredictionRequest, BatchPredictionResponse,
    CustomerInput, ExplainResponse, FeatureContribution, HealthResponse,
    MetricsResponse, ModelInfoResponse, PredictionResponse,
)
from src.feature_engineering import clean_raw_data, engineer_features
from src.preprocessing import ChurnPreprocessor, transform
from src.recommendations import generate_recommendation

MODELS_DIR = ROOT_DIR / "models"

# Loaded once at startup, not per-request -- model/preprocessor loading
# and SHAP explainer construction are relatively expensive.
_state: dict = {}


def _load_artifacts():
    import joblib

    try:
        _state["model"] = joblib.load(MODELS_DIR / "churn_model.pkl")
        _state["preprocessor"] = ChurnPreprocessor.load(str(MODELS_DIR / "preprocessor.pkl"))
        _state["explainer"] = shap.TreeExplainer(_state["model"])
        with open(MODELS_DIR / "metrics.json") as f:
            _state["metrics"] = json.load(f)
        with open(MODELS_DIR / "feature_columns.json") as f:
            _state["feature_columns"] = json.load(f)
    except FileNotFoundError as e:
        # Don't crash the whole app on startup -- /health will report
        # model_loaded=False and prediction endpoints will return a
        # clear 503 instead of an opaque import-time failure.
        _state["load_error"] = str(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield
    _state.clear()


app = FastAPI(
    title="Telecom Retention Intelligence API",
    description=(
        "Predicts telecom customer churn risk, explains individual "
        "predictions with SHAP, and generates grounded retention "
        "recommendations. See notebooks/Analysis.ipynb for the full "
        "model comparison and validation methodology."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_model():
    if "model" not in _state:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artifacts not loaded. Run `python -m src.train` "
                "to generate them, then restart the API."
                + (f" (load error: {_state['load_error']})" if "load_error" in _state else "")
            ),
        )


def _customer_to_features(customer: CustomerInput) -> pd.DataFrame:
    raw = pd.DataFrame([{**customer.model_dump(), "customerID": "api-request", "Churn": "No"}])
    cleaned = clean_raw_data(raw)
    preprocessor: ChurnPreprocessor = _state["preprocessor"]
    engineered = engineer_features(
        cleaned,
        revenue_bin_edges=preprocessor.revenue_bin_edges,
        spend_bin_edges=preprocessor.spend_bin_edges,
    ).drop(columns=["Churn"])
    return transform(engineered, preprocessor)


def _priority(proba: float) -> str:
    if proba >= 0.7:
        return "Urgent — assign to a retention specialist"
    if proba >= 0.4:
        return "High — proactive outreach within the week"
    return "Monitor — include in next scheduled campaign"


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health():
    """Liveness/readiness check. Returns model_loaded=False (still HTTP
    200) if artifacts failed to load, so orchestrators can distinguish
    "process is up but not ready" from "process is down"."""
    return HealthResponse(status="ok", model_loaded="model" in _state)


@app.get("/model_info", response_model=ModelInfoResponse, tags=["Operations"])
def model_info():
    _require_model()
    metrics = _state["metrics"]
    return ModelInfoResponse(
        model_name=metrics["model_name"],
        model_type=type(_state["model"]).__name__,
        n_features=len(_state["feature_columns"]),
        feature_names=_state["feature_columns"],
        training_strategy=(
            "ADASYN class-imbalance handling + Gradient Boosting, selected "
            "by F1 score across a 10-model comparison. See "
            "notebooks/Analysis.ipynb Phases 3-4 for full methodology."
        ),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Operations"])
def metrics():
    _require_model()
    m = _state["metrics"]
    return MetricsResponse(**m)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerInput):
    """Predict churn risk for a single customer."""
    _require_model()
    try:
        X = _customer_to_features(customer)
        proba = float(_state["model"].predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to process customer input: {e}")

    return PredictionResponse(
        churn_probability=proba,
        prediction="Churn" if proba >= 0.5 else "No Churn",
        priority=_priority(proba),
    )


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    """Predict churn risk for up to 500 customers in one call."""
    _require_model()
    results = []
    for i, customer in enumerate(request.customers):
        try:
            X = _customer_to_features(customer)
            proba = float(_state["model"].predict_proba(X)[0, 1])
        except Exception as e:
            raise HTTPException(
                status_code=422, detail=f"Failed to process customer at index {i}: {e}"
            )
        results.append(BatchPredictionItem(
            index=i,
            churn_probability=proba,
            prediction="Churn" if proba >= 0.5 else "No Churn",
            priority=_priority(proba),
        ))
    return BatchPredictionResponse(predictions=results, count=len(results))


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
def explain(customer: CustomerInput):
    """Explain a single customer's prediction with SHAP, and return a
    grounded retention recommendation built from their top risk drivers."""
    _require_model()
    try:
        X = _customer_to_features(customer)
        proba = float(_state["model"].predict_proba(X)[0, 1])
        shap_values = _state["explainer"](X)
        shap_row = shap_values[0]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to explain customer: {e}")

    contributions = pd.Series(shap_row.values, index=X.columns).sort_values(key=abs, ascending=False)
    top_contributions = [
        FeatureContribution(
            feature=feat,
            shap_value=float(val),
            direction="toward_churn" if val > 0 else "toward_staying",
        )
        for feat, val in contributions.head(10).items()
    ]

    top_risk_features = contributions[contributions > 0].head(6).index.tolist()
    rec = generate_recommendation(top_risk_features, proba)

    return ExplainResponse(
        churn_probability=proba,
        base_value=float(shap_row.base_values),
        top_contributions=top_contributions,
        top_reasons=rec["top_reasons"],
        recommended_actions=rec["recommended_actions"],
        priority=rec["priority"],
    )


@app.get("/", tags=["Operations"])
def root():
    return {"message": "Telecom Retention Intelligence API — see /docs for interactive documentation."}

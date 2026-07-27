"""
Batch scoring for the customer retention pipeline.

Takes a raw customer dataframe (same schema as the source Telco CSV,
minus the target column) and returns it scored with churn probability,
prediction, and priority — reusing the exact same feature engineering,
preprocessing, and model as single-customer prediction (`src/train.py`,
`backend/main.py`), so batch and single-record scores are guaranteed
consistent.
"""

import pandas as pd

from src.feature_engineering import clean_raw_data, engineer_features
from src.preprocessing import ChurnPreprocessor, transform

REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


class BatchValidationError(ValueError):
    """Raised when an uploaded CSV is missing required columns. Caught
    explicitly by the app to show a clear message instead of a raw
    traceback from three layers down in feature engineering."""


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise BatchValidationError(
            f"Uploaded file is missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}"
            + (" (plus optional 'customerID')." if "customerID" not in missing else ".")
        )


def score_dataframe(
    df: pd.DataFrame, model, preprocessor: ChurnPreprocessor
) -> pd.DataFrame:
    """Score a batch of customers. Returns the ORIGINAL columns plus
    ChurnProbability, Prediction, and Priority — never silently drops
    or reorders the customer's own data, since this is meant to be
    downloaded and used directly by a retention team."""
    validate_columns(df)

    working = df.copy()
    has_id = "customerID" in working.columns
    ids = working["customerID"] if has_id else pd.Series(range(len(working)), name="customerID")

    # Churn column isn't present in a real batch-upload file (that's
    # what we're predicting) -- clean_raw_data/engineer_features expect
    # it for parity with the training path, so add a placeholder that
    # never affects the prediction (dropped before scoring).
    scoring_input = working.copy()
    scoring_input["Churn"] = "No"
    if not has_id:
        scoring_input.insert(0, "customerID", ids)

    cleaned = clean_raw_data(scoring_input)
    engineered = engineer_features(
        cleaned,
        revenue_bin_edges=preprocessor.revenue_bin_edges,
        spend_bin_edges=preprocessor.spend_bin_edges,
    ).drop(columns=["Churn"])

    X = transform(engineered, preprocessor)
    probabilities = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["ChurnProbability"] = probabilities
    result["Prediction"] = ["Churn" if p >= 0.5 else "No Churn" for p in probabilities]
    result["Priority"] = [
        "Urgent" if p >= 0.7 else "High" if p >= 0.4 else "Monitor"
        for p in probabilities
    ]
    return result


def summarize_batch(scored: pd.DataFrame) -> dict:
    """Business summary for a scored batch -- the numbers a retention
    team actually wants after a bulk upload, not just the raw table."""
    total = len(scored)
    predicted_churners = int((scored["Prediction"] == "Churn").sum())
    revenue_at_risk = float(
        scored.loc[scored["Prediction"] == "Churn", "MonthlyCharges"].sum()
    ) if "MonthlyCharges" in scored.columns else None

    return {
        "total_customers": total,
        "predicted_churners": predicted_churners,
        "predicted_churn_rate": predicted_churners / total if total else 0.0,
        "avg_churn_probability": float(scored["ChurnProbability"].mean()) if total else 0.0,
        "monthly_revenue_at_risk": revenue_at_risk,
        "urgent_count": int((scored["Priority"] == "Urgent").sum()),
        "high_count": int((scored["Priority"] == "High").sum()),
        "monitor_count": int((scored["Priority"] == "Monitor").sum()),
    }

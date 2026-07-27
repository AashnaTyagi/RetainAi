"""
Feature engineering for the Telecom Customer Retention Intelligence Platform.

Mirrors the logic developed and validated in notebooks/Analysis.ipynb
(Phase 2), extracted into a standalone, reusable module so the same
transformations run identically in training, the FastAPI backend, and
the Streamlit app -- no copy-pasted logic drifting out of sync.
"""

import numpy as np
import pandas as pd

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

EXPECTED_REMAINING_MONTHS = {
    "Month-to-month": 3,
    "One year": 12,
    "Two year": 24,
}

# Ordinal risk ranking derived directly from the churn-rate findings in
# Phase 1 of the notebook: Month-to-month (42.7% churn) > One year (11.3%)
# > Two year (2.8%).
CONTRACT_RISK_MAP = {"Month-to-month": 2, "One year": 1, "Two year": 0}


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning applied before any feature engineering: drop the
    customer ID (not predictive), coerce TotalCharges to numeric, and
    fill the resulting nulls (new customers with 0 tenure) with the
    median."""
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


def compute_quantile_bins(df: pd.DataFrame) -> dict:
    """Compute RevenueSegment / SpendCategory quantile bin edges from a
    full training dataset. Must be called once at training time and the
    resulting edges reused at inference -- `pd.qcut` needs a real
    distribution to compute quantiles from and breaks on a single-row
    input (the live Customer Prediction case), which `pd.cut` with
    fixed edges does not."""
    _, revenue_edges = pd.qcut(df["TotalCharges"], q=4, retbins=True, duplicates="drop")
    _, spend_edges = pd.qcut(df["MonthlyCharges"], q=3, retbins=True, duplicates="drop")
    # Extend the outer edges to +/-inf so any out-of-training-range value
    # (e.g. a live customer with charges slightly outside the training
    # min/max) still falls into the nearest bin instead of becoming NaN.
    revenue_edges = np.array(revenue_edges, dtype=float)
    spend_edges = np.array(spend_edges, dtype=float)
    revenue_edges[0], revenue_edges[-1] = -np.inf, np.inf
    spend_edges[0], spend_edges[-1] = -np.inf, np.inf
    return {"revenue_bin_edges": revenue_edges, "spend_bin_edges": spend_edges}


def engineer_features(
    df: pd.DataFrame,
    revenue_bin_edges: np.ndarray = None,
    spend_bin_edges: np.ndarray = None,
) -> pd.DataFrame:
    """Add the 12 engineered features validated in Phase 2 of the
    notebook. Expects `clean_raw_data` to have already run.

    `revenue_bin_edges` / `spend_bin_edges` should be the fixed edges
    from `compute_quantile_bins()` on the TRAINING data -- required for
    single-row (live inference) input, and recommended even for batch
    use so a re-run doesn't silently produce different bin boundaries."""
    df = df.copy()

    # Total subscribed services -- a strong, well-known churn signal:
    # more embedded services means more switching friction.
    df["TotalServicesSubscribed"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)
    df["TotalServicesSubscribed"] += (df["InternetService"] != "No").astype(int)

    # Spend intensity independent of tenure length (avoids double-counting
    # the tenure signal that raw TotalCharges carries).
    df["AverageMonthlySpend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    # Customer lifetime value: historical (to date) and a simple forward
    # projection using contract-type-implied remaining months.
    df["CustomerLifetimeValueToDate"] = df["TotalCharges"]
    df["ProjectedCLV"] = df["MonthlyCharges"] * (
        df["tenure"] + df["Contract"].map(EXPECTED_REMAINING_MONTHS)
    )

    # Risk / value / segment flags
    df["LongTermCustomerFlag"] = (df["tenure"] >= 24).astype(int)
    df["LowTenureFlag"] = (df["tenure"] <= 12).astype(int)
    df["HighValueCustomer"] = (
        df["MonthlyCharges"] >= df["MonthlyCharges"].quantile(0.75)
    ).astype(int)
    df["HighMonthlyChargesFlag"] = (df["MonthlyCharges"] > 70).astype(int)
    df["AutoPaymentUser"] = df["PaymentMethod"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype(int)
    df["PremiumInternetUser"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["ContractRiskLevel"] = df["Contract"].map(CONTRACT_RISK_MAP)

    # Binned groups for interpretability and non-linear effects
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"],
    )

    if revenue_bin_edges is not None:
        df["RevenueSegment"] = pd.cut(
            df["TotalCharges"], bins=revenue_bin_edges,
            labels=["Low", "Medium", "High", "Premium"][:len(revenue_bin_edges) - 1],
        )
    else:
        # Only safe when df has enough rows to form real quantiles
        # (training-time use); single-row batches must pass fixed edges.
        df["RevenueSegment"] = pd.qcut(
            df["TotalCharges"], q=4, labels=["Low", "Medium", "High", "Premium"], duplicates="drop"
        )

    if spend_bin_edges is not None:
        df["SpendCategory"] = pd.cut(
            df["MonthlyCharges"], bins=spend_bin_edges,
            labels=["Budget", "Standard", "Premium"][:len(spend_bin_edges) - 1],
        )
    else:
        df["SpendCategory"] = pd.qcut(
            df["MonthlyCharges"], q=3, labels=["Budget", "Standard", "Premium"], duplicates="drop"
        )

    return df

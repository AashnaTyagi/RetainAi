"""
Encoding and scaling for the Telecom Customer Retention Intelligence
Platform.

Mirrors notebooks/Analysis.ipynb (Phase 2), including the real bug fix
found and validated there: text columns must be selected with
`include=["object", "string", "category"]`, not `== "object"`, because
this pandas version stores text as a dedicated `string` dtype. Nominal
categorical columns are one-hot encoded (never LabelEncoder) since
LabelEncoder silently imposes a false ordinal relationship on
categories like Contract or PaymentMethod that have no natural order.
"""

from dataclasses import dataclass, field

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

BINNED_COLUMNS = ["TenureGroup", "RevenueSegment", "SpendCategory"]

NUMERIC_COLUMNS_TO_SCALE = [
    "tenure", "MonthlyCharges", "TotalCharges", "AverageMonthlySpend",
    "CustomerLifetimeValueToDate", "ProjectedCLV", "TotalServicesSubscribed",
]


@dataclass
class ChurnPreprocessor:
    """Fitted state needed to transform new data identically to how the
    training data was transformed: the one-hot column layout (so a
    single new customer row expands to the same columns as training),
    the fitted StandardScaler, and which binary columns map to which
    0/1 values."""

    feature_columns: list = field(default_factory=list)
    binary_mappings: dict = field(default_factory=dict)
    nominal_columns: list = field(default_factory=list)
    scaler: StandardScaler = None
    revenue_bin_edges: object = None
    spend_bin_edges: object = None

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ChurnPreprocessor":
        return joblib.load(path)


def fit_transform(
    df: pd.DataFrame,
    revenue_bin_edges=None,
    spend_bin_edges=None,
) -> tuple[pd.DataFrame, pd.Series, ChurnPreprocessor]:
    """Fit encoding + scaling on a full (already feature-engineered)
    training dataframe. Returns (X, y, fitted_preprocessor).

    `revenue_bin_edges` / `spend_bin_edges` (from
    `feature_engineering.compute_quantile_bins`) are attached to the
    returned preprocessor directly -- required for single-row (live
    inference) requests to work. Passing them here, rather than setting
    them as a separate manual step after the call, means a caller can't
    silently skip it."""
    df = df.copy()

    categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != "Churn"]

    binary_cols = [c for c in categorical_cols if df[c].nunique() == 2]
    binary_mappings = {}
    for col in binary_cols:
        mapping = {val: i for i, val in enumerate(sorted(df[col].unique()))}
        binary_mappings[col] = mapping
        df[col] = df[col].map(mapping)

    nominal_cols = [c for c in categorical_cols if df[c].nunique() > 2 and c not in BINNED_COLUMNS]
    nominal_cols += BINNED_COLUMNS

    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    scale_cols = [c for c in NUMERIC_COLUMNS_TO_SCALE if c in X.columns]
    scaler = StandardScaler()
    X[scale_cols] = scaler.fit_transform(X[scale_cols])

    preprocessor = ChurnPreprocessor(
        feature_columns=X.columns.tolist(),
        binary_mappings=binary_mappings,
        nominal_columns=nominal_cols,
        scaler=scaler,
        revenue_bin_edges=revenue_bin_edges,
        spend_bin_edges=spend_bin_edges,
    )
    return X, y, preprocessor


def transform(df: pd.DataFrame, preprocessor: ChurnPreprocessor) -> pd.DataFrame:
    """Transform new (already feature-engineered) data using a
    previously fitted preprocessor -- guarantees the exact same column
    layout the model was trained on, which is what the API and app
    need for live predictions on a single customer at a time."""
    df = df.copy()

    for col, mapping in preprocessor.binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    df = pd.get_dummies(df, columns=[c for c in preprocessor.nominal_columns if c in df.columns])

    # Align to the exact training-time column set: add any missing
    # one-hot columns as 0, drop anything unexpected, and reorder.
    for col in preprocessor.feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[preprocessor.feature_columns]

    scale_cols = [c for c in NUMERIC_COLUMNS_TO_SCALE if c in df.columns]
    df[scale_cols] = preprocessor.scaler.transform(df[scale_cols])

    return df

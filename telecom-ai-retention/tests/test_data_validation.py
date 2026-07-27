"""Data validation tests for the raw source dataset. These guard
against silent schema drift if the data file is ever swapped or
updated -- if a required column disappears or a category value
changes, these fail loudly instead of the pipeline breaking three
steps downstream with a confusing error."""

import pandas as pd

EXPECTED_COLUMNS = {
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
}

EXPECTED_CATEGORICAL_VALUES = {
    "gender": {"Male", "Female"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}


class TestSchema:
    def test_expected_columns_present(self, raw_df):
        assert EXPECTED_COLUMNS.issubset(set(raw_df.columns))

    def test_no_duplicate_customer_ids(self, raw_df):
        assert raw_df["customerID"].duplicated().sum() == 0

    def test_no_fully_duplicate_rows(self, raw_df):
        assert raw_df.duplicated().sum() == 0

    def test_row_count_within_expected_range(self, raw_df):
        # Sanity bound, not an exact match -- catches a badly truncated
        # or accidentally concatenated file without being brittle.
        assert 5000 <= len(raw_df) <= 10000


class TestCategoricalValues:
    def test_categorical_columns_have_only_expected_values(self, raw_df):
        for col, expected_values in EXPECTED_CATEGORICAL_VALUES.items():
            actual_values = set(raw_df[col].unique())
            unexpected = actual_values - expected_values
            assert not unexpected, f"{col} has unexpected values: {unexpected}"

    def test_senior_citizen_is_binary(self, raw_df):
        assert set(raw_df["SeniorCitizen"].unique()).issubset({0, 1})


class TestNumericRanges:
    def test_tenure_non_negative_and_bounded(self, raw_df):
        assert raw_df["tenure"].min() >= 0
        assert raw_df["tenure"].max() <= 100  # generous upper bound

    def test_monthly_charges_positive(self, raw_df):
        assert raw_df["MonthlyCharges"].min() > 0

    def test_total_charges_parses_as_numeric_with_known_blank_count(self, raw_df):
        # TotalCharges is stored as a string with some blank entries
        # (new customers, 0 tenure) -- this pins down that known quirk
        # so a future data refresh with a different blank count is
        # caught rather than silently changing downstream behavior.
        coerced = pd.to_numeric(raw_df["TotalCharges"], errors="coerce")
        n_blank = coerced.isnull().sum()
        assert n_blank == 11, (
            f"Expected 11 blank TotalCharges (documented in Phase 1 EDA), "
            f"found {n_blank} — data may have changed, review before proceeding."
        )


class TestTargetDistribution:
    def test_churn_rate_within_expected_range(self, raw_df):
        # Pins down the imbalance ratio driving the ADASYN decision in
        # Phase 3 -- a big shift here would invalidate that choice.
        churn_rate = (raw_df["Churn"] == "Yes").mean()
        assert 0.20 <= churn_rate <= 0.35

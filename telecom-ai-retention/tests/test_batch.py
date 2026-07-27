"""Unit tests for src/batch.py."""

import pandas as pd
import pytest

from src.batch import (
    REQUIRED_COLUMNS, BatchValidationError, score_dataframe,
    summarize_batch, validate_columns,
)


@pytest.fixture
def raw_batch_sample(raw_df):
    """A small real batch: 15 customers from the source CSV, columns
    only (no target) -- exactly what a user would upload."""
    sample = raw_df.head(15).drop(columns=["Churn"]).copy()
    return sample


class TestValidateColumns:
    def test_valid_dataframe_passes(self, raw_batch_sample):
        validate_columns(raw_batch_sample)  # should not raise

    def test_missing_column_raises(self, raw_batch_sample):
        bad = raw_batch_sample.drop(columns=["Contract"])
        with pytest.raises(BatchValidationError, match="Contract"):
            validate_columns(bad)

    def test_error_message_lists_all_missing(self, raw_batch_sample):
        bad = raw_batch_sample.drop(columns=["Contract", "tenure"])
        with pytest.raises(BatchValidationError) as exc_info:
            validate_columns(bad)
        assert "Contract" in str(exc_info.value)
        assert "tenure" in str(exc_info.value)

    def test_all_required_columns_defined(self):
        # Sanity check the constant itself isn't accidentally empty/wrong
        assert len(REQUIRED_COLUMNS) == 19
        assert "MonthlyCharges" in REQUIRED_COLUMNS


class TestScoreDataframe:
    def test_returns_original_columns_plus_three(self, raw_batch_sample, trained_artifacts):
        scored = score_dataframe(
            raw_batch_sample, trained_artifacts["model"], trained_artifacts["preprocessor"]
        )
        expected_cols = set(raw_batch_sample.columns) | {"ChurnProbability", "Prediction", "Priority"}
        assert set(scored.columns) == expected_cols

    def test_row_count_preserved(self, raw_batch_sample, trained_artifacts):
        scored = score_dataframe(
            raw_batch_sample, trained_artifacts["model"], trained_artifacts["preprocessor"]
        )
        assert len(scored) == len(raw_batch_sample)

    def test_probabilities_in_valid_range(self, raw_batch_sample, trained_artifacts):
        scored = score_dataframe(
            raw_batch_sample, trained_artifacts["model"], trained_artifacts["preprocessor"]
        )
        assert scored["ChurnProbability"].between(0, 1).all()

    def test_prediction_matches_probability_threshold(self, raw_batch_sample, trained_artifacts):
        scored = score_dataframe(
            raw_batch_sample, trained_artifacts["model"], trained_artifacts["preprocessor"]
        )
        for _, row in scored.iterrows():
            expected = "Churn" if row["ChurnProbability"] >= 0.5 else "No Churn"
            assert row["Prediction"] == expected

    def test_priority_matches_probability_bands(self, raw_batch_sample, trained_artifacts):
        scored = score_dataframe(
            raw_batch_sample, trained_artifacts["model"], trained_artifacts["preprocessor"]
        )
        for _, row in scored.iterrows():
            p = row["ChurnProbability"]
            expected = "Urgent" if p >= 0.7 else "High" if p >= 0.4 else "Monitor"
            assert row["Priority"] == expected

    def test_works_without_customer_id_column(self, raw_batch_sample, trained_artifacts):
        no_id = raw_batch_sample.copy()  # source CSV rows here have no customerID anyway
        scored = score_dataframe(no_id, trained_artifacts["model"], trained_artifacts["preprocessor"])
        assert len(scored) == len(no_id)

    def test_single_row_batch_does_not_crash(self, raw_batch_sample, trained_artifacts):
        # Regression coverage for the single-row qcut bug class, at the
        # batch entry point specifically (a 1-row "batch" upload).
        one_row = raw_batch_sample.head(1)
        scored = score_dataframe(one_row, trained_artifacts["model"], trained_artifacts["preprocessor"])
        assert len(scored) == 1
        assert not scored["ChurnProbability"].isnull().any()

    def test_matches_single_customer_prediction(self, raw_batch_sample, trained_artifacts):
        """The batch path and the single-customer API path must agree
        on the same customer -- this is the whole point of sharing
        src/ between them."""
        from src.feature_engineering import clean_raw_data, engineer_features
        from src.preprocessing import transform

        model = trained_artifacts["model"]
        preprocessor = trained_artifacts["preprocessor"]

        one_customer = raw_batch_sample.head(1)
        batch_result = score_dataframe(one_customer, model, preprocessor)

        single = one_customer.copy()
        single["Churn"] = "No"
        cleaned = clean_raw_data(single)
        engineered = engineer_features(
            cleaned,
            revenue_bin_edges=preprocessor.revenue_bin_edges,
            spend_bin_edges=preprocessor.spend_bin_edges,
        ).drop(columns=["Churn"])
        X = transform(engineered, preprocessor)
        single_proba = model.predict_proba(X)[0, 1]

        assert batch_result["ChurnProbability"].iloc[0] == pytest.approx(single_proba)


class TestSummarizeBatch:
    def test_summary_keys(self, raw_batch_sample, trained_artifacts):
        scored = score_dataframe(
            raw_batch_sample, trained_artifacts["model"], trained_artifacts["preprocessor"]
        )
        summary = summarize_batch(scored)
        assert set(summary.keys()) == {
            "total_customers", "predicted_churners", "predicted_churn_rate",
            "avg_churn_probability", "monthly_revenue_at_risk",
            "urgent_count", "high_count", "monitor_count",
        }

    def test_total_matches_input_size(self, raw_batch_sample, trained_artifacts):
        scored = score_dataframe(
            raw_batch_sample, trained_artifacts["model"], trained_artifacts["preprocessor"]
        )
        summary = summarize_batch(scored)
        assert summary["total_customers"] == len(raw_batch_sample)

    def test_priority_counts_sum_to_total(self, raw_batch_sample, trained_artifacts):
        scored = score_dataframe(
            raw_batch_sample, trained_artifacts["model"], trained_artifacts["preprocessor"]
        )
        summary = summarize_batch(scored)
        assert (
            summary["urgent_count"] + summary["high_count"] + summary["monitor_count"]
            == summary["total_customers"]
        )

    def test_revenue_at_risk_is_nonnegative(self, raw_batch_sample, trained_artifacts):
        scored = score_dataframe(
            raw_batch_sample, trained_artifacts["model"], trained_artifacts["preprocessor"]
        )
        summary = summarize_batch(scored)
        assert summary["monthly_revenue_at_risk"] >= 0

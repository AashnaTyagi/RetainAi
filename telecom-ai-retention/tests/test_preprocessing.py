"""Unit tests for src/preprocessing.py."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import clean_raw_data, compute_quantile_bins, engineer_features
from src.preprocessing import ChurnPreprocessor, fit_transform, transform


@pytest.fixture(scope="module")
def engineered_full(raw_df):
    cleaned = clean_raw_data(raw_df)
    bins = compute_quantile_bins(cleaned)
    engineered = engineer_features(cleaned, **bins)
    return engineered, bins


@pytest.fixture(scope="module")
def fitted_preprocessor(engineered_full):
    """Uses the real production call pattern from src/train.py: bin
    edges are passed directly into fit_transform, not attached as a
    separate manual step afterward (that used to be a real gap --
    see test_missing_bin_edges_reproduces_known_bug below)."""
    engineered, bins = engineered_full
    X_train, y_train, preprocessor = fit_transform(
        engineered, revenue_bin_edges=bins["revenue_bin_edges"], spend_bin_edges=bins["spend_bin_edges"]
    )
    return X_train, y_train, preprocessor


class TestFitTransform:
    def test_returns_numeric_only(self, engineered_full):
        engineered, _ = engineered_full
        X, y, _ = fit_transform(engineered)
        non_numeric = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
        assert non_numeric == [], f"Non-numeric columns leaked through: {non_numeric}"

    def test_no_nulls_in_output(self, engineered_full):
        engineered, _ = engineered_full
        X, y, _ = fit_transform(engineered)
        assert X.isnull().sum().sum() == 0

    def test_target_is_binary(self, engineered_full):
        engineered, _ = engineered_full
        _, y, _ = fit_transform(engineered)
        assert set(y.unique()) == {0, 1}

    def test_row_count_preserved(self, engineered_full):
        engineered, _ = engineered_full
        X, y, _ = fit_transform(engineered)
        assert len(X) == len(engineered) == len(y)

    def test_no_duplicate_columns(self, engineered_full):
        # Regression test for the Phase 4 bug: binned columns
        # (TenureGroup etc.) were being one-hot encoded twice.
        engineered, _ = engineered_full
        X, y, _ = fit_transform(engineered)
        assert X.columns.duplicated().sum() == 0

    def test_preprocessor_captures_feature_columns(self, engineered_full):
        engineered, _ = engineered_full
        X, y, preprocessor = fit_transform(engineered)
        assert preprocessor.feature_columns == X.columns.tolist()


class TestTransform:
    def test_single_row_matches_training_columns(self, fitted_preprocessor, sample_customer_df):
        X_train, y_train, preprocessor = fitted_preprocessor

        cleaned_single = clean_raw_data(sample_customer_df)
        engineered_single = engineer_features(
            cleaned_single,
            revenue_bin_edges=preprocessor.revenue_bin_edges,
            spend_bin_edges=preprocessor.spend_bin_edges,
        ).drop(columns=["Churn"])

        X_single = transform(engineered_single, preprocessor)
        assert list(X_single.columns) == preprocessor.feature_columns
        assert len(X_single) == 1

    def test_single_row_output_is_numeric(self, fitted_preprocessor, sample_customer_df):
        X_train, y_train, preprocessor = fitted_preprocessor

        cleaned_single = clean_raw_data(sample_customer_df)
        engineered_single = engineer_features(
            cleaned_single,
            revenue_bin_edges=preprocessor.revenue_bin_edges,
            spend_bin_edges=preprocessor.spend_bin_edges,
        ).drop(columns=["Churn"])

        X_single = transform(engineered_single, preprocessor)
        assert X_single.select_dtypes(exclude=[np.number, "bool"]).columns.tolist() == []
        assert X_single.isnull().sum().sum() == 0

    def test_missing_bin_edges_reproduces_known_bug(self, engineered_full, sample_customer_df):
        """Documents the Phase 11/15 fix: fit_transform now REQUIRES bin
        edges to be passed explicitly (no more separate manual
        attachment step to forget). If a caller skips them anyway --
        the old pattern -- single-row inference breaks again, exactly
        as it did before the fix."""
        import pytest as _pytest

        engineered, _ = engineered_full
        _, _, preprocessor = fit_transform(engineered)  # bin edges omitted
        assert preprocessor.revenue_bin_edges is None

        cleaned_single = clean_raw_data(sample_customer_df)
        with _pytest.raises(ValueError):
            engineer_features(
                cleaned_single,
                revenue_bin_edges=preprocessor.revenue_bin_edges,
                spend_bin_edges=preprocessor.spend_bin_edges,
            )


class TestChurnPreprocessorSaveLoad:
    def test_roundtrip_preserves_feature_columns(self, engineered_full, tmp_path):
        engineered, _ = engineered_full
        _, _, preprocessor = fit_transform(engineered)

        save_path = tmp_path / "preprocessor.pkl"
        preprocessor.save(str(save_path))
        loaded = ChurnPreprocessor.load(str(save_path))

        assert loaded.feature_columns == preprocessor.feature_columns
        assert loaded.binary_mappings == preprocessor.binary_mappings

    def test_roundtrip_preserves_bin_edges(self, engineered_full, tmp_path):
        engineered, bins = engineered_full
        _, _, preprocessor = fit_transform(engineered)
        preprocessor.revenue_bin_edges = bins["revenue_bin_edges"]
        preprocessor.spend_bin_edges = bins["spend_bin_edges"]

        save_path = tmp_path / "preprocessor.pkl"
        preprocessor.save(str(save_path))
        loaded = ChurnPreprocessor.load(str(save_path))

        np.testing.assert_array_equal(loaded.revenue_bin_edges, bins["revenue_bin_edges"])
        np.testing.assert_array_equal(loaded.spend_bin_edges, bins["spend_bin_edges"])

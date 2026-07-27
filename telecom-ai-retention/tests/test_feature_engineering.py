"""Unit tests for src/feature_engineering.py."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    clean_raw_data, compute_quantile_bins, engineer_features,
)


class TestCleanRawData:
    def test_drops_customer_id(self, raw_df):
        cleaned = clean_raw_data(raw_df)
        assert "customerID" not in cleaned.columns

    def test_total_charges_becomes_numeric(self, raw_df):
        cleaned = clean_raw_data(raw_df)
        assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])

    def test_no_nulls_remain_in_total_charges(self, raw_df):
        # The raw data has 11 blank TotalCharges (new customers with 0
        # tenure) -- clean_raw_data must fill them, not just coerce to NaN.
        cleaned = clean_raw_data(raw_df)
        assert cleaned["TotalCharges"].isnull().sum() == 0

    def test_row_count_unchanged(self, raw_df):
        cleaned = clean_raw_data(raw_df)
        assert len(cleaned) == len(raw_df)


class TestComputeQuantileBins:
    def test_returns_expected_keys(self, raw_df):
        cleaned = clean_raw_data(raw_df)
        bins = compute_quantile_bins(cleaned)
        assert set(bins.keys()) == {"revenue_bin_edges", "spend_bin_edges"}

    def test_revenue_bins_have_4_segments(self, raw_df):
        cleaned = clean_raw_data(raw_df)
        bins = compute_quantile_bins(cleaned)
        assert len(bins["revenue_bin_edges"]) == 5  # 4 bins -> 5 edges

    def test_outer_edges_are_infinite(self, raw_df):
        # Required so live single-customer inference with an
        # out-of-training-range value doesn't produce NaN.
        cleaned = clean_raw_data(raw_df)
        bins = compute_quantile_bins(cleaned)
        assert bins["revenue_bin_edges"][0] == -np.inf
        assert bins["revenue_bin_edges"][-1] == np.inf
        assert bins["spend_bin_edges"][0] == -np.inf
        assert bins["spend_bin_edges"][-1] == np.inf


class TestEngineerFeatures:
    def test_adds_expected_columns(self, raw_df):
        cleaned = clean_raw_data(raw_df)
        bins = compute_quantile_bins(cleaned)
        engineered = engineer_features(cleaned, **bins)
        expected_new_cols = {
            "TotalServicesSubscribed", "AverageMonthlySpend",
            "CustomerLifetimeValueToDate", "ProjectedCLV",
            "LongTermCustomerFlag", "LowTenureFlag", "HighValueCustomer",
            "HighMonthlyChargesFlag", "AutoPaymentUser", "PremiumInternetUser",
            "ContractRiskLevel", "TenureGroup", "RevenueSegment", "SpendCategory",
        }
        assert expected_new_cols.issubset(set(engineered.columns))

    def test_row_count_unchanged(self, raw_df):
        cleaned = clean_raw_data(raw_df)
        bins = compute_quantile_bins(cleaned)
        engineered = engineer_features(cleaned, **bins)
        assert len(engineered) == len(cleaned)

    def test_single_row_does_not_crash(self, sample_customer_df, raw_df):
        """Regression test for the Phase 11 bug: pd.qcut-based binning
        broke on single-row (live inference) input. Must use fixed
        bin edges from training data, not recompute qcut per-call."""
        cleaned_full = clean_raw_data(raw_df)
        bins = compute_quantile_bins(cleaned_full)

        cleaned_single = clean_raw_data(sample_customer_df)
        engineered = engineer_features(cleaned_single, **bins)

        assert len(engineered) == 1
        assert not engineered["RevenueSegment"].isnull().any()
        assert not engineered["SpendCategory"].isnull().any()

    def test_without_bin_edges_uses_qcut_fallback(self, raw_df):
        # Batch use without pre-computed edges (e.g. standalone notebook
        # use) should still work via the qcut fallback path.
        cleaned = clean_raw_data(raw_df)
        engineered = engineer_features(cleaned)
        assert not engineered["RevenueSegment"].isnull().any()

    def test_total_services_subscribed_in_valid_range(self, raw_df):
        cleaned = clean_raw_data(raw_df)
        bins = compute_quantile_bins(cleaned)
        engineered = engineer_features(cleaned, **bins)
        # 8 add-on services + internet itself = max 9
        assert engineered["TotalServicesSubscribed"].between(0, 9).all()

    def test_contract_risk_level_matches_known_ordering(self, raw_df):
        # Encodes the Phase 1 finding: Month-to-month is highest risk,
        # Two year is lowest -- verify the ordinal mapping wasn't flipped.
        cleaned = clean_raw_data(raw_df)
        bins = compute_quantile_bins(cleaned)
        engineered = engineer_features(cleaned, **bins)
        risk_by_contract = engineered.groupby(cleaned["Contract"])["ContractRiskLevel"].first()
        assert risk_by_contract["Month-to-month"] > risk_by_contract["One year"]
        assert risk_by_contract["One year"] > risk_by_contract["Two year"]

    def test_average_monthly_spend_no_division_by_zero(self, raw_df):
        cleaned = clean_raw_data(raw_df)
        bins = compute_quantile_bins(cleaned)
        engineered = engineer_features(cleaned, **bins)
        assert not engineered["AverageMonthlySpend"].isin([np.inf, -np.inf]).any()
        assert not engineered["AverageMonthlySpend"].isnull().any()

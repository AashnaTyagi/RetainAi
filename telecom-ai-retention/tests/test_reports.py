"""Tests for src/reports.py. Uses real trained artifacts and real batch
scoring output (via conftest.trained_artifacts), not mocks -- the point
is to catch actual rendering/data-shape breaks, not just confirm the
function doesn't raise on fake input."""

import pandas as pd
import pytest
from pypdf import PdfReader
import io

from src.batch import score_dataframe, summarize_batch
from src.reports import generate_pdf_report, _top_recommendations


@pytest.fixture(scope="module")
def scored_batch(trained_artifacts, raw_df):
    sample = raw_df.drop(columns=["Churn"]).head(30)
    scored = score_dataframe(sample, trained_artifacts["model"], trained_artifacts["preprocessor"])
    summary = summarize_batch(scored)
    return scored, summary


class TestGeneratePdfReport:
    def test_returns_nonempty_bytes(self, scored_batch, trained_artifacts):
        scored, summary = scored_batch
        pdf_bytes = generate_pdf_report(
            scored, summary,
            model=trained_artifacts["model"],
            feature_names=trained_artifacts["preprocessor"].feature_columns,
        )
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000

    def test_produces_valid_readable_pdf(self, scored_batch, trained_artifacts):
        scored, summary = scored_batch
        pdf_bytes = generate_pdf_report(
            scored, summary,
            model=trained_artifacts["model"],
            feature_names=trained_artifacts["preprocessor"].feature_columns,
        )
        reader = PdfReader(io.BytesIO(pdf_bytes))
        assert len(reader.pages) >= 2

    def test_contains_real_summary_numbers(self, scored_batch, trained_artifacts):
        scored, summary = scored_batch
        pdf_bytes = generate_pdf_report(
            scored, summary,
            model=trained_artifacts["model"],
            feature_names=trained_artifacts["preprocessor"].feature_columns,
        )
        reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = "".join(p.extract_text() for p in reader.pages)
        assert str(summary["total_customers"]) in full_text
        assert "RetainAI" in full_text

    def test_works_without_model(self, scored_batch):
        # model/feature_names are optional -- report should still build
        # (just without the feature-importance chart section).
        scored, summary = scored_batch
        pdf_bytes = generate_pdf_report(scored, summary)
        assert len(pdf_bytes) > 500

    def test_works_with_small_batch(self, trained_artifacts, raw_df):
        # Edge case: fewer than 20 customers (the high-risk table's
        # default top_n) shouldn't error or produce a broken table.
        sample = raw_df.drop(columns=["Churn"]).head(3)
        scored = score_dataframe(sample, trained_artifacts["model"], trained_artifacts["preprocessor"])
        summary = summarize_batch(scored)
        pdf_bytes = generate_pdf_report(scored, summary)
        assert len(pdf_bytes) > 500

    def test_handles_batch_with_no_high_risk_customers(self):
        # Synthetic all-low-risk batch -- recommendations section must
        # degrade gracefully, not crash on an empty high-risk slice.
        scored = pd.DataFrame({
            "customerID": ["a", "b"],
            "ChurnProbability": [0.05, 0.10],
            "Prediction": ["No Churn", "No Churn"],
            "Priority": ["Monitor", "Monitor"],
            "Contract": ["Two year", "Two year"],
            "MonthlyCharges": [50.0, 55.0],
        })
        summary = summarize_batch(scored)
        pdf_bytes = generate_pdf_report(scored, summary)
        assert len(pdf_bytes) > 500


class TestTopRecommendations:
    def test_empty_high_risk_returns_no_action_message(self):
        scored = pd.DataFrame({
            "Priority": ["Monitor", "Monitor"],
            "Contract": ["Two year", "Two year"],
        })
        recs = _top_recommendations(scored)
        assert len(recs) == 1
        assert "no immediate action" in recs[0].lower()

    def test_dominant_month_to_month_signal_detected(self):
        scored = pd.DataFrame({
            "Priority": ["Urgent"] * 5,
            "Contract": ["Month-to-month"] * 5,
        })
        recs = _top_recommendations(scored)
        assert any("contract" in r.lower() for r in recs)

    def test_respects_max_recommendations(self):
        scored = pd.DataFrame({
            "Priority": ["Urgent"] * 5,
            "Contract": ["Month-to-month"] * 5,
            "PaymentMethod": ["Electronic check"] * 5,
            "InternetService": ["Fiber optic"] * 5,
            "tenure": [2] * 5,
        })
        recs = _top_recommendations(scored, max_recommendations=2)
        assert len(recs) <= 2

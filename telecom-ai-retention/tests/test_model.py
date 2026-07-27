"""Model tests: sanity and regression checks against the trained
artifacts in models/. These don't re-validate the full modeling
methodology (that's notebooks/Analysis.ipynb's job) -- they catch a
degraded or corrupted production artifact, e.g. from a bad retrain or
a bug introduced in src/train.py."""

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

# Floor, not the notebook's actual validated numbers -- catches a
# genuinely broken retrain (e.g. F1 collapsing to 0.1) without being so
# tight that ordinary run-to-run variance fails the suite.
MIN_ACCEPTABLE = {"accuracy": 0.65, "f1_score": 0.45, "roc_auc": 0.70}


@pytest.fixture(scope="module")
def metrics():
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        pytest.skip("models/metrics.json not found — run `python -m src.train` first.")
    with open(path) as f:
        return json.load(f)


class TestModelMetricsFloor:
    @pytest.mark.parametrize("metric_name,floor", list(MIN_ACCEPTABLE.items()))
    def test_metric_above_floor(self, metrics, metric_name, floor):
        assert metrics[metric_name] >= floor, (
            f"{metric_name}={metrics[metric_name]:.4f} is below the "
            f"regression floor of {floor} — investigate before shipping this model."
        )

    def test_metrics_are_valid_probabilities(self, metrics):
        for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
            assert 0.0 <= metrics[key] <= 1.0


class TestModelConsistency:
    def test_model_predicts_deterministically(self, trained_artifacts, sample_customer_df):
        from src.feature_engineering import clean_raw_data, engineer_features
        from src.preprocessing import transform

        model = trained_artifacts["model"]
        preprocessor = trained_artifacts["preprocessor"]

        cleaned = clean_raw_data(sample_customer_df)
        engineered = engineer_features(
            cleaned,
            revenue_bin_edges=preprocessor.revenue_bin_edges,
            spend_bin_edges=preprocessor.spend_bin_edges,
        ).drop(columns=["Churn"])
        X = transform(engineered, preprocessor)

        proba_1 = model.predict_proba(X)[0, 1]
        proba_2 = model.predict_proba(X)[0, 1]
        assert proba_1 == proba_2

    def test_higher_risk_profile_scores_higher(self, trained_artifacts):
        """Directional sanity check grounded in Phase 1's own findings:
        a month-to-month, low-tenure, electronic-check customer should
        score meaningfully higher than a two-year, long-tenure, autopay
        customer with otherwise similar charges. If this ever flips,
        something in the pipeline broke the sign of a core relationship,
        even if the aggregate metrics still look fine."""
        from src.feature_engineering import clean_raw_data, engineer_features
        from src.preprocessing import transform

        model = trained_artifacts["model"]
        preprocessor = trained_artifacts["preprocessor"]

        base = {
            "customerID": "t", "gender": "Female", "SeniorCitizen": 0,
            "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No",
            "PaperlessBilling": "Yes", "MonthlyCharges": 80.0,
            "Churn": "No",
        }
        high_risk = {**base, "tenure": 2, "Contract": "Month-to-month",
                     "PaymentMethod": "Electronic check", "TotalCharges": 160.0}
        low_risk = {**base, "tenure": 60, "Contract": "Two year",
                    "PaymentMethod": "Credit card (automatic)", "TotalCharges": 4800.0}

        def score(record):
            df = pd.DataFrame([record])
            cleaned = clean_raw_data(df)
            engineered = engineer_features(
                cleaned,
                revenue_bin_edges=preprocessor.revenue_bin_edges,
                spend_bin_edges=preprocessor.spend_bin_edges,
            ).drop(columns=["Churn"])
            X = transform(engineered, preprocessor)
            return model.predict_proba(X)[0, 1]

        assert score(high_risk) > score(low_risk)


class TestArtifactIntegrity:
    def test_feature_columns_file_matches_preprocessor(self, trained_artifacts):
        with open(MODELS_DIR / "feature_columns.json") as f:
            saved_columns = json.load(f)
        assert saved_columns == trained_artifacts["preprocessor"].feature_columns

    def test_model_n_features_matches_preprocessor(self, trained_artifacts):
        model = trained_artifacts["model"]
        n_expected = len(trained_artifacts["preprocessor"].feature_columns)
        assert model.n_features_in_ == n_expected

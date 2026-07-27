"""Tests for the FastAPI backend (backend/main.py). Uses TestClient
against the real trained artifacts -- skips if they don't exist yet
(see conftest.trained_artifacts) rather than mocking the model, since
the point is to catch real integration breaks (schema mismatches,
missing files, shape errors) that a mocked model would hide."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(trained_artifacts):
    from backend.main import app
    with TestClient(app) as c:
        yield c


class TestHealthAndInfo:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["model_loaded"] is True

    def test_model_info(self, client):
        r = client.get("/model_info")
        assert r.status_code == 200
        body = r.json()
        assert body["model_name"] == "Gradient Boosting"
        assert body["n_features"] > 0
        assert len(body["feature_names"]) == body["n_features"]

    def test_metrics(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["accuracy"] <= 1
        assert 0 <= body["f1_score"] <= 1

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200


class TestPredict:
    def test_valid_customer_returns_prediction(self, client, sample_customer_dict):
        r = client.post("/predict", json=sample_customer_dict)
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["churn_probability"] <= 1
        assert body["prediction"] in ("Churn", "No Churn")
        assert body["priority"]

    def test_prediction_consistent_with_probability(self, client, sample_customer_dict):
        r = client.post("/predict", json=sample_customer_dict)
        body = r.json()
        expected = "Churn" if body["churn_probability"] >= 0.5 else "No Churn"
        assert body["prediction"] == expected

    def test_invalid_enum_rejected(self, client, sample_customer_dict):
        bad = {**sample_customer_dict, "gender": "Other"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_missing_fields_rejected(self, client):
        r = client.post("/predict", json={"gender": "Female"})
        assert r.status_code == 422

    def test_negative_tenure_rejected(self, client, sample_customer_dict):
        bad = {**sample_customer_dict, "tenure": -5}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_total_charges_below_monthly_rejected(self, client, sample_customer_dict):
        bad = {**sample_customer_dict, "MonthlyCharges": 100.0, "TotalCharges": 10.0}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422


class TestPredictBatch:
    def test_batch_returns_matching_count(self, client, sample_customer_dict):
        r = client.post("/predict_batch", json={"customers": [sample_customer_dict] * 3})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 3
        assert len(body["predictions"]) == 3

    def test_batch_preserves_order_via_index(self, client, sample_customer_dict):
        r = client.post("/predict_batch", json={"customers": [sample_customer_dict] * 3})
        indices = [p["index"] for p in r.json()["predictions"]]
        assert indices == [0, 1, 2]

    def test_empty_batch_rejected(self, client):
        r = client.post("/predict_batch", json={"customers": []})
        assert r.status_code == 422

    def test_identical_customers_get_identical_predictions(self, client, sample_customer_dict):
        r = client.post("/predict_batch", json={"customers": [sample_customer_dict] * 2})
        probs = [p["churn_probability"] for p in r.json()["predictions"]]
        assert probs[0] == probs[1]


class TestExplain:
    def test_returns_shap_contributions(self, client, sample_customer_dict):
        r = client.post("/explain", json=sample_customer_dict)
        assert r.status_code == 200
        body = r.json()
        assert len(body["top_contributions"]) == 10
        for c in body["top_contributions"]:
            assert c["direction"] in ("toward_churn", "toward_staying")

    def test_returns_recommendation(self, client, sample_customer_dict):
        r = client.post("/explain", json=sample_customer_dict)
        body = r.json()
        assert len(body["recommended_actions"]) > 0
        assert body["priority"]

    def test_probability_matches_predict_endpoint(self, client, sample_customer_dict):
        r_predict = client.post("/predict", json=sample_customer_dict)
        r_explain = client.post("/explain", json=sample_customer_dict)
        assert r_predict.json()["churn_probability"] == pytest.approx(
            r_explain.json()["churn_probability"], abs=1e-6
        )

    def test_invalid_input_rejected(self, client, sample_customer_dict):
        bad = {**sample_customer_dict, "Contract": "Three year"}
        r = client.post("/explain", json=bad)
        assert r.status_code == 422

"""Unit tests for src/recommendations.py."""

from src.recommendations import (
    generate_recommendation, humanize_feature, priority_for_probability,
)


class TestPriorityForProbability:
    def test_high_probability_is_urgent(self):
        assert "Urgent" in priority_for_probability(0.85)

    def test_boundary_at_0_7_is_urgent(self):
        assert "Urgent" in priority_for_probability(0.7)

    def test_mid_probability_is_high(self):
        assert "High" in priority_for_probability(0.5)

    def test_boundary_at_0_4_is_high(self):
        assert "High" in priority_for_probability(0.4)

    def test_low_probability_is_monitor(self):
        assert "Monitor" in priority_for_probability(0.1)

    def test_zero_probability_is_monitor(self):
        assert "Monitor" in priority_for_probability(0.0)


class TestHumanizeFeature:
    def test_one_hot_column_becomes_readable(self):
        result = humanize_feature("PaymentMethod_Electronic check")
        assert "PaymentMethod" in result
        assert "Electronic check" in result

    def test_plain_column_passes_through(self):
        assert humanize_feature("tenure") == "tenure"


class TestGenerateRecommendation:
    def test_known_feature_produces_rulebook_reason(self):
        rec = generate_recommendation(["ContractRiskLevel"], proba=0.8)
        assert len(rec["top_reasons"]) == 1
        assert "contract" in rec["top_reasons"][0].lower()

    def test_known_feature_produces_actions(self):
        rec = generate_recommendation(["ContractRiskLevel"], proba=0.8)
        assert len(rec["recommended_actions"]) > 0

    def test_unknown_feature_falls_back_gracefully(self):
        rec = generate_recommendation(["SomeOneHotColumn_ValueX"], proba=0.5)
        assert len(rec["top_reasons"]) == 1
        assert "ValueX" in rec["top_reasons"][0]

    def test_empty_features_still_returns_valid_structure(self):
        rec = generate_recommendation([], proba=0.6)
        assert rec["recommended_actions"] != []  # falls back to generic action
        assert rec["top_reasons"] == []

    def test_priority_matches_probability(self):
        rec = generate_recommendation(["tenure"], proba=0.9)
        assert "Urgent" in rec["priority"]

    def test_actions_are_deduplicated(self):
        # ContractRiskLevel and AutoPaymentUser have distinct action
        # sets, but ensure repeating a feature doesn't duplicate actions.
        rec = generate_recommendation(["ContractRiskLevel", "ContractRiskLevel"], proba=0.7)
        assert len(rec["recommended_actions"]) == len(set(rec["recommended_actions"]))

    def test_respects_top_n_limit(self):
        many_features = ["ContractRiskLevel", "AutoPaymentUser", "PremiumInternetUser",
                          "LowTenureFlag", "HighMonthlyChargesFlag", "TotalServicesSubscribed"]
        rec = generate_recommendation(many_features, proba=0.7, top_n=2)
        assert len(rec["top_reasons"]) == 2

    def test_actions_capped_at_5(self):
        many_features = ["ContractRiskLevel", "AutoPaymentUser", "PremiumInternetUser",
                          "LowTenureFlag", "HighMonthlyChargesFlag", "TotalServicesSubscribed"]
        rec = generate_recommendation(many_features, proba=0.7, top_n=6)
        assert len(rec["recommended_actions"]) <= 5

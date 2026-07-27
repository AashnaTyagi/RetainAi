"""
Retention recommendation engine.

Mirrors notebooks/Analysis.ipynb (Phase 10): maps a customer's top
SHAP risk drivers to plain-English reasons and concrete retention
actions, grounded in the business findings from Phases 1, 3, and 8.
Shared by the Streamlit app and (in a later phase) the FastAPI backend
so the recommendation logic lives in exactly one place.
"""

RETENTION_RULEBOOK = {
    "ContractRiskLevel": {
        "reason": "On a month-to-month or short-term contract (42.7% churn vs. 2.8% on two-year)",
        "actions": [
            "Offer a discounted annual or two-year contract upgrade",
            "Highlight loyalty pricing tiers unlocked by longer terms",
        ],
    },
    "AutoPaymentUser": {
        "reason": "Paying manually rather than by autopay (~3x higher churn for manual payment methods)",
        "actions": [
            "Offer a one-time bill credit for enrolling in autopay",
            "Simplify the autopay signup flow at the next touchpoint",
        ],
    },
    "PremiumInternetUser": {
        "reason": "On fiber internet, the segment with the highest churn rate despite being the premium product",
        "actions": [
            "Proactive service-quality check-in call",
            "Review pricing versus competitor fiber offers in their area",
        ],
    },
    "LowTenureFlag": {
        "reason": "Still in their first year, the highest-risk tenure window",
        "actions": [
            "Enroll in an early-life onboarding / engagement program",
            "Personal check-in call from a retention specialist within the first 90 days",
        ],
    },
    "HighMonthlyChargesFlag": {
        "reason": "Paying above-average monthly charges",
        "actions": [
            "Review current plan for bundling or right-sizing opportunities",
            "Offer a loyalty discount tied to a contract-length upgrade",
        ],
    },
    "TotalServicesSubscribed": {
        "reason": "Subscribed to few add-on services, meaning low switching cost",
        "actions": [
            "Offer a free trial of a bundled add-on (security, backup, streaming)",
            "Cross-sell a service bundle discount",
        ],
    },
    "tenure": {
        "reason": "Relatively short tenure with the company",
        "actions": [
            "Loyalty milestone outreach (e.g. anniversary discount)",
            "Early-life engagement program",
        ],
    },
    "MonthlyCharges": {
        "reason": "Monthly charges are a significant factor in this customer's risk",
        "actions": ["Plan review for cost-saving bundle options", "Targeted discount offer"],
    },
}


def humanize_feature(feature_name: str) -> str:
    """Fallback label for one-hot encoded columns not in the rulebook,
    e.g. 'PaymentMethod_Electronic check' -> readable text."""
    if "_" in feature_name:
        base, value = feature_name.split("_", 1)
        return f"{base} = '{value}'"
    return feature_name


def priority_for_probability(proba: float) -> str:
    if proba >= 0.7:
        return "Urgent — assign to a retention specialist"
    if proba >= 0.4:
        return "High — proactive outreach within the week"
    return "Monitor — include in next scheduled campaign"


def generate_recommendation(top_contributing_features: list, proba: float, top_n: int = 4) -> dict:
    """Build a structured retention recommendation from a customer's
    top positive-SHAP feature names (highest risk contributors first)
    and their model churn probability."""
    reasons, actions = [], []
    for feature_name in top_contributing_features[:top_n]:
        rule = RETENTION_RULEBOOK.get(feature_name)
        if rule:
            reasons.append(rule["reason"])
            actions.extend(rule["actions"])
        else:
            reasons.append(f"Elevated risk contribution from {humanize_feature(feature_name)}")

    seen = set()
    unique_actions = [a for a in actions if not (a in seen or seen.add(a))]

    return {
        "churn_probability": proba,
        "priority": priority_for_probability(proba),
        "top_reasons": reasons,
        "recommended_actions": unique_actions[:5] or ["Standard retention outreach — monitor engagement"],
    }

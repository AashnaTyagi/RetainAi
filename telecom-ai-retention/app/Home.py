import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loaders import load_business_findings, load_metrics

st.set_page_config(
    page_title="Telecom Retention Intelligence Platform",
    page_icon="📡",
    layout="wide",
)

metrics = load_metrics()
findings = load_business_findings()["overview"]

st.title("📡 Telecom Customer Retention Intelligence Platform")
st.caption(
    "An end-to-end churn prediction and retention system — from raw "
    "customer data to explainable, per-customer retention actions."
)

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Overall Churn Rate", f"{findings['overall_churn_rate']:.1%}")
col2.metric("Monthly Revenue Lost", f"${findings['monthly_revenue_lost']:,.0f}")
col3.metric("Model F1 Score", f"{metrics['f1_score']:.3f}")
col4.metric("Model ROC AUC", f"{metrics['roc_auc']:.3f}")

st.markdown("---")

st.markdown(
    f"""
### What this platform does

Every month, **{findings['churned_customers']:,} of {findings['total_customers']:,} customers**
(**{findings['overall_churn_rate']:.1%}**) leave, costing roughly
**${findings['monthly_revenue_lost']:,.0f}/month** in recurring revenue. This
platform predicts which customers are at risk *before* they leave, explains
*why* for each individual customer, and turns that into a concrete retention
action — not just a probability score.

### How it's built

- **Data & Modeling** — 20 engineered features, ADASYN class-imbalance
  handling, and a 10-model comparison (Gradient Boosting won on F1 score).
  Full methodology in `notebooks/Analysis.ipynb`.
- **Explainability** — SHAP and LIME explain every prediction at both the
  global and individual-customer level.
- **Business Intelligence** — customer segmentation and revenue-at-risk
  analysis translate model output into a prioritized retention strategy.
- **This app** — a live demo: predict a new customer's risk, see why, and
  get a recommended retention action, in seconds.

Use the sidebar to navigate between pages: Dashboard, EDA, Model Comparison,
Customer Prediction, Explain Prediction, Business Insights, Customer
Segmentation, Recommendations, and About.
"""
)

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loaders import load_business_findings, load_metrics

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Dashboard")

metrics = load_metrics()
findings = load_business_findings()

overview = findings["overview"]
top_decile = findings["top_decile_targeting"]

st.subheader("Business KPIs")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{overview['total_customers']:,}")
c2.metric("Churned Customers", f"{overview['churned_customers']:,}")
c3.metric("Monthly Revenue Lost", f"${overview['monthly_revenue_lost']:,.0f}")
c4.metric("Annualized Revenue Lost", f"${overview['annualized_revenue_lost']:,.0f}")

st.subheader("Model KPIs")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
c2.metric("Precision", f"{metrics['precision']:.1%}")
c3.metric("Recall", f"{metrics['recall']:.1%}")
c4.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

st.subheader("Retention Opportunity")
c1, c2, c3 = st.columns(3)
c1.metric("Top-Decile Churn Capture", f"{top_decile['capture_rate']:.0%}")
c2.metric("Lift Over Random Targeting", f"{top_decile['lift_over_random']:.2f}x")
c3.metric("Addressable Monthly Revenue", f"${top_decile['addressable_monthly_revenue']:,.0f}")

st.markdown("---")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Churn Rate by Contract Type")
    contract_df = findings["risk_by_contract"]
    fig = px.bar(
        contract_df, x="segment", y="churn_rate",
        labels={"segment": "Contract", "churn_rate": "Churn Rate"},
        color="churn_rate", color_continuous_scale="Reds",
    )
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch", key="dash_churn_by_contract")

with col_b:
    st.subheader("Revenue by Customer Segment")
    seg_df = findings["customer_segments"]
    fig = px.pie(
        seg_df, names="segment", values="monthly_revenue",
        title=None,
    )
    st.plotly_chart(fig, width="stretch", key="dash_revenue_pie")

st.caption(
    "All figures come from the offline analysis in `notebooks/Analysis.ipynb` "
    "(Phases 4, 6, 8, 9) — see the About page for the full methodology."
)

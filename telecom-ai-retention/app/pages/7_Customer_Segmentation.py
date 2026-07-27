import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loaders import load_business_findings

st.set_page_config(page_title="Customer Segmentation", page_icon="🧩", layout="wide")
st.title("🧩 Customer Segmentation")
st.caption(
    "From notebooks/Analysis.ipynb Phase 9 — KMeans clustering on tenure, "
    "monthly charges, total charges, and services subscribed, auto-labeled "
    "by risk / value / loyalty rules."
)

findings = load_business_findings()
seg_df = pd.DataFrame(findings["customer_segments"]).sort_values("churn_rate", ascending=False)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Churn Rate by Segment")
    fig = px.bar(
        seg_df, x="segment", y="churn_rate", color="churn_rate",
        color_continuous_scale="Reds", text_auto=".1%",
    )
    fig.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-15)
    st.plotly_chart(fig, width="stretch", key="seg_churn_bar")

with col2:
    st.subheader("Revenue Share by Segment")
    fig = px.pie(seg_df, names="segment", values="monthly_revenue")
    st.plotly_chart(fig, width="stretch", key="seg_revenue_pie")

st.subheader("Segment Details")
display_df = seg_df.copy()
display_df["churn_rate"] = display_df["churn_rate"].apply(lambda v: f"{v:.1%}")
display_df["monthly_revenue"] = display_df["monthly_revenue"].apply(lambda v: f"${v:,.0f}")
display_df.columns = ["Segment", "Customers", "Churn Rate", "Monthly Revenue"]
st.dataframe(display_df, width="stretch")

st.markdown("---")
st.subheader("How to Use These Segments")
st.markdown(
    """
- **High Risk / Premium segments** — highest churn *and* high revenue per
  customer. This is where retention spend has the best return: losing these
  customers is the most expensive outcome, and they respond to targeted offers.
- **High Risk / Budget / New Customers** — likely early-life churn.
  Onboarding and first-90-days engagement matters more here than discounting.
- **Low Risk / Long-Term Loyal segments** — minimal churn risk. Retention
  spend here has low marginal return; budget is better spent elsewhere.
"""
)

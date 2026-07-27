import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loaders import load_business_findings

st.set_page_config(page_title="Business Insights", page_icon="💼", layout="wide")
st.title("💼 Business Insights")
st.caption("From notebooks/Analysis.ipynb Phase 8 — full business analytics.")

findings = load_business_findings()
overview = findings["overview"]

st.subheader("Revenue Impact of Churn")
c1, c2, c3 = st.columns(3)
c1.metric("Monthly Revenue Lost", f"${overview['monthly_revenue_lost']:,.0f}")
c2.metric("Annualized Revenue Lost", f"${overview['annualized_revenue_lost']:,.0f}")
c3.metric("Historical Billed Revenue Lost", f"${overview['historical_revenue_lost']:,.0f}")
st.caption(
    "Monthly recurring revenue lost is the number that matters most going "
    "forward — it recurs every month until those customers are replaced or retained."
)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["By Contract", "By Payment Method", "By Internet Service"])

with tab1:
    df = pd.DataFrame(findings["risk_by_contract"])
    fig = px.bar(df, x="segment", y="monthly_revenue_at_risk", color="churn_rate",
                 color_continuous_scale="Reds", labels={"monthly_revenue_at_risk": "Monthly Revenue at Risk ($)"})
    st.plotly_chart(fig, width="stretch", key="bi_contract_revenue")
    st.dataframe(df, width="stretch")
    st.caption(
        "Month-to-month customers are both the largest segment and the "
        "highest-churn segment — the single biggest revenue-at-risk line "
        "in the business, and the clearest target for a term-contract incentive."
    )

with tab2:
    df = pd.DataFrame(findings["risk_by_payment_method"])
    fig = px.bar(df, x="segment", y="churn_rate", color="churn_rate", color_continuous_scale="Reds")
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch", key="bi_payment_churn")
    st.dataframe(df, width="stretch")
    st.caption(
        "Electronic check payers churn at roughly 3x the rate of autopay "
        "users — an autopay incentive is a plausible high-ROI retention lever."
    )

with tab3:
    df = pd.DataFrame(findings["risk_by_internet_service"])
    fig = px.bar(df, x="segment", y="churn_rate", color="avg_monthly_charge", color_continuous_scale="Oranges")
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch", key="bi_internet_churn")
    st.dataframe(df, width="stretch")
    st.caption(
        "Fiber customers churn the most AND pay the most — a product/ops "
        "flag worth investigating (pricing vs. perceived value or service "
        "reliability), separate from anything the model alone can fix."
    )

st.markdown("---")
st.subheader("Retention Opportunity Sizing")
top_decile = findings["top_decile_targeting"]
st.markdown(
    f"""
If a retention team acts only on the model's **top-risk decile**:

- They reach **~{top_decile['capture_rate']:.0%}** of all customers who actually churn
  (a **{top_decile['lift_over_random']:.2f}x** lift over random targeting)
- That's **~${top_decile['addressable_monthly_revenue']:,.0f}/month** of revenue that becomes addressable
- Even a conservative 20% save rate on that outreach retains
  **~${top_decile['illustrative_retained_revenue_at_20pct_save_rate']:,.0f}/month**

*(The 20% save rate is an illustrative assumption — replace with real numbers from a pilot campaign.)*
"""
)

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loaders import load_raw_data

st.set_page_config(page_title="EDA", page_icon="🔎", layout="wide")
st.title("🔎 Exploratory Data Analysis")
st.caption(
    "Interactive version of the deep EDA in notebooks/Analysis.ipynb (Phase 1). "
    "Filter by segment to see how distributions shift."
)

df = load_raw_data().copy()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

with st.sidebar:
    st.header("Filters")
    contract_filter = st.multiselect(
        "Contract", options=df["Contract"].unique().tolist(),
        default=df["Contract"].unique().tolist(),
    )
    internet_filter = st.multiselect(
        "Internet Service", options=df["InternetService"].unique().tolist(),
        default=df["InternetService"].unique().tolist(),
    )

filtered = df[df["Contract"].isin(contract_filter) & df["InternetService"].isin(internet_filter)]
st.caption(f"Showing {len(filtered):,} of {len(df):,} customers")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Distribution")
    fig = px.histogram(filtered, x="Churn", color="Churn", color_discrete_sequence=["#2ca02c", "#d62728"])
    st.plotly_chart(fig, width="stretch", key="eda_churn_dist")

with col2:
    st.subheader("Tenure Distribution by Churn")
    fig = px.histogram(filtered, x="tenure", color="Churn", barmode="overlay", opacity=0.6)
    st.plotly_chart(fig, width="stretch", key="eda_tenure_dist")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Monthly Charges by Churn")
    fig = px.box(filtered, x="Churn", y="MonthlyCharges", color="Churn")
    st.plotly_chart(fig, width="stretch", key="eda_monthly_box")

with col4:
    st.subheader("Churn Rate by Contract")
    rate_df = filtered.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean()).reset_index()
    rate_df.columns = ["Contract", "ChurnRate"]
    fig = px.bar(rate_df, x="Contract", y="ChurnRate", color="ChurnRate", color_continuous_scale="Reds")
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch", key="eda_churn_by_contract_fixed")

st.subheader("Churn Rate Across All Categorical Features")
categorical_features = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
selected_feature = st.selectbox("Choose a feature", categorical_features, index=13)
rate_df = filtered.groupby(selected_feature)["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values(ascending=False).reset_index()
rate_df.columns = [selected_feature, "ChurnRate"]
fig = px.bar(rate_df, x=selected_feature, y="ChurnRate", color="ChurnRate", color_continuous_scale="Reds")
fig.update_layout(yaxis_tickformat=".0%")
st.plotly_chart(fig, width="stretch", key="eda_churn_by_selected_feature")

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loaders import load_model, load_preprocessor
from src.batch import BatchValidationError, REQUIRED_COLUMNS, score_dataframe, summarize_batch
from src.reports import generate_pdf_report

st.set_page_config(page_title="Batch Prediction", page_icon="📁", layout="wide")
st.title("📁 Batch Prediction")
st.caption(
    "Upload a CSV of customers to score all of them at once — same model "
    "and pipeline as the single-customer prediction page, applied in bulk."
)

with st.expander("Expected CSV format"):
    st.write(f"Required columns: `{'`, `'.join(REQUIRED_COLUMNS)}`")
    st.write("An optional `customerID` column is preserved in the output if present.")

uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read this file as CSV: {e}")
        st.stop()

    model = load_model()
    preprocessor = load_preprocessor()

    try:
        scored = score_dataframe(raw_df, model, preprocessor)
    except BatchValidationError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Scoring failed: {e}")
        st.stop()

    summary = summarize_batch(scored)

    st.markdown("---")
    st.subheader("Business Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers Scored", f"{summary['total_customers']:,}")
    c2.metric("Predicted Churners", f"{summary['predicted_churners']:,}")
    c3.metric("Predicted Churn Rate", f"{summary['predicted_churn_rate']:.1%}")
    if summary["monthly_revenue_at_risk"] is not None:
        c4.metric("Monthly Revenue at Risk", f"${summary['monthly_revenue_at_risk']:,.0f}")
    else:
        c4.metric("Avg Churn Probability", f"{summary['avg_churn_probability']:.1%}")

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Urgent", summary["urgent_count"])
    c2.metric("🟠 High", summary["high_count"])
    c3.metric("🟢 Monitor", summary["monitor_count"])

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Risk Distribution")
        fig = px.histogram(scored, x="ChurnProbability", nbins=20, color_discrete_sequence=["indianred"])
        fig.update_layout(xaxis_title="Predicted churn probability", yaxis_title="Customers")
        st.plotly_chart(fig, width="stretch", key="batch_risk_dist")

    with col_b:
        st.subheader("Priority Breakdown")
        priority_counts = scored["Priority"].value_counts().reindex(["Urgent", "High", "Monitor"]).fillna(0)
        fig = px.pie(
            names=priority_counts.index, values=priority_counts.values,
            color=priority_counts.index,
            color_discrete_map={"Urgent": "#d62728", "High": "#ff7f0e", "Monitor": "#2ca02c"},
        )
        st.plotly_chart(fig, width="stretch", key="batch_priority_pie")

    st.markdown("---")
    st.subheader("Highest-Risk Customers")
    top_risk = scored.sort_values("ChurnProbability", ascending=False).head(20)
    st.dataframe(top_risk, width="stretch")

    st.markdown("---")
    st.subheader("Full Scored Results")
    st.dataframe(scored, width="stretch")

    csv_bytes = scored.to_csv(index=False).encode("utf-8")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇️ Download scored CSV",
            data=csv_bytes,
            file_name="retainiq_batch_predictions.csv",
            mime="text/csv",
        )
    with col_dl2:
        model = load_model()
        preprocessor = load_preprocessor()
        pdf_bytes = generate_pdf_report(
            scored, summary, model=model, feature_names=preprocessor.feature_columns,
        )
        st.download_button(
            "📄 Download executive PDF report",
            data=pdf_bytes,
            file_name="retainiq_executive_report.pdf",
            mime="application/pdf",
        )

    st.caption(
        "Business summary and CSV both draw from the same scored table above — "
        "the numbers are consistent by construction, not computed separately."
    )
else:
    st.info("Upload a CSV to get started, or try the sample file below.")
    sample_note = (
        "No sample file is bundled with this app — the source dataset at "
        "`data/Telco-Customer-Churn.csv` has the right columns (drop the "
        "`Churn` column first) if you want to try this page with real data."
    )
    st.caption(sample_note)

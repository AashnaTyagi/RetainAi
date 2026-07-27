import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loaders import load_model_comparison, load_metrics

st.set_page_config(page_title="Model Comparison", page_icon="🏆", layout="wide")
st.title("🏆 Model Comparison")

comparison = load_model_comparison()
metrics = load_metrics()

st.markdown(
    f"**Winner: {comparison['winner']}** — selected by "
    f"*{comparison['winner_selection_criterion']}*."
)
st.info(comparison["note"])

comp_df = pd.DataFrame(comparison["phase4_comparison"]).sort_values("f1_score", ascending=False)

st.subheader("Phase 4: 10-Model Comparison")
fig = px.bar(
    comp_df, x="model", y=["precision", "recall", "f1_score", "roc_auc"],
    barmode="group", labels={"value": "Score", "model": "Model", "variable": "Metric"},
)
fig.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig, width="stretch", key="mc_comparison_bar")

st.dataframe(
    comp_df.style.format({
        "accuracy": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}",
        "f1_score": "{:.3f}", "roc_auc": "{:.3f}",
    }).background_gradient(subset=["f1_score"], cmap="Greens"),
    width="stretch",
)

st.subheader("Phase 5: Hyperparameter Tuning")
st.warning(comparison["phase5_tuning_note"])

st.subheader("Production Model — Live Test-Set Performance")
st.caption(
    "Reproduced by `src/train.py` (may differ slightly from the notebook's "
    "numbers above due to a different train/test split draw — both are "
    "valid; the notebook numbers are the ones referenced throughout this app)."
)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
c2.metric("Precision", f"{metrics['precision']:.3f}")
c3.metric("Recall", f"{metrics['recall']:.3f}")
c4.metric("F1 Score", f"{metrics['f1_score']:.3f}")
c5.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

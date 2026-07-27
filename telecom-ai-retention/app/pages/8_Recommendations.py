import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.recommendations import generate_recommendation

st.set_page_config(page_title="Recommendations", page_icon="💡", layout="wide")
st.title("💡 Retention Recommendations")

if "last_prediction" not in st.session_state or "top_risk_features" not in st.session_state:
    st.info(
        "Go to **Customer Prediction**, submit a customer, then open "
        "**Explain Prediction** first — recommendations are grounded in "
        "that customer's specific SHAP risk drivers."
    )
    st.stop()

proba = st.session_state["last_prediction"]["proba"]
top_risk_features = st.session_state["top_risk_features"]

rec = generate_recommendation(top_risk_features, proba)

st.metric("Predicted Churn Probability", f"{proba:.1%}")

if "Urgent" in rec["priority"]:
    st.error(f"**Priority: {rec['priority']}**")
elif "High" in rec["priority"]:
    st.warning(f"**Priority: {rec['priority']}**")
else:
    st.success(f"**Priority: {rec['priority']}**")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Risk Factors")
    for reason in rec["top_reasons"]:
        st.markdown(f"- {reason}")

with col2:
    st.subheader("Recommended Actions")
    for action in rec["recommended_actions"]:
        st.markdown(f"- ✅ {action}")

st.markdown("---")
st.subheader("Agent-Ready Summary")
reasons_text = ", ".join(rec["top_reasons"][:2]).lower()
summary = (
    f"This customer carries a {proba:.0%} predicted churn risk, driven mainly by "
    f"{reasons_text}. Priority: {rec['priority']}. "
    f"Suggested first action: {rec['recommended_actions'][0]}."
)
st.info(summary)

st.caption(
    "This recommendation is grounded entirely in the structured risk-driver "
    "rulebook (src/recommendations.py) applied to this customer's own SHAP "
    "explanation — see notebooks/Analysis.ipynb Phase 10 for the full design, "
    "including the LLM-narrative extension pattern."
)

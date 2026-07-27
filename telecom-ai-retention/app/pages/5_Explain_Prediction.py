import sys
from pathlib import Path

import matplotlib.pyplot as plt
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loaders import load_model

st.set_page_config(page_title="Explain Prediction", page_icon="🧠", layout="wide")
st.title("🧠 Explain Prediction")

if "last_prediction" not in st.session_state:
    st.info("Go to **Customer Prediction** first and submit a customer to see their explanation here.")
    st.stop()

pred = st.session_state["last_prediction"]
X_live = pred["X_live"]
proba = pred["proba"]

st.metric("Predicted Churn Probability", f"{proba:.1%}")

model = load_model()


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


explainer = get_explainer(model)
shap_values = explainer(X_live)

# GradientBoostingClassifier's TreeExplainer returns a single set of
# values for the positive class directly (binary classification, not
# the 3D multiclass shape some other tree models produce).
shap_row = shap_values[0]

st.subheader("Why this prediction — SHAP Waterfall")
st.caption(
    "Starts at the model's average prediction (base value) and shows which "
    "features pushed THIS customer's score up (red, toward churn) or down "
    "(blue, toward staying)."
)

fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.waterfall(shap_row, max_display=12, show=False)
st.pyplot(fig, width="stretch")
plt.close(fig)

st.subheader("Top Contributing Features")
import pandas as pd

contributions = pd.Series(shap_row.values, index=X_live.columns).sort_values(key=abs, ascending=False)
top_contrib_df = contributions.head(10).reset_index()
top_contrib_df.columns = ["Feature", "SHAP Impact"]
top_contrib_df["Direction"] = top_contrib_df["SHAP Impact"].apply(
    lambda v: "⬆️ Toward Churn" if v > 0 else "⬇️ Toward Staying"
)
st.dataframe(top_contrib_df, width="stretch")

# Store the top positive (risk-driving) feature names for the
# Recommendations page.
top_risk_features = contributions[contributions > 0].head(6).index.tolist()
st.session_state["top_risk_features"] = top_risk_features
st.session_state["last_prediction"]["proba"] = proba

st.success("Open **Recommendations** in the sidebar to see suggested retention actions.")

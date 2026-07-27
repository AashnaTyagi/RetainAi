import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loaders import load_model, load_preprocessor
from src.feature_engineering import clean_raw_data, engineer_features
from src.preprocessing import transform

st.set_page_config(page_title="Customer Prediction", page_icon="🔮", layout="wide")
st.title("🔮 Customer Prediction")
st.caption(
    "Enter a customer's details to get a live churn risk score from the "
    "production model. Continue to the Explain Prediction and "
    "Recommendations pages afterward to see why and what to do about it."
)

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])

        st.markdown("**Tenure & Billing**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges))

    with col2:
        st.markdown("**Contract & Payment**")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

        st.markdown("**Phone Service**")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    with col3:
        st.markdown("**Internet & Add-ons**")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    submitted = st.form_submit_button("Predict Churn Risk", type="primary")

if submitted:
    raw_input = pd.DataFrame([{
        "customerID": "live-input",
        "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
        "MultipleLines": multiple_lines, "InternetService": internet_service,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges, "Churn": "No",  # placeholder, dropped before scoring
    }])

    preprocessor = load_preprocessor()
    cleaned = clean_raw_data(raw_input)
    engineered = engineer_features(
        cleaned,
        revenue_bin_edges=preprocessor.revenue_bin_edges,
        spend_bin_edges=preprocessor.spend_bin_edges,
    )
    engineered_for_model = engineered.drop(columns=["Churn"])

    X_live = transform(engineered_for_model, preprocessor)

    model = load_model()
    proba = model.predict_proba(X_live)[0, 1]
    prediction = "Churn" if proba >= 0.5 else "No Churn"

    st.markdown("---")
    st.subheader("Prediction Result")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Predicted Churn Probability", f"{proba:.1%}")
        st.metric("Prediction", prediction)
    with c2:
        if proba >= 0.7:
            st.error("🔴 **Urgent risk** — this customer should be prioritized for retention outreach.")
        elif proba >= 0.4:
            st.warning("🟠 **Elevated risk** — proactive outreach recommended within the week.")
        else:
            st.success("🟢 **Low risk** — no immediate action needed; include in standard engagement.")

    # Store for the Explain Prediction and Recommendations pages
    st.session_state["last_prediction"] = {
        "X_live": X_live,
        "proba": proba,
        "raw_input": raw_input.drop(columns=["customerID", "Churn"]).to_dict(orient="records")[0],
    }
    st.success("Saved — open **Explain Prediction** or **Recommendations** in the sidebar to continue.")

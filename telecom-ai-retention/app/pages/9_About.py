import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
st.title("ℹ️ About This Project")

st.markdown(
    """
## Telecom Customer Retention Intelligence Platform

An end-to-end machine learning system that predicts telecom customer churn,
explains individual predictions, and converts them into concrete retention
actions — built as a full production-style project, not a single notebook.

### Methodology (see `notebooks/Analysis.ipynb` for full detail)

1. **EDA** — data quality, distributions, and churn-rate breakdowns across
   every categorical feature, each with a business interpretation.
2. **Feature Engineering & Preprocessing** — 12 engineered features
   (CLV, risk flags, tenure/revenue segments); fixed a real dtype bug in
   categorical encoding and replaced ordinal-implying LabelEncoder with
   one-hot encoding for nominal features.
3. **Class Imbalance Handling** — compared No Balancing, SMOTE, ADASYN, and
   Class Weights; ADASYN won on recall/F1.
4. **Model Comparison** — 10 models (Logistic Regression through Stacking
   Classifier) compared on Accuracy/Precision/Recall/F1/ROC AUC/Training
   Time; Gradient Boosting won.
5. **Hyperparameter Tuning** — RandomizedSearchCV + GridSearchCV on the top
   4 candidates; an honest negative result (tuning didn't beat the untuned
   winner) is reported as-is.
6. **Model Validation** — 5/10-fold CV, learning/validation curves,
   bias-variance analysis, ROC/PR curves, calibration curve, lift/gain chart.
7. **Explainability** — SHAP (global + per-customer waterfall), LIME
   cross-check, permutation importance.
8. **Business Analytics** — revenue lost to churn, risk by contract/payment/
   internet service, CLV insights, retention-opportunity sizing.
9. **Customer Segmentation** — KMeans, DBSCAN, and hierarchical clustering,
   auto-labeled into actionable Risk × Value × Loyalty segments.
10. **AI Retention Engine** — per-customer recommendations grounded in SHAP
    drivers, with a working LLM-prompt integration pattern.
11. **This Streamlit App** — a live demo of the trained model.

### Tech Stack

**ML/Data:** Python, Pandas, NumPy, scikit-learn, XGBoost, LightGBM, CatBoost,
imbalanced-learn, SHAP, LIME
**App:** Streamlit, Plotly, Matplotlib, Seaborn
**Project structure:** modular `src/` package shared between the notebook,
this app, and (in later phases) a FastAPI backend

### Resume Bullet Points

> Built an AI-powered Telecom Customer Retention Intelligence Platform using
> ensemble machine learning, explainable AI, and business analytics — compared
> 10 classification models, selected the best performer by F1 score, and
> validated it with cross-validation, calibration analysis, and a lift/gain
> chart tying model output to revenue impact.

> Developed an end-to-end ML pipeline featuring 12 engineered features,
> ADASYN class-imbalance handling, SHAP/LIME explainability, KMeans/DBSCAN
> customer segmentation, and a grounded AI retention recommendation engine,
> deployed as an interactive Streamlit application.

### Project Structure

```
RetainAI/
├── src/                    # Shared, reusable pipeline modules
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   └── recommendations.py
├── models/                 # Trained model + reference artifacts
├── data/                   # Source dataset
├── app/                    # This Streamlit application
│   ├── Home.py
│   └── pages/
├── notebooks/              # Full exploratory analysis (Analysis.ipynb)
├── tests/                  # (Phase 15)
└── requirements.txt
```
"""
)
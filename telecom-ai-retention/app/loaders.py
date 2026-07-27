"""Shared, cached data/model loaders for the Streamlit app. Every page
imports from here instead of re-loading artifacts itself, so the model
and reference data are only ever read from disk once per session."""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "churn_model.pkl")


@st.cache_resource
def load_preprocessor():
    from src.preprocessing import ChurnPreprocessor
    return ChurnPreprocessor.load(str(MODELS_DIR / "preprocessor.pkl"))


@st.cache_data
def load_metrics() -> dict:
    with open(MODELS_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_model_comparison() -> dict:
    with open(MODELS_DIR / "model_comparison.json") as f:
        return json.load(f)


@st.cache_data
def load_business_findings() -> dict:
    with open(MODELS_DIR / "business_findings.json") as f:
        return json.load(f)


@st.cache_data
def load_test_reference() -> pd.DataFrame:
    return pd.read_csv(MODELS_DIR / "test_reference.csv")


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "Telco-Customer-Churn.csv")


@st.cache_data
def load_feature_columns() -> list:
    with open(MODELS_DIR / "feature_columns.json") as f:
        return json.load(f)

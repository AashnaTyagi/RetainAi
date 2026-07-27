"""
Trains and saves the production model for the Telecom Customer
Retention Intelligence Platform.

This trains the SPECIFIC winning configuration already established and
validated across Phases 1-6 of notebooks/Analysis.ipynb: ADASYN-balanced
Gradient Boosting. It does not re-run the full 10-model comparison or
hyperparameter search -- that exploratory work lives in the notebook,
which is the right place for it. This script is the reproducible,
production path: clean data in, tracked run + registered model out.

Every run is logged to MLflow (params, metrics, model artifact,
confusion matrix, feature importance plot) and evaluated for promotion
to the "Production" stage of the model registry: a new run is only
promoted if it beats the current Production model's F1 score, so the
registry can never silently regress.

Usage:
    python -m src.train
    python -m src.train --no-promote   # log the run but skip registry promotion
"""

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split

from src.feature_engineering import clean_raw_data, compute_quantile_bins, engineer_features
from src.logger import get_logger
from src.preprocessing import fit_transform

logger = get_logger("train")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "Telco-Customer-Churn.csv"
MODELS_DIR = ROOT_DIR / "models"
DOCS_DIR = ROOT_DIR / "docs"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
EXPERIMENT_NAME = "telecom-churn-prediction"
REGISTERED_MODEL_NAME = "telecom-churn-model"

MODEL_PARAMS = {
    "model_type": "GradientBoostingClassifier",
    "random_state": 42,
    "imbalance_strategy": "ADASYN",
    "test_size": 0.2,
}


def load_and_prepare_raw(data_path: Path = DATA_PATH):
    df = pd.read_csv(data_path)
    df = clean_raw_data(df)
    return df


def _log_confusion_matrix_artifact(cm: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["No Churn", "Churn"])
    ax.set_yticks([0, 1], ["No Churn", "Churn"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im)
    path = MODELS_DIR / "confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _log_feature_importance_artifact(model, feature_names) -> Path:
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax)
    ax.set_title("Top 15 Feature Importances")
    path = MODELS_DIR / "feature_importance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _get_current_production_f1(client: MlflowClient) -> float:
    """Returns the F1 score of the model version currently holding the
    'production' alias, or -1 if no version holds it yet (so the first
    run is always promoted). Uses the modern registry aliases API --
    the older stage-based API (Staging/Production as fixed stages) is
    deprecated as of MLflow 2.9."""
    try:
        version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "production")
    except Exception:
        return -1.0
    run = client.get_run(version.run_id)
    return run.data.metrics.get("f1_score", -1.0)


def train(promote: bool = True):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    logger.info("Loading and preparing data...")
    df = load_and_prepare_raw()

    logger.info("Computing quantile bin edges from training data "
                "(reused at inference so single-row predictions don't break)")
    bins = compute_quantile_bins(df)
    df = engineer_features(df, **bins)

    logger.info("Fitting encoders/scaler and building the feature matrix...")
    X, y, preprocessor = fit_transform(
        df, revenue_bin_edges=bins["revenue_bin_edges"], spend_bin_edges=bins["spend_bin_edges"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=MODEL_PARAMS["test_size"], random_state=MODEL_PARAMS["random_state"], stratify=y
    )

    with mlflow.start_run() as run:
        logger.info(f"MLflow run started: {run.info.run_id}")
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_param("n_features", X.shape[1])

        logger.info("Balancing training data with ADASYN "
                    "(winning strategy from Phase 3 of the notebook)")
        adasyn = ADASYN(random_state=MODEL_PARAMS["random_state"])
        X_train_bal, y_train_bal = adasyn.fit_resample(X_train, y_train)
        mlflow.log_param("n_train_samples_after_adasyn", int(len(X_train_bal)))

        logger.info("Training Gradient Boosting "
                    "(winning model from Phase 4 of the notebook)")
        start = time.time()
        model = GradientBoostingClassifier(random_state=MODEL_PARAMS["random_state"])
        model.fit(X_train_bal, y_train_bal)
        train_time = time.time() - start

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "train_time_seconds": round(train_time, 3),
        }
        mlflow.log_metrics(metrics)

        logger.info("Test set performance: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        logger.info("\n" + classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        cm_path = _log_confusion_matrix_artifact(cm)
        fi_path = _log_feature_importance_artifact(model, X.columns)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(fi_path))

        model_info = mlflow.sklearn.log_model(
            model, name="model", registered_model_name=REGISTERED_MODEL_NAME,
        )
        new_version = model_info.registered_model_version
        logger.info(f"Registered as {REGISTERED_MODEL_NAME} version {new_version}")

        # --- Registry gating: only promote if this run actually beats
        # whatever is currently in Production, so the registry can't regress.
        current_prod_f1 = _get_current_production_f1(client)
        will_promote = promote and metrics["f1_score"] > current_prod_f1

        if will_promote:
            client.set_registered_model_alias(
                name=REGISTERED_MODEL_NAME, alias="production", version=new_version,
            )
            client.set_model_version_tag(REGISTERED_MODEL_NAME, new_version, "stage", "production")
            logger.info(
                f"Promoted version {new_version} to 'production' alias "
                f"(F1 {metrics['f1_score']:.4f} > previous production F1 {current_prod_f1:.4f})"
            )
        else:
            client.set_model_version_tag(REGISTERED_MODEL_NAME, new_version, "stage", "staging")
            reason = "promotion disabled (--no-promote)" if not promote else (
                f"F1 {metrics['f1_score']:.4f} did not beat current production F1 {current_prod_f1:.4f}"
            )
            logger.info(f"Kept version {new_version} in staging — {reason}")

        metrics_out = {
            "model_name": "Gradient Boosting",
            **metrics,
            "n_features": X.shape[1],
            "n_train_samples": int(len(X_train_bal)),
            "n_test_samples": int(len(X_test)),
            "mlflow_run_id": run.info.run_id,
            "registered_version": new_version,
            "promoted_to_production": will_promote,
        }

        # --- Save the flat-file artifacts the app/API load directly (fast
        # path, no MLflow client round-trip needed at serving time). These
        # always reflect the LATEST run, independent of registry stage --
        # app/API deployment against a specific registry stage is a
        # deliberate later-phase (deployment) decision, not this script's.
        logger.info(f"Saving flat-file artifacts to {MODELS_DIR} ...")
        joblib.dump(model, MODELS_DIR / "churn_model.pkl")
        preprocessor.save(str(MODELS_DIR / "preprocessor.pkl"))

        with open(MODELS_DIR / "feature_columns.json", "w") as f:
            json.dump(preprocessor.feature_columns, f, indent=2)
        with open(MODELS_DIR / "metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2)
        with open(MODELS_DIR / "confusion_matrix.json", "w") as f:
            json.dump(cm.tolist(), f, indent=2)

        test_reference = X_test.copy()
        test_reference["Churn"] = y_test.values
        test_reference["PredictedProbability"] = y_proba
        test_reference.to_csv(MODELS_DIR / "test_reference.csv", index=False)

        logger.info("Done.")
        return model, preprocessor, metrics_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-promote", action="store_true",
        help="Log the run and register a version, but skip Production promotion.",
    )
    args = parser.parse_args()
    train(promote=not args.no_promote)

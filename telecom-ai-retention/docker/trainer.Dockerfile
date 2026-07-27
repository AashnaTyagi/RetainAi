# Standalone retraining utility image. NOT required for backend/frontend
# to work -- those now train themselves at build time (see the
# multi-stage builds in backend.Dockerfile / frontend.Dockerfile) and
# are fully self-contained. This image exists for the separate MLOps
# workflow from Phase 13: running a retrain against a SHARED, persistent
# MLflow tracking server (not the ephemeral build-time one), so a new
# run can be evaluated for promotion to the registry's "production"
# alias alongside every other run's history.
#
# Typical use: a scheduled job (cron, GitHub Actions, cloud scheduler)
# runs this image periodically against a real MLflow tracking server:
#
#   docker run --rm \
#     -e MLFLOW_TRACKING_URI=https://your-mlflow-server \
#     telecom-churn-trainer
#
# (Registry promotion still requires the explicit --no-promote flag to
# be OMITTED -- see src/train.py's CLI for the gating logic.)

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

COPY src/ ./src/
COPY data/ ./data/

ENTRYPOINT ["python", "-m", "src.train"]

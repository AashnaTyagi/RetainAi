# Frontend image: same multi-stage pattern as backend.Dockerfile --
# trains at build time in an isolated stage, copies only the trained
# artifacts (not training dependencies) into the lean runtime image.

# ---- Stage 1: train ----
FROM python:3.11-slim AS trainer

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

COPY src/ ./src/
COPY data/ ./data/

RUN python -m src.train --no-promote

# ---- Stage 2: runtime ----
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-frontend.txt .
RUN pip install --no-cache-dir -r requirements-frontend.txt

COPY src/ ./src/
COPY app/ ./app/
COPY data/ ./data/

COPY --from=trainer /app/models/churn_model.pkl ./models/churn_model.pkl
COPY --from=trainer /app/models/preprocessor.pkl ./models/preprocessor.pkl
COPY --from=trainer /app/models/feature_columns.json ./models/feature_columns.json
COPY --from=trainer /app/models/metrics.json ./models/metrics.json
COPY models/model_comparison.json models/business_findings.json ./models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health').read()" || exit 1

CMD ["streamlit", "run", "app/Home.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

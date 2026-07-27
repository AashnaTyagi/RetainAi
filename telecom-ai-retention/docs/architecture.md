# RetainAI — Architecture

```mermaid
flowchart TB
    subgraph Research["Research (offline, one-time per methodology change)"]
        CSV[(Telco-Customer-Churn.csv)]
        NB[notebooks/Analysis.ipynb<br/>10-phase validated study:<br/>EDA → features → imbalance →<br/>model comparison → tuning →<br/>validation → explainability →<br/>business analytics → segmentation →<br/>retention engine]
        CSV --> NB
    end

    subgraph Core["src/ — shared source of truth"]
        FE[feature_engineering.py<br/>12 engineered features]
        PP[preprocessing.py<br/>encoding + scaling]
        TR[train.py<br/>ADASYN + Gradient Boosting<br/>+ MLflow tracking]
        RE[recommendations.py<br/>SHAP-driver → action rulebook]
        FE --> PP --> TR
    end

    subgraph MLOps
        MLF[(MLflow Registry<br/>production alias,<br/>gated by F1 comparison)]
        DVC[[DVC pipeline<br/>data.csv.dvc → train stage]]
        TR --> MLF
        DVC -.drives.-> TR
    end

    subgraph Artifacts["models/ (generated)"]
        PKL[churn_model.pkl<br/>preprocessor.pkl<br/>feature_columns.json<br/>metrics.json]
    end
    TR --> PKL

    subgraph Serving
        APP[Streamlit App<br/>11 pages]
        API[FastAPI Backend<br/>/predict /explain /metrics ...]
    end
    PKL --> APP
    PKL --> API
    RE --> APP
    RE --> API

    NB -. validated methodology,<br/>hand-distilled .-> FE

    subgraph Deploy
        DOCKER[docker/ multi-stage builds<br/>self-train at image-build time]
        CLOUD[Render / Railway / AWS /<br/>Azure / GCP]
    end
    APP --> DOCKER
    API --> DOCKER
    DOCKER --> CLOUD
```

## Design decisions worth calling out

**Why `src/` exists separately from the notebook.** The notebook is where the
10-phase methodology was actually developed and validated — every modeling
decision (ADASYN over SMOTE, Gradient Boosting over the other 9 models,
tuning that didn't help) was tested there first. `src/` is a hand-distilled,
production version of only the *winning* configuration, shared identically by
training, the app, and the API. This is a deliberate one-way flow: findings
move from notebook → `src/`, not the other way, and `src/` never re-runs the
full exploratory comparison.

**Why the app and API don't call each other.** Both independently import
`src/` rather than the app calling the API over HTTP. This avoids a network
hop for what's otherwise the same in-process computation, at the cost of not
demonstrating the two services actually integrated — a real trade-off, noted
honestly in `deploy/DEPLOYMENT.md` as a documented gap rather than glossed
over.

**Why Docker images self-train at build time.** An earlier design used a
shared volume and a one-shot "trainer" container that `backend`/`frontend`
waited on. That pattern is docker-compose-specific and doesn't port to
Render, Railway, App Runner, Container Apps, or Cloud Run, which don't have
"run this once, then start these" as a first-class concept. Multi-stage
builds (train in an isolated build stage, copy only the trained artifacts
into the runtime image) work identically everywhere, including plain
`docker build` with no orchestration at all.

**Why MLflow registry promotion is gated.** A new training run is only
promoted to the `production` alias if its F1 score beats whatever is
currently there — verified by testing, not just documented: a tied rerun
correctly stayed in staging instead of falsely re-promoting (see project
history / test suite for the specifics).

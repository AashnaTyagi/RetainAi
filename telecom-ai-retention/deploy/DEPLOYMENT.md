# Cloud Deployment Guide

Every option below deploys the same two self-contained images built from
`docker/backend.Dockerfile` and `docker/frontend.Dockerfile`. Both are
multi-stage builds that train the model fresh at image-build time — there's no
separate "run training, then deploy" step on any platform, and no shared volume
or service-ordering dependency to get wrong.

**Honest status:** these configs were authored and validated as far as this
environment allows — YAML/JSON syntax checked, shell scripts checked with
`sh -n`, every file path referenced actually exists, and the underlying
Dockerfiles/commands are verified working outside a container (see the main
README's testing notes). None of these have been run against a real cloud
account — this sandbox has no cloud credentials or CLI access. Treat this as a
solid, reviewed starting point, not a "click deploy and it definitely works"
guarantee. Each section below notes the one or two things most likely to need
a small adjustment for your specific account/region.

---

## Render (`deploy/render.yaml`)

Simplest option — a single Blueprint file, no CLI setup needed.

1. Push this repo to GitHub.
2. In the Render dashboard: **New +** → **Blueprint** → point it at the repo
   and select `deploy/render.yaml`.
3. Render builds both services from their Dockerfiles automatically.

Likely adjustment: the `starter` plan may need to change to `free` or a paid
tier depending on your account and how much RAM the SHAP/sklearn install needs
at build time (typically fine on `starter`).

---

## Railway (`deploy/railway-backend.json`, `deploy/railway-frontend.json`)

1. `railway init` in the repo root, or connect the GitHub repo in the Railway dashboard.
2. Create two services, each pointed at the respective Dockerfile
   (`docker/backend.Dockerfile`, `docker/frontend.Dockerfile`).
3. Copy the matching JSON file's content into that service's settings, or
   place it at the repo root as `railway.json` per service if deploying them
   as separate repos/branches.

Railway injects a dynamic `$PORT` — both configs' `startCommand` already bind
to it rather than the Dockerfile's hardcoded port, so this should work as-is.

---

## AWS (`deploy/aws/apprunner-backend.yaml`, `apprunner-frontend.yaml`)

Uses AWS App Runner — the simplest AWS path for a two-container app (no ECS
cluster/task-definition management needed).

```bash
aws ecr create-repository --repository-name telecom-churn-backend
aws ecr create-repository --repository-name telecom-churn-frontend

aws ecr get-login-password | docker login --username AWS \
    --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com

docker build -f docker/backend.Dockerfile -t telecom-churn-backend .
docker tag telecom-churn-backend:latest \
    <account-id>.dkr.ecr.<region>.amazonaws.com/telecom-churn-backend:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/telecom-churn-backend:latest

# repeat build/tag/push for frontend with docker/frontend.Dockerfile

aws apprunner create-service --cli-input-yaml file://deploy/aws/apprunner-backend.yaml
aws apprunner create-service --cli-input-yaml file://deploy/aws/apprunner-frontend.yaml
```

**Required edit before running:** replace `<account-id>` and `<region>` in
both YAML files' `ImageIdentifier` with your actual AWS account ID and region.

---

## Azure (`deploy/azure/deploy.sh`)

Uses Azure Container Apps with `az acr build`, which builds images in the
cloud — no local Docker daemon required, matching how this project was
actually developed and tested.

```bash
az login
ACR_NAME=<your-acr-name> RESOURCE_GROUP=telecom-churn-rg LOCATION=eastus \
    ./deploy/azure/deploy.sh
```

**Required edit:** set `ACR_NAME` to an Azure Container Registry you've
created (`az acr create --name <name> --resource-group <rg> --sku Basic`), or
add that step to the top of the script.

---

## GCP (`deploy/gcp/cloudbuild-backend.yaml`, `cloudbuild-frontend.yaml`)

Uses Cloud Build (cloud-side build, same rationale as the Azure option) to
build and deploy straight to Cloud Run.

```bash
gcloud auth login
gcloud config set project <your-project-id>

gcloud builds submit --config deploy/gcp/cloudbuild-backend.yaml .
gcloud builds submit --config deploy/gcp/cloudbuild-frontend.yaml .
```

No edits needed — `$PROJECT_ID` and `$SHORT_SHA` are populated automatically
by Cloud Build from your active gcloud project and the current commit.

---

## Connecting frontend → backend

The Streamlit app currently calls the model directly (via `src/` modules), not
through the deployed backend API — so `frontend` and `backend` can be deployed
independently of each other as-is. If you want the frontend to call the live
API instead (e.g. to demonstrate the two services actually integrated), the
`BACKEND_URL` environment variable is threaded through in the Render and Azure
configs above as a starting point; wiring the Streamlit pages to actually call
it via `requests` instead of importing `src/` directly is a small follow-up
change, not something already implemented.

#!/bin/bash
# Deploy to Azure Container Apps.
#
# Prereqs: az CLI logged in (`az login`), an Azure Container Registry
# (ACR) to push images to.
#
# Usage:
#   RESOURCE_GROUP=telecom-churn-rg \
#   ACR_NAME=telecomchurnacr \
#   LOCATION=eastus \
#   ./deploy/azure/deploy.sh

set -e

RESOURCE_GROUP="${RESOURCE_GROUP:-telecom-churn-rg}"
ACR_NAME="${ACR_NAME:?Set ACR_NAME to your Azure Container Registry name}"
LOCATION="${LOCATION:-eastus}"
ENV_NAME="telecom-churn-env"

echo "Creating resource group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

echo "Building and pushing images to ACR (az acr build runs the build in the cloud -- no local Docker daemon required)..."
az acr build --registry "$ACR_NAME" --image telecom-churn-backend:latest \
    --file docker/backend.Dockerfile .
az acr build --registry "$ACR_NAME" --image telecom-churn-frontend:latest \
    --file docker/frontend.Dockerfile .

echo "Creating Container Apps environment..."
az containerapp env create \
    --name "$ENV_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION"

echo "Deploying backend..."
az containerapp create \
    --name telecom-churn-backend \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$ENV_NAME" \
    --image "$ACR_NAME.azurecr.io/telecom-churn-backend:latest" \
    --registry-server "$ACR_NAME.azurecr.io" \
    --target-port 8000 \
    --ingress external \
    --cpu 1.0 --memory 2.0Gi \
    --min-replicas 1 --max-replicas 3

BACKEND_URL=$(az containerapp show --name telecom-churn-backend \
    --resource-group "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv)

echo "Deploying frontend (backend at https://$BACKEND_URL)..."
az containerapp create \
    --name telecom-churn-frontend \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$ENV_NAME" \
    --image "$ACR_NAME.azurecr.io/telecom-churn-frontend:latest" \
    --registry-server "$ACR_NAME.azurecr.io" \
    --target-port 8501 \
    --ingress external \
    --env-vars "BACKEND_URL=https://$BACKEND_URL" \
    --cpu 1.0 --memory 2.0Gi \
    --min-replicas 1 --max-replicas 3

echo "Done. Frontend URL:"
az containerapp show --name telecom-churn-frontend \
    --resource-group "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv

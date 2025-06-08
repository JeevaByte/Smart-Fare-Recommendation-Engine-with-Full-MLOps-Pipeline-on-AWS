#!/bin/bash
# Script to set up MLflow tracking server

# Set AWS credentials
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="eu-west-2"

# Get S3 bucket name from Terraform output
cd infra/terraform
MODEL_ARTIFACTS_BUCKET=$(terraform output -raw model_artifacts_bucket)
cd ../..

# Create MLflow container
docker run -d \
  --name mlflow-server \
  -p 5000:5000 \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION \
  -e MLFLOW_S3_ENDPOINT_URL=https://s3.$AWS_DEFAULT_REGION.amazonaws.com \
  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  -e BACKEND_URI=sqlite:///mlflow.db \
  -e ARTIFACT_ROOT=s3://$MODEL_ARTIFACTS_BUCKET/mlflow-artifacts \
  ghcr.io/mlflow/mlflow:latest \
  mlflow server \
  --host 0.0.0.0 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://$MODEL_ARTIFACTS_BUCKET/mlflow-artifacts

echo "MLflow tracking server started at http://localhost:5000"
echo "Artifact storage: s3://$MODEL_ARTIFACTS_BUCKET/mlflow-artifacts"
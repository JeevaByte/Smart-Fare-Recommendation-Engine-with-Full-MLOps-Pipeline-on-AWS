# PowerShell script to set up MLflow tracking server

# Set AWS credentials
$env:AWS_ACCESS_KEY_ID = "your-access-key"
$env:AWS_SECRET_ACCESS_KEY = "your-secret-key"
$env:AWS_DEFAULT_REGION = "eu-west-2"

# Get S3 bucket name from Terraform output
Set-Location -Path ".\infra\terraform"
$MODEL_ARTIFACTS_BUCKET = terraform output -raw model_artifacts_bucket
Set-Location -Path "..\..\"

# Create MLflow container
docker run -d `
  --name mlflow-server `
  -p 5000:5000 `
  -e AWS_ACCESS_KEY_ID=$env:AWS_ACCESS_KEY_ID `
  -e AWS_SECRET_ACCESS_KEY=$env:AWS_SECRET_ACCESS_KEY `
  -e AWS_DEFAULT_REGION=$env:AWS_DEFAULT_REGION `
  -e MLFLOW_S3_ENDPOINT_URL="https://s3.$env:AWS_DEFAULT_REGION.amazonaws.com" `
  -e MLFLOW_TRACKING_URI="http://localhost:5000" `
  -e BACKEND_URI="sqlite:///mlflow.db" `
  -e ARTIFACT_ROOT="s3://$MODEL_ARTIFACTS_BUCKET/mlflow-artifacts" `
  ghcr.io/mlflow/mlflow:latest `
  mlflow server `
  --host 0.0.0.0 `
  --backend-store-uri sqlite:///mlflow.db `
  --default-artifact-root "s3://$MODEL_ARTIFACTS_BUCKET/mlflow-artifacts"

Write-Host "MLflow tracking server started at http://localhost:5000"
Write-Host "Artifact storage: s3://$MODEL_ARTIFACTS_BUCKET/mlflow-artifacts"
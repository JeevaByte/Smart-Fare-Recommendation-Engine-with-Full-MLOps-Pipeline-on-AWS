# PowerShell script to run the initial data generation and model training

# Set environment variables
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:AWS_ACCESS_KEY_ID = "your-access-key"
$env:AWS_SECRET_ACCESS_KEY = "your-secret-key"
$env:AWS_DEFAULT_REGION = "eu-west-2"

# Get S3 bucket names from Terraform output
Set-Location -Path ".\infra\terraform"
$RAW_DATA_BUCKET = terraform output -raw raw_data_bucket
$PROCESSED_DATA_BUCKET = terraform output -raw processed_data_bucket
Set-Location -Path "..\..\"

Write-Host "Using buckets:"
Write-Host "Raw data: $RAW_DATA_BUCKET"
Write-Host "Processed data: $PROCESSED_DATA_BUCKET"

# Generate data
Write-Host "Generating synthetic data..."
python src/data/generate_data.py `
  --samples 50000 `
  --output data/raw/train_fares.csv `
  --upload-s3 `
  --s3-bucket $RAW_DATA_BUCKET `
  --s3-prefix raw/

# Process data with PySpark
Write-Host "Processing data with PySpark..."
python -m src.data.process_data `
  --input "s3://$RAW_DATA_BUCKET/raw/train_fares.csv" `
  --output "s3://$PROCESSED_DATA_BUCKET/processed/initial/" `
  --save-pipeline "s3://$PROCESSED_DATA_BUCKET/pipelines/initial/"

# Train model
Write-Host "Training model..."
python -m src.model.train `
  --data-path "s3://$PROCESSED_DATA_BUCKET/processed/initial/" `
  --model-name fare_recommendation_model `
  --mlflow-tracking-uri $env:MLFLOW_TRACKING_URI

# Evaluate model
Write-Host "Evaluating model..."
python -m src.model.evaluate `
  --model-uri "models:/fare_recommendation_model/latest" `
  --data-path "s3://$PROCESSED_DATA_BUCKET/processed/initial/" `
  --output-dir "./evaluation_results"

Write-Host "Initial pipeline completed successfully!"
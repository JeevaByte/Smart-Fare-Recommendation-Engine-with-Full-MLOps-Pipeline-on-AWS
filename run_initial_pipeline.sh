#!/bin/bash
# Script to run the initial data generation and model training

# Set environment variables
export MLFLOW_TRACKING_URI="http://localhost:5000"
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="eu-west-2"

# Get S3 bucket names from Terraform output
cd infra/terraform
RAW_DATA_BUCKET=$(terraform output -raw raw_data_bucket)
PROCESSED_DATA_BUCKET=$(terraform output -raw processed_data_bucket)
cd ../..

echo "Using buckets:"
echo "Raw data: $RAW_DATA_BUCKET"
echo "Processed data: $PROCESSED_DATA_BUCKET"

# Generate data
echo "Generating synthetic data..."
python src/data/generate_data.py \
  --samples 50000 \
  --output data/raw/train_fares.csv \
  --upload-s3 \
  --s3-bucket $RAW_DATA_BUCKET \
  --s3-prefix raw/

# Process data with PySpark
echo "Processing data with PySpark..."
python -m src.data.process_data \
  --input s3://$RAW_DATA_BUCKET/raw/train_fares.csv \
  --output s3://$PROCESSED_DATA_BUCKET/processed/initial/ \
  --save-pipeline s3://$PROCESSED_DATA_BUCKET/pipelines/initial/

# Train model
echo "Training model..."
python -m src.model.train \
  --data-path s3://$PROCESSED_DATA_BUCKET/processed/initial/ \
  --model-name fare_recommendation_model \
  --mlflow-tracking-uri $MLFLOW_TRACKING_URI

# Evaluate model
echo "Evaluating model..."
python -m src.model.evaluate \
  --model-uri models:/fare_recommendation_model/latest \
  --data-path s3://$PROCESSED_DATA_BUCKET/processed/initial/ \
  --output-dir ./evaluation_results

echo "Initial pipeline completed successfully!"
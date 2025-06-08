#!/bin/bash
# Script to test the API and monitoring systems

# Set environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="eu-west-2"

# Get API endpoint from Terraform output
cd infra/terraform
API_ENDPOINT=$(terraform output -raw api_endpoint)
cd ../..

echo "API endpoint: $API_ENDPOINT"

# Build and run the Docker container locally for testing
echo "Building Docker image..."
docker build -t fare-recommendation-api -f infra/docker/Dockerfile .

echo "Running Docker container..."
docker run -d --name fare-api-test -p 8000:8000 \
  -e MLFLOW_TRACKING_URI="http://host.docker.internal:5000" \
  -e MODEL_URI="models:/fare_recommendation_model/latest" \
  fare-recommendation-api

echo "Waiting for API to start..."
sleep 5

# Test health endpoint
echo "Testing health endpoint..."
curl -s http://localhost:8000/health | jq

# Test prediction endpoint
echo "Testing prediction endpoint..."
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "origin_station": "London Kings Cross",
    "destination_station": "Edinburgh Waverley",
    "booking_days_ahead": 7,
    "travel_time_minutes": 240,
    "time_of_day": 10,
    "day_of_week": 1,
    "train_operator": "LNER",
    "travel_class": "standard",
    "user_type": "standard",
    "is_peak": 0,
    "is_weekend": 0,
    "is_holiday": 0,
    "distance_miles": 400
  }' | jq

# Test monitoring
echo "Testing drift detection..."
python -m src.monitoring.drift_detection \
  --reference-data s3://$PROCESSED_DATA_BUCKET/processed/initial/ \
  --current-data s3://$PROCESSED_DATA_BUCKET/processed/initial/ \
  --model-uri models:/fare_recommendation_model/latest \
  --output-dir ./drift_reports

echo "API and monitoring tests completed!"
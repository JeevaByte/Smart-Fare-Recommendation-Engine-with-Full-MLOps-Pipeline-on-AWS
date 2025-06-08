# PowerShell script to test the API and monitoring systems

# Set environment variables
$env:AWS_ACCESS_KEY_ID = "your-access-key"
$env:AWS_SECRET_ACCESS_KEY = "your-secret-key"
$env:AWS_DEFAULT_REGION = "eu-west-2"

# Get API endpoint from Terraform output
Set-Location -Path ".\infra\terraform"
$API_ENDPOINT = terraform output -raw api_endpoint
Set-Location -Path "..\..\"

Write-Host "API endpoint: $API_ENDPOINT"

# Build and run the Docker container locally for testing
Write-Host "Building Docker image..."
docker build -t fare-recommendation-api -f infra/docker/Dockerfile .

Write-Host "Running Docker container..."
docker run -d --name fare-api-test -p 8000:8000 `
  -e MLFLOW_TRACKING_URI="http://host.docker.internal:5000" `
  -e MODEL_URI="models:/fare_recommendation_model/latest" `
  fare-recommendation-api

Write-Host "Waiting for API to start..."
Start-Sleep -Seconds 5

# Test health endpoint
Write-Host "Testing health endpoint..."
Invoke-RestMethod -Uri "http://localhost:8000/health" | ConvertTo-Json

# Test prediction endpoint
Write-Host "Testing prediction endpoint..."
$body = @{
    origin_station = "London Kings Cross"
    destination_station = "Edinburgh Waverley"
    booking_days_ahead = 7
    travel_time_minutes = 240
    time_of_day = 10
    day_of_week = 1
    train_operator = "LNER"
    travel_class = "standard"
    user_type = "standard"
    is_peak = 0
    is_weekend = 0
    is_holiday = 0
    distance_miles = 400
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json" | ConvertTo-Json

# Test monitoring
Write-Host "Testing drift detection..."
python -m src.monitoring.drift_detection `
  --reference-data "s3://$PROCESSED_DATA_BUCKET/processed/initial/" `
  --current-data "s3://$PROCESSED_DATA_BUCKET/processed/initial/" `
  --model-uri "models:/fare_recommendation_model/latest" `
  --output-dir "./drift_reports"

Write-Host "API and monitoring tests completed!"
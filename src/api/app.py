#!/usr/bin/env python
"""
FastAPI application for serving fare recommendation model.
"""

import os
import logging
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import mlflow
import mlflow.lightgbm
import time
from datetime import datetime
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fare Recommendation API",
    description="API for predicting optimal train fares",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request models
class FareRequest(BaseModel):
    origin_station: str = Field(..., description="Origin station name")
    destination_station: str = Field(..., description="Destination station name")
    booking_days_ahead: int = Field(..., description="Days ahead of travel for booking")
    travel_time_minutes: int = Field(..., description="Travel time in minutes")
    time_of_day: int = Field(..., description="Hour of day (0-23)")
    day_of_week: int = Field(..., description="Day of week (0=Monday, 6=Sunday)")
    train_operator: str = Field(..., description="Train operator name")
    travel_class: str = Field(..., description="Class of travel (standard or first)")
    user_type: str = Field(..., description="User type (standard, business, student, senior, family)")
    is_peak: int = Field(..., description="Is peak time (0 or 1)")
    is_weekend: int = Field(..., description="Is weekend (0 or 1)")
    is_holiday: Optional[int] = Field(0, description="Is holiday (0 or 1)")
    distance_miles: int = Field(..., description="Distance in miles")

class FareResponse(BaseModel):
    predicted_fare: float = Field(..., description="Predicted optimal fare")
    confidence: float = Field(..., description="Confidence score (0-1)")
    request_id: str = Field(..., description="Unique request identifier")
    model_version: str = Field(..., description="Model version used for prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

# Global variables
model = None
model_version = "unknown"
feature_names = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, model_version, feature_names
    
    # Get model URI from environment variable or use default
    model_uri = os.getenv("MODEL_URI", "models:/fare_recommendation_model/latest")
    
    try:
        logger.info(f"Loading model from {model_uri}")
        
        # Set MLflow tracking URI if provided
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Load the model
        model = mlflow.lightgbm.load_model(model_uri)
        
        # Get model version
        if "models:/" in model_uri:
            model_name = model_uri.split("/")[0].replace("models:", "")
            model_version = model_uri.split("/")[1]
            logger.info(f"Loaded {model_name} version {model_version}")
        else:
            model_version = "custom"
            logger.info("Loaded custom model")
        
        # Get feature names from model
        if hasattr(model, "feature_name"):
            feature_names = model.feature_name()
        else:
            feature_names = None
            logger.warning("Could not extract feature names from model")
        
        logger.info("Model loaded successfully")
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Fare Recommendation API",
        "status": "active",
        "model_version": model_version,
        "endpoints": ["/predict", "/health", "/metrics"]
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {"status": "healthy", "model_version": model_version}

@app.post("/predict", response_model=FareResponse)
async def predict(request: FareRequest):
    """Predict fare based on input features"""
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        input_data = {
            "origin_station": [request.origin_station],
            "destination_station": [request.destination_station],
            "booking_days_ahead": [request.booking_days_ahead],
            "travel_time_minutes": [request.travel_time_minutes],
            "time_of_day": [request.time_of_day],
            "day_of_week": [request.day_of_week],
            "train_operator": [request.train_operator],
            "class": [request.travel_class],
            "user_type": [request.user_type],
            "is_peak": [request.is_peak],
            "is_weekend": [request.is_weekend],
            "is_holiday": [request.is_holiday],
            "distance_miles": [request.distance_miles]
        }
        
        df = pd.DataFrame(input_data)
        
        # Feature engineering (same as in training)
        df = feature_engineering(df)
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Generate request ID
        request_id = f"req_{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        # Log prediction
        logger.info(f"Prediction: {prediction:.2f}, Request ID: {request_id}")
        
        # Return response
        return FareResponse(
            predicted_fare=round(float(prediction), 2),
            confidence=0.95,  # Placeholder for confidence score
            request_id=request_id,
            model_version=model_version,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Return API metrics"""
    return {
        "model_version": model_version,
        "uptime_seconds": time.time() - startup_time,
        "request_count": request_count,
        "error_count": error_count
    }

def feature_engineering(df):
    """Perform feature engineering on input data"""
    # Convert time_of_day to categorical time buckets
    df["time_bucket"] = pd.cut(
        df["time_of_day"],
        bins=[-1, 6, 10, 16, 20, 24],
        labels=["night_off_peak", "morning_peak", "day_off_peak", "evening_peak", "night_off_peak"]
    )
    
    # Bucket booking days ahead
    df["booking_window"] = pd.cut(
        df["booking_days_ahead"],
        bins=[-1, 7, 14, 30, 60, float('inf')],
        labels=["last_minute", "one_week", "two_weeks", "one_month", "advance"]
    )
    
    # Bucket travel time
    df["journey_length"] = pd.cut(
        df["travel_time_minutes"],
        bins=[-1, 60, 120, 240, float('inf')],
        labels=["short", "medium", "long", "very_long"]
    )
    
    # Create route feature
    df["route"] = df["origin_station"] + "-" + df["destination_station"]
    
    return df

# Global variables for metrics
startup_time = time.time()
request_count = 0
error_count = 0

@app.middleware("http")
async def track_metrics(request: Request, call_next):
    """Middleware to track API metrics"""
    global request_count, error_count
    
    # Increment request count
    request_count += 1
    
    try:
        # Process request
        response = await call_next(request)
        return response
    
    except Exception as e:
        # Increment error count
        error_count += 1
        
        # Re-raise exception
        raise

if __name__ == "__main__":
    # Run the API server
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
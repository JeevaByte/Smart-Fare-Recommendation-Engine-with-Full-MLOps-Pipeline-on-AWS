#!/usr/bin/env python
"""
Tests for the fare recommendation API.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the FastAPI app
from src.api.app import app

class TestAPI(unittest.TestCase):
    """Test cases for the fare recommendation API"""
    
    def setUp(self):
        """Set up test client"""
        self.client = TestClient(app)
        
        # Sample request data
        self.sample_request = {
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
        }
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = self.client.get("/")
        
        # Check response status code
        self.assertEqual(response.status_code, 200)
        
        # Check response content
        data = response.json()
        self.assertEqual(data["message"], "Fare Recommendation API")
        self.assertEqual(data["status"], "active")
        self.assertIn("endpoints", data)
    
    def test_health_endpoint(self):
        """Test the health endpoint"""
        # Mock the model global variable
        with patch("src.api.app.model", MagicMock()):
            response = self.client.get("/health")
            
            # Check response status code
            self.assertEqual(response.status_code, 200)
            
            # Check response content
            data = response.json()
            self.assertEqual(data["status"], "healthy")
    
    def test_health_endpoint_no_model(self):
        """Test the health endpoint when model is not loaded"""
        # Mock the model global variable as None
        with patch("src.api.app.model", None):
            response = self.client.get("/health")
            
            # Check response status code
            self.assertEqual(response.status_code, 503)
            
            # Check response content
            data = response.json()
            self.assertEqual(data["detail"], "Model not loaded")
    
    @patch("src.api.app.model")
    def test_predict_endpoint(self, mock_model):
        """Test the predict endpoint"""
        # Mock the model prediction
        mock_model.predict.return_value = [75.50]
        
        # Make request to predict endpoint
        response = self.client.post("/predict", json=self.sample_request)
        
        # Check response status code
        self.assertEqual(response.status_code, 200)
        
        # Check response content
        data = response.json()
        self.assertAlmostEqual(data["predicted_fare"], 75.50, places=2)
        self.assertIn("confidence", data)
        self.assertIn("request_id", data)
        self.assertIn("model_version", data)
        self.assertIn("processing_time_ms", data)
    
    def test_predict_endpoint_invalid_input(self):
        """Test the predict endpoint with invalid input"""
        # Missing required fields
        invalid_request = {
            "origin_station": "London Kings Cross",
            "destination_station": "Edinburgh Waverley"
            # Missing other required fields
        }
        
        response = self.client.post("/predict", json=invalid_request)
        
        # Check response status code
        self.assertEqual(response.status_code, 422)
    
    @patch("src.api.app.model")
    def test_predict_endpoint_model_error(self, mock_model):
        """Test the predict endpoint when model raises an error"""
        # Mock the model to raise an exception
        mock_model.predict.side_effect = Exception("Model prediction error")
        
        # Make request to predict endpoint
        response = self.client.post("/predict", json=self.sample_request)
        
        # Check response status code
        self.assertEqual(response.status_code, 500)
        
        # Check response content
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("Prediction error", data["detail"])

if __name__ == '__main__':
    unittest.main()
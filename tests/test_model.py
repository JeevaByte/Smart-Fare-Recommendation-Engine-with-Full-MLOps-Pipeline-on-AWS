#!/usr/bin/env python
"""
Tests for the fare recommendation model.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.train import train_model, evaluate_model
from src.data.generate_data import generate_data

class TestModel(unittest.TestCase):
    """Test cases for the fare recommendation model"""
    
    def setUp(self):
        """Set up test data"""
        # Generate a small test dataset
        self.test_data = generate_data(num_samples=100)
        
        # Split features and target
        self.X = self.test_data.drop(['base_fare'], axis=1)
        self.y = self.test_data['base_fare']
    
    def test_data_generation(self):
        """Test data generation function"""
        data = generate_data(num_samples=50)
        
        # Check data shape
        self.assertEqual(len(data), 50)
        
        # Check required columns
        required_columns = [
            'origin_station', 'destination_station', 'booking_days_ahead',
            'travel_time_minutes', 'time_of_day', 'train_operator',
            'class', 'user_type', 'base_fare'
        ]
        for col in required_columns:
            self.assertIn(col, data.columns)
        
        # Check that base_fare is positive
        self.assertTrue((data['base_fare'] > 0).all())
        
        # Check that origin and destination are different
        self.assertTrue((data['origin_station'] != data['destination_station']).all())
    
    @patch('lightgbm.train')
    def test_train_model(self, mock_train):
        """Test model training function"""
        # Mock the LightGBM train function
        mock_model = MagicMock()
        mock_train.return_value = mock_model
        
        # Call the train_model function
        model = train_model(self.X, self.y)
        
        # Check that LightGBM train was called
        mock_train.assert_called_once()
        
        # Check that a model was returned
        self.assertEqual(model, mock_model)
    
    def test_evaluate_model(self):
        """Test model evaluation function"""
        # Create a mock model that returns the mean of the target
        mock_model = MagicMock()
        mean_fare = self.y.mean()
        mock_model.predict.return_value = np.full(len(self.y), mean_fare)
        
        # Evaluate the mock model
        metrics = evaluate_model(mock_model, self.X, self.y)
        
        # Check that metrics were calculated
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        
        # Check predictions
        np.testing.assert_array_equal(metrics['predictions'], np.full(len(self.y), mean_fare))
        
        # Check actuals
        np.testing.assert_array_equal(metrics['actuals'], self.y)

if __name__ == '__main__':
    unittest.main()
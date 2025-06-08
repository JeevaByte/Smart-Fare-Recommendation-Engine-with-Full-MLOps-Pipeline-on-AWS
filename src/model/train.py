#!/usr/bin/env python
"""
Model training script for fare recommendation engine.
Uses LightGBM for regression and MLflow for experiment tracking.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from datetime import datetime

def load_data(data_path):
    """Load processed data from parquet files"""
    if data_path.startswith('s3://'):
        # For S3 paths
        import boto3
        from io import BytesIO
        import pyarrow.parquet as pq
        
        s3_path = data_path.replace('s3://', '')
        bucket_name = s3_path.split('/')[0]
        key = '/'.join(s3_path.split('/')[1:])
        
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        data = BytesIO(obj['Body'].read())
        
        # Read parquet file
        table = pq.read_table(data)
        df = table.to_pandas()
    else:
        # For local paths
        df = pd.read_parquet(data_path)
    
    return df

def prepare_data(df, test_size=0.2, random_state=42):
    """Prepare data for training"""
    # Extract features and target
    X = df.drop(['label', 'base_fare'], axis=1, errors='ignore')
    y = df['label'] if 'label' in df.columns else df['base_fare']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, params=None):
    """Train LightGBM model"""
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
    
    # Create dataset for LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred,
        'actuals': y_test
    }

def log_to_mlflow(model, params, metrics, model_name="fare_recommendation_model"):
    """Log model and metrics to MLflow"""
    # Set experiment name
    mlflow.set_experiment("Fare Recommendation Engine")
    
    # Start run and log parameters
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Log metrics
        mlflow.log_metric("rmse", metrics['rmse'])
        mlflow.log_metric("mae", metrics['mae'])
        mlflow.log_metric("r2", metrics['r2'])
        
        # Log model
        mlflow.lightgbm.log_model(model, "model", registered_model_name=model_name)
        
        # Get run ID
        run_id = mlflow.active_run().info.run_id
        
    return run_id

def main():
    parser = argparse.ArgumentParser(description='Train fare recommendation model')
    parser.add_argument('--data-path', type=str, required=True, 
                        help='Path to processed data (local or s3://)')
    parser.add_argument('--model-name', type=str, default='fare_recommendation_model',
                        help='Name for the registered model in MLflow')
    parser.add_argument('--mlflow-tracking-uri', type=str, 
                        help='MLflow tracking server URI')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (fraction of data)')
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI if provided
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    
    # Load and prepare data
    print(f"Loading data from {args.data_path}")
    df = load_data(args.data_path)
    print(f"Data shape: {df.shape}")
    
    X_train, X_test, y_train, y_test = prepare_data(df, test_size=args.test_size)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Define model parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train, params)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print(f"Model performance:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    
    # Log to MLflow
    print("Logging to MLflow...")
    run_id = log_to_mlflow(model, params, metrics, args.model_name)
    
    print(f"Model training completed. MLflow run ID: {run_id}")

if __name__ == "__main__":
    main()
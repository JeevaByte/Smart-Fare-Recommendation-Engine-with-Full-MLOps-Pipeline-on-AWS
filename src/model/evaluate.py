#!/usr/bin/env python
"""
Model evaluation script for fare recommendation engine.
Evaluates model performance and detects potential drift.
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
import mlflow
import mlflow.lightgbm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.metrics import *

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

def load_model(model_uri):
    """Load model from MLflow model registry or local path"""
    return mlflow.lightgbm.load_model(model_uri)

def evaluate_model(model, data, target_col='label'):
    """Evaluate model performance on given data"""
    # Extract features and target
    X = data.drop([target_col, 'base_fare'], axis=1, errors='ignore')
    y = data[target_col] if target_col in data.columns else data['base_fare']
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Create results dataframe with predictions
    results_df = pd.DataFrame({
        'actual': y,
        'predicted': y_pred,
        'error': y - y_pred,
        'abs_error': np.abs(y - y_pred),
        'squared_error': (y - y_pred) ** 2
    })
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'results_df': results_df,
        'predictions': y_pred,
        'actuals': y
    }

def detect_drift(reference_data, current_data, target_col='label'):
    """Detect data and target drift using Evidently"""
    # Create Evidently report
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        RegressionPreset()
    ])
    
    # Run the report
    data_drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping={
            'target': target_col,
            'prediction': 'predicted'
        }
    )
    
    return data_drift_report

def save_evaluation_results(metrics, output_path):
    """Save evaluation metrics to JSON file"""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare metrics for serialization (convert numpy values to Python types)
    serializable_metrics = {
        'rmse': float(metrics['rmse']),
        'mae': float(metrics['mae']),
        'r2': float(metrics['r2'])
    }
    
    # Save metrics to JSON file
    with open(output_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"Evaluation results saved to {output_path}")

def plot_predictions(results_df, output_path):
    """Create and save prediction plots"""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Actual vs Predicted
    axes[0, 0].scatter(results_df['actual'], results_df['predicted'], alpha=0.5)
    axes[0, 0].plot([results_df['actual'].min(), results_df['actual'].max()], 
                   [results_df['actual'].min(), results_df['actual'].max()], 
                   'r--')
    axes[0, 0].set_xlabel('Actual Fare')
    axes[0, 0].set_ylabel('Predicted Fare')
    axes[0, 0].set_title('Actual vs Predicted Fares')
    
    # Error Distribution
    axes[0, 1].hist(results_df['error'], bins=50)
    axes[0, 1].axvline(0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Prediction Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Error Distribution')
    
    # Error vs Actual
    axes[1, 0].scatter(results_df['actual'], results_df['error'], alpha=0.5)
    axes[1, 0].axhline(0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Actual Fare')
    axes[1, 0].set_ylabel('Prediction Error')
    axes[1, 0].set_title('Error vs Actual Fare')
    
    # Residual Plot
    axes[1, 1].scatter(results_df['predicted'], results_df['error'], alpha=0.5)
    axes[1, 1].axhline(0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Predicted Fare')
    axes[1, 1].set_ylabel('Prediction Error')
    axes[1, 1].set_title('Residual Plot')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Prediction plots saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate fare recommendation model')
    parser.add_argument('--model-uri', type=str, required=True, 
                        help='URI of the model to evaluate (MLflow model URI)')
    parser.add_argument('--data-path', type=str, required=True, 
                        help='Path to evaluation data (local or s3://)')
    parser.add_argument('--reference-data-path', type=str, 
                        help='Path to reference data for drift detection (local or s3://)')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--target-col', type=str, default='label',
                        help='Name of the target column')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_uri}")
    model = load_model(args.model_uri)
    
    # Load evaluation data
    print(f"Loading evaluation data from {args.data_path}")
    eval_data = load_data(args.data_path)
    print(f"Evaluation data shape: {eval_data.shape}")
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, eval_data, args.target_col)
    
    print(f"Model performance:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save evaluation results
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    save_evaluation_results(metrics, metrics_path)
    
    # Create and save plots
    plots_path = os.path.join(args.output_dir, 'prediction_plots.png')
    plot_predictions(metrics['results_df'], plots_path)
    
    # Detect drift if reference data is provided
    if args.reference_data_path:
        print(f"Loading reference data from {args.reference_data_path}")
        reference_data = load_data(args.reference_data_path)
        print(f"Reference data shape: {reference_data.shape}")
        
        # Add predictions to both datasets
        reference_data['predicted'] = model.predict(reference_data.drop([args.target_col, 'base_fare'], axis=1, errors='ignore'))
        eval_data['predicted'] = metrics['predictions']
        
        print("Detecting data drift...")
        drift_report = detect_drift(reference_data, eval_data, args.target_col)
        
        # Save drift report
        drift_report_path = os.path.join(args.output_dir, 'drift_report.html')
        drift_report.save_html(drift_report_path)
        print(f"Drift report saved to {drift_report_path}")
    
    print("Model evaluation completed")

if __name__ == "__main__":
    main()
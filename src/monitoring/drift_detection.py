#!/usr/bin/env python
"""
Data and model drift detection for fare recommendation engine.
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
import boto3
import mlflow
import logging
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.metrics import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_path):
    """Load data from parquet files"""
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

def send_alert(message, sns_topic_arn=None):
    """Send alert via SNS or log it"""
    logger.warning(f"ALERT: {message}")
    
    if sns_topic_arn:
        try:
            sns = boto3.client('sns')
            sns.publish(
                TopicArn=sns_topic_arn,
                Subject="Fare Recommendation Model Alert",
                Message=message
            )
            logger.info(f"Alert sent to SNS topic: {sns_topic_arn}")
        except Exception as e:
            logger.error(f"Failed to send SNS alert: {str(e)}")

def analyze_drift_metrics(drift_report):
    """Analyze drift metrics and determine if action is needed"""
    # Extract metrics from report
    metrics = drift_report.as_dict()
    
    alerts = []
    
    # Check data drift
    data_drift_share = metrics.get('metrics', {}).get('data_drift', {}).get('share_of_drifted_columns', 0)
    if data_drift_share > 0.3:  # If more than 30% of columns have drifted
        alerts.append(f"Data drift detected in {data_drift_share:.1%} of features")
    
    # Check prediction drift
    prediction_drift = metrics.get('metrics', {}).get('prediction_drift', {}).get('drift_score', 0)
    if prediction_drift > 0.1:  # If prediction drift score is high
        alerts.append(f"Prediction drift detected with score {prediction_drift:.3f}")
    
    # Check performance
    rmse = metrics.get('metrics', {}).get('regression_performance', {}).get('rmse', {}).get('current', 0)
    rmse_ref = metrics.get('metrics', {}).get('regression_performance', {}).get('rmse', {}).get('reference', 0)
    if rmse > rmse_ref * 1.2:  # If RMSE increased by more than 20%
        alerts.append(f"Model performance degraded: RMSE increased from {rmse_ref:.2f} to {rmse:.2f}")
    
    return alerts

def main():
    parser = argparse.ArgumentParser(description='Detect data and model drift')
    parser.add_argument('--reference-data', type=str, required=True, 
                        help='Path to reference data (local or s3://)')
    parser.add_argument('--current-data', type=str, required=True, 
                        help='Path to current data (local or s3://)')
    parser.add_argument('--model-uri', type=str, required=True, 
                        help='URI of the model to evaluate (MLflow model URI)')
    parser.add_argument('--target-col', type=str, default='label',
                        help='Name of the target column')
    parser.add_argument('--output-dir', type=str, default='./drift_reports',
                        help='Directory to save drift reports')
    parser.add_argument('--sns-topic-arn', type=str,
                        help='ARN of SNS topic for alerts')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading reference data from {args.reference_data}")
    reference_data = load_data(args.reference_data)
    logger.info(f"Reference data shape: {reference_data.shape}")
    
    logger.info(f"Loading current data from {args.current_data}")
    current_data = load_data(args.current_data)
    logger.info(f"Current data shape: {current_data.shape}")
    
    # Load model
    logger.info(f"Loading model from {args.model_uri}")
    model = load_model(args.model_uri)
    
    # Add predictions to both datasets
    logger.info("Generating predictions for reference data")
    reference_data['predicted'] = model.predict(reference_data.drop([args.target_col, 'base_fare'], axis=1, errors='ignore'))
    
    logger.info("Generating predictions for current data")
    current_data['predicted'] = model.predict(current_data.drop([args.target_col, 'base_fare'], axis=1, errors='ignore'))
    
    # Detect drift
    logger.info("Detecting data and model drift")
    drift_report = detect_drift(reference_data, current_data, args.target_col)
    
    # Save drift report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(args.output_dir, f'drift_report_{timestamp}.html')
    drift_report.save_html(report_path)
    logger.info(f"Drift report saved to {report_path}")
    
    # Analyze drift metrics
    alerts = analyze_drift_metrics(drift_report)
    
    # Send alerts if needed
    if alerts:
        alert_message = "Fare Recommendation Model Alerts:\n" + "\n".join(alerts)
        send_alert(alert_message, args.sns_topic_arn)
        
        # Save alerts to file
        alerts_path = os.path.join(args.output_dir, f'alerts_{timestamp}.json')
        with open(alerts_path, 'w') as f:
            json.dump({"alerts": alerts, "timestamp": timestamp}, f, indent=2)
        logger.info(f"Alerts saved to {alerts_path}")
    else:
        logger.info("No drift alerts detected")

if __name__ == "__main__":
    main()
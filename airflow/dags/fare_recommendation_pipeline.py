"""
Airflow DAG for the fare recommendation pipeline.
Orchestrates data processing, model training, and deployment.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.emr import EmrAddStepsOperator
from airflow.providers.amazon.aws.sensors.emr import EmrStepSensor
from airflow.providers.amazon.aws.operators.ecs import EcsRunTaskOperator
from airflow.models import Variable
import boto3
import os

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'fare_recommendation_pipeline',
    default_args=default_args,
    description='Fare recommendation pipeline',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['fare', 'recommendation', 'mlops'],
)

# Get configuration from Airflow Variables
try:
    config = {
        'raw_data_bucket': Variable.get('raw_data_bucket'),
        'processed_data_bucket': Variable.get('processed_data_bucket'),
        'model_artifacts_bucket': Variable.get('model_artifacts_bucket'),
        'emr_cluster_id': Variable.get('emr_cluster_id'),
        'ecs_cluster': Variable.get('ecs_cluster'),
        'ecs_task_definition': Variable.get('ecs_task_definition'),
        'ecs_subnets': Variable.get('ecs_subnets').split(','),
        'ecs_security_group': Variable.get('ecs_security_group'),
    }
except KeyError as e:
    # Use default values for local development
    config = {
        'raw_data_bucket': 'fare-recommendation-dev-raw-data',
        'processed_data_bucket': 'fare-recommendation-dev-processed-data',
        'model_artifacts_bucket': 'fare-recommendation-dev-model-artifacts',
        'emr_cluster_id': 'j-XXXXXXXXXX',
        'ecs_cluster': 'fare-recommendation-dev-cluster',
        'ecs_task_definition': 'fare-recommendation-dev-api',
        'ecs_subnets': ['subnet-xxxxxxxxxx'],
        'ecs_security_group': 'sg-xxxxxxxxxx',
    }

# Generate data
def generate_data(**kwargs):
    """Generate synthetic train fare data"""
    from datetime import datetime
    import subprocess
    import sys
    
    # Generate data with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"/tmp/train_fares_{timestamp}.csv"
    
    # Run data generation script
    cmd = [
        sys.executable,
        "/opt/airflow/dags/scripts/generate_data.py",
        "--samples", "50000",
        "--output", output_file,
        "--upload-s3",
        "--s3-bucket", config['raw_data_bucket'],
        "--s3-prefix", f"raw/{timestamp}/"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Data generation failed: {result.stderr}")
    
    # Return S3 path for next task
    s3_path = f"s3://{config['raw_data_bucket']}/raw/{timestamp}/train_fares_{timestamp}.csv"
    kwargs['ti'].xcom_push(key='raw_data_path', value=s3_path)
    
    return s3_path

# Define EMR steps for data processing
def get_data_processing_steps(**kwargs):
    """Get EMR steps for data processing"""
    ti = kwargs['ti']
    raw_data_path = ti.xcom_pull(key='raw_data_path', task_ids='generate_data')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"s3://{config['processed_data_bucket']}/processed/{timestamp}/"
    pipeline_path = f"s3://{config['model_artifacts_bucket']}/pipelines/{timestamp}/"
    
    steps = [
        {
            'Name': 'Process Train Fare Data',
            'ActionOnFailure': 'CONTINUE',
            'HadoopJarStep': {
                'Jar': 'command-runner.jar',
                'Args': [
                    'spark-submit',
                    '--deploy-mode', 'cluster',
                    '--master', 'yarn',
                    '--conf', 'spark.dynamicAllocation.enabled=true',
                    '--conf', 'spark.shuffle.service.enabled=true',
                    '--conf', 'spark.executor.instances=2',
                    '--conf', 'spark.executor.cores=2',
                    '--conf', 'spark.executor.memory=4g',
                    's3://{}/scripts/process_data.py'.format(config['model_artifacts_bucket']),
                    '--input', raw_data_path,
                    '--output', output_path,
                    '--save-pipeline', pipeline_path
                ]
            }
        }
    ]
    
    # Store output path for next task
    kwargs['ti'].xcom_push(key='processed_data_path', value=output_path)
    kwargs['ti'].xcom_push(key='pipeline_path', value=pipeline_path)
    
    return steps

# Train model
def train_model(**kwargs):
    """Train the fare recommendation model"""
    import subprocess
    import sys
    
    ti = kwargs['ti']
    processed_data_path = ti.xcom_pull(key='processed_data_path', task_ids='submit_data_processing')
    
    # Run training script
    cmd = [
        sys.executable,
        "/opt/airflow/dags/scripts/train.py",
        "--data-path", processed_data_path,
        "--model-name", "fare_recommendation_model",
        "--mlflow-tracking-uri", os.environ.get('MLFLOW_TRACKING_URI', '')
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Model training failed: {result.stderr}")
    
    # Extract run ID from output
    import re
    run_id_match = re.search(r"MLflow run ID: ([\w\d]+)", result.stdout)
    if run_id_match:
        run_id = run_id_match.group(1)
        kwargs['ti'].xcom_push(key='mlflow_run_id', value=run_id)
        return run_id
    else:
        raise Exception("Could not extract MLflow run ID from output")

# Evaluate model
def evaluate_model(**kwargs):
    """Evaluate the trained model"""
    import subprocess
    import sys
    import json
    
    ti = kwargs['ti']
    processed_data_path = ti.xcom_pull(key='processed_data_path', task_ids='submit_data_processing')
    run_id = ti.xcom_pull(key='mlflow_run_id', task_ids='train_model')
    
    # Get model URI
    model_uri = f"runs:/{run_id}/model"
    
    # Run evaluation script
    output_dir = "/tmp/evaluation_results"
    cmd = [
        sys.executable,
        "/opt/airflow/dags/scripts/evaluate.py",
        "--model-uri", model_uri,
        "--data-path", processed_data_path,
        "--output-dir", output_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Model evaluation failed: {result.stderr}")
    
    # Load metrics
    try:
        with open(f"{output_dir}/metrics.json", 'r') as f:
            metrics = json.load(f)
        
        # Push metrics to XCom
        for key, value in metrics.items():
            kwargs['ti'].xcom_push(key=f'metric_{key}', value=value)
        
        return metrics
    except Exception as e:
        raise Exception(f"Failed to load evaluation metrics: {str(e)}")

# Deploy model
def deploy_model(**kwargs):
    """Deploy the model to production"""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    ti = kwargs['ti']
    run_id = ti.xcom_pull(key='mlflow_run_id', task_ids='train_model')
    
    # Set MLflow tracking URI
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', '')
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Create MLflow client
    client = MlflowClient()
    
    # Get model details
    model_name = "fare_recommendation_model"
    run = client.get_run(run_id)
    model_uri = f"runs:/{run_id}/model"
    
    # Register model if not exists
    try:
        client.get_registered_model(model_name)
    except:
        client.create_registered_model(model_name)
    
    # Create new version
    model_version = mlflow.register_model(model_uri, model_name)
    
    # Transition to Production
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production"
    )
    
    # Return model version
    kwargs['ti'].xcom_push(key='model_version', value=model_version.version)
    return model_version.version

# Task definitions
generate_data_task = PythonOperator(
    task_id='generate_data',
    python_callable=generate_data,
    provide_context=True,
    dag=dag,
)

submit_data_processing = EmrAddStepsOperator(
    task_id='submit_data_processing',
    job_flow_id=config['emr_cluster_id'],
    aws_conn_id='aws_default',
    steps=get_data_processing_steps,
    dag=dag,
)

wait_for_data_processing = EmrStepSensor(
    task_id='wait_for_data_processing',
    job_flow_id=config['emr_cluster_id'],
    step_id="{{ task_instance.xcom_pull(task_ids='submit_data_processing')[0] }}",
    aws_conn_id='aws_default',
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    provide_context=True,
    dag=dag,
)

deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    provide_context=True,
    dag=dag,
)

# Define task dependencies
generate_data_task >> submit_data_processing >> wait_for_data_processing >> train_model_task >> evaluate_model_task >> deploy_model_task
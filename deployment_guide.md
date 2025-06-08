# Deployment Guide for Fare Recommendation Engine

This guide provides step-by-step instructions for deploying the complete MLOps pipeline for the fare recommendation engine.

## Prerequisites

- AWS account with appropriate permissions
- Terraform installed (v1.3.0+)
- Docker installed
- Python 3.9+ installed
- Git installed

## Step 1: Deploy Infrastructure with Terraform

### Windows:
```powershell
# Edit deploy_terraform.ps1 to add your AWS credentials
.\deploy_terraform.ps1
```

### Linux/macOS:
```bash
# Edit deploy_terraform.sh to add your AWS credentials
chmod +x deploy_terraform.sh
./deploy_terraform.sh
```

After deployment, note the outputs including S3 bucket names, ECR repository URL, and API endpoint.

## Step 2: Configure MLflow for Model Tracking

### Windows:
```powershell
# Edit setup_mlflow.ps1 to add your AWS credentials
.\setup_mlflow.ps1
```

### Linux/macOS:
```bash
# Edit setup_mlflow.sh to add your AWS credentials
chmod +x setup_mlflow.sh
./setup_mlflow.sh
```

Verify MLflow is running by accessing the UI at http://localhost:5000

## Step 3: Set Up GitHub CI/CD Pipeline

1. Push your code to GitHub
2. Configure GitHub secrets as described in `setup_github_secrets.md`
3. Verify the GitHub Actions workflow is set up correctly

## Step 4: Run Initial Data Generation and Model Training

### Windows:
```powershell
# Edit run_initial_pipeline.ps1 to add your AWS credentials
.\run_initial_pipeline.ps1
```

### Linux/macOS:
```bash
# Edit run_initial_pipeline.sh to add your AWS credentials
chmod +x run_initial_pipeline.sh
./run_initial_pipeline.sh
```

Check MLflow UI to verify the model was registered successfully.

## Step 5: Test the API and Monitoring Systems

### Windows:
```powershell
# Edit test_api.ps1 to add your AWS credentials
.\test_api.ps1
```

### Linux/macOS:
```bash
# Edit test_api.sh to add your AWS credentials
chmod +x test_api.sh
./test_api.sh
```

Verify that the API responds correctly and monitoring is working.

## Step 6: Deploy the LangChain Agent

### Windows:
```powershell
# Edit deploy_langchain_agent.ps1 to add your OpenAI API key
.\deploy_langchain_agent.ps1
```

### Linux/macOS:
```bash
# Edit deploy_langchain_agent.sh to add your OpenAI API key
chmod +x deploy_langchain_agent.sh
./deploy_langchain_agent.sh
```

Access the Streamlit UI to interact with the fare recommendation engine using natural language.

## Troubleshooting

### Common Issues:

1. **Terraform errors**: Check AWS credentials and permissions
2. **MLflow connection issues**: Verify network connectivity and S3 bucket permissions
3. **API deployment failures**: Check ECR repository and ECS service logs
4. **Model training errors**: Check data paths and MLflow tracking URI

### Logs:

- CloudWatch Logs for ECS services
- MLflow UI for experiment tracking
- GitHub Actions logs for CI/CD pipeline

## Maintenance

- Monitor model drift using the drift detection script
- Update the model regularly with new data
- Check CloudWatch metrics for API performance
- Review security settings periodically
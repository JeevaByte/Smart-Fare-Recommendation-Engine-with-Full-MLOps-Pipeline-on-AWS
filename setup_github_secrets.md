# Setting Up GitHub Secrets for CI/CD Pipeline

To set up the CI/CD pipeline in GitHub, you need to configure the following secrets in your GitHub repository:

1. Navigate to your GitHub repository
2. Go to Settings > Secrets and variables > Actions
3. Click on "New repository secret"
4. Add the following secrets:

## Required Secrets

| Secret Name | Description |
|-------------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS access key with permissions for S3, ECR, ECS, etc. |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key corresponding to the access key |
| `RAW_DATA_BUCKET` | Name of the S3 bucket for raw data (from Terraform output) |
| `PROCESSED_DATA_BUCKET` | Name of the S3 bucket for processed data (from Terraform output) |
| `MLFLOW_TRACKING_URI` | URI of the MLflow tracking server |

## Steps to Get Values

1. **AWS Credentials**: Create an IAM user with appropriate permissions
2. **S3 Bucket Names**: Get from Terraform output after infrastructure deployment:
   ```
   cd infra/terraform
   terraform output raw_data_bucket
   terraform output processed_data_bucket
   ```
3. **MLflow Tracking URI**: Use the URI of your MLflow server (e.g., `http://your-mlflow-server:5000`)

## Enabling GitHub Actions

1. Make sure GitHub Actions is enabled in your repository
2. The workflow file is already created at `.github/workflows/ci_cd.yml`
3. The workflow will run automatically on pushes to `main` and `dev` branches
4. You can also trigger the workflow manually from the Actions tab

## Testing the CI/CD Pipeline

1. Make a small change to a file in the repository
2. Commit and push to the `dev` branch
3. Go to the Actions tab in your GitHub repository
4. Verify that the workflow runs successfully
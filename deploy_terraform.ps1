# PowerShell script to deploy infrastructure using Terraform

# Set AWS credentials
Write-Host "Setting AWS credentials..."
$env:AWS_ACCESS_KEY_ID = "your-access-key"
$env:AWS_SECRET_ACCESS_KEY = "your-secret-key"
$env:AWS_DEFAULT_REGION = "eu-west-2"

# Navigate to Terraform directory
Set-Location -Path ".\infra\terraform"

# Initialize Terraform
Write-Host "Initializing Terraform..."
terraform init

# Validate configuration
Write-Host "Validating Terraform configuration..."
terraform validate

# Plan deployment
Write-Host "Planning Terraform deployment..."
terraform plan -out=tfplan

# Apply configuration
Write-Host "Applying Terraform configuration..."
terraform apply -auto-approve tfplan

# Output important values
Write-Host "Infrastructure deployment complete. Outputs:"
terraform output
#!/bin/bash
# Script to deploy infrastructure using Terraform

# Set AWS credentials
echo "Setting AWS credentials..."
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="eu-west-2"

# Navigate to Terraform directory
cd infra/terraform

# Initialize Terraform
echo "Initializing Terraform..."
terraform init

# Validate configuration
echo "Validating Terraform configuration..."
terraform validate

# Plan deployment
echo "Planning Terraform deployment..."
terraform plan -out=tfplan

# Apply configuration
echo "Applying Terraform configuration..."
terraform apply -auto-approve tfplan

# Output important values
echo "Infrastructure deployment complete. Outputs:"
terraform output
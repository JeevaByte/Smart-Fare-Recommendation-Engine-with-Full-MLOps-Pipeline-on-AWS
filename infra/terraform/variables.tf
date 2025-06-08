variable "aws_region" {
  description = "The AWS region to deploy resources"
  type        = string
  default     = "eu-west-2"  # London region (for UK train data)
}

variable "project_name" {
  description = "The name of the project"
  type        = string
  default     = "fare-recommendation"
}

variable "environment" {
  description = "The deployment environment (dev, staging, production)"
  type        = string
  default     = "dev"
}

variable "vpc_cidr" {
  description = "The CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "The availability zones to use"
  type        = list(string)
  default     = ["eu-west-2a", "eu-west-2b", "eu-west-2c"]
}

variable "private_subnet_cidrs" {
  description = "The CIDR blocks for the private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "The CIDR blocks for the public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "task_cpu" {
  description = "The CPU units for the ECS task"
  type        = string
  default     = "1024"  # 1 vCPU
}

variable "task_memory" {
  description = "The memory for the ECS task"
  type        = string
  default     = "2048"  # 2 GB
}

variable "service_desired_count" {
  description = "The desired number of tasks for the ECS service"
  type        = number
  default     = 2
}

variable "model_uri" {
  description = "The URI of the model in MLflow"
  type        = string
  default     = "models:/fare_recommendation_model/latest"
}

variable "mlflow_tracking_uri" {
  description = "The URI of the MLflow tracking server"
  type        = string
  default     = ""  # Will be set during deployment
}

variable "emr_master_instance_type" {
  description = "The instance type for the EMR master node"
  type        = string
  default     = "m5.xlarge"
}

variable "emr_core_instance_type" {
  description = "The instance type for the EMR core nodes"
  type        = string
  default     = "m5.xlarge"
}

variable "emr_core_instance_count" {
  description = "The number of EMR core nodes"
  type        = number
  default     = 2
}

variable "mwaa_environment_class" {
  description = "The environment class for MWAA"
  type        = string
  default     = "mw1.small"
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {
    Project     = "fare-recommendation"
    ManagedBy   = "terraform"
    Environment = "dev"
  }
}
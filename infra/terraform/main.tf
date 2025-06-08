provider "aws" {
  region = var.aws_region
}

# Random string for unique naming
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  resource_suffix = random_string.suffix.result
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 3.0"

  name = "${local.name_prefix}-vpc"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway = true
  single_nat_gateway = var.environment != "production"
  
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = var.tags
}

# S3 Buckets
resource "aws_s3_bucket" "raw_data" {
  bucket = "${local.name_prefix}-raw-data-${local.resource_suffix}"
  
  tags = merge(var.tags, {
    Name = "${local.name_prefix}-raw-data"
  })
}

resource "aws_s3_bucket" "processed_data" {
  bucket = "${local.name_prefix}-processed-data-${local.resource_suffix}"
  
  tags = merge(var.tags, {
    Name = "${local.name_prefix}-processed-data"
  })
}

resource "aws_s3_bucket" "model_artifacts" {
  bucket = "${local.name_prefix}-model-artifacts-${local.resource_suffix}"
  
  tags = merge(var.tags, {
    Name = "${local.name_prefix}-model-artifacts"
  })
}

# ECR Repository
resource "aws_ecr_repository" "model_api" {
  name = "${local.name_prefix}-model-api"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  tags = var.tags
}

# IAM Roles
resource "aws_iam_role" "ecs_task_execution_role" {
  name = "${local.name_prefix}-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task_role" {
  name = "${local.name_prefix}-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_policy" "s3_access" {
  name        = "${local.name_prefix}-s3-access"
  description = "Policy for accessing S3 buckets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Effect   = "Allow"
        Resource = [
          aws_s3_bucket.raw_data.arn,
          "${aws_s3_bucket.raw_data.arn}/*",
          aws_s3_bucket.processed_data.arn,
          "${aws_s3_bucket.processed_data.arn}/*",
          aws_s3_bucket.model_artifacts.arn,
          "${aws_s3_bucket.model_artifacts.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_s3_access" {
  role       = aws_iam_role.ecs_task_role.name
  policy_arn = aws_iam_policy.s3_access.arn
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${local.name_prefix}-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  
  tags = var.tags
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "ecs_logs" {
  name              = "/ecs/${local.name_prefix}-api"
  retention_in_days = 30
  
  tags = var.tags
}

# ECS Task Definition
resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name_prefix}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name      = "${local.name_prefix}-api"
      image     = "${aws_ecr_repository.model_api.repository_url}:latest"
      essential = true
      
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "MODEL_URI"
          value = var.model_uri
        },
        {
          name  = "MLFLOW_TRACKING_URI"
          value = var.mlflow_tracking_uri
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_logs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
    }
  ])

  tags = var.tags
}

# Security Group for ECS Service
resource "aws_security_group" "ecs_service" {
  name        = "${local.name_prefix}-ecs-service-sg"
  description = "Security group for ECS service"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.tags
}

# ECS Service
resource "aws_ecs_service" "api" {
  name            = "${local.name_prefix}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.service_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = module.vpc.private_subnets
    security_groups  = [aws_security_group.ecs_service.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "${local.name_prefix}-api"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.api]

  tags = var.tags
}

# ALB
resource "aws_security_group" "alb" {
  name        = "${local.name_prefix}-alb-sg"
  description = "Security group for ALB"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.tags
}

resource "aws_lb" "api" {
  name               = "${local.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets

  enable_deletion_protection = var.environment == "production"

  tags = var.tags
}

resource "aws_lb_target_group" "api" {
  name        = "${local.name_prefix}-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = module.vpc.vpc_id
  target_type = "ip"

  health_check {
    path                = "/health"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 3
    unhealthy_threshold = 3
    matcher             = "200"
  }

  tags = var.tags
}

resource "aws_lb_listener" "api" {
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# EMR Cluster (for Spark jobs)
resource "aws_emr_cluster" "spark" {
  name          = "${local.name_prefix}-emr-cluster"
  release_label = "emr-6.6.0"
  applications  = ["Spark", "Hadoop"]

  ec2_attributes {
    subnet_id                         = module.vpc.private_subnets[0]
    instance_profile                  = aws_iam_instance_profile.emr_profile.name
    emr_managed_master_security_group = aws_security_group.emr_master.id
    emr_managed_slave_security_group  = aws_security_group.emr_slave.id
    service_access_security_group     = aws_security_group.emr_service_access.id
  }

  master_instance_group {
    instance_type = var.emr_master_instance_type
  }

  core_instance_group {
    instance_type  = var.emr_core_instance_type
    instance_count = var.emr_core_instance_count
  }

  service_role = aws_iam_role.emr_service_role.name

  tags = var.tags
}

# EMR Security Groups
resource "aws_security_group" "emr_master" {
  name        = "${local.name_prefix}-emr-master-sg"
  description = "Security group for EMR master"
  vpc_id      = module.vpc.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.tags
}

resource "aws_security_group" "emr_slave" {
  name        = "${local.name_prefix}-emr-slave-sg"
  description = "Security group for EMR slave"
  vpc_id      = module.vpc.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.tags
}

resource "aws_security_group" "emr_service_access" {
  name        = "${local.name_prefix}-emr-service-access-sg"
  description = "Security group for EMR service access"
  vpc_id      = module.vpc.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.tags
}

# EMR IAM Roles
resource "aws_iam_role" "emr_service_role" {
  name = "${local.name_prefix}-emr-service-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "elasticmapreduce.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "emr_service_role_policy" {
  role       = aws_iam_role.emr_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonElasticMapReduceRole"
}

resource "aws_iam_role" "emr_ec2_role" {
  name = "${local.name_prefix}-emr-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "emr_ec2_role_policy" {
  role       = aws_iam_role.emr_ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonElasticMapReduceforEC2Role"
}

resource "aws_iam_role_policy_attachment" "emr_s3_access" {
  role       = aws_iam_role.emr_ec2_role.name
  policy_arn = aws_iam_policy.s3_access.arn
}

resource "aws_iam_instance_profile" "emr_profile" {
  name = "${local.name_prefix}-emr-profile"
  role = aws_iam_role.emr_ec2_role.name
}

# MWAA (Managed Airflow)
resource "aws_s3_bucket" "airflow" {
  bucket = "${local.name_prefix}-airflow-${local.resource_suffix}"
  
  tags = merge(var.tags, {
    Name = "${local.name_prefix}-airflow"
  })
}

resource "aws_s3_bucket_object" "dags" {
  bucket = aws_s3_bucket.airflow.id
  key    = "dags/"
  source = "/dev/null"
}

resource "aws_s3_bucket_object" "requirements" {
  bucket  = aws_s3_bucket.airflow.id
  key     = "requirements.txt"
  content = <<EOF
apache-airflow-providers-amazon>=2.0.0
boto3>=1.17.0
pandas>=1.3.0
pyspark>=3.1.2
mlflow>=1.20.0
EOF
}

resource "aws_mwaa_environment" "airflow" {
  name               = "${local.name_prefix}-airflow"
  airflow_version    = "2.5.1"
  source_bucket_arn  = aws_s3_bucket.airflow.arn
  dag_s3_path        = "dags"
  requirements_s3_path = "requirements.txt"
  
  execution_role_arn = aws_iam_role.mwaa_execution_role.arn
  
  network_configuration {
    security_group_ids = [aws_security_group.mwaa.id]
    subnet_ids         = module.vpc.private_subnets
  }
  
  environment_class = var.mwaa_environment_class
  
  logging_configuration {
    dag_processing_logs {
      enabled   = true
      log_level = "INFO"
    }
    
    scheduler_logs {
      enabled   = true
      log_level = "INFO"
    }
    
    webserver_logs {
      enabled   = true
      log_level = "INFO"
    }
    
    worker_logs {
      enabled   = true
      log_level = "INFO"
    }
  }
  
  tags = var.tags
}

resource "aws_security_group" "mwaa" {
  name        = "${local.name_prefix}-mwaa-sg"
  description = "Security group for MWAA"
  vpc_id      = module.vpc.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.tags
}

resource "aws_iam_role" "mwaa_execution_role" {
  name = "${local.name_prefix}-mwaa-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "airflow.amazonaws.com"
        }
      },
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "airflow-env.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "mwaa_execution_policy" {
  role       = aws_iam_role.mwaa_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonMWAAServiceRolePolicy"
}

resource "aws_iam_role_policy_attachment" "mwaa_s3_access" {
  role       = aws_iam_role.mwaa_execution_role.name
  policy_arn = aws_iam_policy.s3_access.arn
}

# Outputs
output "vpc_id" {
  description = "The ID of the VPC"
  value       = module.vpc.vpc_id
}

output "raw_data_bucket" {
  description = "The name of the raw data S3 bucket"
  value       = aws_s3_bucket.raw_data.bucket
}

output "processed_data_bucket" {
  description = "The name of the processed data S3 bucket"
  value       = aws_s3_bucket.processed_data.bucket
}

output "model_artifacts_bucket" {
  description = "The name of the model artifacts S3 bucket"
  value       = aws_s3_bucket.model_artifacts.bucket
}

output "ecr_repository_url" {
  description = "The URL of the ECR repository"
  value       = aws_ecr_repository.model_api.repository_url
}

output "api_endpoint" {
  description = "The endpoint URL of the API"
  value       = "http://${aws_lb.api.dns_name}"
}

output "airflow_webserver_url" {
  description = "The URL of the Airflow webserver"
  value       = aws_mwaa_environment.airflow.webserver_url
}

output "emr_cluster_id" {
  description = "The ID of the EMR cluster"
  value       = aws_emr_cluster.spark.id
}
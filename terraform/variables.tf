# General AWS

variable "aws_region" {
  type        = string
  description = "AWS Region"
}

variable "environment" {
  type        = string
  description = "Deployment Environment"
}

# Terraform Cloud

variable "tfe_org_name" {
  type        = string
  description = "Terraform Cloud organization name"
  default     = "ByItsCover"
}

variable "bic_infra_workspace" {
  type        = string
  description = "Terraform Cloud Workspace BIC-Infra name"
}

# Batch

variable "max_duration" {
  type        = number
  description = "Maximum duration for batch task, after which will be terminated"
  default     = 3600
}

variable "batch_vcpu" {
  type        = number
  description = "VCPU count for batch job"
  default     = 1
}

variable "batch_memory" {
  type        = number
  description = "Memory size for batch job"
  default     = 512
}

variable "batch_gpus" {
  type        = number
  description = "Number of GPUs for batch job"
  default     = 1
}

variable "embed_lambda_name" {
  type        = string
  description = "Name of Embed Server Lambda Function"
}

variable "batch_tower_dim" {
  type        = number
  description = "Number of dimensions for two tower model output"
  default     = 64
}

variable "efs_volume_name" {
  type        = string
  description = "Name of EFS volume attached to batch instance"
  default     = "recVolume"
}

variable "efs_path" {
  type        = string
  description = "File path for EFS attached to batch instance"
  default     = "/mount/efs"
}

# EventBridge

variable "full_train_frequency" {
  type        = string
  description = "The cron schedule frequency at which the full train job should run"
}

variable "scheduler_arn" {
  type        = string
  description = "Target ARN for EventBridge scheduler"
  default     = "arn:aws:scheduler:::aws-sdk:batch:submitJob"
}

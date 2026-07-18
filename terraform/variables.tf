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

variable "dotnet_env" {
  type        = string
  description = "The ASPNETCORE_ENVIRONMENT for the AWS batch container"
}

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

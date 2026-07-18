locals {
  ecs_execution_role_arn = data.terraform_remote_state.bic_infra.outputs.ecs_execution_role_arn
  s3_db_uri              = data.terraform_remote_state.bic_infra.outputs.s3_db_uri
  hardcover_secret_arn   = data.terraform_remote_state.bic_infra.outputs.hardcover_secret_arn
}


resource "aws_batch_job_definition" "job" {
  name = "learning_batch_job_definition"
  type = "container"
  container_properties = jsonencode({
    image = data.aws_ecr_image.server_image.image_uri

    executionRoleArn = local.ecs_execution_role_arn

    resourceRequirements = [
      {
        type  = "VCPU"
        value = tostring(var.batch_vcpu)
      },
      {
        type  = "MEMORY"
        value = tostring(var.batch_memory)
      },
      {
        type  = "GPU"
        value = tostring(var.batch_gpus)
      }
    ]

    environment = [
      {
        name  = "ENVIRONMENT"
        value = var.environment
      },
      {
        name  = "DB_URI"
        value = local.s3_db_uri
      },
      {
        name  = "AWS_REGION"
        value = var.aws_region
      }
    ]

    secrets = [
      {
        name      = "HARDCOVER_TOKEN"
        valueFrom = local.hardcover_secret_arn
      }
    ]
  })

  timeout {
    attempt_duration_seconds = var.max_duration
  }
}

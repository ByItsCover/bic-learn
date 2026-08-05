locals {
  batch_queue_arn            = data.terraform_remote_state.bic_infra.outputs.learn_batch_queue_arn
  scheduler_role_arn         = data.terraform_remote_state.bic_infra.outputs.scheduler_role_arn
  eventbridge_deadletter_arn = data.terraform_remote_state.bic_infra.outputs.eventbridge_deadletter_arn
}


resource "aws_scheduler_schedule" "full-train-schedule" {
  name       = "full-train-schedule"
  group_name = "default"

  flexible_time_window {
    mode = "OFF"
  }

  schedule_expression = var.full_train_frequency

  target {
    arn      = var.scheduler_arn
    role_arn = local.scheduler_role_arn

    dead_letter_config {
      arn = local.eventbridge_deadletter_arn
    }

    input = jsonencode({
      "JobName" : "full-train-job",
      "JobDefinition" : aws_batch_job_definition.job.arn,
      "JobQueue" : local.batch_queue_arn,
      "ContainerOverrides" : {
        "ResourceRequirements" : [
          {
            "Type" : "GPU",
            "Value" : tostring(var.batch_gpus)
          }
        ],
        "Environment" : [
          {
            "Name" : "JOB_TYPE",
            "Value" : "full_train"
          }
        ]
      }
    })
  }
}
resource "aws_scheduler_schedule" "tune-users-schedule" {
  name       = "tune-users-schedule"
  group_name = "default"

  flexible_time_window {
    mode = "OFF"
  }

  schedule_expression = var.tune_users_frequency

  target {
    arn      = var.scheduler_arn
    role_arn = local.scheduler_role_arn

    dead_letter_config {
      arn = local.eventbridge_deadletter_arn
    }

    input = jsonencode({
      "JobName" : "full-train-job",
      "JobDefinition" : aws_batch_job_definition.job.arn,
      "JobQueue" : local.batch_queue_arn,
      "ContainerOverrides" : {
        "ResourceRequirements" : [
          {
            "Type" : "GPU",
            "Value" : tostring(var.batch_gpus)
          }
        ],
        "Environment" : [
          {
            "Name" : "JOB_TYPE",
            "Value" : "tune_users"
          }
        ]
      }
    })
  }
}


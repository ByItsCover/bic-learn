locals {
  scheduler_role_name = data.terraform_remote_state.bic_infra.outputs.scheduler_role_name
}


data "aws_iam_policy_document" "scheduler_batch_job_policy" {
  statement {
    actions = [
      "batch:SubmitJob",
      "batch:DescribeJobs",
      "batch:TerminateJob"
    ]

    resources = [aws_batch_job_definition.job.arn]
  }
}

resource "aws_iam_role_policy" "scheduler_batch_job_role_policy" {
  name   = "scheduler_batch_job_policy"
  role   = local.scheduler_role_name
  policy = data.aws_iam_policy_document.scheduler_batch_job_policy.json
}

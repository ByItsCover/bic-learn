aws_region  = "us-east-2"
environment = "PROD"

bic_infra_workspace = "bic-infra-prod"

# Batch

max_duration         = 1800
batch_vcpu           = 1
batch_memory         = 2048
batch_gpus           = 1
embed_lambda_name    = "embed-server-lambda"
full_train_frequency = "cron(0 13 * * ? *)" # Every day at 9 AM EST

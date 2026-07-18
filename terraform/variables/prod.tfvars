aws_region  = "us-east-2"
environment = "prod"

bic_infra_workspace = "bic-infra-prod"

# Batch

dotnet_env   = "Production"
max_duration = 1800
batch_vcpu   = 1
batch_memory = 2048
full_train_frequency = "cron(0 13 * * *)" # Every day at 9 AM EST

<!-- BEGIN_TF_DOCS -->
## Requirements

| Name | Version |
|------|---------|
| <a name="requirement_terraform"></a> [terraform](#requirement\_terraform) | >= 1.2 |
| <a name="requirement_aws"></a> [aws](#requirement\_aws) | ~> 6.0 |

## Providers

| Name | Version |
|------|---------|
| <a name="provider_aws"></a> [aws](#provider\_aws) | 6.56.0 |
| <a name="provider_terraform"></a> [terraform](#provider\_terraform) | n/a |

## Modules

No modules.

## Resources

| Name | Type |
|------|------|
| [aws_batch_job_definition.job](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/batch_job_definition) | resource |
| [aws_scheduler_schedule.full-train-schedule](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/scheduler_schedule) | resource |
| [aws_ecr_image.server_image](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/data-sources/ecr_image) | data source |
| [terraform_remote_state.bic_infra](https://registry.terraform.io/providers/hashicorp/terraform/latest/docs/data-sources/remote_state) | data source |

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| <a name="input_aws_region"></a> [aws\_region](#input\_aws\_region) | AWS Region | `string` | n/a | yes |
| <a name="input_batch_gpus"></a> [batch\_gpus](#input\_batch\_gpus) | Number of GPUs for batch job | `number` | `1` | no |
| <a name="input_batch_memory"></a> [batch\_memory](#input\_batch\_memory) | Memory size for batch job | `number` | `512` | no |
| <a name="input_batch_vcpu"></a> [batch\_vcpu](#input\_batch\_vcpu) | VCPU count for batch job | `number` | `1` | no |
| <a name="input_bic_infra_workspace"></a> [bic\_infra\_workspace](#input\_bic\_infra\_workspace) | Terraform Cloud Workspace BIC-Infra name | `string` | n/a | yes |
| <a name="input_efs_path"></a> [efs\_path](#input\_efs\_path) | File path for EFS attached to batch instance | `string` | `"/mount/efs"` | no |
| <a name="input_efs_volume_name"></a> [efs\_volume\_name](#input\_efs\_volume\_name) | Name of EFS volume attached to batch instance | `string` | `"recVolume"` | no |
| <a name="input_embed_lambda_name"></a> [embed\_lambda\_name](#input\_embed\_lambda\_name) | Name of Embed Server Lambda Function | `string` | n/a | yes |
| <a name="input_environment"></a> [environment](#input\_environment) | Deployment Environment | `string` | n/a | yes |
| <a name="input_full_train_frequency"></a> [full\_train\_frequency](#input\_full\_train\_frequency) | The cron schedule frequency at which the full train job should run | `string` | n/a | yes |
| <a name="input_max_duration"></a> [max\_duration](#input\_max\_duration) | Maximum duration for batch task, after which will be terminated | `number` | `3600` | no |
| <a name="input_scheduler_arn"></a> [scheduler\_arn](#input\_scheduler\_arn) | Target ARN for EventBridge scheduler | `string` | `"arn:aws:scheduler:::aws-sdk:batch:submitJob"` | no |
| <a name="input_tfe_org_name"></a> [tfe\_org\_name](#input\_tfe\_org\_name) | Terraform Cloud organization name | `string` | `"ByItsCover"` | no |

## Outputs

No outputs.
<!-- END_TF_DOCS -->
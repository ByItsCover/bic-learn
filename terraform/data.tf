locals {
  ecr_repo = data.terraform_remote_state.bic_infra.outputs.learn_ecr_name
}


data "terraform_remote_state" "bic_infra" {
  backend = "remote"

  config = {
    organization = var.tfe_org_name
    workspaces = {
      name = var.bic_infra_workspace
    }
  }
}

data "aws_ecr_image" "server_image" {
  repository_name = local.ecr_repo
  image_tag       = "latest"
}

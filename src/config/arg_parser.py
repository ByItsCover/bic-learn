import argparse
from enum import Enum
from .env_config import EnvDefault


class Environments(str, Enum):
    development = "Development"
    prod = "PROD"

class JobType(str, Enum):
    full_train = "full_train"
    create_tables = "create_tables"

def parse_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--aws_region', action=EnvDefault, envvar='AWS_REGION')
    parser.add_argument('--env', action=EnvDefault, envvar='ENVIRONMENT', type=Environments, choices=list(Environments))
    parser.add_argument('--job_type', action=EnvDefault, envvar='JOB_TYPE', type=JobType, choices=list(JobType))
    parser.add_argument('--db_uri', action=EnvDefault, envvar='DB_URI')
    parser.add_argument('--embed_lambda_name', action=EnvDefault, envvar='EMBED_LAMBDA_NAME')
    parser.add_argument('--hardcover_token', action=EnvDefault, envvar='HARDCOVER_TOKEN', required=False)

    parser.add_argument('--model_root_dir', action=EnvDefault, envvar='MODEL_ROOT_DIR')
    parser.add_argument('--epochs', action=EnvDefault, envvar='EPOCHS', type=int, default=10)
    parser.add_argument('--batch_size', action=EnvDefault, envvar='BATCH_SIZE', type=int, default=2)
    parser.add_argument('--shuffle', action=EnvDefault, envvar='SHUFFLE', type=bool, default=True)
    parser.add_argument('--user_lr', action=EnvDefault, envvar='USER_LR', type=float, default=1e-3)
    parser.add_argument('--item_lr', action=EnvDefault, envvar='ITEM_LR', type=float, default=1e-3)

    parser.add_argument('--popular_count', action=EnvDefault, envvar='POPULAR_COUNT', type=int, default=10)
    parser.add_argument('--trending_count', action=EnvDefault, envvar='TRENDING_COUNT', type=int, default=10)

    return parser.parse_args(args)

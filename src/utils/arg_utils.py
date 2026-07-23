import argparse
from enum import Enum
from .env_config import EnvDefault


class Environments(str, Enum):
    development = "Development"
    prod = "PROD"

class JobType(str, Enum):
    full_train = "full_train"

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', action=EnvDefault, envvar='ENVIRONMENT', type=Environments, choices=list(Environments))
    parser.add_argument('--job_type', action=EnvDefault, envvar='JOB_TYPE', type=JobType, choices=list(JobType))
    parser.add_argument('--db_uri', action=EnvDefault, envvar='DB_URI')
    parser.add_argument('--hardcover_token', action=EnvDefault, envvar='HARDCOVER_TOKEN', required=False)
    parser.add_argument('--model_root_dir', action=EnvDefault, envvar='MODEL_ROOT_DIR')
    parser.add_argument('--tower_dim', action=EnvDefault, envvar='TOWER_DIM', type=int)

    return parser.parse_args(args)

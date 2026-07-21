import sys
import logging
from utils import parse_args, Environments, JobType
from jobs.full_train import full_train
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    params = parse_args(sys.argv[1:])

    if params.env == Environments.development:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    elif params.env == Environments.prod:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    logger.debug("Args: %s %s %s", params.env, params.job_type, params.db_uri)

    if params.job_type == JobType.full_train:
        full_train()

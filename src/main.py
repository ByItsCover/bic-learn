import asyncio
import sys
import logging
from dotenv import load_dotenv
from config.arg_parser import parse_args, Environments, JobType
from jobs.full_train import full_train

load_dotenv()
logger = logging.getLogger(__name__)

asyncio.set_event_loop(asyncio.new_event_loop())


if __name__ == '__main__':
    params = parse_args(sys.argv[1:])

    if params.env == Environments.development:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    elif params.env == Environments.prod:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    logger.debug(
        "Args: %s %s %s %s $s",
        params.env, params.job_type,params.db_uri, params.model_root_dir, params.tower_dim
    )

    loop = asyncio.get_event_loop()
    if params.job_type == JobType.full_train:
        loop.run_until_complete(full_train(
            params.aws_region, params.db_uri, params.embed_lambda_name, params.hardcover_token,
            params.model_root_dir, params.epochs, params.user_lr, params.item_lr,
            params.popular_count, params.trending_count
        ))

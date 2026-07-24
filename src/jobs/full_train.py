import asyncio
import lancedb
import torch
from torch.utils.data import DataLoader
import logging
from helpers.datasets import PopularCoversDataSet
from helpers.db_tables import get_db, get_cover_table
from helpers.models import UserTower, ItemTower, train_models, save_models
from helpers.hardcover import get_hardcover_client, get_popular_covers, get_trending_covers
from helpers.embed_call import get_lambda_client, embed_covers

logger = logging.getLogger(__name__)


async def full_train(
        aws_region: str, db_uri: str, embed_lambda: str, hardcover_token: str, tower_dim: int, model_dir: str, epochs: int, user_lr: float,
        item_lr: float, popular_count: int, trending_count: int
    ):
    logger.info("CUDA availability: %s", torch.cuda.is_available())
    logger.info("CUDA device name: %s", torch.cuda.get_device_name(0))

    db_task = asyncio.create_task(get_db(db_uri))
    hardcover_client = get_hardcover_client(hardcover_token)
    hardcover_session_task = asyncio.create_task(hardcover_client.connect_async(reconnecting=True))
    hardcover_session = await hardcover_session_task
    popular_covers_task = asyncio.create_task(get_popular_covers(hardcover_session, popular_count))
    trending_covers_task = asyncio.create_task(get_trending_covers(hardcover_session, trending_count))

    db = await db_task
    cover_table_task = asyncio.create_task(get_cover_table(db, tower_dim))

    cover_table = await cover_table_task
    popular_covers = await popular_covers_task
    trending_covers = await trending_covers_task
    hot_covers_map = {cover.id: cover for cover in popular_covers + trending_covers}
    hot_covers = list(hot_covers_map.values())
    lambda_client = get_lambda_client(aws_region)
    embed_covers_task = asyncio.create_task(embed_covers(hot_covers, lambda_client, embed_lambda))

    hot_cover_ids = list(hot_covers_map.keys())
    dataset = PopularCoversDataSet(cover_table, cover_ids=hot_cover_ids)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    user_tower = UserTower(tower_dim)
    item_tower = ItemTower(tower_dim)

    await embed_covers_task
    train_models(user_tower, item_tower, dataloader, epochs, user_lr, item_lr)
    save_models(user_tower, item_tower, model_dir)

    await hardcover_client.close_async()

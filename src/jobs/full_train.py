import asyncio
import torch
from torch.utils.data import DataLoader, ConcatDataset
import logging
from helpers.datasets import HotCoversDataSet, FeedbackDataSet
from helpers.db_tables import get_db, get_cover_table, get_user_table, get_feedback_table
from helpers.models import UserTower, ItemTower, train_models, save_models
from helpers.hardcover import get_hardcover_client, get_popular_covers, get_trending_covers
from helpers.feedback_ops import get_hot_covers_map
from helpers.embed_call import get_lambda_client, embed_covers
from helpers.inference import update_all_users, update_all_covers

logger = logging.getLogger(__name__)


async def full_train(
        aws_region: str, db_uri: str, embed_lambda: str, hardcover_token: str,
        model_dir: str, epochs: int, early_stop: int, batch_size: int, shuffle: bool, user_lr: float,
        item_lr: float, user_weight_decay: float, item_weight_decay: float, popular_count: int, trending_count: int
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
    cover_table_task = asyncio.create_task(get_cover_table(db))
    user_table_task = asyncio.create_task(get_user_table(db))
    feedback_table_task = asyncio.create_task(get_feedback_table(db))

    popular_covers = await popular_covers_task
    trending_covers = await trending_covers_task
    hot_covers_map = get_hot_covers_map(popular_covers, trending_covers)
    hot_covers = [cover[0] for cover in hot_covers_map.values()]
    lambda_client = get_lambda_client(aws_region)
    embed_covers_task = asyncio.create_task(embed_covers(hot_covers, lambda_client, embed_lambda))

    user_tower = UserTower()
    item_tower = ItemTower()

    cover_table = await cover_table_task
    hot_dataset = HotCoversDataSet(cover_table, covers_map=hot_covers_map)
    user_table = await user_table_task
    feedback_table = await feedback_table_task
    feedback_dataset = FeedbackDataSet(feedback_table, user_table, cover_table)
    full_dataset = ConcatDataset([hot_dataset, feedback_dataset])
    dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle)

    await embed_covers_task
    train_models(
        user_tower, item_tower, dataloader, epochs, early_stop,
        user_lr, item_lr, user_weight_decay, item_weight_decay
    )

    update_all_covers_task = asyncio.create_task(update_all_covers(cover_table, item_tower))
    update_all_users_task = asyncio.create_task(update_all_users(user_table, user_tower))

    await update_all_users_task
    await update_all_covers_task
    save_models(user_tower, item_tower, model_dir)

    await hardcover_client.close_async()

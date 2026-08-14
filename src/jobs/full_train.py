import asyncio
import torch
from torch.utils.data import DataLoader
from datetime import datetime, UTC
import logging
from helpers.datasets import HotCoversDataSet, FeedbackDataSet, BatchConcatDataset, collate_fn
from helpers.db_tables import get_db, get_duckdb, get_cover_table, get_user_table, get_feedback_table, get_hot_covers_table, get_runlog_table
from helpers.models import UserTower, ItemTower, train_all_models, save_models
from helpers.inference import update_all_users, update_all_covers
from helpers.runlog import log_run

logger = logging.getLogger(__name__)


async def full_train(
        device: str, db_uri: str, model_dir: str, epochs: int, early_stop: int, 
        batch_size: int, shuffle: bool, user_lr: float, item_lr: float
    ):
    start_time = datetime.now(UTC)
    logger.info("Start time: %s", start_time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info(f"Using {device} device")
    if device == "cuda":
        logger.info("CUDA availability: %s", torch.cuda.is_available())
        logger.info("CUDA device name: %s", torch.cuda.get_device_name(0))

    db_task = asyncio.create_task(get_db(db_uri))
    duckdb_task = asyncio.create_task(get_duckdb(db_uri))

    cover_table_task = asyncio.create_task(get_cover_table(db_task))
    user_table_task = asyncio.create_task(get_user_table(db_task))
    feedback_table_task = asyncio.create_task(get_feedback_table(db_task))
    hot_covers_table_task = asyncio.create_task(get_hot_covers_table(db_task))
    runlog_table_task = asyncio.create_task(get_runlog_table(db_task))

    await hot_covers_table_task
    cover_table = await cover_table_task
    duckdb = await duckdb_task
    hot_dataset = HotCoversDataSet(duckdb, device=device)

    user_table = await user_table_task
    await feedback_table_task
    feedback_dataset = FeedbackDataSet(duckdb, device=device)

    if len(feedback_dataset) == 0:
        logger.warning("No feedback existing in database yet. Training only default user.")
        full_dataset = hot_dataset
    else:
        full_dataset = BatchConcatDataset([hot_dataset, feedback_dataset])

    dataloader = DataLoader(full_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)

    user_tower = UserTower().to(device)
    item_tower = ItemTower().to(device)

    train_all_models(
        user_tower, item_tower, dataloader,
        epochs, early_stop, user_lr, item_lr
    )

    update_all_covers_task = asyncio.create_task(update_all_covers(cover_table, item_tower, device))
    update_all_users_task = asyncio.create_task(update_all_users(user_table, user_tower, device))

    await update_all_users_task
    await update_all_covers_task
    save_models(model_dir, user_tower=user_tower, item_tower=item_tower)

    log_run_task = asyncio.create_task(log_run(runlog_table_task, start_time))
    await log_run_task

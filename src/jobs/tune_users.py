import asyncio
import torch
from torch.utils.data import DataLoader
from datetime import datetime, UTC
import logging
from helpers.datasets import FeedbackDataSet
from helpers.db_tables import get_db, get_cover_table, get_user_table, get_feedback_table, get_runlog_table
from helpers.models import UserTower, ItemTower, tune_user_model, load_models, save_models
from helpers.inference import update_user_list
from helpers.runlog import fetch_last_run, log_run

logger = logging.getLogger(__name__)


async def tune_users(
        device: str, db_uri: str, model_dir: str, epochs: int, early_stop: int,
        batch_size: int, shuffle: bool, user_lr: float
    ):
    start_time = datetime.now(UTC)
    logger.info("Start time: %s", start_time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info(f"Using {device} device")
    if device == "cuda":
        logger.info("CUDA availability: %s", torch.cuda.is_available())
        logger.info("CUDA device name: %s", torch.cuda.get_device_name(0))

    db_task = asyncio.create_task(get_db(db_uri))

    db = await db_task
    cover_table_task = asyncio.create_task(get_cover_table(db))
    user_table_task = asyncio.create_task(get_user_table(db))
    feedback_table_task = asyncio.create_task(get_feedback_table(db))
    runlog_table_task = asyncio.create_task(get_runlog_table(db))

    runlog_table = await runlog_table_task
    last_run_task = asyncio.create_task(fetch_last_run(runlog_table))

    cover_table = await cover_table_task
    user_table = await user_table_task
    feedback_table = await feedback_table_task
    last_run = await last_run_task
    dataset = FeedbackDataSet(feedback_table, user_table, cover_table, last_runtime=last_run, device=device)

    if len(dataset) == 0:
        logger.warning("No new feedback to fine tune on of since last run.")
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        user_tower = UserTower().to(device)
        item_tower = ItemTower().to(device)

        load_models(user_tower, item_tower, model_dir)
        user_id_list = tune_user_model(
            user_tower, item_tower, dataloader,
            epochs, early_stop, user_lr
        )

        update_user_list_task = asyncio.create_task(update_user_list(user_table, user_tower, user_id_list))

        await update_user_list_task
        save_models(model_dir, user_tower=user_tower)

        log_run_task = asyncio.create_task(log_run(runlog_table, start_time))
        await log_run_task

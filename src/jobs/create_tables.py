import asyncio
import logging
from helpers.db_tables import get_db, get_cover_table, get_user_table, get_feedback_table, get_hot_covers_table, get_runlog_table

logger = logging.getLogger(__name__)


async def create_tables(db_uri: str):
    logger.info("Creating/opening tables...")

    db_task = asyncio.create_task(get_db(db_uri))

    cover_table_task = asyncio.create_task(get_cover_table(db_task))
    user_table_task = asyncio.create_task(get_user_table(db_task))
    feedback_table_task = asyncio.create_task(get_feedback_table(db_task))
    hot_covers_table_task = asyncio.create_task(get_hot_covers_table(db_task))
    runlog_table_task = asyncio.create_task(get_runlog_table(db_task))

    await cover_table_task
    await user_table_task
    await feedback_table_task
    await hot_covers_table_task
    await runlog_table_task

    logger.info("Tables created")

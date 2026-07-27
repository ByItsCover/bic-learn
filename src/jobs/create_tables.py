import asyncio
import logging
from helpers.db_tables import get_db, get_cover_table, get_user_table, get_feedback_table

logger = logging.getLogger(__name__)


async def create_tables(db_uri: str):
    db_task = asyncio.create_task(get_db(db_uri))

    db = await db_task

    logger.info("Creating/opening tables...")

    cover_table_task = asyncio.create_task(get_cover_table(db))
    user_table_task = asyncio.create_task(get_user_table(db))
    feedback_table_task = asyncio.create_task(get_feedback_table(db))

    await cover_table_task
    await user_table_task
    await feedback_table_task

    logger.info("Tables created")

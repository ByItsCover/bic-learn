import asyncio
from asyncio import Task
from lancedb import Table
from datetime import datetime
import logging
from helpers.db_tables import RunlogEnum, Runlog, runlog_adapter

logger = logging.getLogger(__name__)


async def log_run(runlog_table_task: Task[Table], start_time: datetime):
    runlog_table = await runlog_table_task
    return await asyncio.to_thread(log_run_sync, runlog_table, start_time)

def log_run_sync(runlog_table: Table, start_time: datetime):
    runlog = [Runlog(type=RunlogEnum.learn_job, last_run=start_time)]

    logger.info(
        "Saving %s runlog for time: %s",
        RunlogEnum.learn_job, start_time.strftime('%Y-%m-%d %H:%M:%S')
    )

    (
        runlog_table.merge_insert("type")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(runlog_adapter.dump_python(runlog))
    )

async def fetch_last_run(runlog_table_task: Task[Table]) -> datetime | None:
    runlog_table = await runlog_table_task
    return await asyncio.to_thread(fetch_last_run_sync, runlog_table)

def fetch_last_run_sync(runlog_table: Table) -> datetime | None:
    runlog = (
        runlog_table.search()
        .where(f"type = '{RunlogEnum.learn_job.value}'")
        .select(["type", "last_run"])
        .limit(1)
    ).to_pydantic(Runlog)

    if len(runlog) == 0:
        logger.warning("First time job has run. Training on all records")
        return None

    last_run = runlog[0].last_run
    logger.info(
        "Retrieved past %s runlog. Training from time: %s",
        RunlogEnum.learn_job, last_run.strftime('%Y-%m-%d %H:%M:%S')
    )

    return last_run

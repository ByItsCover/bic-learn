import asyncio
import lancedb
from lancedb import Table
from lancedb.db import DBConnection
from lancedb.index import BTree
from lancedb.pydantic import LanceModel, Vector
from pydantic import PlainSerializer, TypeAdapter
import pyarrow as pa
from enum import Enum
import uuid
from datetime import datetime
from typing import Optional, Annotated
from config.constants import (TOWER_DIM, CLIP_DIM,
                              COVER_TABLE_NAME, USER_TABLE_NAME,
                              FEEDBACK_TABLE_NAME, RUNLOG_TABLE_NAME)


class Cover(LanceModel):
    cover_id: int
    book_id: int
    isbn_13: str
    cover_url: str
    cover_embedding: Vector(CLIP_DIM)  # pyright: ignore[reportInvalidTypeForm]
    tower_embedding: Optional[Vector(TOWER_DIM)] = None  # pyright: ignore[reportInvalidTypeForm, reportInvalidTypeArguments]

class User(LanceModel):
    user_id: Annotated[uuid.UUID, PlainSerializer(lambda x: x.bytes, return_type=bytes)]
    tower_embedding: Optional[Vector(TOWER_DIM)] = None  # pyright: ignore[reportInvalidTypeForm, reportInvalidTypeArguments]

users_adapter = TypeAdapter(list[User])

class RunlogEnum(str, Enum):
    learn_job = 'learn_job'

class Runlog(LanceModel):
    type: RunlogEnum
    last_run: datetime

runlog_adapter = TypeAdapter(list[Runlog])


async def get_db(uri: str) -> DBConnection:
    return await asyncio.to_thread(get_db_sync, uri)

def get_db_sync(uri: str) -> DBConnection:
    return lancedb.connect(uri)

async def get_cover_table(db: DBConnection) -> Table:
    return await asyncio.to_thread(get_cover_table_sync, db)

def get_cover_table_sync(db: DBConnection) -> Table:
    cover_table = db.create_table(
        COVER_TABLE_NAME,
        schema=Cover.to_arrow_schema(),
        exist_ok=True,
    )

    id_stats = cover_table.index_stats("cover_id_idx")
    if not id_stats:
        cover_table.create_index("cover_id", config=BTree(), name="cover_id_idx")

    cover_schema = cover_table.schema
    if "tower_embedding" not in cover_schema.names:
        cover_table.add_columns({"tower_embedding": f"arrow_cast(NULL, 'FixedSizeList({TOWER_DIM}, Float32)')"})
    if "cover_embedding" not in cover_schema.names:
        cover_table.alter_columns({"path": "embedding", "rename": "cover_embedding"})

    return cover_table

async def get_user_table(db: DBConnection) -> Table:
    return await asyncio.to_thread(get_user_table_sync, db)

def get_user_table_sync(db: DBConnection) -> Table:
    user_schema = pa.schema(
        [
            pa.field("user_id", pa.uuid(), nullable=False),
            pa.field("tower_embedding", pa.list_(pa.float32(), TOWER_DIM), nullable=True),
        ]
    )
    user_table = db.create_table(
        USER_TABLE_NAME,
        schema=user_schema,
        exist_ok=True,
        storage_options={
            "new_table_enable_stable_row_ids": "true"
        }
    )

    id_stats = user_table.index_stats("user_id_idx")
    if not id_stats:
        user_table.create_index("user_id", config=BTree(), name="user_id_idx")

    return user_table

async def get_feedback_table(db: DBConnection) -> Table:
    return await asyncio.to_thread(get_feedback_table_sync, db)

def get_feedback_table_sync(db: DBConnection) -> Table:
    feedback_schema = pa.schema(
        [
            pa.field("user_id", pa.uuid(), nullable=False),
            pa.field("cover_id", pa.int64(), nullable=False),
            pa.field("type", pa.string(), nullable=False),
            pa.field("score", pa.int64(), nullable=False),
            pa.field("timestamp", pa.timestamp('us'), nullable=False),
        ]
    )
    feedback_table = db.create_table(
        FEEDBACK_TABLE_NAME,
        schema=feedback_schema,
        exist_ok=True,
    )

    user_id_stats = feedback_table.index_stats("user_id_idx")
    cover_id_stats = feedback_table.index_stats("cover_id_idx")
    type_stats = feedback_table.index_stats("type_idx")
    if not user_id_stats or not cover_id_stats or not type_stats:
        feedback_table.create_index("user_id", config=BTree(), name="user_id_idx")
        feedback_table.create_index("cover_id", config=BTree(), name="cover_id_idx")
        feedback_table.create_index("type", config=BTree(), name="type_idx")

    return feedback_table

async def get_runlog_table(db: DBConnection) -> Table:
    return await asyncio.to_thread(get_runlog_table_sync, db)

def get_runlog_table_sync(db: DBConnection) -> Table:
    runlog_schema = pa.schema(
        [
            pa.field("type", pa.string(), nullable=False),
            pa.field("last_run", pa.timestamp('us'), nullable=False),
        ]
    )
    runlog_table = db.create_table(
        RUNLOG_TABLE_NAME,
        schema=runlog_schema,
        exist_ok=True,
    )

    type_stats = runlog_table.index_stats("type_idx")
    if not type_stats:
        runlog_table.create_index("type", config=BTree(), name="type_idx")

    return runlog_table

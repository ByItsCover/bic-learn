import asyncio
import lancedb
from lancedb import Table
from lancedb.db import DBConnection
from lancedb.index import BTree
from lancedb.pydantic import LanceModel, Vector
from pydantic import PlainSerializer, TypeAdapter
from typing import Optional, Annotated
import uuid
from config.constants import TOWER_DIM, CLIP_DIM, COVER_TABLE_NAME, USER_TABLE_NAME


class Cover(LanceModel):
    cover_id: int
    book_id: int
    isbn_13: str
    cover_url: str
    cover_embedding: Vector(CLIP_DIM)  # pyright: ignore[reportInvalidTypeForm]
    tower_embedding: Optional[Vector(TOWER_DIM)] = None  # pyright: ignore[reportInvalidTypeForm, reportInvalidTypeArguments]

class User(LanceModel):
    user_id: Annotated[uuid.UUID, PlainSerializer(lambda x: x.bytes, return_type=bytes)]
    tower_embedding: Vector(TOWER_DIM)  # pyright: ignore[reportInvalidTypeForm]

users_adapter = TypeAdapter(list[Cover])

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
    user_table = db.create_table(
        USER_TABLE_NAME,
        schema=User.to_arrow_schema(),
        exist_ok=True,
    )

    id_stats = user_table.index_stats("user_id_idx")
    if not id_stats:
        user_table.create_index("user_id", config=BTree(), name="user_id_idx")

    return user_table

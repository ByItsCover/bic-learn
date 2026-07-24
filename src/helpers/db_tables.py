import asyncio
import lancedb
from lancedb import Table
from lancedb.db import DBConnection
from lancedb.index import BTree
from config.constants import COVER_TABLE_NAME


async def get_db(uri: str) -> DBConnection:
    return await asyncio.to_thread(get_db_sync, uri)

def get_db_sync(uri: str) -> DBConnection:
    return lancedb.connect(uri)

async def get_cover_table(db: DBConnection, tower_dim: int) -> Table:
    return await asyncio.to_thread(get_cover_table_sync, db, tower_dim)

def get_cover_table_sync(db: DBConnection, tower_dim: int) -> Table:
    cover_table = db.open_table(COVER_TABLE_NAME)

    id_stats = cover_table.index_stats("cover_id_idx")
    if not id_stats:
        cover_table.create_index("cover_id", config=BTree(), name="cover_id_idx")

    cover_schema = cover_table.schema
    if "tower_embedding" not in cover_schema.names:
        cover_table.add_columns({"tower_embedding": f"arrow_cast(NULL, 'FixedSizeList({tower_dim}, Float32)')"})
    if "cover_embedding" not in cover_schema.names:
        cover_table.alter_columns({"path": "embedding", "rename": "cover_embedding"})

    return cover_table

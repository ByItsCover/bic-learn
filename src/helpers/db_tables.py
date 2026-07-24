from lancedb import AsyncTable
from lancedb.db import AsyncConnection
from lancedb.index import BTree
from config.constants import COVER_TABLE_NAME


async def get_cover_table(db: AsyncConnection, clip_dim: int, tower_dim: int) -> AsyncTable:
    cover_table = await db.open_table(COVER_TABLE_NAME)

    id_stats = await cover_table.index_stats("cover_id_idx")
    if not id_stats:
        await cover_table.create_index("cover_id", config=BTree(), name="cover_id_idx")

    await cover_table.add_columns({"tower_embedding": f"arrow_cast(NULL, 'FixedSizeList({tower_dim}, Float32)')"})
    await cover_table.alter_columns([{"path": "embedding", "rename": "cover_embedding"}])

    return cover_table
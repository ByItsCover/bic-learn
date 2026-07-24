from lancedb import AsyncTable
from lancedb.db import AsyncConnection
from lancedb.index import BTree
import pyarrow as pa
from config.constants import COVER_TABLE_NAME


async def get_cover_table(db: AsyncConnection, clip_dim: int, tower_dim: int) -> AsyncTable:
    cover_schema = pa.schema(
        [
            pa.field("cover_id", pa.int64(), nullable=False),
            pa.field("book_id", pa.int64(), nullable=False),
            pa.field("isbn_13", pa.string(), nullable=False),
            pa.field("cover_url", pa.string(), nullable=False),
            pa.field("embedding", pa.fixed_shape_tensor(pa.float32(), (clip_dim,)), nullable=False), #cover_embedding
            #pa.field("tower_embedding", pa.list_(pa.float32(), tower_dim), nullable=True),
        ]
    )
    cover_table = await db.create_table(
        COVER_TABLE_NAME, schema=cover_schema, exist_ok=True
    )

    id_stats = await cover_table.index_stats("cover_id_idx")
    if not id_stats:
        await cover_table.create_index("cover_id", config=BTree(), name="cover_id_idx")

    await cover_table.add_columns({"tower_embedding": f"arrow_cast(NULL, 'FixedSizeList({tower_dim}, Float32)')"})
    await cover_table.alter_columns([{"path": "embedding", "rename": "cover_embedding"}])

    return cover_table
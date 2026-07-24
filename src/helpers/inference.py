import asyncio
import torch
from lancedb import Table
from lancedb.pydantic import LanceModel, Vector
from pydantic import TypeAdapter
import uuid
import logging
from helpers.models import UserTower, ItemTower
from helpers.db_tables import User, users_adapter
from helpers.datasets import process_user_id
from config.constants import TOWER_DIM

logger = logging.getLogger(__name__)


class CoverUpdate(LanceModel):
    cover_id: int
    tower_embedding: Vector(TOWER_DIM) #pyright: ignore[reportInvalidTypeForm]

cover_updates_adapter = TypeAdapter(list[CoverUpdate])

async def update_all_users(user_table: Table, user_tower: UserTower):
    return await asyncio.to_thread(update_all_users_sync, user_table, user_tower)

def update_all_users_sync(user_table: Table, user_tower: UserTower):
    user_tower.eval()
    user_table.checkout_latest()

    default_user = uuid.UUID(int=0)
    user_ids = [default_user]
    db_user_dict = user_table.search().select(["user_id"]).to_list()
    for user in db_user_dict:
        if user["user_id"] == default_user:
            continue;
        user_ids.append(user["user_id"])

    user_tensors = torch.vstack([process_user_id(uid) for uid in user_ids])
    with torch.no_grad():
        user_embeddings_tensor = user_tower(user_tensors)
        logger.info("User update shape: %s", user_embeddings_tensor.shape)

    user_embedding_list = torch.unbind(user_embeddings_tensor, dim=0)
    user_list = [
        User(user_id=uid, tower_embedding=tensor)
        for uid, tensor in zip(user_ids, user_embedding_list)
    ]

    (
        user_table.merge_insert("user_id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(users_adapter.dump_python(user_list))
    )

async def update_all_covers(cover_table: Table, item_tower: ItemTower):
    return await asyncio.to_thread(update_all_covers_sync, cover_table, item_tower)

def update_all_covers_sync(cover_table: Table, item_tower: ItemTower):
    item_tower.eval()
    cover_table.checkout_latest()

    cover_ids = []
    cover_embeddings = []
    db_cover_dict = cover_table.search().select(["cover_id", "cover_embedding"]).to_list()
    for cover in db_cover_dict:
        cover_ids.append(cover["cover_id"])
        cover_embeddings.append(cover["cover_embedding"])

    cover_tensors = torch.vstack([torch.tensor(embed) for embed in cover_embeddings])
    with torch.no_grad():
        tower_embeddings_tensor = item_tower(cover_tensors)
        logger.info("Cover update shape: %s", tower_embeddings_tensor.shape)

    tower_embedding_list = torch.unbind(tower_embeddings_tensor, dim=0)
    cover_update_list = [
        CoverUpdate(cover_id=cid, tower_embedding=tensor)
        for cid, tensor in zip(cover_ids, tower_embedding_list)
    ]

    (
        cover_table.merge_insert("cover_id")
        .when_matched_update_all()
        .execute(cover_updates_adapter.dump_python(cover_update_list))
    )

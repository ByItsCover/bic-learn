import asyncio
import torch
from lancedb import Table
from lancedb.pydantic import LanceModel, Vector
from pydantic import TypeAdapter
import uuid
import logging
from helpers.models import UserTower, ItemTower
from helpers.db_tables import User, users_adapter
from helpers.tensor_ops import normalize
from config.constants import TOWER_DIM, DEFAULT_USER_OFFSET

logger = logging.getLogger(__name__)


class CoverUpdate(LanceModel):
    cover_id: int
    tower_embedding: Vector(TOWER_DIM) # type: ignore[PyTypeChecker]

cover_updates_adapter = TypeAdapter(list[CoverUpdate])

async def update_all_users(user_table: Table, user_tower: UserTower):
    return await asyncio.to_thread(update_all_users_sync, user_table, user_tower)

def update_all_users_sync(user_table: Table, user_tower: UserTower):
    user_tower.eval()
    user_table.checkout_latest()

    default_user = uuid.UUID(int=0)
    default_user_id = 0
    users = [default_user]
    user_row_ids = [default_user_id]
    db_user_dict = (
        user_table.search()
        .with_row_id(True)
        .select(["user_id"])
    ).to_list()
    for user in db_user_dict:
        if user["user_id"] == default_user:
            continue;
        user_row_ids.append(user["_rowid"] + DEFAULT_USER_OFFSET)
        users.append(user["user_id"])

    user_id_tensors = torch.tensor(user_row_ids)
    with torch.no_grad():
        user_embeddings_tensor_raw = user_tower(user_id_tensors)
        user_embeddings_tensor = normalize(user_embeddings_tensor_raw)
        logger.info("User update shape: %s", user_embeddings_tensor.shape)

    user_embedding_list = torch.unbind(user_embeddings_tensor, dim=0)
    user_list = [
        User(user_id=uid, tower_embedding=tensor)
        for uid, tensor in zip(users, user_embedding_list)
    ]

    (
        user_table.merge_insert("user_id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(users_adapter.dump_python(user_list))
    )

async def update_user_list(user_table: Table, user_tower: UserTower, user_id_list: list[int], device: str = "cuda"):
    return await asyncio.to_thread(update_user_list_sync, user_table, user_tower, user_id_list, device)

def update_user_list_sync(user_table: Table, user_tower: UserTower, user_id_list: list[int], device: str):
    user_tower.eval()
    user_table.checkout_latest()

    users = []
    user_row_ids = []
    db_user_dict = (
        user_table.take_row_ids(user_id_list)
        .with_row_id()
        .select(["user_id"])
    ).to_list()
    for user in db_user_dict:
        user_row_ids.append(user["_rowid"] + DEFAULT_USER_OFFSET)
        users.append(user["user_id"])

    user_id_tensors = torch.tensor(user_row_ids, device=device)
    with torch.no_grad():
        user_embeddings_tensor_raw = user_tower(user_id_tensors)
        user_embeddings_tensor = normalize(user_embeddings_tensor_raw)
        logger.info("User update shape: %s", user_embeddings_tensor.shape)

    user_embedding_list = torch.unbind(user_embeddings_tensor, dim=0)
    user_list = [
        User(user_id=uid, tower_embedding=tensor)
        for uid, tensor in zip(users, user_embedding_list)
    ]

    (
        user_table.merge_insert("user_id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(users_adapter.dump_python(user_list))
    )

async def update_all_covers(cover_table: Table, item_tower: ItemTower, device: str = "cuda"):
    return await asyncio.to_thread(update_all_covers_sync, cover_table, item_tower, device)

def update_all_covers_sync(cover_table: Table, item_tower: ItemTower, device: str):
    item_tower.eval()
    cover_table.checkout_latest()

    added_covers = set()
    cover_ids = []
    cover_embeddings = []
    db_cover_dict = (
        cover_table.search()
        .select(["cover_id", "cover_embedding"])
    ).to_list()
    for cover in db_cover_dict:
        if cover["cover_id"] not in added_covers:
            cover_ids.append(cover["cover_id"])
            cover_embeddings.append(cover["cover_embedding"])
            added_covers.add(cover["cover_id"])

    cover_tensors = torch.vstack([torch.tensor(embed, device=device) for embed in cover_embeddings])
    cover_id_tensors = torch.tensor(cover_ids, device=device)
    with torch.no_grad():
        tower_embeddings_tensor_raw = item_tower(cover_tensors, cover_id_tensors)
        tower_embeddings_tensor = normalize(tower_embeddings_tensor_raw)
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

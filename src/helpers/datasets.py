import torch
from torch import Tensor
from torch.utils.data import Dataset
from lancedb import Table
from lancedb.permutation import Permutation, permutation_builder
from lancedb.pydantic import LanceModel, Vector
from datetime import datetime
from typing import Optional
from helpers.hardcover import CoverRecord
from helpers.feedback_ops import FeedbackMap
from config.constants import HOT_FEEDBACK_TYPE, CLIP_DIM, DEFAULT_USER_OFFSET


class CoverBackdate(LanceModel):
    cover_id: int
    cover_embedding: Vector(CLIP_DIM)  # pyright: ignore[reportInvalidTypeForm]


class HotCoversDataSet(Dataset):
    def __init__(
            self, cover_table: Table, covers_map: dict[int, tuple[CoverRecord, float]],
            id_field: str = "cover_id", embedding_field: str = "cover_embedding",
            feedback_type: str = HOT_FEEDBACK_TYPE
        ):
        self.cover_table = cover_table
        self.covers_map = covers_map
        self.id_field = id_field
        self.embedding_field = embedding_field
        self.min_rating = FeedbackMap[feedback_type].value[0]
        self.max_rating = FeedbackMap[feedback_type].value[1]
        self.default_user_id = 0
        self.rating_arr = torch.tensor([self.max_rating])
        self.perm: Permutation | None = None

    def __len__(self):
        return len(self.covers_map)

    def _ensure_permutation(self):
        if self.perm is None:
            id_strings = [f'{cid}' for cid, _ in self.covers_map.items()]
            self.cover_table.checkout_latest()
            permutation_tbl = (
                permutation_builder(self.cover_table)
                .filter(f"{self.id_field} IN ({', '.join(id_strings)})")
                .execute()
            )
            permutation = (
                Permutation.from_tables(self.cover_table, permutation_tbl)
                .select_columns([self.id_field, self.embedding_field])
            )
            self.perm = permutation

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        self._ensure_permutation()
        cover = self.perm.__getitem__(idx)[0]
        user_id_arr = torch.tensor(self.default_user_id)
        item_arr = torch.tensor(cover[self.embedding_field])
        item_id_arr = torch.tensor(cover[self.id_field])
        rating_arr = torch.tensor([self.covers_map[cover[self.id_field]][1]])
        min_rating_arr = torch.tensor([self.min_rating])
        max_rating_arr = torch.tensor([self.max_rating])

        return user_id_arr, item_arr, item_id_arr, rating_arr, min_rating_arr, max_rating_arr


class FeedbackDataSet(Dataset):
    def __init__(
            self, feedback_table: Table, user_table: Table,
            cover_table: Table, last_runtime: Optional[datetime] = None, uid_field: str = "user_id", cid_field: str = "cover_id",
            embedding_field: str = "cover_embedding", row_field: str = "_rowid"
        ):
        self.feedback_table = feedback_table
        self.user_table = user_table
        self.cover_table = cover_table
        self.uid_field = uid_field
        self.cid_field = cid_field
        self.embedding_field = embedding_field
        self.row_field = row_field
        self.feedback_perm = self._load_feedback_perm(last_runtime)

    def __len__(self):
        return len(self.feedback_perm)

    def _load_feedback_perm(self, after_time: Optional[datetime]) -> Permutation:
        if after_time is None:
            return (
                Permutation.identity(self.feedback_table)
                .select_columns([self.uid_field, self.cid_field, "type", "score", "timestamp"])
            )

        permutation_tbl = (
            permutation_builder(self.feedback_table)
            .filter(f"timestamp >= timestamp {after_time.strftime('%Y-%m-%d %H:%M:%S')}'")
            .execute()
        )
        return (
            Permutation.from_tables(self.feedback_table, permutation_tbl)
            .select_columns([self.uid_field, self.cid_field, "type", "score", "timestamp"])
        )

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        feedback = self.feedback_perm.__getitem__(idx)[0]
        user = (
            self.user_table.search()
            .where(f"{self.uid_field} = X'{feedback[self.uid_field].hex}'")
            .with_row_id(True)
            .select(["_rowid"])
            .limit(1)
        ).to_list()[0]
        cover = (
            self.cover_table.search()
            .where(f"{self.cid_field} = {feedback[self.cid_field]}")
            .select([self.cid_field, self.embedding_field])
            .limit(1)
        ).to_pydantic(CoverBackdate)[0]

        user_id_arr = torch.tensor(user["_rowid"] + DEFAULT_USER_OFFSET)
        item_arr = torch.tensor(cover.cover_embedding)
        item_id_arr = torch.tensor(cover.cover_id)
        rating_arr = torch.tensor([feedback["score"]])
        min_rating_arr = torch.tensor([FeedbackMap[feedback["type"]].value[0]])
        max_rating_arr = torch.tensor([FeedbackMap[feedback["type"]].value[1]])

        return user_id_arr, item_arr, item_id_arr, rating_arr, min_rating_arr, max_rating_arr

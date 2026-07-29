import torch
from torch import Tensor
from torch.utils.data import Dataset
from lancedb import Table
from lancedb.permutation import Permutation, permutation_builder
from lancedb.pydantic import LanceModel, Vector
from enum import Enum
import uuid
from helpers.db_tables import Feedback
from helpers.hardcover import CoverRecord
from helpers.tensor_ops import process_user_id, id_hash
from config.constants import HOT_FEEDBACK_TYPE, CLIP_DIM


class FeedbackMap(tuple[int, int], Enum):
    Rating = (0, 3)

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
        self.default_user_id = uuid.UUID(int=0)
        self.default_user = self._get_default_user()
        self.rating_arr = torch.tensor([self.max_rating])
        self.perm: Permutation | None = None

    def __len__(self):
        return len(self.covers_map)

    def _get_default_user(self) -> Tensor:
        return process_user_id(self.default_user_id)

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
        item_arr = torch.tensor(cover[self.embedding_field])
        item_id_arr = torch.tensor(id_hash(cover[self.id_field]))
        rating_arr = torch.tensor([self.covers_map[cover[self.id_field]][1]])
        min_rating_arr = torch.tensor([self.min_rating])
        max_rating_arr = torch.tensor([self.max_rating])

        return self.default_user, item_arr, item_id_arr, rating_arr, min_rating_arr, max_rating_arr


class FeedbackDataSet(Dataset):
    def __init__(
            self, feedback_table: Table, cover_table: Table,
            uid_field: str = "user_id", cid_field: str = "cover_id",
            embedding_field: str = "cover_embedding"
        ):
        self.feedback_table = feedback_table
        self.cover_table = cover_table
        self.uid_field = uid_field
        self.cid_field = cid_field
        self.embedding_field = embedding_field
        self.feedback_list = self._load_feedback_list()

    def __len__(self):
        return len(self.feedback_list)

    def _load_feedback_list(self) -> list[Feedback]:
        return (
            self.feedback_table.search()
            .select([self.uid_field, self.cid_field, "type", "score", "timestamp"])
        ).to_pydantic(Feedback)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        feedback = self.feedback_list[idx]
        cover = (
            self.cover_table.search()
            .where(f"{self.cid_field} = {feedback.cover_id}")
            .select([self.cid_field, self.embedding_field])
            .limit(1)
        ).to_pydantic(CoverBackdate)[0]

        user_arr = process_user_id(feedback.user_id)
        item_arr = torch.tensor(cover.cover_embedding)
        item_id_arr = torch.tensor(id_hash(cover.cover_id))
        rating_arr = torch.tensor([feedback.score])
        min_rating_arr = torch.tensor([FeedbackMap[feedback.type.value].value[0]])
        max_rating_arr = torch.tensor([FeedbackMap[feedback.type.value].value[1]])

        return user_arr, item_arr, item_id_arr, rating_arr, min_rating_arr, max_rating_arr

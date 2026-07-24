import torch
from torch import Tensor
from torch.utils.data import Dataset
from lancedb import Table
from lancedb.permutation import Permutation, permutation_builder
import uuid


class PopularCoversDataSet(Dataset):
    def __init__(
            self, table: Table, cover_ids: list[int], cover_id_field: str = "cover_id",
            embedding_field: str = "cover_embedding", min_rating: float = 0.0, max_rating: float = 5.0
        ):
        self.table = table
        self.cover_ids = cover_ids
        self.cover_id_field = cover_id_field
        self.embedding_field = embedding_field
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.default_user_id = uuid.UUID(int=0)
        self.default_user = self._get_default_user()
        self.rating_arr = torch.tensor([self.max_rating])
        self.perm: Permutation | None = None

    def __len__(self):
        return len(self.cover_ids)

    def _get_default_user(self) -> Tensor:
        return (
            torch.frombuffer(self.default_user_id.bytes_le, dtype=torch.int32)
            .to(dtype=torch.float32).unsqueeze(0)
        )

    def _ensure_permutation(self):
        if self.perm is None:
            id_strings = [f'{cid}' for cid in self.cover_ids]
            self.table.checkout_latest()
            permutation_tbl = (
                permutation_builder(self.table)
                .filter(f"{self.cover_id_field} IN ({', '.join(id_strings)})")
                .execute()
            )
            permutation = (
                Permutation.from_tables(self.table, permutation_tbl)
                .select_columns(["cover_id", "cover_embedding"])
            )
            self.perm = permutation

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        self._ensure_permutation()
        cover = self.perm.__getitem__(idx)[0]
        item_arr = torch.tensor(cover[self.embedding_field])
        rating_arr = torch.tensor([self.max_rating])
        min_rating_arr = torch.tensor([self.min_rating])
        max_rating_arr = torch.tensor([self.max_rating])

        return self.default_user, item_arr, rating_arr, min_rating_arr, max_rating_arr

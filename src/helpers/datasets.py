import torch
from torch import Tensor
from torch.utils.data import Dataset
from duckdb import DuckDBPyConnection
import polars as pl
import numpy as np
from datetime import datetime
from typing import Optional, Iterable
from helpers.feedback_maps import FeedbackMap, feedback_dict, hot_ratings_dict
from config.constants import HOT_COVERS_TABLE_NAME, COVER_TABLE_NAME, FEEDBACK_TABLE_NAME, USER_TABLE_NAME, DEFAULT_USER_OFFSET


def collate_fn(items: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    (other_features, embeddings) = items
    return (
        other_features[:, 0], embeddings, other_features[:, 1],
        other_features[:, 2], other_features[:, 3], other_features[:, 4]
    )


class BatchConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: Iterable[torch.utils.data.Dataset]) -> None:
        super().__init__(datasets)
        self.cumulative_sizes_arr = np.array(self.cumulative_sizes)

    def __getitems__(self, indices: list[int]):
        indices_arr = np.array(indices)
        if (-indices_arr[indices_arr < 0] > 165).any():
            raise ValueError(
                "absolute value of index should not exceed dataset length"
            )
        indices_arr[indices_arr < 0] += len(self)

        dataset_indices = np.searchsorted(self.cumulative_sizes_arr, indices_arr, side="right")
        indices_arr[dataset_indices > 0] -= self.cumulative_sizes_arr[dataset_indices[dataset_indices > 0] - 1]

        res1 = np.empty(indices_arr.shape[0], dtype=object)
        res2 = np.empty(indices_arr.shape[0], dtype=object)
        for ind in np.unique(dataset_indices):
            mask = dataset_indices == ind
            (items1, items2) = self.datasets[ind].__getitems__(indices_arr[mask].tolist())
            res1[mask] = np.fromiter(items1.unbind(dim=0), dtype=object)
            res2[mask] = np.fromiter(items2.unbind(dim=0), dtype=object)
        return (
            torch.vstack(res1.tolist()),
            torch.vstack(res2.tolist())
        )


class HotCoversDataSet(Dataset):
    def __init__(
            self, duckdb: DuckDBPyConnection, id_field: str = "cover_id",
            embedding_field: str = "cover_embedding", device: str = "cuda"
        ):
        self.duckdb = duckdb
        self.id_field = id_field
        self.embedding_field = embedding_field
        self.min_rating = FeedbackMap.Rating.value[0]
        self.max_rating = FeedbackMap.Rating.value[1]
        self.default_user_id = 0
        self.device = device
        self.hot_covers_df = self._load_hot_covers_df()

    def __len__(self):
        return len(self.hot_covers_df)

    def _load_hot_covers_df(self) -> pl.DataFrame:
        df = self.duckdb.execute(f"""
            SELECT covers.{self.id_field}, covers.{self.embedding_field}, hot_covers.type
            FROM lance_ns.main.{HOT_COVERS_TABLE_NAME} AS hot_covers
            LEFT OUTER JOIN lance_ns.main.{COVER_TABLE_NAME} AS covers
                ON (hot_covers.{self.id_field} = covers.{self.id_field})
        """).pl()

        return df.with_columns(
            pl.col("type").replace_strict(hot_ratings_dict).alias("score"),
            pl.lit(self.min_rating).alias("min_rating"),
            pl.lit(self.max_rating).alias("max_rating"),
            pl.lit(self.default_user_id).alias("user_row_id"),
        ).select([
            "user_row_id", self.embedding_field, self.id_field, "score", "min_rating", "max_rating"
        ]).with_row_index()

    def __getitems__(self, indices: list[int]) -> tuple[Tensor, Tensor]:
        res_df = (pl.DataFrame({"index": indices})
                  .join(self.hot_covers_df, on="index", how="left"))
        return (
            res_df.drop([self.embedding_field, "index"]).to_torch().to(self.device),
            res_df[self.embedding_field].to_torch().to(self.device)
        )


class FeedbackDataSet(Dataset):
    def __init__(
            self, duckdb: DuckDBPyConnection, last_runtime: Optional[datetime] = None,
            uid_field: str = "user_id", cid_field: str = "cover_id",
            embedding_field: str = "cover_embedding", user_row_field: str = "user_row_id",
            device: str = "cuda"
        ):
        self.duckdb = duckdb
        self.uid_field = uid_field
        self.cid_field = cid_field
        self.embedding_field = embedding_field
        self.user_row_field = user_row_field
        self.device = device
        self.feedback_df = self._load_feedback_df(last_runtime)

    def __len__(self):
        return len(self.feedback_df)

    def _load_feedback_df(self, last_runtime: Optional[datetime]) -> pl.DataFrame:
        df = self.duckdb.execute(f"""
            SELECT users.rowid as user_row_id, covers.{self.cid_field}, covers.{self.embedding_field}, feedback.type, feedback.score
            FROM lance_ns.main.{FEEDBACK_TABLE_NAME} AS feedback
            LEFT OUTER JOIN lance_ns.main.{COVER_TABLE_NAME} AS covers
                ON (feedback.{self.cid_field} = covers.{self.cid_field})    
            LEFT OUTER JOIN lance_ns.main.{USER_TABLE_NAME} AS users
                ON (feedback.{self.uid_field} = users.{self.uid_field})
            {f"WHERE feedback.timestamp >= timestamp '{last_runtime.strftime('%Y-%m-%d %H:%M:%S')}'" if last_runtime else ""}
        """).pl()

        return df.with_columns(
            pl.col("type")
                .replace_strict(feedback_dict).list
                .to_struct(fields=["min_rating", "max_rating"])
                .alias("ratings_range"),
            pl.col(self.user_row_field) + DEFAULT_USER_OFFSET
        ).unnest("ratings_range").select([
            self.user_row_field, self.embedding_field, self.cid_field, "score", "min_rating", "max_rating"
        ]).with_row_index()

    def __getitems__(self, indices: list[int]) -> tuple[Tensor, Tensor]:
        res_df = (pl.DataFrame({"index": indices})
                  .join(self.feedback_df, on="index", how="left"))
        return (
            res_df.drop([self.embedding_field, "index"]).to_torch().to(self.device),
            res_df[self.embedding_field].to_torch().to(self.device)
        )

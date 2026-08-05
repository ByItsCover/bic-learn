import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
import os
import logging
from config.constants import (TOWER_DIM, CLIP_DIM, FEATURE_WEIGHT_INIT,
                              ID_WEIGHT_INIT, ITEM_WEIGHT_DIFF_PENALTY,
                              MAX_ITEM_COUNT, ITEM_ID_BUCKET_COUNT,
                              HIDDEN_DIM, DROPOUT, DEFAULT_USER_OFFSET,
                              FINE_TUNING_SHRINK_FACTOR)
from helpers.tensor_ops import normalize

logger = logging.getLogger(__name__)


class UserTower(torch.nn.Module):
    def __init__(
            self, output_dim: int = TOWER_DIM, max_item_count: int = MAX_ITEM_COUNT,
            id_bucket_count: int = ITEM_ID_BUCKET_COUNT, hidden_dim: int = HIDDEN_DIM,
            dropout: float = DROPOUT
    ):
        super().__init__()
        self.id_bucket_count = id_bucket_count
        self.q_vocab_size = (max_item_count // self.id_bucket_count) + 1
        self.r_vocab_size = self.id_bucket_count
        self.q_layer = nn.Embedding(self.q_vocab_size, output_dim)
        self.r_layer = nn.Embedding(self.r_vocab_size, output_dim)
        self.tower = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, id_x: torch.Tensor):
        quotient_id = torch.floor_divide(id_x, self.id_bucket_count)
        remainder_id = torch.remainder(id_x, self.id_bucket_count)
        quotient_embed = self.q_layer(quotient_id)
        remainder_embed = self.r_layer(remainder_id)

        x = quotient_embed * remainder_embed

        out = self.tower(x)
        return out

class ItemTower(torch.nn.Module):
    def __init__(
            self, output_dim: int = TOWER_DIM, embed_dim: int = CLIP_DIM,
            feature_weight_init: float = FEATURE_WEIGHT_INIT,
            id_weight_init: float = ID_WEIGHT_INIT,
            max_item_count: int = MAX_ITEM_COUNT,
            id_bucket_count: int = ITEM_ID_BUCKET_COUNT,
            hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT
    ):
        super().__init__()
        self.features_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.features_weight = nn.Parameter(torch.tensor(feature_weight_init))

        self.id_bucket_count = id_bucket_count
        self.q_vocab_size = (max_item_count // self.id_bucket_count) + 1
        self.r_vocab_size = self.id_bucket_count
        self.q_layer = nn.Embedding(self.q_vocab_size, output_dim)
        self.r_layer = nn.Embedding(self.r_vocab_size, output_dim)
        self.id_layer = nn.Dropout(p=dropout)
        self.id_weight = nn.Parameter(torch.tensor(id_weight_init))

        self.tower = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features_x: torch.Tensor, id_x: torch.Tensor):
        features_embed = self.features_layer(features_x)
        weighted_features = self.features_weight / (self.features_weight + self.id_weight) * features_embed

        quotient_id = torch.floor_divide(id_x, self.id_bucket_count)
        remainder_id = torch.remainder(id_x, self.id_bucket_count)
        quotient_embed = self.q_layer(quotient_id)
        remainder_embed = self.r_layer(remainder_id)

        id_embed = self.id_layer(quotient_embed * remainder_embed)
        weighted_id = self.id_weight / (self.features_weight + self.id_weight) * id_embed

        x = weighted_features + weighted_id
        out = self.tower(x)
        return out

def train_all_models(
        user_tower: UserTower, item_tower: ItemTower, dataloader: DataLoader,
        epochs: int, early_stop: int, user_lr: float, item_lr: float
):
    user_optimizer = torch.optim.Adam(user_tower.parameters(), lr=user_lr)
    item_optimizer = torch.optim.Adam(item_tower.parameters(), lr=item_lr)
    user_tower.train()
    item_tower.train()

    best_loss = None
    not_lose_streak = 0

    logger.info("Training start")

    for epoch in tqdm(range(epochs)):
        total_training_loss = 0

        logger.info("Epoch %s", epoch)
        for batch_ind, batch in enumerate(tqdm(dataloader)):
            user_id, item, item_id, rating, min_rating, max_rating = batch
            user_optimizer.zero_grad()
            item_optimizer.zero_grad()
            user_pred = user_tower(user_id)
            item_pred = item_tower(item, item_id)

            ratings_pred = ((min_rating + 1
                             + F.cosine_similarity(
                                normalize(user_pred),
                                normalize(item_pred), dim=-1
                            ).unsqueeze(0).T)
                            * (max_rating / 2))

            ratings_loss = torch.square(rating - ratings_pred).mean()
            weight_loss = ITEM_WEIGHT_DIFF_PENALTY * torch.abs(item_tower.features_weight - item_tower.id_weight)
            loss = ratings_loss + weight_loss
            logger.info("Batch %s loss: %s", batch_ind, loss.item())

            loss.backward()
            user_optimizer.step()
            item_optimizer.step()
            total_training_loss += loss.item()

        avg_training_loss = total_training_loss / len(dataloader)
        logger.info("Average loss for epoch %s: %s", epoch, avg_training_loss)

        if best_loss is None or avg_training_loss < best_loss:
            not_lose_streak = 0
            best_loss = avg_training_loss
        else:
            not_lose_streak += 1

        if 0 < early_stop <= not_lose_streak:
            logger.info("Accuracy has not improved in %s rounds. Stopping early...", not_lose_streak)
            break;

    for name, param in item_tower.named_parameters():
        if name.endswith("_weight"):
            logger.info("Item tower final %s: %s", name, str(param.data))

    logger.info("Training end")

def tune_user_model(
        user_tower: UserTower, item_tower: ItemTower, dataloader: DataLoader,
        epochs: int, early_stop: int, user_lr: float
) -> list[int]:
    user_optimizer = torch.optim.Adam(user_tower.parameters(), lr=user_lr * FINE_TUNING_SHRINK_FACTOR)
    user_tower.train()
    item_tower.eval()

    for param in item_tower.parameters():
        param.requires_grad = False

    unique_ids = set()
    best_loss = None
    not_lose_streak = 0

    logger.info("Training start")

    for epoch in tqdm(range(int(epochs * FINE_TUNING_SHRINK_FACTOR))):
        total_training_loss = 0

        logger.info("Epoch %s", epoch)
        for batch_ind, batch in enumerate(tqdm(dataloader)):
            user_id, item, item_id, rating, min_rating, max_rating = batch
            user_optimizer.zero_grad()
            user_pred = user_tower(user_id)
            item_pred = item_tower(item, item_id)

            ratings_pred = ((min_rating + 1
                             + F.cosine_similarity(
                                normalize(user_pred),
                                normalize(item_pred), dim=-1
                            ).unsqueeze(0).T)
                            * (max_rating / 2))

            loss = torch.square(rating - ratings_pred).mean()
            logger.info("Batch %s loss: %s", batch_ind, loss.item())

            loss.backward()
            user_optimizer.step()
            total_training_loss += loss.item()
            unique_ids |= set((user_id - DEFAULT_USER_OFFSET).tolist())

        avg_training_loss = total_training_loss / len(dataloader)
        logger.info("Average loss for epoch %s: %s", epoch, avg_training_loss)

        if best_loss is None or avg_training_loss < best_loss:
            not_lose_streak = 0
            best_loss = avg_training_loss
        else:
            not_lose_streak += 1

        if 0 < early_stop <= not_lose_streak:
            logger.info("Accuracy has not improved in %s rounds. Stopping early...", not_lose_streak)
            break;

    logger.info("Training end. %s Users trained for", len(unique_ids))
    return list(unique_ids)

def save_models(model_dir: str, user_tower: Optional[UserTower] = None, item_tower: Optional[ItemTower] = None):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"No such directory: {model_dir}")

    if user_tower is not None:
        user_tower.eval()

        user_input_tensor = torch.ones(2, dtype=torch.int32)
        user_tower_onnx_path = os.path.join(model_dir, "user_tower.onnx")
        torch.onnx.export(
            user_tower,
            (user_input_tensor),
            user_tower_onnx_path,
            input_names=['users'],
            output_names=['embeddings'],
            dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
            external_data=False
        )

        user_tower_torch_path = os.path.join(model_dir, "user_tower_weights.pth")
        torch.save(user_tower.state_dict(), user_tower_torch_path)

    if item_tower is not None:
        item_tower.eval()

        item_input_tensor = torch.ones((2, CLIP_DIM), dtype=torch.float32)
        item_id_input_tensor = torch.ones(2, dtype=torch.int32)
        item_tower_onnx_path = os.path.join(model_dir, "item_tower.onnx")
        torch.onnx.export(
            item_tower,
            (item_input_tensor, item_id_input_tensor),
            item_tower_onnx_path,
            input_names=['items', 'ids'],
            output_names=['embeddings'],
            dynamic_shapes=(
                {0: torch.export.Dim.DYNAMIC},
                {0: torch.export.Dim.DYNAMIC},
            ),
            external_data=False
        )

        item_tower_torch_path = os.path.join(model_dir, "item_tower_weights.pth")
        torch.save(item_tower.state_dict(), item_tower_torch_path)

def load_models(user_tower: UserTower, item_tower: ItemTower, model_dir: str):
    user_tower_torch_path = os.path.join(model_dir, "user_tower_weights.pth")
    item_tower_torch_path = os.path.join(model_dir, "item_tower_weights.pth")

    if not (os.path.isfile(user_tower_torch_path) and os.path.isfile(item_tower_torch_path)):
        logger.warning("Torch models have not been saved. Training job likely hasn't run yet.")
        return;

    user_tower.load_state_dict(torch.load(user_tower_torch_path, weights_only=True))
    item_tower.load_state_dict(torch.load(item_tower_torch_path, weights_only=True))

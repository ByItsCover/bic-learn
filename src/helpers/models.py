import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
from config.constants import CLIP_DIM, USER_ID_DIM

logger = logging.getLogger(__name__)


class UserTower(torch.nn.Module):
    def __init__(self, output_dim: int, input_dim: int = USER_ID_DIM):
        super().__init__()
        self.layer_1 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor):
        out = self.layer_1(x)
        return out

class ItemTower(torch.nn.Module):
    def __init__(self, output_dim: int, input_dim: int = CLIP_DIM):
        super().__init__()
        self.layer_1 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor):
        out = self.layer_1(x)
        return out

def train_models(user_tower: UserTower, item_tower: ItemTower, dataloader: DataLoader, epochs: int, user_lr: float, item_lr: float):
    user_optimizer = torch.optim.Adam(user_tower.parameters(), lr=user_lr)
    item_optimizer = torch.optim.Adam(item_tower.parameters(), lr=item_lr)
    user_tower.train()
    item_tower.train()

    logger.info("Training start")

    for epoch in tqdm(range(epochs)):
        total_training_loss = 0

        logger.info("Epoch %s", epoch)
        for batch_ind, batch in enumerate(tqdm(dataloader)):
            user, item, rating, min_rating, max_rating = batch
            user_optimizer.zero_grad()
            item_optimizer.zero_grad()
            user_pred = user_tower(user)
            item_pred = item_tower(item)

            ratings_pred = ((min_rating + 1 + (user_pred / user_pred.norm(dim=-1, keepdim=True))
                             @ (item_pred / item_pred.norm(dim=-1, keepdim=True)).T)
                            * (max_rating / 2))
            loss = torch.square(rating - ratings_pred).mean()
            logger.info("Batch %s loss: %s", batch_ind, loss.item())

            loss.backward()
            user_optimizer.step()
            item_optimizer.step()
            total_training_loss += loss.item()

        avg_training_loss = total_training_loss / len(dataloader)
        logger.info("Average loss for epoch %s: %s", epoch, avg_training_loss)

    logger.info("Training end")

def save_models(user_tower: UserTower, item_tower: ItemTower, model_dir: str):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"No such directory: {model_dir}")

    user_tower.eval()
    item_tower.eval()

    user_input_tensor = torch.ones((2, USER_ID_DIM), dtype=torch.float32)
    user_tower_path = os.path.join(model_dir, "user_tower.onnx")
    torch.onnx.export(
        user_tower,
        (user_input_tensor),
        user_tower_path,
        input_names=['users'],
        output_names=['embeddings'],
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
        external_data=False
    )

    item_input_tensor = torch.ones((2, CLIP_DIM), dtype=torch.float32)
    item_tower_path = os.path.join(model_dir, "item_tower.onnx")
    torch.onnx.export(
        item_tower,
        (item_input_tensor),
        item_tower_path,
        input_names=['items'],
        output_names=['embeddings'],
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
        external_data=False
    )

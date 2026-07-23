import torch
import logging
from models import UserTower, ItemTower
from utils.model_utils import save_models

logger = logging.getLogger(__name__)


def full_train(tower_dim: int, model_dir: str):
    logger.info("CUDA availability: %s", torch.cuda.is_available())
    logger.info("CUDA device name: %s", torch.cuda.get_device_name(0))

    user_tower = UserTower(tower_dim)
    item_tower = ItemTower(tower_dim)

    user_tower.eval()
    item_tower.eval()
    save_models(user_tower, item_tower, model_dir)

import torch
import os
from models import UserTower, ItemTower


def save_models(user_tower: UserTower, item_tower: ItemTower, model_dir: str):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"No such directory: {model_dir}")

    user_input_tensor = torch.ones((2, 4), dtype=torch.float32)
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

    item_input_tensor = torch.ones((2, 512), dtype=torch.float32)
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

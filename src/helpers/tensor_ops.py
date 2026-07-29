import torch
import uuid
from config.constants import ITEM_ID_BUCKET_SIZE


def normalize(tensor: torch.Tensor):
    return tensor / tensor.norm(dim=-1, keepdim=True)

def process_user_id(user_id: uuid.UUID) -> torch.Tensor:
    bytes_copy = bytearray(user_id.bytes_le)
    return (
        torch.frombuffer(bytes_copy, dtype=torch.int32)
        .to(dtype=torch.float32).unsqueeze(0)
    )

def process_item_id(id_val: int, hash_dim: int = ITEM_ID_BUCKET_SIZE) -> int:
    return hash(id_val) % hash_dim

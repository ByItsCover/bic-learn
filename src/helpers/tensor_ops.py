import torch


def normalize(tensor: torch.Tensor):
    return tensor / tensor.norm(dim=-1, keepdim=True)

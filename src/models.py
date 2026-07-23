import torch


class UserTower(torch.nn.Module):
    def __init__(self, output_dim: int, input_dim: int = 4):
        super().__init__()
        self.layer_1 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor):
        out = self.layer_1(x)
        return out

class ItemTower(torch.nn.Module):
    def __init__(self, output_dim: int, input_dim: int = 512):
        super().__init__()
        self.layer_1 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor):
        out = self.layer_1(x)
        return out

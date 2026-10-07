from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, num_users: int, embedding_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(self.user_embedding(ids)), p=2, dim=-1)


class ItemTower(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(self.item_embedding(ids)), p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        temperature: float,
    ):
        super().__init__()
        self.user_tower = UserTower(num_users, embedding_dim, hidden_dim, output_dim)
        self.item_tower = ItemTower(num_items, embedding_dim, hidden_dim, output_dim)
        self.temperature = float(temperature)

    def encode_users(self, ids: torch.Tensor) -> torch.Tensor:
        return self.user_tower(ids)

    def encode_items(self, ids: torch.Tensor) -> torch.Tensor:
        return self.item_tower(ids)


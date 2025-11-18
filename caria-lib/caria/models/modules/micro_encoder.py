"""Encoder para señales microeconómicas (fundamentales)."""

from __future__ import annotations

import torch
from torch import nn


class MicroEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev, dim), nn.ReLU(), nn.BatchNorm1d(dim)])
            prev = dim
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


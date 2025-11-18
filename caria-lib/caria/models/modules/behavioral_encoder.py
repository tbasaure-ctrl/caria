"""Encoder para señales de comportamiento (sentiment)."""

from __future__ import annotations

import torch
from torch import nn


class BehavioralEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


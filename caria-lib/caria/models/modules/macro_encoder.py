"""Encoder macro basado en LSTM."""

from __future__ import annotations

import torch
from torch import nn


class MacroEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(x)
        return outputs[:, -1, :]


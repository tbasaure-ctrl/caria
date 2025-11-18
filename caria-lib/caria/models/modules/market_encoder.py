"""Encoder de estructura de mercado con self-attention."""

from __future__ import annotations

import torch
from torch import nn


class MarketEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x)
        return attended.mean(dim=1)


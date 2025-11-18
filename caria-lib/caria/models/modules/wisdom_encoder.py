"""Encoder para sabiduría histórica mediante transformers."""

from __future__ import annotations

import torch
from torch import nn

from transformers import AutoModel


class WisdomEncoder(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]


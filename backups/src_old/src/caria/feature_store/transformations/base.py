"""Transformaciones de features Caria."""

from __future__ import annotations

from typing import Protocol

import polars as pl


class Transformation(Protocol):
    def __call__(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Aplica la transformación sobre un DataFrame y retorna otro."""


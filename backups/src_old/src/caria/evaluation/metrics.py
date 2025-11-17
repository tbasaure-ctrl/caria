"""Módulo para métricas de evaluación del modelo."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def sharpe_ratio(returns: Iterable[float]) -> float:
    arr = np.asarray(list(returns))
    if arr.std() == 0:
        return 0.0
    return float(arr.mean() / arr.std())


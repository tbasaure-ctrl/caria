"""Utilidades para servir el modelo de Caria en producción."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from caria.config.settings import Settings

# Importaciones opcionales para modelo legacy (deprecated)
try:
    from caria.models.training.datamodule import CariaDataModule
    from caria.models.training.workflow import SimpleFusionModel
except ImportError:
    # pytorch_lightning no está instalado
    CariaDataModule = None
    SimpleFusionModel = None


@dataclass(slots=True)
class ModelBundle:
    """Agrupa los artefactos necesarios para inferencia."""

    model: SimpleFusionModel
    feature_mean: np.ndarray
    feature_std: np.ndarray
    device: torch.device

    def predict(self, rows: Iterable[Iterable[float]]) -> np.ndarray:
        """Genera predicciones normalizando con las estadísticas aprendidas."""

        matrix = np.asarray(list(rows), dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        normalized = (matrix - self.feature_mean) / self.feature_std
        with torch.no_grad():
            tensor = torch.from_numpy(normalized).to(self.device)
            outputs = self.model(tensor).squeeze(-1).cpu().numpy()
        return outputs


def load_model_bundle(settings: Settings, checkpoint_path: Path | str) -> ModelBundle:
    """Carga el modelo entrenado más las estadísticas de features."""

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint en {checkpoint}")

    gold_path = settings.get("storage", "gold_path", default="data/gold")
    data_root = Path(gold_path)

    datamodule = CariaDataModule(data_root=data_root, settings=settings)
    datamodule.setup(stage="predict")
    feature_dim = datamodule.get_feature_dim()
    feature_mean, feature_std = datamodule.get_feature_stats()

    model = SimpleFusionModel.load_from_checkpoint(checkpoint, input_dim=feature_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return ModelBundle(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        device=device,
    )



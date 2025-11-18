"""DataModule para PyTorch Lightning."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
try:
    import pytorch_lightning as pl
except ImportError:
    # Crear clase stub para compatibilidad cuando pytorch_lightning no está instalado
    class _DummyModule:
        class LightningDataModule:
            pass
    pl = _DummyModule()
import torch
from torch.utils.data import DataLoader, Dataset

from caria.config.settings import Settings


class CariaDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[idx]
        features = row["features"]
        if not isinstance(features, np.ndarray):
            features = np.asarray(features, dtype=np.float32)
        else:
            features = features.astype(np.float32, copy=False)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if self.feature_mean is not None and self.feature_std is not None:
            features = (features - self.feature_mean) / self.feature_std

        features_tensor = torch.from_numpy(features)
        target_tensor = torch.tensor(row["target"], dtype=torch.float32)
        return {"features": features_tensor, "target": target_tensor}


class CariaDataModule(pl.LightningDataModule):
    def __init__(self, data_root: Path, settings: Settings) -> None:
        super().__init__()
        self.data_root = data_root
        self.settings = settings
        self.train_ds: CariaDataset | None = None
        self.val_ds: CariaDataset | None = None
        self.test_ds: CariaDataset | None = None
        self._feature_dim: int | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._stats_path = self.data_root / "metadata" / "feature_stats.json"

    def _load_frame(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"No se encuentra dataset {path}")
        frame = pd.read_parquet(path)
        numeric_cols = frame.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            frame.loc[:, numeric_cols] = frame.loc[:, numeric_cols].replace([np.inf, -np.inf], np.nan)
        if "target_return_20d" in frame.columns:
            frame = frame.dropna(subset=["target_return_20d"])
        if "target" in frame.columns:
            frame = frame.dropna(subset=["target"])
        if "features" in frame.columns and not frame.empty:
            frame["features"] = frame["features"].apply(
                lambda values: np.nan_to_num(
                    np.asarray(values, dtype=np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )
            first_features = frame.iloc[0]["features"]
            if isinstance(first_features, (list, tuple, np.ndarray)):
                self._feature_dim = len(first_features)
        return frame

    def _compute_stats(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            raise ValueError("El dataset de entrenamiento esta vacio; no se pueden inferir estadisticas")
        features_array = np.stack(frame["features"].values).astype(np.float64)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        self._feature_dim = features_array.shape[1]
        mean = features_array.mean(axis=0)
        std = features_array.std(axis=0)
        std[std < 1e-8] = 1.0
        self._feature_mean = mean.astype(np.float32)
        self._feature_std = std.astype(np.float32)
        self._persist_stats()

    def _persist_stats(self) -> None:
        self._stats_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_dim": int(self._feature_dim) if self._feature_dim is not None else None,
            "mean": self._feature_mean.tolist() if self._feature_mean is not None else None,
            "std": self._feature_std.tolist() if self._feature_std is not None else None,
        }
        self._stats_path.write_text(json.dumps(payload))

    def _load_stats_from_disk(self) -> bool:
        if not self._stats_path.exists():
            return False
        payload = json.loads(self._stats_path.read_text())
        mean = payload.get("mean")
        std = payload.get("std")
        dim = payload.get("feature_dim")
        if mean is None or std is None or dim is None:
            return False
        self._feature_mean = np.asarray(mean, dtype=np.float32)
        self._feature_std = np.asarray(std, dtype=np.float32)
        self._feature_dim = int(dim)
        return True

    def setup(self, stage: str | None = None) -> None:
        train_path = self.data_root / "train.parquet"
        val_path = self.data_root / "val.parquet"
        test_path = self.data_root / "test.parquet"
        train_frame = self._load_frame(train_path)

        if stage in (None, "fit", "train"):
            self._compute_stats(train_frame)
        else:
            stats_loaded = self._load_stats_from_disk()
            if not stats_loaded:
                self._compute_stats(train_frame)

        val_frame = self._load_frame(val_path)
        test_frame = self._load_frame(test_path)

        self.train_ds = CariaDataset(train_frame, self._feature_mean, self._feature_std)
        self.val_ds = CariaDataset(val_frame, self._feature_mean, self._feature_std)
        self.test_ds = CariaDataset(test_frame, self._feature_mean, self._feature_std)

    def get_feature_dim(self) -> int:
        if self._feature_dim is None:
            if not self._load_stats_from_disk():
                train_frame = self._load_frame(self.data_root / "train.parquet")
                self._compute_stats(train_frame)
        if self._feature_dim is None:
            raise RuntimeError("No se pudo determinar feature_dim")
        return self._feature_dim

    def get_feature_stats(self) -> tuple[np.ndarray, np.ndarray]:
        if self._feature_mean is None or self._feature_std is None:
            if not self._load_stats_from_disk():
                raise RuntimeError("Las estadisticas de features aun no han sido calculadas; llama setup() primero")
        if self._feature_mean is None or self._feature_std is None:
            raise RuntimeError("No se pudieron recuperar las estadisticas de features")
        return self._feature_mean, self._feature_std

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        batch_size = self.settings.get("training", "batch_size", default=64)
        return DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        batch_size = self.settings.get("training", "batch_size", default=64)
        return DataLoader(self.val_ds, batch_size=batch_size)

    def test_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        batch_size = self.settings.get("training", "batch_size", default=64)
        return DataLoader(self.test_ds, batch_size=batch_size)


"""Construcción del modelo multi-modal.

DEPRECATED: Este modelo monolítico será reemplazado por los 4 sistemas especializados:
- Sistema I: HMM Régimen (regime_hmm_pipeline.py)
- Sistema II: RAG (rag_service.py)
- Sistema III: Factores (factor_screener.py)
- Sistema IV: Valuación (dcf_valuator.py, scorecard_valuator.py)

Este módulo se mantiene solo para compatibilidad temporal y será eliminado en el futuro.
"""

from __future__ import annotations

import warnings

import torch
import pytorch_lightning as pl
from torch import nn

from caria.config.settings import Settings

# Advertir sobre deprecación
warnings.warn(
    "SimpleFusionModel está deprecated. Usa los sistemas especializados: "
    "Sistema I (HMM Régimen), Sistema II (RAG), Sistema III (Factores), Sistema IV (Valuación).",
    DeprecationWarning,
    stacklevel=2,
)


def _ensure_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype not in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        tensor = tensor.float()
    return tensor


class SimpleFusionModel(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        learning_rate: float,
        dropout: float = 0.4,
        weight_decay: float = 1e-3,
    ) -> None:
        super().__init__()
        # Reducir hidden_dim si es muy grande vs input_dim para evitar overfitting
        if hidden_dim > input_dim * 2:
            hidden_dim = max(input_dim, int(input_dim * 1.5))
        self.save_hyperparameters()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, 1)
        self.loss_fn = nn.MSELoss()
        self._validation_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._test_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.backbone(x)
        return self.head(latent)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        preds = self.forward(batch["features"]).squeeze(-1)
        target = _ensure_float_tensor(batch["target"])
        loss = self.loss_fn(preds, target)
        self.log("train_loss", loss, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:  # noqa: ARG002
        preds = self.forward(batch["features"]).squeeze(-1)
        target = _ensure_float_tensor(batch["target"])
        self._validation_outputs.append((preds.detach().cpu(), target.detach().cpu()))

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:  # noqa: ARG002
        preds = self.forward(batch["features"]).squeeze(-1)
        target = _ensure_float_tensor(batch["target"])
        self._test_outputs.append((preds.detach().cpu(), target.detach().cpu()))

    def on_validation_epoch_end(self) -> None:
        if not self._validation_outputs:
            return
        preds, targets = self._stack_outputs(self._validation_outputs)
        metrics = self._compute_regression_metrics(preds, targets)
        self.log_dict(
            {f"val_{name}": value for name, value in metrics.items()},
            prog_bar=True,
            on_epoch=True,
            sync_dist=False,
        )
        self._validation_outputs.clear()

    def on_test_epoch_end(self) -> None:
        if not self._test_outputs:
            return
        preds, targets = self._stack_outputs(self._test_outputs)
        metrics = self._compute_regression_metrics(preds, targets)
        self.log_dict(
            {f"test_{name}": value for name, value in metrics.items()},
            prog_bar=True,
            on_epoch=True,
            sync_dist=False,
        )
        self._test_outputs.clear()

    @staticmethod
    def _stack_outputs(outputs: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        preds = torch.cat([pred for pred, _ in outputs], dim=0)
        targets = torch.cat([target for _, target in outputs], dim=0)
        return preds, targets

    def _compute_regression_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        preds = preds.float()
        targets = targets.float()
        mse = torch.mean((preds - targets) ** 2)
        rmse = torch.sqrt(mse + 1e-12)
        mae = torch.mean(torch.abs(preds - targets))
        target_mean = torch.mean(targets)
        ss_tot = torch.sum((targets - target_mean) ** 2)
        ss_res = torch.sum((targets - preds) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        r2 = torch.clamp(r2, min=-1.0, max=1.0)
        return {
            "loss": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self._weight_decay,
        )


def build_model(settings: Settings, input_dim: int | None = None) -> SimpleFusionModel:
    if input_dim is None:
        input_dim = settings.get("model", "fusion", "latent_dim", default=512)
    hidden_dim = settings.get("model", "fusion", "latent_dim", default=512)
    learning_rate = settings.get("training", "learning_rate", default=1e-4)
    dropout = settings.get("training", "dropout", default=0.4)
    weight_decay = settings.get("training", "weight_decay", default=1e-3)
    return SimpleFusionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        dropout=dropout,
        weight_decay=weight_decay,
    )


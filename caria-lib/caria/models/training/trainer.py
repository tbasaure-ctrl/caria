"""Construcción del objeto Trainer de Lightning."""

from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from caria.config.settings import Settings


def build_trainer(settings: Settings) -> pl.Trainer:
    log_dir = settings.get("training", "log_dir", default="lightning_logs")
    experiment_name = settings.get("training", "experiment_name", default="caria")
    logger = TensorBoardLogger(save_dir=log_dir, name=experiment_name, default_hp_metric=False)

    monitor_metric = settings.get("training", "monitor_metric", default="val_loss")
    mode = settings.get("training", "monitor_mode", default="min")
    patience = settings.get("training", "early_stopping_patience", default=7)
    min_delta = settings.get("training", "early_stopping_min_delta", default=1e-5)
    save_top_k = settings.get("training", "checkpoint_save_top_k", default=3)

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode=mode,
        filename="epoch={epoch:02d}-val_loss={val_loss:.4f}",
        save_top_k=save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        mode=mode,
        min_delta=min_delta,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=settings.get("training", "max_epochs", default=50),
        accelerator="auto",
        devices="auto",
        precision=settings.get("training", "precision", default=16),
        gradient_clip_val=settings.get("training", "gradient_clip_val", default=1.0),
        log_every_n_steps=10,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    return trainer


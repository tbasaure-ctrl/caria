"""Pipeline de entrenamiento para el modelo multi-modal de Caria."""

from __future__ import annotations

import logging
from pathlib import Path

import pytorch_lightning as pl
from prefect import flow, task

from caria.config.settings import Settings
from caria.models.training.datamodule import CariaDataModule
from caria.models.training.trainer import build_trainer
from caria.models.training.workflow import build_model


LOGGER = logging.getLogger("caria.pipelines.training")


@task(name="prepare-data")
def prepare_data(settings: Settings) -> CariaDataModule:
    data_root = Path(settings.get("storage", "gold_path", default="data/gold"))
    return CariaDataModule(data_root=data_root, settings=settings)


@task(name="train-model")
def train_model(trainer: pl.Trainer, model: pl.LightningModule, datamodule: CariaDataModule) -> None:
    trainer.fit(model=model, datamodule=datamodule)


@flow(name="caria-training-pipeline")
def training_flow(settings: Settings) -> None:
    datamodule = prepare_data(settings)
    model = build_model(settings)
    trainer = build_trainer(settings)
    train_model(trainer, model, datamodule)


def run(settings: Settings, pipeline_config_path: str | None = None) -> None:
    training_flow(settings=settings)


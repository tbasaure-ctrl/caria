"""Evaluar un checkpoint entrenado sobre splits de validación o test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pytorch_lightning as pl
from dotenv import load_dotenv

# Ajustar sys.path para que las importaciones de `caria` funcionen al ejecutar desde notebooks
CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parents[2]
POTENTIAL_SRC_DIRS = [
    BASE_DIR / "src",
    BASE_DIR / "caria" / "src",
    BASE_DIR.parent / "caria" / "src",
]
for candidate in POTENTIAL_SRC_DIRS:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from caria.config.settings import Settings
from caria.models.training.datamodule import CariaDataModule
from caria.models.training.workflow import SimpleFusionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa un checkpoint de Caria")
    parser.add_argument(
        "--config",
        default=str((BASE_DIR / "caria" / "configs" / "base.yaml").resolve()),
        help="Ruta al archivo YAML de configuración",
    )
    parser.add_argument("--checkpoint", required=True, help="Ruta al checkpoint (.ckpt) a evaluar")
    parser.add_argument(
        "--split",
        choices=("val", "test", "both"),
        default="both",
        help="Splits a evaluar (validación, test o ambos)",
    )
    return parser.parse_args()


def build_datamodule(settings: Settings) -> CariaDataModule:
    gold_path = settings.get("storage", "gold_path", default="data/gold")
    data_root = Path(gold_path)
    return CariaDataModule(data_root=data_root, settings=settings)


def run_validation(trainer: pl.Trainer, model: SimpleFusionModel, datamodule: CariaDataModule) -> dict[str, float]:
    results = trainer.validate(model=model, dataloaders=datamodule.val_dataloader())
    return results[0] if results else {}


def run_test(trainer: pl.Trainer, model: SimpleFusionModel, datamodule: CariaDataModule) -> dict[str, float]:
    results = trainer.test(model=model, dataloaders=datamodule.test_dataloader())
    return results[0] if results else {}


def main() -> None:
    load_dotenv()
    args = parse_args()
    settings = Settings.from_yaml(Path(args.config))

    datamodule = build_datamodule(settings)
    datamodule.setup(stage="test")

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (BASE_DIR / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint en {checkpoint_path}")

    model = SimpleFusionModel.load_from_checkpoint(
        checkpoint_path,
        input_dim=datamodule.get_feature_dim(),
    )
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,
        enable_checkpointing=False,
    )

    summary: dict[str, dict[str, float]] = {}
    if args.split in ("val", "both"):
        summary["validation"] = run_validation(trainer, model, datamodule)
    if args.split in ("test", "both"):
        summary["test"] = run_test(trainer, model, datamodule)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()



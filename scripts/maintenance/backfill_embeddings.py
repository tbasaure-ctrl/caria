"""Backfill de embeddings históricos."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from caria.config.settings import Settings
from caria.services.workers.jobs.rag_refresh import run as run_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill embeddings into vector store")
    parser.add_argument("dataset", help="Ruta al dataset parquet/JSONL con contenidos")
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    settings = Settings.from_yaml(Path(args.config))
    run_job(Path(args.dataset), settings=settings)


if __name__ == "__main__":
    main()


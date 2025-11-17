"""Sincroniza el schema de bases operacionales."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from caria.config.settings import Settings
from caria.data_access.jobs_registry import ensure_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync operational schemas")
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    settings = Settings.from_yaml(Path(args.config))
    connection_uri = settings.get("vector_store", "connection")
    if not connection_uri:
        raise RuntimeError("vector_store.connection no configurado")
    ensure_tables(connection_uri)


if __name__ == "__main__":
    main()


"""Orquestador CLI para ingerir textos de sabiduría y poblar pgvector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parents[2]
POTENTIAL_SRC_DIRS = [
    BASE_DIR / "src",
    BASE_DIR / "caria" / "src",
]
for candidate in POTENTIAL_SRC_DIRS:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from caria.config.settings import Settings
from caria.pipelines.wisdom_pipeline import wisdom_ingest_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingesta de textos de sabiduría y embeddings en pgvector")
    parser.add_argument("--config", default="caria/configs/base.yaml", help="Ruta al archivo de configuración")
    parser.add_argument("--raw-dir", help="Directorio raíz con documentos de sabiduría (por defecto usa storage.raw_path/wisdom)")
    parser.add_argument("--version", help="Versión o etiqueta del índice de sabiduría")
    parser.add_argument("--chunk-size", type=int, default=800, help="Tamaño de fragmento en palabras para dividir textos")
    parser.add_argument("--overlap", type=int, default=120, help="Solapamiento en palabras entre fragmentos consecutivos")
    return parser.parse_args()


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return path


def main() -> None:
    load_dotenv()
    args = parse_args()
    settings = Settings.from_yaml(_resolve_path(args.config))
    raw_root = Path(args.raw_dir) if args.raw_dir else Path(settings.get("storage", "raw_path", default="data/raw")) / "wisdom"
    wisdom_ingest_flow(
        settings=settings,
        raw_dir=raw_root,
        version=args.version,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()

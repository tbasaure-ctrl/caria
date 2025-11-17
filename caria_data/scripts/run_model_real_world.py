from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import func, select

CURRENT_FILE = Path(__file__).resolve()
DEFAULT_BASE_PATH = CURRENT_FILE.parents[1]

POTENTIAL_SRC_DIRS = [
    DEFAULT_BASE_PATH / "src",
    DEFAULT_BASE_PATH / "caria" / "src",
    DEFAULT_BASE_PATH.parent / "caria" / "src",
]
for candidate in POTENTIAL_SRC_DIRS:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from caria.config.settings import Settings  # noqa: E402  (import after sys.path tweak)
from caria.data_access.db import factory  # noqa: E402
from caria.embeddings.generator import EmbeddingGenerator  # noqa: E402
from caria.models.training.inference import load_model_bundle  # noqa: E402
from caria.pipelines.wisdom_pipeline import run as run_wisdom_pipeline  # noqa: E402
from caria.retrieval.retrievers import Retriever  # noqa: E402
from caria.retrieval.vector_store import VectorStore  # noqa: E402


def parse_val_loss(checkpoint_name: str) -> float | None:
    if "val_loss=" not in checkpoint_name:
        return None
    try:
        suffix = checkpoint_name.split("val_loss=")[-1]
        value_str = suffix.split("-")[0].replace(".ckpt", "")
        return float(value_str)
    except ValueError:
        return None


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    candidates: list[tuple[float, Path]] = []
    for path in checkpoint_dir.glob("*.ckpt"):
        if path.name == "last.ckpt":
            continue
        val_loss = parse_val_loss(path.name)
        if val_loss is not None:
            candidates.append((val_loss, path))
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]
    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt
    raise FileNotFoundError(
        f"No se encontró un checkpoint en {checkpoint_dir}. Ejecuta el pipeline de entrenamiento primero."
    )


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_true - y_pred
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    r2 = float(
        1.0
        - np.sum(diff**2)
        / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12)
    )
    return {"rmse": rmse, "mae": mae, "r2": r2}


def format_metrics(metrics: dict[str, float]) -> dict[str, Any]:
    pct_metrics = {k: v * 100 for k, v in metrics.items() if k in {"rmse", "mae"}}
    return {
        **metrics,
        "rmse_pct": pct_metrics.get("rmse"),
        "mae_pct": pct_metrics.get("mae"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalúa el modelo con datos reales y opcionalmente ejecuta el pipeline de embeddings."
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=DEFAULT_BASE_PATH,
        help="Ruta base del proyecto (default: carpeta que contiene este script).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Ruta al checkpoint .ckpt; si no se especifica se usa el mejor en lightning_logs.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directorio que contiene lightning_logs/caria/version_xx (default: BASE_PATH/lightning_logs/caria/version_15).",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="Directorio donde copiar el checkpoint seleccionado (default: BASE_PATH/artifacts/models).",
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        help="Ruta al parquet de test (default: BASE_PATH/data/gold/test.parquet).",
    )
    parser.add_argument(
        "--query",
        default="impacto de crisis financieras en valoración de empresas",
        help="Consulta para verificar recuperación semántica.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Número de documentos a recuperar (default: retrieval.top_k).",
    )
    parser.add_argument(
        "--run-vector-pipeline",
        action="store_true",
        help="Si se proporciona, ejecuta run_wisdom_pipeline antes de consultar pgvector.",
    )
    parser.add_argument(
        "--wisdom-version",
        default=os.getenv("WISDOM_VERSION", "2025-11-08"),
        help="Versión del índice de sabiduría a procesar (cuando se usa --run-vector-pipeline).",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="No copiar el checkpoint al directorio de artefactos.",
    )
    args = parser.parse_args()

    load_dotenv()

    base_path: Path = args.base_path.resolve()
    if not (base_path / "configs" / "base.yaml").exists():
        raise FileNotFoundError(f"No se encontró configs/base.yaml en {base_path}")

    gold_path = (args.gold_path or (base_path / "data/gold/test.parquet")).resolve()
    if not gold_path.exists():
        raise FileNotFoundError(f"No se encontró dataset gold en {gold_path}")

    log_dir = (
        args.log_dir
        or (base_path / "lightning_logs/caria/version_15")
    ).resolve()
    checkpoint_dir = log_dir / "checkpoints"

    export_dir = (args.export_dir or (base_path / "artifacts/models")).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint(checkpoint_dir)
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe el checkpoint {checkpoint_path}")

    if not args.skip_export:
        export_target = export_dir / checkpoint_path.name
        shutil.copy2(checkpoint_path, export_target)
        print(f"Checkpoint copiado a {export_target}")
    else:
        export_target = checkpoint_path

    settings = Settings.from_yaml(base_path / "configs/base.yaml")

    model_bundle = load_model_bundle(settings=settings, checkpoint_path=export_target)
    test_df = pd.read_parquet(gold_path)
    y_true = test_df["target_return_20d"].to_numpy()
    feature_matrix = np.vstack(test_df["features"].values)
    y_pred = model_bundle.predict(feature_matrix)
    baseline_metrics = format_metrics(regression_report(y_true, y_pred))
    print("Métricas (baseline):")
    print(json.dumps(baseline_metrics, indent=2))

    if args.run_vector_pipeline:
        os.environ.setdefault("RETRIEVAL_PROVIDER", "gemini")
        os.environ.setdefault("RETRIEVAL_EMBEDDING_MODEL", "models/text-embedding-004")
        os.environ.setdefault("RETRIEVAL_EMBEDDING_DIM", "768")
        settings = Settings.from_yaml(base_path / "configs/base.yaml")
        raw_dir = base_path / "data/raw/wisdom" / args.wisdom_version
        run_wisdom_pipeline(settings=settings, raw_dir=raw_dir)
        print("Pipeline de wisdom ejecutado correctamente.")

    # Verificación del índice y búsqueda
    settings = Settings.from_yaml(base_path / "configs/base.yaml")
    vector_store = VectorStore.from_settings(settings)
    with factory.get_engine(vector_store.connection_uri).connect() as connection:
        count = connection.execute(
            select(func.count()).select_from(vector_store.table)
        ).scalar()
    print(
        f"Embeddings almacenados en {vector_store.table_name}: "
        f"{int(count) if count is not None else 0}"
    )

    generator = EmbeddingGenerator(settings=settings)
    retriever = Retriever(vector_store=vector_store)
    embedding = generator.embed_text(args.query)
    top_k = args.top_k or int(settings.get("retrieval", "top_k", default=5))
    results = retriever.query(embedding, top_k=top_k)
    print("Resultados de recuperación:")
    for idx, result in enumerate(results, start=1):
        meta = result.metadata
        title = meta.get("title") or meta.get("source", "sin título")
        print(f"[{idx}] score={result.score:.3f} :: {title}")

    # A/B simple (re-evaluamos para asegurar consistencia)
    y_pred_after = model_bundle.predict(feature_matrix)
    post_metrics = format_metrics(regression_report(y_true, y_pred_after))
    delta_rmse_bps = (post_metrics["rmse"] - baseline_metrics["rmse"]) * 10_000
    delta_mae_bps = (post_metrics["mae"] - baseline_metrics["mae"]) * 10_000
    summary = {
        "baseline": baseline_metrics,
        "post_retrieval": post_metrics,
        "delta_rmse_bps": delta_rmse_bps,
        "delta_mae_bps": delta_mae_bps,
    }
    print("Comparativa A/B:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


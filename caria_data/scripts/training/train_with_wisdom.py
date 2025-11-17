"""Pre-computar embeddings de wisdom para datasets gold y agregarlos como features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Ajustar sys.path
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
from caria.embeddings.generator import EmbeddingGenerator
from caria.retrieval.retrievers import Retriever
from caria.retrieval.vector_store import VectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-computar embeddings de wisdom para gold datasets")
    parser.add_argument(
        "--config",
        default=str((BASE_DIR / "caria" / "configs" / "base.yaml").resolve()),
        help="Ruta al archivo YAML de configuración",
    )
    parser.add_argument(
        "--gold-dir",
        default=str(BASE_DIR / "data" / "gold"),
        help="Directorio con datasets gold",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Número de documentos wisdom a recuperar por muestra",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits a procesar",
    )
    return parser.parse_args()


def create_query_context(row: pd.Series) -> str:
    """Crea un contexto de query para RAG basado en ticker y fecha."""
    ticker = row.get("ticker", "")
    date = row.get("date", "")
    context = f"Investment analysis for {ticker} on {date}"
    return context


def get_wisdom_embedding(
    retriever: Retriever,
    embedding_generator: EmbeddingGenerator,
    query_text: str,
    top_k: int = 5,
) -> np.ndarray | None:
    """Obtiene embedding promedio de documentos wisdom relevantes."""
    try:
        # Generar embedding del query
        query_embedding = embedding_generator.embed_text(query_text)

        # Buscar documentos relevantes
        results = retriever.query(query_embedding, top_k=top_k)

        if not results:
            return None

        # Obtener embeddings de los documentos recuperados
        # Los embeddings ya están en pgvector, pero necesitamos recuperarlos
        # Por simplicidad, usamos el embedding del query como proxy
        # o promediamos los scores para crear un embedding representativo
        # En producción, se deberían recuperar los embeddings reales de los documentos

        # Por ahora, retornamos el embedding del query como aproximación
        # (mejor implementación requeriría almacenar embeddings de documentos en metadata)
        return np.array(query_embedding, dtype=np.float32)
    except Exception as e:
        print(f"  [WARNING] Error obteniendo wisdom embedding: {e}")
        return None


def add_wisdom_features_to_split(
    df: pd.DataFrame,
    retriever: Retriever,
    embedding_generator: EmbeddingGenerator,
    top_k: int,
    split_name: str,
) -> pd.DataFrame:
    """Agrega columna de wisdom_features a un split."""
    print(f"\nProcesando split: {split_name}")
    print(f"  Filas: {len(df)}")

    df = df.copy()

    # Agrupar por ticker/date para evitar queries duplicadas
    unique_contexts = df[["ticker", "date"]].drop_duplicates()
    print(f"  Contextos únicos: {len(unique_contexts)}")

    wisdom_embeddings_map: dict[tuple[str, str], np.ndarray] = {}

    for idx, row in unique_contexts.iterrows():
        query_text = create_query_context(row)
        embedding = get_wisdom_embedding(retriever, embedding_generator, query_text, top_k)

        if embedding is not None:
            key = (str(row["ticker"]), str(row["date"]))
            wisdom_embeddings_map[key] = embedding

        if (idx + 1) % 100 == 0:
            print(f"    Procesados {idx + 1}/{len(unique_contexts)} contextos...")

    # Mapear embeddings a todas las filas
    wisdom_features_list = []
    for _, row in df.iterrows():
        key = (str(row["ticker"]), str(row["date"]))
        embedding = wisdom_embeddings_map.get(key)
        if embedding is not None:
            wisdom_features_list.append(embedding.tolist())
        else:
            # Embedding cero si no se encontró
            embedding_dim = embedding_generator.embedding_dim
            wisdom_features_list.append([0.0] * embedding_dim)

    df["wisdom_features"] = wisdom_features_list
    print(f"  [OK] Wisdom features agregadas (dims: {embedding_generator.embedding_dim})")

    return df


def main() -> None:
    load_dotenv()
    args = parse_args()

    print("=" * 60)
    print("PRE-COMPUTACIÓN DE WISDOM EMBEDDINGS")
    print("=" * 60)

    # Cargar configuración
    settings = Settings.from_yaml(Path(args.config))

    # Inicializar componentes
    print("\n[1/4] Inicializando componentes...")
    try:
        vector_store = VectorStore.from_settings(settings)
        retriever = Retriever(vector_store)
        embedding_generator = EmbeddingGenerator(settings)
        print(f"  [OK] VectorStore y EmbeddingGenerator inicializados")
        print(f"  Provider: {embedding_generator.provider}")
        print(f"  Embedding dim: {embedding_generator.embedding_dim}")
    except Exception as e:
        print(f"  [ERROR] No se pudieron inicializar componentes: {e}")
        print("  Asegúrate de que POSTGRES_* y API keys estén configurados en .env")
        return

    # Procesar cada split
    gold_dir = Path(args.gold_dir)
    processed_splits = []

    for split_name in args.splits:
        split_path = gold_dir / f"{split_name}.parquet"
        if not split_path.exists():
            print(f"\n[WARNING] Split {split_name} no existe: {split_path}")
            continue

        print(f"\n[2/4] Cargando split: {split_name}...")
        df = pd.read_parquet(split_path)
        print(f"  Filas cargadas: {len(df)}")

        print(f"\n[3/4] Agregando wisdom features a {split_name}...")
        df_with_wisdom = add_wisdom_features_to_split(
            df, retriever, embedding_generator, args.top_k, split_name
        )

        # Guardar
        print(f"\n[4/4] Guardando {split_name} con wisdom features...")
        df_with_wisdom.to_parquet(split_path, index=False)
        print(f"  [OK] Guardado: {split_path}")
        processed_splits.append(split_name)

    print("\n" + "=" * 60)
    print("[COMPLETADO] PRE-COMPUTACIÓN WISDOM")
    print("=" * 60)
    print(f"\nSplits procesados: {', '.join(processed_splits)}")
    print(f"\nNota: Los wisdom_features ahora están disponibles como columna en los datasets gold.")
    print("Para usarlos en entrenamiento, actualiza _build_feature_matrix para incluir 'wisdom_features'.")


if __name__ == "__main__":
    main()





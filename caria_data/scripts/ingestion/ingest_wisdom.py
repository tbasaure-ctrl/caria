"""
Script para ingestar wisdom chunks en PostgreSQL con embeddings vectoriales.

Uso:
    poetry run python scripts/ingestion/ingest_wisdom.py
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg2
from psycopg2.extras import execute_batch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)

# Configuración
# Path: notebooks/caria_data/scripts/ingestion/ingest_wisdom.py -> notebooks/data/raw/wisdom/
WISDOM_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "wisdom" / "2025-11-08"
BATCH_SIZE = 100  # Insertar en batches para mejor performance

# Modelo de embeddings (768 dimensiones)
# Opciones:
# - "all-mpnet-base-v2" (768 dims, mejor calidad)
# - "all-MiniLM-L6-v2" (384 dims, más rápido)
EMBEDDING_MODEL = "all-mpnet-base-v2"


def get_db_connection():
    """Conectar a PostgreSQL."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "caria_user"),
        password=os.getenv("POSTGRES_PASSWORD", "Theolucas7"),
        database=os.getenv("POSTGRES_DB", "caria"),
    )


def update_embedding_dimension(conn, dimension: int):
    """Actualizar dimensión de embeddings en la tabla si es necesario."""
    with conn.cursor() as cursor:
        # Verificar dimensión actual
        cursor.execute("""
            SELECT column_name, udt_name::regtype
            FROM information_schema.columns
            WHERE table_name = 'document_chunks' AND column_name = 'embedding'
        """)
        result = cursor.fetchone()

        if result:
            current_type = result[1]
            LOGGER.info(f"Tipo actual de embedding: {current_type}")

            # Si la dimensión es diferente, recrear la columna
            if f"vector({dimension})" not in current_type:
                LOGGER.warning(f"Cambiando dimensión de embedding a {dimension}")
                cursor.execute("ALTER TABLE document_chunks DROP COLUMN embedding")
                cursor.execute(f"ALTER TABLE document_chunks ADD COLUMN embedding VECTOR({dimension})")
                conn.commit()
                LOGGER.info("Dimensión actualizada")


def load_jsonl_file(filepath: Path) -> list[dict[str, Any]]:
    """Cargar archivo JSONL."""
    chunks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError as e:
                    LOGGER.warning(f"Error parseando línea en {filepath.name}: {e}")
    return chunks


def parse_json_field(value: str | list) -> list:
    """Parsear campo que puede ser string JSON o lista."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def ingest_chunks(
    chunks: list[dict[str, Any]],
    model: SentenceTransformer,
    conn: psycopg2.extensions.connection,
    user_id: str | None = None,
) -> int:
    """
    Ingestar chunks con embeddings en PostgreSQL.

    Args:
        chunks: Lista de chunks con estructura {text, source, book, author, ...}
        model: Modelo de sentence-transformers
        conn: Conexión a PostgreSQL
        user_id: UUID del usuario (None = público)

    Returns:
        Número de chunks insertados
    """
    if not chunks:
        return 0

    LOGGER.info(f"Generando embeddings para {len(chunks)} chunks...")

    # Generar embeddings en batch
    texts = [chunk.get('text', '') for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    LOGGER.info("Preparando datos para inserción...")

    # Preparar datos para inserción
    insert_data = []
    for chunk, embedding in zip(chunks, embeddings):
        # Preparar metadata
        metadata = {
            "source": chunk.get("source"),
            "book": chunk.get("book"),
            "author": chunk.get("author"),
            "context": chunk.get("context"),
            "themes": parse_json_field(chunk.get("themes", [])),
            "regimes": parse_json_field(chunk.get("regimes", [])),
            "biases": parse_json_field(chunk.get("biases", [])),
        }

        # Filtrar valores None
        metadata = {k: v for k, v in metadata.items() if v is not None}

        insert_data.append((
            str(uuid4()),  # id
            user_id,  # user_id (None = público)
            chunk.get('text', ''),  # content
            embedding.tolist(),  # embedding
            json.dumps(metadata),  # metadata
            chunk.get('source'),  # source
            chunk.get('book'),  # title
            None,  # chunk_index (no lo tenemos)
        ))

    LOGGER.info(f"Insertando {len(insert_data)} chunks en PostgreSQL...")

    # Insertar en batches
    with conn.cursor() as cursor:
        execute_batch(
            cursor,
            """
            INSERT INTO document_chunks (
                id, user_id, content, embedding, metadata, source, title, chunk_index
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            insert_data,
            page_size=BATCH_SIZE
        )
        conn.commit()

    LOGGER.info(f"✅ {len(insert_data)} chunks insertados")
    return len(insert_data)


def main():
    """Script principal."""
    LOGGER.info("=" * 60)
    LOGGER.info("INGESTION DE WISDOM CHUNKS")
    LOGGER.info("=" * 60)

    # Verificar directorio de wisdom
    if not WISDOM_DIR.exists():
        LOGGER.error(f"Directorio no encontrado: {WISDOM_DIR}")
        sys.exit(1)

    LOGGER.info(f"Directorio de wisdom: {WISDOM_DIR}")

    # Listar archivos JSONL
    jsonl_files = list(WISDOM_DIR.glob("*.jsonl"))
    LOGGER.info(f"Encontrados {len(jsonl_files)} archivos JSONL")

    # Filtrar archivos principales (evitar duplicados)
    # Prioridad: wisdom_corpus_unified_final.jsonl
    priority_files = [
        "wisdom_corpus_unified_final.jsonl",
        "wisdom_corpus_unified_EXPANDED.jsonl",
        "all_books_wisdom.jsonl",
    ]

    selected_file = None
    for filename in priority_files:
        filepath = WISDOM_DIR / filename
        if filepath.exists():
            selected_file = filepath
            LOGGER.info(f"✅ Usando archivo: {filename}")
            break

    if not selected_file:
        LOGGER.warning("No se encontró archivo prioritario, usando todos los chunks_*.jsonl")
        jsonl_files = [f for f in jsonl_files if f.name.startswith("chunks_")]
    else:
        jsonl_files = [selected_file]

    LOGGER.info(f"Archivos a procesar: {[f.name for f in jsonl_files]}")

    # Cargar modelo de embeddings
    LOGGER.info(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dim = model.get_sentence_embedding_dimension()
    LOGGER.info(f"Dimensión de embeddings: {embedding_dim}")

    # Conectar a base de datos
    LOGGER.info("Conectando a PostgreSQL...")
    conn = get_db_connection()
    LOGGER.info("✅ Conectado a PostgreSQL")

    # Actualizar dimensión de embeddings si es necesario
    update_embedding_dimension(conn, embedding_dim)

    # Procesar archivos
    total_chunks = 0

    for jsonl_file in jsonl_files:
        LOGGER.info(f"\n📖 Procesando: {jsonl_file.name}")

        # Cargar chunks
        chunks = load_jsonl_file(jsonl_file)
        LOGGER.info(f"   Cargados {len(chunks)} chunks")

        if not chunks:
            continue

        # Ingestar
        inserted = ingest_chunks(chunks, model, conn, user_id=None)
        total_chunks += inserted

    # Cerrar conexión
    conn.close()

    # Resumen final
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("INGESTION COMPLETADA")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Total chunks insertados: {total_chunks}")
    LOGGER.info(f"Modelo de embeddings: {EMBEDDING_MODEL} ({embedding_dim} dims)")
    LOGGER.info("\n✅ Wisdom chunks listos para RAG!")


if __name__ == "__main__":
    main()

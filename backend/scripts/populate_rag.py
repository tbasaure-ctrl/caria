import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path to import caria-lib
sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[2] / "caria-lib"))

from caria.config.settings import Settings
from caria.embeddings.generator import EmbeddingGenerator
from caria.retrieval.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("populate_rag")

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                LOGGER.warning(f"Skipping invalid JSON line in {file_path}")
    return data

def main():
    # Initialize settings and components
    settings = Settings()
    
    # Instantiate directly to be safe
    db_url = os.getenv("DATABASE_URL") or settings.get("vector_store", "connection")
    if not db_url:
        # Construct from env vars
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "postgres")
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.getenv("POSTGRES_DB", "caria")
        db_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

    LOGGER.info(f"Connecting to DB: {db_url.split('@')[-1]}") # Log only host/db

    embedding_dim = int(settings.get("retrieval", "embedding_dim", default=1024))
    table_name = settings.get("vector_store", "embedding_table", default="embeddings")

    try:
        vector_store = VectorStore(connection_uri=db_url, table_name=table_name, embedding_dim=embedding_dim)
        embedding_generator = EmbeddingGenerator(settings)
    except Exception as e:
        LOGGER.error(f"Failed to initialize components: {e}")
        return

    # Path to data
    # Resolve relative to project root
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / "wisdom" / "wisdom_corpus_unified_final.jsonl"
    
    if not data_path.exists():
        LOGGER.warning(f"Data file not found at {data_path}. Skipping population.")
        # Try finding any jsonl in wisdom folder?
        wisdom_dir = project_root / "data" / "raw" / "wisdom"
        if wisdom_dir.exists():
             for f in wisdom_dir.glob("*.jsonl"):
                 LOGGER.info(f"Found alternative file: {f}")
                 data_path = f
                 break
        
    if not data_path.exists():
        LOGGER.error(f"No JSONL file found for ingestion.")
        return

    LOGGER.info(f"Loading data from {data_path}")
    chunks = load_jsonl(data_path)
    LOGGER.info(f"Loaded {len(chunks)} chunks")

    # Process in batches
    batch_size = 50
    total_processed = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        records_to_upsert = []
        
        for item in batch:
            content = item.get("content") or item.get("text")
            if not content:
                continue
            
            # Generate embedding
            try:
                embedding = embedding_generator.embed_text(content)
            except Exception as e:
                LOGGER.error(f"Error embedding chunk {item.get('id')}: {e}")
                continue

            # Prepare record
            # Ensure metadata is JSON serializable
            metadata = item.copy()
            if "embedding" in metadata:
                del metadata["embedding"] # Don't store embedding in metadata
            
            records_to_upsert.append({
                "id": str(item.get("id") or item.get("chunk_id")),
                "embedding": embedding,
                "metadata": metadata
            })

        if records_to_upsert:
            try:
                vector_store.upsert(records_to_upsert)
                total_processed += len(records_to_upsert)
                LOGGER.info(f"Processed {total_processed}/{len(chunks)}")
            except Exception as e:
                LOGGER.error(f"Error upserting batch: {e}")

    LOGGER.info("Population complete")

if __name__ == "__main__":
    main()

# RAG Population Guide

This guide explains how to populate the NEON PostgreSQL database with investor wisdom embeddings for RAG (Retrieval-Augmented Generation).

## Prerequisites

1. **NEON Database Setup**
   - Active NEON PostgreSQL database
   - Connection string (from NEON dashboard)
   - pgvector extension enabled

2. **Python Environment**
   - Python 3.8+
   - Virtual environment (recommended)

3. **Data Files**
   - Wisdom corpus file: `c:\key\wise_adviser_cursor_context\notebooks\data\raw\wisdom\2025-11-08\wisdom_corpus_unified_final.jsonl`

## Step-by-Step Instructions

### 1. Set Up Environment Variables

Create a `.env` file in the `backend` directory with your NEON credentials:

```bash
# NEON PostgreSQL Connection
DATABASE_URL=postgresql+psycopg2://username:password@host/database

# Or individual components:
POSTGRES_USER=your_neon_username
POSTGRES_PASSWORD=your_neon_password
POSTGRES_HOST=your-project.neon.tech
POSTGRES_PORT=5432
POSTGRES_DB=caria

# Embedding Model Settings (optional, uses defaults if not set)
# EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# EMBEDDING_DIM=1024
```

**To get your NEON connection string:**
1. Go to https://console.neon.tech/
2. Select your project
3. Click "Connection Details"
4. Copy the connection string
5. Replace `postgresql://` with `postgresql+psycopg2://`

### 2. Install Dependencies

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Ensure these are installed:
pip install sentence-transformers psycopg2-binary sqlalchemy pgvector
```

### 3. Verify Database Tables

Before running the script, ensure the `embeddings` table exists in NEON:

```sql
-- Run this in NEON SQL Editor
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    embedding VECTOR(1024),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create index for faster similarity search
CREATE INDEX IF NOT EXISTS embeddings_vector_idx 
ON embeddings USING ivfflat (embedding vector_cosine_ops);
```

### 4. Run the Population Script

```bash
cd backend

# Make sure you're in the backend directory
python scripts/populate_rag.py
```

### 5. Monitor Progress

The script will output progress logs:

```
INFO:populate_rag:Connecting to DB: your-project.neon.tech/caria
INFO:populate_rag:Loading data from c:\key\wise_adviser_cursor_context\notebooks\data\raw\wisdom\2025-11-08\wisdom_corpus_unified_final.jsonl
INFO:populate_rag:Loaded 1234 chunks
INFO:populate_rag:Processed 50/1234
INFO:populate_rag:Processed 100/1234
...
INFO:populate_rag:Population complete
```

**Expected Time:** ~2-5 minutes for 1000-2000 chunks (depending on embedding model and CPU)

### 6. Verify Population

Check the database to confirm embeddings were inserted:

```sql
-- Run in NEON SQL Editor
SELECT COUNT(*) FROM embeddings;

-- View sample records
SELECT id, metadata->>'source', metadata->>'chunk_text' 
FROM embeddings 
LIMIT 5;

-- Test similarity search
SELECT id, metadata->>'chunk_text', 
       embedding <=> '[0.1, 0.2, ...]'::vector as distance
FROM embeddings
ORDER BY distance
LIMIT 3;
```

## Troubleshooting

### Error: "Data file not found"
**Solution:** Verify the path in `populate_rag.py` matches your data file location:
```python
data_path = Path(r"c:\key\wise_adviser_cursor_context\notebooks\data\raw\wisdom\2025-11-08\wisdom_corpus_unified_final.jsonl")
```

### Error: "Failed to initialize components"
**Solution:** 
1. Check DATABASE_URL is correct
2. Verify NEON database is accessible
3. Ensure pgvector extension is installed

### Error: "No module named 'caria'"
**Solution:**
```bash
# Add the caria-lib to Python path
cd backend
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/caria-lib"

# Or on Windows:
set PYTHONPATH=%PYTHONPATH%;%CD%;%CD%\caria-lib
```

### Error: "Connection refused"
**Solution:**
1. Check NEON database is running
2. Verify firewall/network access
3. Confirm connection string format: `postgresql+psycopg2://...`

### Slow Performance
**Solution:**
- Reduce batch size in script (set `batch_size = 25`)
- Use a GPU-enabled machine for faster embeddings
- Consider using a lighter embedding model

## Configuration Options

Edit `backend/scripts/populate_rag.py` to customize:

```python
# Line 93: Batch size (lower = slower but safer)
batch_size = 50  # Try 25 for slower connections

# Embedding dimension (must match your model)
embedding_dim = 1024  # or 384 for smaller models
```

## Next Steps

After populating embeddings:

1. **Test RAG Search:**
   ```python
   from caria.retrieval.vector_store import VectorStore
   
   vs = VectorStore(connection_uri=db_url, table_name="embeddings")
   results = vs.search("value investing principles", k=5)
   ```

2. **Integrate with Chat:**
   - Wire `RagService.search()` into chat prompt
   - Add context from top-k relevant chunks
   - Test with investment queries

3. **Monitor Usage:**
   - Track embedding table size
   - Monitor query performance
   - Optimize indexes as needed

## Database Maintenance

```sql
-- Check table size
SELECT pg_size_pretty(pg_total_relation_size('embeddings'));

-- Rebuild index if needed
REINDEX INDEX embeddings_vector_idx;

-- Clear and repopulate (if needed)
TRUNCATE TABLE embeddings;
-- Then run populate_rag.py again
```

## Support

If you encounter issues:
1. Check logs: `populate_rag.py` outputs detailed error messages
2. Verify environment variables are set correctly
3. Test database connection separately
4. Check NEON dashboard for connection limits/quotas

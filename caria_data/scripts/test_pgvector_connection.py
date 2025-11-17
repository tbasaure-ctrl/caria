"""Script para probar la conexión a PostgreSQL con pgvector."""

import os
from pathlib import Path
import sys

# Add src to path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

def test_connection():
    """Test connection to PostgreSQL with pgvector."""

    # Build connection string
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "caria")
    user = os.getenv("POSTGRES_USER", "caria_user")
    password = os.getenv("POSTGRES_PASSWORD", "Theolucas7")

    conn_str = f"postgresql://{user}:{password}@{host}:{port}/{db}"

    print("=" * 60)
    print("Testing PostgreSQL Connection")
    print("=" * 60)
    print(f"Host: {host}:{port}")
    print(f"Database: {db}")
    print(f"User: {user}")
    print()

    try:
        engine = create_engine(conn_str)

        with engine.connect() as conn:
            # Test basic connection
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print("✅ Connection successful!")
            print(f"PostgreSQL version: {version[:50]}...")
            print()

            # Test pgvector extension
            result = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector';"))
            row = result.fetchone()
            if row:
                print(f"✅ pgvector extension installed (version: {row[0]})")
            else:
                print("❌ pgvector extension NOT installed")
                print("   Run: CREATE EXTENSION vector;")
            print()

            # Test rag schema
            result = conn.execute(text("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'rag';"))
            row = result.fetchone()
            if row:
                print("✅ 'rag' schema exists")
            else:
                print("❌ 'rag' schema NOT found")
                print("   Run: CREATE SCHEMA rag;")
            print()

            # Test table count
            result = conn.execute(text("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = 'rag';
            """))
            count = result.fetchone()[0]
            print(f"📊 Tables in 'rag' schema: {count}")

            if count > 0:
                result = conn.execute(text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'rag';
                """))
                tables = [row[0] for row in result.fetchall()]
                print(f"   Tables: {', '.join(tables)}")

            print()
            print("=" * 60)
            print("✅ ALL CHECKS PASSED - Ready for wisdom ingestion!")
            print("=" * 60)
            return True

    except Exception as e:
        print()
        print("=" * 60)
        print("❌ CONNECTION FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check if Docker container is running:")
        print("   docker ps | grep caria-postgres")
        print()
        print("2. Start container if stopped:")
        print("   docker start caria-postgres")
        print()
        print("3. Check logs:")
        print("   docker logs caria-postgres")
        print()
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)

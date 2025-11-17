#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script para instalar la extensión pgvector en PostgreSQL."""

import os
import sys
from pathlib import Path

# Cargar .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    print("[ERROR] psycopg2 no instalado. Ejecuta: pip install psycopg2-binary")
    sys.exit(1)

def install_pgvector():
    """Instala la extensión pgvector en PostgreSQL."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    user = os.getenv("POSTGRES_USER", "caria_user")
    password = os.getenv("POSTGRES_PASSWORD")
    database = os.getenv("POSTGRES_DB", "caria")

    if not password:
        print("[ERROR] POSTGRES_PASSWORD no configurado")
        return False

    print(f"[INFO] Conectando a PostgreSQL en {host}:{port}...")

    try:
        # Conectar a PostgreSQL
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        with conn.cursor() as cur:
            # Verificar si pgvector ya está instalado
            cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
            if cur.fetchone():
                print("[INFO] Extension 'vector' ya esta instalada")

                # Verificar si la tabla embeddings existe
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = 'embeddings'
                    );
                """)
                if cur.fetchone()[0]:
                    print("[OK] Tabla 'embeddings' ya existe")
                else:
                    print("[INFO] Creando tabla 'embeddings'...")
                    cur.execute("""
                        CREATE TABLE embeddings (
                            id VARCHAR(64) PRIMARY KEY,
                            embedding VECTOR(1024),
                            metadata JSON NOT NULL
                        );
                    """)
                    print("[OK] Tabla 'embeddings' creada")

                return True

            # Intentar instalar pgvector
            print("[INFO] Intentando instalar extension 'vector'...")
            try:
                cur.execute("CREATE EXTENSION vector;")
                print("[OK] Extension 'vector' instalada correctamente")

                # Crear tabla embeddings
                print("[INFO] Creando tabla 'embeddings'...")
                cur.execute("""
                    CREATE TABLE embeddings (
                        id VARCHAR(64) PRIMARY KEY,
                        embedding VECTOR(1024),
                        metadata JSON NOT NULL
                    );
                """)
                print("[OK] Tabla 'embeddings' creada")

                return True

            except psycopg2.errors.UndefinedFile as e:
                print(f"\n[ERROR] Extension pgvector no encontrada: {e}")
                print("\n[INFO] SOLUCION:")
                print("   1. Descarga pgvector desde: https://github.com/pgvector/pgvector/releases")
                print("   2. O instala con:")
                print("      - Windows: Usa Stack Builder en PostgreSQL installer")
                print("      - Linux: apt install postgresql-17-pgvector")
                print("      - macOS: brew install pgvector")
                print("\n[INFO] ALTERNATIVA: Desactiva RAG/chat en la API (opcional)")
                return False

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    success = install_pgvector()
    sys.exit(0 if success else 1)

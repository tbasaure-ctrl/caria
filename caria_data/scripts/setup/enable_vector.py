"""Script para habilitar la extensión vector (pgvector)."""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def enable_vector():
    """Habilitar extensión vector."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "caria_user"),
            password=os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("POSTGRES_DB", "caria"),
        )
        
        # CREATE EXTENSION debe ejecutarse fuera de transacción
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        try:
            cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
            print("✓ Extensión vector habilitada exitosamente")
        except Exception as exc:
            error_msg = str(exc)
            if 'already exists' in error_msg.lower() or 'ya existe' in error_msg.lower():
                print("✓ Extensión vector ya está habilitada")
            else:
                print(f"⚠ No se pudo habilitar vector: {exc}")
                print("  Esto puede ser normal si pgvector no está instalado en el contenedor")
                return False
        
        # Verificar que esté habilitada
        cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
        result = cur.fetchone()
        if result:
            print(f"✓ Versión de vector: {result[1]}")
        else:
            print("⚠ Extensión vector no encontrada después de habilitarla")
            return False
        
        conn.close()
        return True
        
    except Exception as exc:
        print(f"❌ Error: {exc}")
        return False

if __name__ == "__main__":
    success = enable_vector()
    sys.exit(0 if success else 1)


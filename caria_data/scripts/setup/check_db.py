"""Script para verificar el estado de la base de datos."""

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

def check_database():
    """Verificar estado de la base de datos."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "caria_user"),
            password=os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("POSTGRES_DB", "caria"),
        )
        
        cur = conn.cursor()
        
        # Verificar versión de PostgreSQL
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        print(f"✓ PostgreSQL conectado: {version[:60]}...")
        
        # Verificar extensiones
        cur.execute("""
            SELECT extname, extversion 
            FROM pg_extension 
            WHERE extname IN ('vector', 'uuid-ossp')
            ORDER BY extname
        """)
        exts = cur.fetchall()
        print(f"\n✓ Extensiones instaladas ({len(exts)}):")
        for ext_name, ext_version in exts:
            print(f"  - {ext_name} {ext_version}")
        
        if len(exts) == 0:
            print("  ⚠ No se encontraron extensiones (vector, uuid-ossp)")
        
        # Verificar tablas
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """)
        tables = [row[0] for row in cur.fetchall()]
        print(f"\n✓ Tablas creadas ({len(tables)}):")
        for table in tables[:15]:
            print(f"  - {table}")
        if len(tables) > 15:
            print(f"  ... y {len(tables) - 15} más")
        
        # Verificar migraciones aplicadas
        cur.execute("SELECT migration_name FROM schema_migrations ORDER BY applied_at")
        migrations = [row[0] for row in cur.fetchall()]
        print(f"\n✓ Migraciones aplicadas ({len(migrations)}):")
        for migration in migrations:
            print(f"  - {migration}")
        
        conn.close()
        print("\n✅ Base de datos configurada correctamente!")
        return True
        
    except Exception as exc:
        print(f"\n❌ Error conectando a la base de datos: {exc}")
        return False

if __name__ == "__main__":
    success = check_database()
    sys.exit(0 if success else 1)


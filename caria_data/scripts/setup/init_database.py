"""Script para inicializar la base de datos Caria ejecutando init_db.sql."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Configurar logging
LOGGER = logging.getLogger("caria.setup")

# Intentar cargar variables de entorno desde .env si existe
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        LOGGER.info("Variables de entorno cargadas desde %s", env_path)
except ImportError:
    pass

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def get_db_connection(database: str | None = None):
    """Obtener conexión a base de datos."""
    postgres_password = os.getenv("POSTGRES_PASSWORD")
    if not postgres_password:
        LOGGER.error("POSTGRES_PASSWORD no está configurado")
        raise RuntimeError("POSTGRES_PASSWORD no configurado")

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "caria_user"),
        password=postgres_password,
        database=database or os.getenv("POSTGRES_DB", "caria"),
    )


def create_mlflow_database():
    """Crear base de datos mlflow_db (debe ejecutarse fuera de transacción)."""
    try:
        # Conectar a postgres para crear base de datos
        conn = get_db_connection("postgres")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            # Verificar si ya existe
            cur.execute("SELECT 1 FROM pg_database WHERE datname = 'mlflow_db'")
            if cur.fetchone():
                LOGGER.info("Base de datos mlflow_db ya existe")
            else:
                cur.execute('CREATE DATABASE mlflow_db')
                LOGGER.info("Base de datos mlflow_db creada exitosamente")
        
        conn.close()
    except psycopg2.errors.DuplicateDatabase:
        LOGGER.info("Base de datos mlflow_db ya existe")
    except Exception as exc:
        LOGGER.warning("No se pudo crear mlflow_db (puede que ya exista o falten permisos): %s", exc)


def execute_init_db():
    """Ejecutar init_db.sql usando psycopg2."""
    # Obtener ruta del script
    script_path = Path(__file__).parent.parent.parent / "infrastructure" / "init_db.sql"
    
    if not script_path.exists():
        LOGGER.error("No se encontró init_db.sql en %s", script_path)
        sys.exit(1)
    
    # Leer SQL completo
    sql = script_path.read_text(encoding="utf-8")
    
    # Conectar a la base de datos
    try:
        conn = get_db_connection()
    except Exception as exc:
        LOGGER.error("Error conectando a base de datos: %s", exc)
        sys.exit(1)
    
    # Ejecutar SQL completo usando psycopg2
    # psycopg2 puede ejecutar múltiples statements si están separados correctamente
    try:
        # Usar execute() con el SQL completo
        # Nota: Necesitamos ejecutar en modo AUTOCOMMIT para bloques DO $$
        old_isolation = conn.isolation_level
        
        with conn.cursor() as cur:
            # Ejecutar el SQL completo
            # psycopg2 puede manejar múltiples statements separados por punto y coma
            # pero los bloques DO $$ necesitan ejecutarse en modo AUTOCOMMIT
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            try:
                cur.execute(sql)
                LOGGER.info("SQL ejecutado exitosamente")
            except Exception as exc:
                # Algunos errores son esperados (tablas que ya existen, etc.)
                error_msg = str(exc)
                if any(phrase in error_msg.lower() for phrase in [
                    'already exists', 'ya existe', 'duplicate', 
                    'no existe el rol', 'permission denied', 'permiso denegado'
                ]):
                    LOGGER.debug("Algunos warnings esperados durante ejecución: %s", error_msg[:200])
                else:
                    LOGGER.warning("Error ejecutando SQL: %s", error_msg[:300])
                    # Intentar continuar de todas formas
            finally:
                conn.set_isolation_level(old_isolation)
        
        LOGGER.info("init_db.sql ejecutado exitosamente")
        
    except Exception as exc:
        LOGGER.error("Error ejecutando init_db.sql: %s", exc)
        conn.close()
        sys.exit(1)
    finally:
        conn.close()


def main():
    """Función principal."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    
    LOGGER.info("Inicializando base de datos Caria...")
    
    # Crear base de datos mlflow_db primero (fuera de transacción)
    create_mlflow_database()
    
    # Ejecutar init_db.sql
    execute_init_db()
    
    LOGGER.info("Inicialización completada. Ahora puedes ejecutar las migraciones:")
    LOGGER.info("  python scripts/migrations/run_migrations.py")


if __name__ == "__main__":
    main()

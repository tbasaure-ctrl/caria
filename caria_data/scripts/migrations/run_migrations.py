"""Script para ejecutar migraciones de base de datos en orden."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Configurar logging primero
LOGGER = logging.getLogger("caria.migrations")

# Intentar cargar variables de entorno desde .env si existe
try:
    from dotenv import load_dotenv
    # Buscar .env en el directorio raíz del proyecto (caria_data/)
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        LOGGER.info("Variables de entorno cargadas desde %s", env_path)
except ImportError:
    # python-dotenv no está instalado, continuar sin él
    pass

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Configurar paths
CURRENT_FILE = Path(__file__).resolve()
MIGRATIONS_DIR = CURRENT_FILE.parent.parent.parent / "infrastructure" / "migrations"
if not MIGRATIONS_DIR.exists():
    MIGRATIONS_DIR = CURRENT_FILE.parent.parent / "infrastructure" / "migrations"


def get_db_connection():
    """Obtener conexión a base de datos desde variables de entorno."""
    import os

    postgres_password = os.getenv("POSTGRES_PASSWORD")
    if not postgres_password:
        LOGGER.error(
            "POSTGRES_PASSWORD no está configurado. "
            "Por favor, configura las variables de entorno:\n"
            "  export POSTGRES_HOST=localhost\n"
            "  export POSTGRES_PORT=5432\n"
            "  export POSTGRES_USER=caria_user\n"
            "  export POSTGRES_PASSWORD=tu_contraseña\n"
            "  export POSTGRES_DB=caria\n\n"
            "O crea un archivo .env y usa python-dotenv para cargarlo."
        )
        raise RuntimeError("POSTGRES_PASSWORD no configurado")

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "caria_user"),
        password=postgres_password,
        database=os.getenv("POSTGRES_DB", "caria"),
    )


def get_applied_migrations(conn) -> set[str]:
    """Obtener lista de migraciones ya aplicadas."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT migration_name FROM schema_migrations ORDER BY id")
            return {row[0] for row in cur.fetchall()}
    except (psycopg2.errors.UndefinedTable, psycopg2.errors.InsufficientPrivilege) as exc:
        # Tabla schema_migrations no existe o no hay permisos
        LOGGER.warning(
            "No se pudo acceder a schema_migrations: %s\n"
            "Esto puede deberse a:\n"
            "  1. La tabla no existe (ejecuta init_db.sql primero)\n"
            "  2. El usuario no tiene permisos (usa un usuario con permisos o ejecuta como superusuario)\n"
            "Intentando continuar asumiendo que no hay migraciones aplicadas...",
            exc
        )
        # Intentar crear la tabla si no existe (puede fallar por permisos)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        id SERIAL PRIMARY KEY,
                        migration_name VARCHAR(255) UNIQUE NOT NULL,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                LOGGER.info("Tabla schema_migrations creada exitosamente")
        except psycopg2.errors.InsufficientPrivilege:
            LOGGER.error(
                "No se tienen permisos para crear la tabla schema_migrations.\n"
                "Opciones:\n"
                "  1. Ejecuta init_db.sql como superusuario primero\n"
                "  2. Concede permisos al usuario: GRANT ALL ON SCHEMA public TO caria_user;\n"
                "  3. Ejecuta este script como superusuario de PostgreSQL"
            )
            raise
        return set()


def split_sql_statements(sql: str) -> list[str]:
    """Dividir SQL en statements individuales, manejando comentarios y líneas múltiples."""
    statements = []
    current_statement = []
    in_comment = False
    
    for line in sql.split('\n'):
        stripped = line.strip()
        
        # Manejar comentarios de bloque /* */
        if '/*' in stripped:
            in_comment = True
            # Continuar hasta encontrar */
            if '*/' in stripped:
                in_comment = False
            continue
        
        if '*/' in stripped:
            in_comment = False
            continue
        
        if in_comment:
            continue
        
        # Saltar comentarios de línea y líneas vacías
        if not stripped or stripped.startswith('--'):
            continue
        
        current_statement.append(line)
        
        # Si la línea termina con punto y coma, es el final de un statement
        if stripped.endswith(';'):
            statement = '\n'.join(current_statement).strip()
            if statement:
                statements.append(statement)
            current_statement = []
    
    # Si queda algo sin punto y coma, agregarlo
    if current_statement:
        statement = '\n'.join(current_statement).strip()
        if statement:
            statements.append(statement)
    
    return statements


def apply_migration(conn, migration_file: Path) -> None:
    """Aplicar una migración SQL."""
    migration_name = migration_file.name
    LOGGER.info("Aplicando migración: %s", migration_name)

    # Leer SQL completo
    sql = migration_file.read_text(encoding="utf-8")
    
    # Ejecutar SQL completo directamente
    # Los bloques DO $$ necesitan ejecutarse en modo AUTOCOMMIT
    old_isolation = conn.isolation_level
    
    try:
        # Ejecutar en modo AUTOCOMMIT para manejar CREATE EXTENSION y bloques DO $$
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cur:
            cur.execute(sql)
        
        # Registrar migración
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO schema_migrations (migration_name) VALUES (%s) ON CONFLICT DO NOTHING",
                    (migration_name,),
                )
        except Exception as exc:
            LOGGER.warning("No se pudo registrar migración en schema_migrations: %s", exc)
        
        LOGGER.info("Migración %s aplicada exitosamente", migration_name)
        
    except Exception as exc:
        error_msg = str(exc)
        # Algunos errores son esperados y no críticos
        if any(phrase in error_msg.lower() for phrase in [
            'already exists', 'ya existe', 'duplicate', 
            'no existe el rol', 'permission denied', 'permiso denegado',
            'no se pudieron otorgar permisos', 'omitiendo grant'
        ]):
            LOGGER.warning("Migración %s completada con warnings (no críticos): %s", migration_name, error_msg[:200])
            # Registrar migración de todas formas
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO schema_migrations (migration_name) VALUES (%s) ON CONFLICT DO NOTHING",
                        (migration_name,),
                    )
            except Exception:
                pass
        else:
            LOGGER.error("Error ejecutando migración %s: %s", migration_name, error_msg)
            raise
    finally:
        conn.set_isolation_level(old_isolation)


def check_prerequisites(conn) -> None:
    """Verificar que las tablas base necesarias existan."""
    required_tables = ['users']
    missing_tables = []
    
    with conn.cursor() as cur:
        for table in required_tables:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                )
            """, (table,))
            exists = cur.fetchone()[0]
            if not exists:
                missing_tables.append(table)
    
    if missing_tables:
        LOGGER.error(
            "Las siguientes tablas base no existen: %s\n"
            "Por favor, ejecuta init_db.sql primero para crear las tablas base:\n"
            "  psql -U postgres -f infrastructure/init_db.sql\n\n"
            "O si usas pgAdmin, ejecuta el contenido de infrastructure/init_db.sql",
            ', '.join(missing_tables)
        )
        raise RuntimeError(f"Tablas base faltantes: {', '.join(missing_tables)}")


def run_migrations() -> None:
    """Ejecutar todas las migraciones pendientes."""
    # Obtener conexión
    try:
        conn = get_db_connection()
    except Exception as exc:
        LOGGER.error("Error conectando a base de datos: %s", exc)
        sys.exit(1)
    
    # Asegurar que las extensiones necesarias estén habilitadas
    # CREATE EXTENSION debe ejecutarse fuera de transacción
    try:
        old_isolation = conn.isolation_level
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
        conn.set_isolation_level(old_isolation)
        LOGGER.info("Extensión uuid-ossp verificada/habilitada")
    except Exception as exc:
        LOGGER.warning("No se pudo habilitar uuid-ossp (puede que ya esté habilitada o falten permisos): %s", exc)
        # Restaurar isolation level si cambió
        if 'old_isolation' in locals():
            conn.set_isolation_level(old_isolation)

    # Verificar que las tablas base existan
    try:
        check_prerequisites(conn)
    except RuntimeError:
        conn.close()
        sys.exit(1)

    # Obtener migraciones aplicadas
    applied = get_applied_migrations(conn)

    # Listar archivos de migración
    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not migration_files:
        LOGGER.warning("No se encontraron archivos de migración en %s", MIGRATIONS_DIR)
        conn.close()
        return

    # Aplicar migraciones pendientes
    applied_count = 0
    for migration_file in migration_files:
        if migration_file.name not in applied:
            try:
                apply_migration(conn, migration_file)
                applied_count += 1
            except Exception as exc:
                LOGGER.error("Error aplicando migración %s: %s", migration_file.name, exc)
                try:
                    conn.rollback()
                except Exception:
                    pass  # Puede que no haya transacción activa
                conn.close()
                sys.exit(1)
        else:
            LOGGER.info("Migración %s ya aplicada, omitiendo", migration_file.name)

    try:
        conn.close()
    except Exception:
        pass  # Ignorar errores al cerrar

    if applied_count > 0:
        LOGGER.info("Aplicadas %d migraciones nuevas", applied_count)
    else:
        LOGGER.info("Todas las migraciones ya están aplicadas")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_migrations()

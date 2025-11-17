"""Utility helpers to ensure database schema and seed data are present on startup."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

LOGGER = logging.getLogger("caria.api.db_bootstrap")

ROOT_DIR = Path(__file__).resolve().parents[2]  # points to notebooks/
MIGRATIONS_DIR = ROOT_DIR / "caria_data" / "infrastructure" / "migrations"
INIT_SQL = ROOT_DIR / "caria_data" / "migrations" / "init.sql"

DEFAULT_USER_USERNAME = "TBL"
DEFAULT_USER_EMAIL = "tbl@example.com"
DEFAULT_USER_PASSWORD = "Theolucas7"
DEFAULT_USER_FULL_NAME = "TBL Admin"


def _connection_kwargs() -> dict[str, object]:
    """Get connection parameters, supporting both DATABASE_URL and individual variables."""
    # Railway proporciona DATABASE_URL automáticamente cuando agregas PostgreSQL
    # Cloud SQL usa formato: postgresql://user:password@/dbname?host=/cloudsql/instance
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        try:
            from urllib.parse import parse_qs
            parsed = urlparse(database_url)

            # Extraer parámetros de query string
            query_params = parse_qs(parsed.query)

            # Verificar si hay un socket Unix de Cloud SQL en el query string
            unix_socket_host = None
            if 'host' in query_params:
                unix_socket_host = query_params['host'][0]

            # Si hay socket Unix (Cloud SQL), usarlo
            if unix_socket_host:
                return {
                    "host": unix_socket_host,
                    "user": parsed.username,
                    "password": parsed.password,
                    "database": parsed.path.lstrip('/'),
                }
            # Si no, usar conexión normal con hostname y port
            elif parsed.hostname:
                return {
                    "host": parsed.hostname,
                    "port": parsed.port or 5432,
                    "user": parsed.username,
                    "password": parsed.password,
                    "database": parsed.path.lstrip('/'),
                }
        except Exception as e:
            LOGGER.warning(f"Error parsing DATABASE_URL: {e}. Using individual variables...")

    # Fallback a variables individuales
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "user": os.getenv("POSTGRES_USER", "caria_user"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "database": os.getenv("POSTGRES_DB", "caria"),
    }


def _connect() -> Optional[psycopg2.extensions.connection]:
    params = _connection_kwargs()
    password = params.get("password")
    if not password:
        LOGGER.warning("POSTGRES_PASSWORD or DATABASE_URL not set – skipping bootstrap tasks.")
        return None

    try:
        conn = psycopg2.connect(**params)
        conn.autocommit = False
        return conn
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Could not connect to PostgreSQL for bootstrap tasks: %s", exc)
        return None


def _run_sql_file(conn: psycopg2.extensions.connection, file_path: Path) -> None:
    if not file_path.exists():
        LOGGER.error("Migration file not found: %s", file_path)
        return

    LOGGER.info("Applying SQL migration: %s", file_path.name)
    sql_text = file_path.read_text(encoding="utf-8")
    with conn.cursor() as cursor:
        cursor.execute(sql_text)
    conn.commit()


def _table_exists(conn: psycopg2.extensions.connection, table_name: str) -> bool:
    with conn.cursor() as cursor:
        cursor.execute(
            sql.SQL("SELECT to_regclass(%s)"),
            (f"public.{table_name}",),
        )
        return cursor.fetchone()[0] is not None


def ensure_auth_tables(conn: psycopg2.extensions.connection) -> None:
    """Make sure auxiliary auth tables exist by replaying migration 001 if needed."""
    if _table_exists(conn, "audit_logs") and _table_exists(conn, "refresh_tokens"):
        return

    migration_file = MIGRATIONS_DIR / "001_add_auth_tables.sql"
    LOGGER.warning("Auth tables missing – applying %s", migration_file.name)
    _run_sql_file(conn, migration_file)


def ensure_community_tables(conn: psycopg2.extensions.connection) -> None:
    """Make sure community module tables exist by replaying migration 002 if needed."""
    if _table_exists(conn, "community_posts") and _table_exists(conn, "community_votes"):
        return

    migration_file = MIGRATIONS_DIR / "002_add_community_tables.sql"
    LOGGER.warning("Community tables missing – applying %s", migration_file.name)
    _run_sql_file(conn, migration_file)


def ensure_holdings_table(conn: psycopg2.extensions.connection) -> None:
    """Make sure holdings table exists by replaying migration if needed."""
    if _table_exists(conn, "holdings"):
        return

    migration_file = MIGRATIONS_DIR / "add_holdings_table.sql"
    LOGGER.warning("Holdings table missing – applying %s", migration_file.name)
    _run_sql_file(conn, migration_file)


def ensure_default_user(conn: psycopg2.extensions.connection) -> None:
    """Create the default user if it does not exist."""
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(
            "SELECT id FROM users WHERE username = %s",
            (DEFAULT_USER_USERNAME,),
        )
        if cursor.fetchone():
            LOGGER.info("Default user %s already exists", DEFAULT_USER_USERNAME)
            return

    # Import lazily to avoid circular imports during module import time
    from caria.services.auth_service import AuthService

    hashed_password = AuthService.hash_password(DEFAULT_USER_PASSWORD)

    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO users (email, username, full_name, hashed_password, is_active, is_verified, is_superuser)
            VALUES (%s, %s, %s, %s, TRUE, TRUE, TRUE)
            """,
            (
                DEFAULT_USER_EMAIL,
                DEFAULT_USER_USERNAME,
                DEFAULT_USER_FULL_NAME,
                hashed_password,
            ),
        )
        conn.commit()
        LOGGER.info("Default user %s created successfully", DEFAULT_USER_USERNAME)


def ensure_vector_extension(conn: psycopg2.extensions.connection) -> None:
    """Ensure the pgvector extension is enabled in PostgreSQL."""
    with conn.cursor() as cursor:
        # Check if extension exists
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        if cursor.fetchone():
            LOGGER.info("Vector extension already exists")
            return
        
        # Try to create the extension
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            LOGGER.info("Vector extension created successfully")
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            LOGGER.warning("Could not create vector extension (may require superuser privileges): %s", exc)
            # Don't fail bootstrap if extension creation fails - VectorStore will handle it


def ensure_init_schema(conn: psycopg2.extensions.connection) -> None:
    """Ensure base schema exists by running init.sql if needed."""
    # Check if users table exists as a proxy for schema initialization
    if _table_exists(conn, "users"):
        LOGGER.info("Base schema already initialized")
        return

    # Run init.sql to create base schema
    if not INIT_SQL.exists():
        LOGGER.warning("init.sql not found at %s, skipping schema initialization", INIT_SQL)
        return

    LOGGER.info("Initializing database schema from init.sql")
    _run_sql_file(conn, INIT_SQL)
    LOGGER.info("Base schema initialized successfully")


def run_bootstrap_tasks() -> None:
    """Entry point executed on API startup."""
    conn = _connect()
    if conn is None:
        return

    try:
        ensure_vector_extension(conn)
        ensure_init_schema(conn)  # Run init.sql first to create base tables
        ensure_auth_tables(conn)
        ensure_community_tables(conn)
        ensure_holdings_table(conn)
        ensure_default_user(conn)
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        LOGGER.exception("Bootstrap tasks failed: %s", exc)
    finally:
        conn.close()



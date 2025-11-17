#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script para verificar conexión a PostgreSQL."""

import os
import sys
from pathlib import Path

# Cargar variables de entorno desde .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[INFO] Variables cargadas desde: {env_path}")
    else:
        print(f"[WARNING] No se encontro archivo .env en: {env_path}")
except ImportError:
    print("[WARNING] python-dotenv no instalado, usando variables del sistema")

try:
    import psycopg2
except ImportError:
    print("[ERROR] psycopg2 no esta instalado.")
    print("   Instalalo con: pip install psycopg2-binary")
    sys.exit(1)

def check_connection():
    """Verifica conexión a PostgreSQL."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    user = os.getenv("POSTGRES_USER", "caria_user")
    password = os.getenv("POSTGRES_PASSWORD")
    database = os.getenv("POSTGRES_DB", "caria")
    
    if not password:
        print("[WARNING] POSTGRES_PASSWORD no esta configurado.")
        print("\nConfiguralo con:")
        print("   PowerShell: $env:POSTGRES_PASSWORD='tu_password'")
        print("   CMD: set POSTGRES_PASSWORD=tu_password")
        return False

    print(f"[INFO] Intentando conectar a PostgreSQL...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   User: {user}")
    print(f"   Database: {database}")
    
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
        
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            print(f"\n[OK] Conexion exitosa!")
            print(f"   PostgreSQL version: {version.split(',')[0]}")

            # Verificar tablas existentes
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = [row[0] for row in cur.fetchall()]

            print(f"\n[INFO] Tablas existentes ({len(tables)}):")
            for table in tables[:10]:  # Mostrar primeras 10
                print(f"   - {table}")
            if len(tables) > 10:
                print(f"   ... y {len(tables) - 10} mas")

            # Verificar tablas criticas
            critical_tables = ["users", "holdings", "refresh_tokens", "audit_logs"]
            missing_tables = [t for t in critical_tables if t not in tables]

            if missing_tables:
                print(f"\n[WARNING] Tablas faltantes: {', '.join(missing_tables)}")
                print("   Necesitas ejecutar las migraciones SQL")
            else:
                print("\n[OK] Todas las tablas criticas existen")
        
        conn.close()
        return True
        
    except psycopg2.OperationalError as e:
        print(f"\n[ERROR] Error de conexion: {e}")
        print("\n[INFO] Verifica:")
        print("   - PostgreSQL esta corriendo")
        print("   - Las credenciales son correctas")
        print("   - La base de datos 'caria' existe")
        print("   - El usuario tiene permisos")
        return False
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        return False

if __name__ == "__main__":
    success = check_connection()
    sys.exit(0 if success else 1)


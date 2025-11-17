#!/usr/bin/env python3
"""Script para ejecutar la migración de holdings."""

import os
import sys
from pathlib import Path

try:
    import psycopg2
except ImportError:
    print("❌ Error: psycopg2 no está instalado.")
    print("   Instálalo con: pip install psycopg2-binary")
    sys.exit(1)

def get_db_config():
    """Obtiene configuración de base de datos desde variables de entorno."""
    config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "user": os.getenv("POSTGRES_USER", "caria_user"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "database": os.getenv("POSTGRES_DB", "caria"),
    }
    
    if not config["password"]:
        print("⚠️  POSTGRES_PASSWORD no está configurado.")
        print("\nOpciones:")
        print("1. Configurar variable de entorno:")
        print("   PowerShell: $env:POSTGRES_PASSWORD='tu_password'")
        print("   CMD: set POSTGRES_PASSWORD=tu_password")
        print("   Linux/Mac: export POSTGRES_PASSWORD=tu_password")
        print("\n2. O pasar la contraseña como argumento:")
        print("   python run_migration.py --password tu_password")
        print("\n3. O ingresarla interactivamente:")
        password = input("\nIngresa la contraseña de PostgreSQL: ")
        config["password"] = password
    
    return config

def run_migration(migration_file: Path, db_config: dict):
    """Ejecuta la migración SQL."""
    print(f"📝 Leyendo migración desde: {migration_file}")
    
    if not migration_file.exists():
        print(f"❌ Error: Archivo de migración no encontrado: {migration_file}")
        return False
    
    # Intentar leer con diferentes encodings
    encodings = ["utf-8", "latin-1", "cp1252"]
    sql_content = None
    for encoding in encodings:
        try:
            sql_content = migration_file.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    
    if sql_content is None:
        print(f"❌ Error: No se pudo leer el archivo con ningún encoding")
        return False
    
    print(f"🔌 Conectando a PostgreSQL...")
    print(f"   Host: {db_config['host']}")
    print(f"   Port: {db_config['port']}")
    print(f"   User: {db_config['user']}")
    print(f"   Database: {db_config['database']}")
    
    try:
        conn = psycopg2.connect(**db_config)
        print("✅ Conexión exitosa")
        
        with conn.cursor() as cur:
            print("🚀 Ejecutando migración...")
            cur.execute(sql_content)
            conn.commit()
            print("✅ Migración ejecutada exitosamente")
        
        # Verificar que la tabla se creó
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'holdings'
                );
            """)
            exists = cur.fetchone()[0]
            
            if exists:
                print("✅ Tabla 'holdings' verificada en la base de datos")
            else:
                print("⚠️  Advertencia: La tabla 'holdings' no se encontró después de la migración")
        
        conn.close()
        return True
        
    except psycopg2.OperationalError as e:
        print(f"❌ Error de conexión: {e}")
        print("\nVerifica:")
        print("  - PostgreSQL está corriendo")
        print("  - Las credenciales son correctas")
        print("  - La base de datos 'caria' existe")
        return False
    except psycopg2.Error as e:
        print(f"❌ Error ejecutando migración: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ejecuta migración de tabla holdings")
    parser.add_argument(
        "--migration-file",
        type=Path,
        default=Path(__file__).parent.parent.parent / "caria_data" / "infrastructure" / "migrations" / "add_holdings_table.sql",
        help="Ruta del archivo de migración SQL"
    )
    parser.add_argument(
        "--password",
        type=str,
        help="Contraseña de PostgreSQL (alternativa a variable de entorno)"
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host de PostgreSQL (default: localhost)"
    )
    parser.add_argument(
        "--user",
        type=str,
        help="Usuario de PostgreSQL (default: caria_user)"
    )
    parser.add_argument(
        "--database",
        type=str,
        help="Nombre de la base de datos (default: caria)"
    )
    
    args = parser.parse_args()
    
    # Obtener configuración
    db_config = get_db_config()
    
    # Overrides desde argumentos
    if args.password:
        db_config["password"] = args.password
    if args.host:
        db_config["host"] = args.host
    if args.user:
        db_config["user"] = args.user
    if args.database:
        db_config["database"] = args.database
    
    if not db_config["password"]:
        print("❌ Error: No se pudo obtener la contraseña de PostgreSQL")
        sys.exit(1)
    
    # Ejecutar migración
    success = run_migration(args.migration_file, db_config)
    
    if success:
        print("\n🎉 ¡Migración completada exitosamente!")
        print("   La tabla 'holdings' está lista para usar.")
    else:
        print("\n❌ La migración falló. Revisa los errores arriba.")
        sys.exit(1)

if __name__ == "__main__":
    main()


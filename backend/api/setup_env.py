#!/usr/bin/env python3
"""Script para configurar variables de entorno rápidamente."""

import os
import sys
from pathlib import Path

# Valores por defecto
DEFAULT_ENV = {
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USER": "caria_user",
    "POSTGRES_PASSWORD": "",  # Debe ser configurado por el usuario
    "POSTGRES_DB": "caria",
    "FMP_API_KEY": "your-fmp-api-key-here",  # API key específica para precios en tiempo real
    "LLAMA_API_KEY": "",  # Groq API key
    "LLAMA_API_URL": "https://api.groq.com/openai/v1/chat/completions",
    "LLAMA_MODEL": "llama-3.1-8b-instruct",
    "JWT_SECRET_KEY": "",  # Se generará si está vacío
    "CORS_ORIGINS": "http://localhost:3000,http://localhost:5173",
}

def generate_jwt_secret():
    """Genera un JWT secret seguro."""
    import secrets
    return secrets.token_urlsafe(32)

def create_env_file(env_path: Path, overwrite: bool = False):
    """Crea archivo .env con valores por defecto."""
    if env_path.exists() and not overwrite:
        print(f"⚠️  Archivo {env_path} ya existe. Usa --overwrite para sobrescribir.")
        return False
    
    lines = []
    lines.append("# Variables de entorno para CARIA API")
    lines.append("# Configurado automáticamente por setup_env.py\n")
    
    for key, default_value in DEFAULT_ENV.items():
        if key == "JWT_SECRET_KEY" and not default_value:
            default_value = generate_jwt_secret()
            lines.append(f"# JWT Secret generado automáticamente")
        elif key == "POSTGRES_PASSWORD" and not default_value:
            lines.append(f"# {key}={default_value}  # ⚠️ CONFIGURA ESTE VALOR")
            continue
        
        lines.append(f"{key}={default_value}")
    
    env_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Archivo {env_path} creado exitosamente")
    return True

def set_env_vars():
    """Configura variables de entorno en el proceso actual."""
    import secrets
    
    for key, default_value in DEFAULT_ENV.items():
        if key == "JWT_SECRET_KEY" and not default_value:
            default_value = generate_jwt_secret()
        elif key == "POSTGRES_PASSWORD" and not default_value:
            print(f"⚠️  {key} no configurado. Configúralo manualmente.")
            continue
        
        os.environ[key] = default_value
        print(f"✅ {key} = {default_value[:20]}..." if len(str(default_value)) > 20 else f"✅ {key} = {default_value}")

def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configura variables de entorno para CARIA API")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(__file__).parent / ".env",
        help="Ruta del archivo .env (default: .env en el mismo directorio)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribir archivo .env existente"
    )
    parser.add_argument(
        "--set-env",
        action="store_true",
        help="Configurar variables en el proceso actual (no crear archivo)"
    )
    
    args = parser.parse_args()
    
    if args.set_env:
        print("🔧 Configurando variables de entorno en el proceso actual...")
        set_env_vars()
        print("\n✅ Variables configuradas. Estas solo estarán disponibles en este proceso.")
        print("💡 Para hacerlas permanentes, crea un archivo .env o configúralas en tu sistema.")
    else:
        print(f"📝 Creando archivo {args.file}...")
        if create_env_file(args.file, args.overwrite):
            print("\n✅ Configuración completada!")
            print(f"📋 Revisa y edita {args.file} si necesitas cambiar valores.")
            print("⚠️  IMPORTANTE: Configura POSTGRES_PASSWORD antes de iniciar la API")

if __name__ == "__main__":
    main()


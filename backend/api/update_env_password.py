#!/usr/bin/env python3
"""Script para actualizar la contraseña en el archivo .env"""

import os
from pathlib import Path

API_DIR = Path(__file__).parent
ENV_FILE = API_DIR / ".env"

PASSWORD = "Theolucas7"

def update_env_file():
    """Actualiza o crea el archivo .env con la contraseña."""
    lines = []
    password_found = False
    
    if ENV_FILE.exists():
        # Leer archivo existente
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            # Si la línea contiene POSTGRES_PASSWORD (comentada o no)
            if "POSTGRES_PASSWORD" in line:
                # Reemplazar con línea sin comentar
                lines.append(f"POSTGRES_PASSWORD={PASSWORD}")
                password_found = True
            else:
                lines.append(line)
        
        # Si no existía, agregarlo después de POSTGRES_USER
        if not password_found:
            new_lines = []
            for line in lines:
                new_lines.append(line)
                if line.strip().startswith("POSTGRES_USER"):
                    new_lines.append(f"POSTGRES_PASSWORD={PASSWORD}")
            lines = new_lines
    else:
        # Crear archivo nuevo
        lines = [
            "# Variables de entorno para CARIA API",
            "",
            "# Base de datos PostgreSQL",
            "POSTGRES_HOST=localhost",
            "POSTGRES_PORT=5432",
            "POSTGRES_USER=caria_user",
            f"POSTGRES_PASSWORD={PASSWORD}",
            "POSTGRES_DB=caria",
            "",
            "# FMP API Key",
            "FMP_API_KEY=79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq",
            "",
            "# JWT Secret Key",
            "# JWT_SECRET_KEY=tu_secret_key_aqui",
        ]
    
    ENV_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Archivo .env actualizado en {ENV_FILE}")
    print(f"   POSTGRES_PASSWORD configurado: {PASSWORD}")
    
    # Verificar que se guardó correctamente
    content = ENV_FILE.read_text(encoding="utf-8")
    if f"POSTGRES_PASSWORD={PASSWORD}" in content:
        print(f"   ✅ Verificación: Contraseña encontrada en .env")
    else:
        print(f"   ⚠️  Advertencia: No se pudo verificar la contraseña")

if __name__ == "__main__":
    update_env_file()


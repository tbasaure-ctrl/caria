#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script para iniciar la API de CARIA con configuración correcta."""

import os
import sys
from pathlib import Path

# Agregar directorios necesarios al path para imports
API_DIR = Path(__file__).parent.resolve()
SERVICES_DIR = API_DIR.parent  # services/
NOTEBOOKS_DIR = SERVICES_DIR.parent  # notebooks/
CARIA_DATA_SRC = NOTEBOOKS_DIR / "caria_data" / "src"

# Configurar PYTHONPATH como variable de entorno para que el proceso hijo lo herede
pythonpath_parts = []
if str(SERVICES_DIR) not in pythonpath_parts:
    pythonpath_parts.append(str(SERVICES_DIR))
if str(CARIA_DATA_SRC) not in pythonpath_parts and CARIA_DATA_SRC.exists():
    pythonpath_parts.append(str(CARIA_DATA_SRC))

# Agregar al PYTHONPATH existente o crear uno nuevo
existing_pythonpath = os.environ.get("PYTHONPATH", "")
if existing_pythonpath:
    pythonpath_parts.insert(0, existing_pythonpath)
os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

# También agregar al sys.path del proceso actual
for path in pythonpath_parts:
    if path not in sys.path:
        sys.path.insert(0, path)

# Cambiar al directorio de la API para que los imports funcionen
os.chdir(API_DIR)

# Cargar variables de entorno desde .env si existe
env_file = API_DIR / ".env"
if env_file.exists():
    print(f"[INFO] Cargando variables de entorno desde {env_file}")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()

if __name__ == "__main__":
    import uvicorn
    
    # Verificar variables críticas
    required_vars = ["FMP_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print("[WARNING] Advertencia: Variables de entorno faltantes:")
        for var in missing:
            print(f"   - {var}")
        print("\n[INFO] Configuralas o crea un archivo .env")
        print("   Ejecuta: python setup_env.py")
        print("\n[WARNING] Continuando de todas formas...")

    print("[INFO] Iniciando API de CARIA...")
    print(f"   Directorio de trabajo: {os.getcwd()}")
    print(f"   PYTHONPATH: {os.environ.get('PYTHONPATH', 'No configurado')}")
    fmp_status = "[OK] Configurada" if os.getenv('FMP_API_KEY') else "[ERROR] No configurada"
    print(f"   FMP_API_KEY: {fmp_status}")
    print(f"   Servidor: http://0.0.0.0:8000")
    print(f"   Docs: http://localhost:8000/docs\n")
    
    try:
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=[str(API_DIR)],
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n\n[INFO] API detenida por el usuario")
    except Exception as e:
        print(f"\n[ERROR] Error iniciando la API: {e}")
        print("\n[INFO] Verifica:")
        print("   - Estas en el directorio services/api")
        print("   - Todas las dependencias estan instaladas")
        print("   - Las variables de entorno estan configuradas")
        sys.exit(1)


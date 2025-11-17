"""Helper para configurar paths correctamente en scripts de caria_data.

Este módulo debe importarse al inicio de cualquier script que use módulos de caria.
Asegura que sys.path incluya el directorio src/ para encontrar el módulo caria.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Obtener el directorio base (caria_data/)
# Este archivo está en scripts/_setup_paths.py, así que subimos 1 nivel
BASE_DIR = Path(__file__).resolve().parent.parent

# Agregar src/ al path si no está ya
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# También agregar el directorio scripts/ por si acaso
SCRIPTS_DIR = BASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

__all__ = ["BASE_DIR", "SRC_DIR", "SCRIPTS_DIR"]


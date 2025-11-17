"""Configuración de paths para encontrar módulo caria.

Este módulo debe importarse al inicio de cualquier archivo que use módulos de caria.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Configurar paths antes de importar módulos de caria
# Desde backend/api/, subir hasta notebooks/, luego entrar a caria-lib/
CURRENT_FILE = Path(__file__).resolve()
CARIA_LIB = CURRENT_FILE.parent.parent.parent / "caria-lib"
if CARIA_LIB.exists() and str(CARIA_LIB) not in sys.path:
    sys.path.insert(0, str(CARIA_LIB))

# También verificar ruta Docker
DOCKER_CARIA_LIB = Path("/app/caria-lib")
if DOCKER_CARIA_LIB.exists() and str(DOCKER_CARIA_LIB) not in sys.path:
    sys.path.insert(0, str(DOCKER_CARIA_LIB))

__all__ = ["CARIA_LIB", "DOCKER_CARIA_LIB"]


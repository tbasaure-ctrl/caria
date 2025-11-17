#!/usr/bin/env python3
"""Script para probar la validación de contraseñas."""

import sys
from pathlib import Path

# Agregar path de caria
CARIA_SRC = Path(__file__).parent.parent.parent / "caria_data" / "src"
sys.path.insert(0, str(CARIA_SRC))

from caria.models.auth import UserRegister

# Probar diferentes contraseñas
test_cases = [
    ("test123", "Corta simple"),
    ("password123", "Media simple"),
    ("MyP@ssw0rd123", "Media con símbolos"),
    ("EstaEsMiContraseña123", "Media con acentos"),
    ("a" * 50, "50 caracteres ASCII"),
    ("a" * 72, "72 caracteres ASCII (límite)"),
    ("a" * 73, "73 caracteres ASCII (debería fallar)"),
    ("ñ" * 36, "36 caracteres con tilde (72 bytes)"),
    ("ñ" * 37, "37 caracteres con tilde (74 bytes, debería fallar)"),
]

print("Probando validación de contraseñas:\n")
print(f"{'Contraseña':<40} {'Bytes':<10} {'Resultado':<20}")
print("-" * 70)

for password, description in test_cases:
    bytes_len = len(password.encode('utf-8'))
    try:
        test_data = {
            'email': 'test@test.com',
            'username': 'testuser',
            'password': password
        }
        user = UserRegister(**test_data)
        result = "✅ OK"
    except ValueError as e:
        result = f"❌ Error: {str(e)[:50]}"
    except Exception as e:
        result = f"⚠️  {type(e).__name__}: {str(e)[:50]}"
    
    print(f"{description:<40} {bytes_len:<10} {result}")


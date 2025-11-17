#!/usr/bin/env python3
"""Script simple para probar solo el login y ver qué está pasando."""

import requests
import json

BASE_URL = "http://localhost:8000"

print("="*70)
print("TEST SIMPLE DE LOGIN")
print("="*70)

# 1. Verificar que la API esté corriendo
print("\n1. Verificando que la API esté corriendo...")
try:
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    if response.status_code == 200:
        print("✅ API está corriendo")
        print(f"   Response: {response.json()}")
    else:
        print(f"❌ API respondió con código {response.status_code}")
        print(f"   Response: {response.text}")
except requests.exceptions.ConnectionError:
    print("❌ No se pudo conectar a la API")
    print("   Asegúrate de que la API esté corriendo:")
    print("   python start_api.py")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# 2. Intentar login
print("\n2. Intentando login...")
login_data = {
    "username": "testuser",
    "password": "Test123!"  # Contraseña más corta
}

try:
    response = requests.post(
        f"{BASE_URL}/api/auth/login",
        data=login_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=10
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Login exitoso!")
        if "access_token" in data:
            token = data["access_token"]
            print(f"Token: {token[:50]}...")
        elif "token" in data and "access_token" in data["token"]:
            token = data["token"]["access_token"]
            print(f"Token: {token[:50]}...")
        else:
            print(f"Response completa: {json.dumps(data, indent=2)}")
    else:
        print(f"❌ Login falló con código {response.status_code}")
        try:
            error_data = response.json()
            print(f"Error: {json.dumps(error_data, indent=2)}")
        except:
            print(f"Error (texto): {response.text}")
            
except Exception as e:
    print(f"❌ Error inesperado: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)













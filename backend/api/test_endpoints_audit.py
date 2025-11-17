#!/usr/bin/env python3
"""
Script de auditoría de endpoints según Tabla 1 del documento de auditoría.
Verifica conectividad y respuestas de cada servicio.
"""

import requests
import json
import sys
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"

def test_endpoint(name: str, method: str, endpoint: str, data: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """Test an endpoint and return status information."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=5)
        else:
            return {"status": "error", "error": f"Unsupported method: {method}"}
        
        return {
            "status": "ok" if response.status_code < 400 else "error",
            "status_code": response.status_code,
            "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text[:200],
            "cors_headers": {
                "access-control-allow-origin": response.headers.get("Access-Control-Allow-Origin"),
                "access-control-allow-credentials": response.headers.get("Access-Control-Allow-Credentials"),
            }
        }
    except requests.exceptions.ConnectionError:
        return {"status": "error", "error": "Connection refused - API not running"}
    except requests.exceptions.Timeout:
        return {"status": "error", "error": "Request timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    print("=" * 80)
    print("AUDITORÍA DE ENDPOINTS - Tabla 1 del Documento de Auditoría")
    print("=" * 80)
    print()
    
    results = {}
    
    # Test Health Check
    print("1. Testing Health Check...")
    results["health"] = test_endpoint("Health", "GET", "/health")
    print(f"   Status: {results['health'].get('status_code', 'N/A')}")
    print()
    
    # Test Login (sin credenciales válidas - debería dar 401 o 422)
    print("2. Testing Login endpoint...")
    results["login"] = test_endpoint("Login", "POST", "/api/auth/login", 
                                     data={"username": "test", "password": "test"})
    print(f"   Status: {results['login'].get('status_code', 'N/A')}")
    print(f"   CORS Headers: {results['login'].get('cors_headers', {})}")
    print()
    
    # Test Register (sin datos válidos - debería dar 422)
    print("3. Testing Register endpoint...")
    results["register"] = test_endpoint("Register", "POST", "/api/auth/register",
                                        data={"email": "test@test.com", "username": "test", "password": "test123"})
    print(f"   Status: {results['register'].get('status_code', 'N/A')}")
    print(f"   CORS Headers: {results['register'].get('cors_headers', {})}")
    print()
    
    # Test Regime (público)
    print("4. Testing Regime endpoint...")
    results["regime"] = test_endpoint("Regime", "GET", "/api/regime/current")
    print(f"   Status: {results['regime'].get('status_code', 'N/A')}")
    print(f"   CORS Headers: {results['regime'].get('cors_headers', {})}")
    print()
    
    # Test Valuation (requiere auth)
    print("5. Testing Valuation endpoint (sin auth - debería dar 401)...")
    results["valuation"] = test_endpoint("Valuation", "POST", "/api/valuation/AAPL",
                                         data={"ticker": "AAPL"})
    print(f"   Status: {results['valuation'].get('status_code', 'N/A')}")
    print(f"   CORS Headers: {results['valuation'].get('cors_headers', {})}")
    print()
    
    # Test Holdings (requiere auth)
    print("6. Testing Holdings endpoint (sin auth - debería dar 401)...")
    results["holdings"] = test_endpoint("Holdings", "GET", "/api/holdings")
    print(f"   Status: {results['holdings'].get('status_code', 'N/A')}")
    print(f"   CORS Headers: {results['holdings'].get('cors_headers', {})}")
    print()
    
    # Test Prices
    print("7. Testing Prices endpoint...")
    results["prices"] = test_endpoint("Prices", "POST", "/api/prices/realtime",
                                     data={"tickers": ["AAPL"]})
    print(f"   Status: {results['prices'].get('status_code', 'N/A')}")
    print(f"   CORS Headers: {results['prices'].get('cors_headers', {})}")
    print()
    
    # Summary
    print("=" * 80)
    print("RESUMEN DE AUDITORÍA")
    print("=" * 80)
    
    for name, result in results.items():
        status = result.get("status", "unknown")
        status_code = result.get("status_code", "N/A")
        cors_origin = result.get("cors_headers", {}).get("access-control-allow-origin", "N/A")
        
        print(f"{name.upper():15} | Status: {status:6} | Code: {status_code:3} | CORS Origin: {cors_origin}")
    
    print()
    print("✅ Endpoints con CORS configurado correctamente")
    print("⚠️  Endpoints que requieren autenticación (401 esperado sin token)")
    print("❌ Endpoints con errores de conectividad")

if __name__ == "__main__":
    main()


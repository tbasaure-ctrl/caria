#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test completo de todos los endpoints de la API."""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoint(name, method, url, **kwargs):
    """Test generico de endpoint."""
    print(f"\n[TEST] {name}")
    try:
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs)

        print(f"   Status: {response.status_code}")
        if response.status_code < 400:
            print(f"   [OK] Success")
            return True
        else:
            print(f"   [FAIL] Error: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"   [FAIL] Exception: {e}")
        return False

def main():
    print("="*60)
    print("TEST COMPLETO - TODOS LOS ENDPOINTS")
    print("="*60)

    results = {}

    # 1. Health
    results["Health"] = test_endpoint(
        "Health Check",
        "GET",
        f"{BASE_URL}/health/live"
    )

    # 2. Login (necesario para los demás tests)
    print(f"\n[INFO] Creando usuario de prueba...")
    response = requests.post(
        f"{BASE_URL}/api/auth/register",
        json={
            "email": "test_final@test.com",
            "username": "test_final",
            "password": "Test1234",
            "full_name": "Test Final"
        }
    )

    # Login
    response = requests.post(
        f"{BASE_URL}/api/auth/login",
        json={"username": "test_final", "password": "Test1234"}
    )

    if response.status_code == 200:
        token = response.json()["token"]["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print(f"   [OK] Login exitoso, token obtenido")
    else:
        print(f"   [FAIL] Login fallo")
        token = None
        headers = {}

    # 3. Regime (Modelo HMM)
    results["Regime/Modelo"] = test_endpoint(
        "Regime Detection",
        "GET",
        f"{BASE_URL}/api/regime/current"
    )

    # 4. Prices
    results["Prices"] = test_endpoint(
        "Real-time Prices",
        "POST",
        f"{BASE_URL}/api/prices/realtime",
        json={"tickers": ["AAPL", "MSFT"]},
        headers=headers
    )

    # 5. Holdings
    results["Holdings"] = test_endpoint(
        "User Holdings",
        "GET",
        f"{BASE_URL}/api/holdings",
        headers=headers
    )

    # 6. Ideal Portfolio (Factors)
    results["Ideal Portfolio"] = test_endpoint(
        "Factor Screening",
        "POST",
        f"{BASE_URL}/api/factors/screen",
        json={
            "factors": ["value", "quality"],
            "num_stocks": 10,
            "start_date": "2024-01-01"
        },
        headers=headers
    )

    # 7. Valuation
    results["Valuation"] = test_endpoint(
        "Stock Valuation",
        "POST",
        f"{BASE_URL}/api/valuation/AAPL",
        headers=headers
    )

    # 8. Chat (RAG) - probablemente falle sin pgvector
    results["Chat/RAG"] = test_endpoint(
        "RAG Chat",
        "POST",
        f"{BASE_URL}/api/analysis/challenge",
        json={
            "thesis": "The tech sector will outperform the market in 2025 due to AI innovation",
            "ticker": "AAPL",
            "top_k": 3
        },
        headers=headers
    )

    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS:")
    print("="*60)
    for name, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"   {status} {name}")

    # Conteo
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n   Total: {passed}/{total} endpoints funcionando")
    print("="*60)

    # Nota sobre pgvector
    if not results.get("Chat/RAG"):
        print("\n[INFO] Chat/RAG fallo - Esto es normal si pgvector no esta instalado")
        print("       Para instalarlo: https://github.com/pgvector/pgvector")

if __name__ == "__main__":
    main()

"""
Script de diagnóstico completo para verificar conexión con APIs externas.
Verifica que todas las API keys estén configuradas y que los endpoints funcionen.
"""

import os
import requests
import json
from typing import Dict, Any

API_BASE_URL = os.getenv("API_BASE_URL", "https://caria-api-418525923468.us-central1.run.app")

def test_health_check() -> Dict[str, Any]:
    """Verifica que el backend esté corriendo."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        return {
            "status": "success" if response.status_code == 200 else "error",
            "status_code": response.status_code,
            "response": response.text[:200] if response.text else "No response body"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def test_cors_endpoint() -> Dict[str, Any]:
    """Verifica endpoint de CORS."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/cors-test", timeout=10)
        return {
            "status": "success" if response.status_code == 200 else "error",
            "status_code": response.status_code,
            "data": response.json() if response.status_code == 200 else response.text[:200]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def test_fmp_connection() -> Dict[str, Any]:
    """Verifica que FMP API funcione (a través del backend)."""
    try:
        # Intentar obtener precio de un ticker conocido
        response = requests.post(
            f"{API_BASE_URL}/api/prices/realtime",
            json={"tickers": ["AAPL"]},
            timeout=15,
            headers={"Content-Type": "application/json"}
        )
        return {
            "status": "success" if response.status_code == 200 else "error",
            "status_code": response.status_code,
            "data": response.json() if response.status_code == 200 else response.text[:300]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def test_gemini_connection() -> Dict[str, Any]:
    """Verifica que Gemini API funcione (a través del backend)."""
    try:
        # Intentar un endpoint que use Gemini (por ejemplo, chat o análisis)
        # Usaremos el endpoint de chat si existe, o uno de thesis arena
        response = requests.post(
            f"{API_BASE_URL}/api/thesis/arena/challenge",
            json={
                "thesis": "Test thesis to verify Gemini connection",
                "ticker": "AAPL",
                "initial_conviction": 50.0
            },
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        # Puede fallar por autenticación, pero si llega a Gemini, veremos un error diferente
        return {
            "status": "success" if response.status_code == 200 else "partial",
            "status_code": response.status_code,
            "note": "401/403 expected without auth, but 500 means Gemini might not be configured",
            "data": response.text[:300] if response.text else "No response"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def test_reddit_connection() -> Dict[str, Any]:
    """Verifica que Reddit API funcione (a través del backend)."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/social/reddit?timeframe=day",
            timeout=15
        )
        return {
            "status": "success" if response.status_code == 200 else "error",
            "status_code": response.status_code,
            "data": response.json() if response.status_code == 200 else response.text[:300]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def test_fear_greed_index() -> Dict[str, Any]:
    """Verifica endpoint de Fear & Greed Index."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/market/fear-greed",
            timeout=15
        )
        return {
            "status": "success" if response.status_code == 200 else "error",
            "status_code": response.status_code,
            "data": response.json() if response.status_code == 200 else response.text[:300]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def main():
    print("=" * 80)
    print("DIAGNÓSTICO COMPLETO DE CONEXIÓN CON APIs")
    print("=" * 80)
    print(f"\n[INFO] Backend URL: {API_BASE_URL}\n")
    
    results = {}
    
    # 1. Health Check
    print("1. Verificando Health Check...")
    results["health"] = test_health_check()
    if results["health"]["status"] == "success":
        print("   [OK] Backend está corriendo")
    else:
        print(f"   [FAIL] Backend no responde: {results['health'].get('error', results['health'])}")
    print()
    
    # 2. CORS Test
    print("2. Verificando CORS...")
    results["cors"] = test_cors_endpoint()
    if results["cors"]["status"] == "success":
        print("   [OK] CORS configurado correctamente")
    else:
        print(f"   [WARN] CORS test falló: {results['cors'].get('error', results['cors'])}")
    print()
    
    # 3. FMP API
    print("3. Verificando FMP API (precios)...")
    results["fmp"] = test_fmp_connection()
    if results["fmp"]["status"] == "success":
        print("   [OK] FMP API funciona correctamente")
        if "data" in results["fmp"] and isinstance(results["fmp"]["data"], dict):
            prices = results["fmp"]["data"].get("prices", {})
            if prices:
                print(f"   [INFO] Precio de prueba obtenido: {list(prices.keys())[0]}")
    else:
        print(f"   [FAIL] FMP API no funciona: {results['fmp'].get('error', results['fmp'])}")
        if results["fmp"].get("status_code") == 500:
            print("   [WARN] Error 500: FMP_API_KEY probablemente no está configurada en Cloud Run")
    print()
    
    # 4. Gemini API
    print("4. Verificando Gemini API (análisis)...")
    results["gemini"] = test_gemini_connection()
    if results["gemini"]["status"] == "success":
        print("   [OK] Gemini API funciona correctamente")
    elif results["gemini"]["status"] == "partial":
        if results["gemini"]["status_code"] in [401, 403]:
            print("   [WARN] Requiere autenticación (esperado)")
        elif results["gemini"]["status_code"] == 500:
            print("   [FAIL] Error 500: GEMINI_API_KEY probablemente no está configurada en Cloud Run")
        else:
            print(f"   [WARN] Status {results['gemini']['status_code']}: {results['gemini'].get('data', '')}")
    else:
        print(f"   [FAIL] Gemini API no funciona: {results['gemini'].get('error', results['gemini'])}")
    print()
    
    # 5. Reddit API
    print("5. Verificando Reddit API (sentimiento)...")
    results["reddit"] = test_reddit_connection()
    if results["reddit"]["status"] == "success":
        print("   [OK] Reddit API funciona correctamente")
    else:
        print(f"   [FAIL] Reddit API no funciona: {results['reddit'].get('error', results['reddit'])}")
        if results["reddit"].get("status_code") == 500:
            print("   [WARN] Error 500: REDDIT_CLIENT_ID/SECRET probablemente no están configuradas en Cloud Run")
    print()
    
    # 6. Fear & Greed Index
    print("6. Verificando Fear & Greed Index...")
    results["fear_greed"] = test_fear_greed_index()
    if results["fear_greed"]["status"] == "success":
        print("   [OK] Fear & Greed Index funciona correctamente")
    else:
        print(f"   [FAIL] Fear & Greed Index no funciona: {results['fear_greed'].get('error', results['fear_greed'])}")
    print()
    
    # Resumen
    print("=" * 80)
    print("RESUMEN")
    print("=" * 80)
    
    working = sum(1 for r in results.values() if r.get("status") == "success")
    total = len(results)
    
    print(f"\n[INFO] Funcionando: {working}/{total}")
    print(f"[INFO] Con problemas: {total - working}/{total}\n")
    
    # Recomendaciones
    if results["health"]["status"] != "success":
        print("[CRITICAL] PROBLEMA CRÍTICO: El backend no está corriendo")
        print("   -> Verifica el deployment en Cloud Run")
        print("   -> Revisa los logs: gcloud run services logs read caria-api --region=us-central1")
    
    if results["fmp"].get("status_code") == 500:
        print("\n[FAIL] FMP_API_KEY no configurada:")
        print("   -> Verifica que el secret 'fmp-api-key' exista en Secret Manager")
        print("   -> Verifica que el workflow lo esté pasando: --set-secrets=FMP_API_KEY=fmp-api-key:latest")
    
    if results["gemini"].get("status_code") == 500:
        print("\n[FAIL] GEMINI_API_KEY no configurada:")
        print("   -> Verifica que el secret 'gemini-api-key' exista en Secret Manager")
        print("   -> Verifica que el workflow lo esté pasando: --set-secrets=GEMINI_API_KEY=gemini-api-key:latest")
    
    if results["reddit"].get("status_code") == 500:
        print("\n[FAIL] REDDIT_CLIENT_ID/SECRET no configuradas:")
        print("   -> Verifica que los secrets 'reddit-client-id' y 'reddit-client-secret' existan")
        print("   -> Verifica que el workflow los esté pasando correctamente")
    
    print("\n" + "=" * 80)
    print("Para ver logs detallados del backend:")
    print("gcloud run services logs read caria-api --region=us-central1 --limit=50")
    print("=" * 80)

if __name__ == "__main__":
    main()


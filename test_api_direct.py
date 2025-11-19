"""
Test directo de endpoints de API para diagnosticar problemas específicos.
"""

import requests
import os
import json

API_BASE_URL = "https://caria-api-418525923468.us-central1.run.app"

def test_reddit_endpoint():
    """Test Reddit API endpoint directamente."""
    print("\n" + "="*60)
    print("TESTING REDDIT API")
    print("="*60)
    
    try:
        # Reddit endpoint no requiere auth según el código
        response = requests.get(
            f"{API_BASE_URL}/api/social/reddit?timeframe=day",
            timeout=15
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Reddit API funciona!")
            print(f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        elif response.status_code == 500:
            print(f"❌ Error 500 - Revisar logs del backend")
            print(f"Error detail: {response.text}")
        elif response.status_code == 401:
            print(f"❌ Error 401 - Credenciales inválidas")
        else:
            print(f"⚠️ Status inesperado: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_fmp_endpoint():
    """Test FMP API endpoint directamente."""
    print("\n" + "="*60)
    print("TESTING FMP API")
    print("="*60)
    
    try:
        # FMP endpoint requiere auth, pero probemos primero sin auth para ver el error
        response = requests.post(
            f"{API_BASE_URL}/api/prices/realtime",
            json={"tickers": ["AAPL"]},
            timeout=15,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        
        if response.status_code == 200:
            print(f"✅ FMP API funciona!")
        elif response.status_code == 403:
            print(f"⚠️ Requiere autenticación (esperado)")
        elif response.status_code == 500:
            print(f"❌ Error 500 - Problema interno del backend")
            print(f"Error detail: {response.text}")
        else:
            print(f"⚠️ Status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_fear_greed_endpoint():
    """Test Fear & Greed Index endpoint."""
    print("\n" + "="*60)
    print("TESTING FEAR & GREED INDEX")
    print("="*60)
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/market/fear-greed",
            timeout=15
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        
        if response.status_code == 200:
            print(f"✅ Fear & Greed funciona!")
        elif response.status_code == 403:
            print(f"⚠️ Requiere autenticación")
        elif response.status_code == 500:
            print(f"❌ Error 500 - Problema interno")
        else:
            print(f"⚠️ Status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_gemini_via_thesis_arena():
    """Test Gemini API a través del endpoint de Thesis Arena."""
    print("\n" + "="*60)
    print("TESTING GEMINI API (via Thesis Arena)")
    print("="*60)
    
    try:
        # Este endpoint requiere auth, pero podemos ver el error
        response = requests.post(
            f"{API_BASE_URL}/api/thesis/arena/challenge",
            json={
                "thesis": "Test thesis",
                "ticker": "AAPL",
                "initial_conviction": 50.0
            },
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        
        if response.status_code == 200:
            print(f"✅ Gemini API funciona!")
        elif response.status_code == 401:
            print(f"⚠️ Requiere autenticación (esperado)")
        elif response.status_code == 500:
            print(f"❌ Error 500 - Revisar si Gemini API key está configurada")
            print(f"Error detail: {response.text}")
        else:
            print(f"⚠️ Status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def main():
    print("="*60)
    print("DIAGNÓSTICO DIRECTO DE ENDPOINTS API")
    print("="*60)
    print(f"\nBackend URL: {API_BASE_URL}\n")
    
    test_reddit_endpoint()
    test_fmp_endpoint()
    test_fear_greed_endpoint()
    test_gemini_via_thesis_arena()
    
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    print("\nSi ves errores 500, revisa los logs del backend:")
    print("gcloud run services logs read caria-api --region=us-central1 --project=caria-backend --limit=50")

if __name__ == "__main__":
    main()


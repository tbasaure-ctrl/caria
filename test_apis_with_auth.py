"""
Test completo de APIs con autenticación para verificar FMP y Gemini.
"""

import requests
import json

API_BASE_URL = "https://caria-api-418525923468.us-central1.run.app"

def test_with_auth():
    """Test completo con autenticación."""
    print("="*60)
    print("TEST COMPLETO DE APIs CON AUTENTICACIÓN")
    print("="*60)
    
    # 1. Crear usuario de prueba o hacer login
    print("\n1. Creando usuario de prueba...")
    test_email = "test_api_user@caria.com"
    test_password = "TestPassword123!"
    
    try:
        # Intentar registro
        register_response = requests.post(
            f"{API_BASE_URL}/api/auth/register",
            json={
                "email": test_email,
                "username": "test_api_user",
                "password": test_password,
                "full_name": "Test API User"
            },
            timeout=10
        )
        
        if register_response.status_code == 201:
            print("✅ Usuario creado exitosamente")
            print(f"Response: {register_response.text[:500]}")
            data = register_response.json()
            token = data.get("access_token") or data.get("token")
            if not token:
                print(f"⚠️ Respuesta completa: {json.dumps(data, indent=2)}")
        elif register_response.status_code == 400:
            # Usuario ya existe, intentar login
            print("⚠️ Usuario ya existe, haciendo login...")
            login_response = requests.post(
                f"{API_BASE_URL}/api/auth/login",
                json={
                    "email": test_email,
                    "username": "test_api_user",
                    "password": test_password
                },
                timeout=10
            )
            if login_response.status_code == 200:
                data = login_response.json()
                # Token puede estar en data["access_token"] o data["token"]["access_token"]
                if "token" in data and isinstance(data["token"], dict):
                    token = data["token"].get("access_token")
                else:
                    token = data.get("access_token")
                print("✅ Login exitoso")
                if not token:
                    print(f"⚠️ No se encontró access_token. Response keys: {list(data.keys())}")
                    print(f"Full response: {json.dumps(data, indent=2)}")
            else:
                print(f"❌ Error en login: {login_response.status_code} - {login_response.text}")
                return
        else:
            print(f"❌ Error en registro: {register_response.status_code} - {register_response.text}")
            return
        
        if not token:
            print("❌ No se obtuvo token de autenticación")
            return
        
        headers = {"Authorization": f"Bearer {token}"}
        print(f"✅ Token obtenido: {token[:20]}...")
        
    except Exception as e:
        print(f"❌ Error en autenticación: {e}")
        return
    
    # 2. Test FMP API con autenticación
    print("\n2. Probando FMP API (precios)...")
    try:
        fmp_response = requests.post(
            f"{API_BASE_URL}/api/prices/realtime",
            json={"tickers": ["AAPL", "MSFT", "GOOGL"]},
            headers=headers,
            timeout=15
        )
        print(f"Status Code: {fmp_response.status_code}")
        if fmp_response.status_code == 200:
            data = fmp_response.json()
            prices = data.get("prices", {})
            print(f"✅ FMP API funciona! Precios obtenidos: {list(prices.keys())}")
            for ticker, price_data in list(prices.items())[:3]:
                print(f"   {ticker}: ${price_data.get('price', 'N/A')}")
        elif fmp_response.status_code == 500:
            error_detail = fmp_response.json().get("detail", "")
            print(f"❌ Error 500: {error_detail}")
            if "FMP_API_KEY" in error_detail or "no configurado" in error_detail:
                print("   → Problema: FMP_API_KEY no está siendo leída correctamente")
        else:
            print(f"⚠️ Status: {fmp_response.status_code}")
            print(f"Response: {fmp_response.text[:300]}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # 3. Test Gemini API (Thesis Arena)
    print("\n3. Probando Gemini API (Thesis Arena)...")
    try:
        gemini_response = requests.post(
            f"{API_BASE_URL}/api/thesis/arena/challenge",
            json={
                "thesis": "Apple will outperform the market due to strong iPhone sales",
                "ticker": "AAPL",
                "initial_conviction": 75.0
            },
            headers=headers,
            timeout=30
        )
        print(f"Status Code: {gemini_response.status_code}")
        if gemini_response.status_code == 200:
            data = gemini_response.json()
            print("✅ Gemini API funciona!")
            print(f"   Comunidades respondieron: {len(data.get('community_responses', []))}")
            if data.get('community_responses'):
                first_response = data['community_responses'][0]
                print(f"   Ejemplo: {first_response.get('community')} - {first_response.get('response', '')[:100]}...")
        elif gemini_response.status_code == 500:
            error_detail = gemini_response.json().get("detail", "")
            print(f"❌ Error 500: {error_detail}")
            if "GEMINI_API_KEY" in error_detail or "not configured" in error_detail:
                print("   → Problema: GEMINI_API_KEY no está siendo leída correctamente")
        else:
            print(f"⚠️ Status: {gemini_response.status_code}")
            print(f"Response: {gemini_response.text[:300]}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # 4. Test Fear & Greed (debe funcionar sin auth ahora)
    print("\n4. Probando Fear & Greed (sin auth)...")
    try:
        fg_response = requests.get(
            f"{API_BASE_URL}/api/market/fear-greed",
            timeout=10
        )
        print(f"Status Code: {fg_response.status_code}")
        if fg_response.status_code == 200:
            data = fg_response.json()
            print(f"✅ Fear & Greed funciona! Valor: {data.get('value')} ({data.get('classification')})")
        else:
            print(f"⚠️ Status: {fg_response.status_code}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # 5. Test Reddit (debe funcionar sin auth)
    print("\n5. Probando Reddit API (sin auth)...")
    try:
        reddit_response = requests.get(
            f"{API_BASE_URL}/api/social/reddit?timeframe=day",
            timeout=15
        )
        print(f"Status Code: {reddit_response.status_code}")
        if reddit_response.status_code == 200:
            data = reddit_response.json()
            print(f"✅ Reddit funciona! Stocks: {len(data.get('stocks', []))}")
            if data.get('mock_data'):
                print("   ⚠️ Usando datos mock (API real no disponible)")
            else:
                print("   ✅ Usando datos reales de Reddit")
        else:
            print(f"⚠️ Status: {reddit_response.status_code}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    print("\nSi FMP o Gemini dan error 500, el problema es que los secrets")
    print("no se están leyendo correctamente en el código, aunque estén")
    print("configurados en Cloud Run.")

if __name__ == "__main__":
    test_with_auth()


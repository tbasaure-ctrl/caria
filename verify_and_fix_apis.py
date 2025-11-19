"""
Script completo para verificar y arreglar problemas de APIs.
"""

import requests
import json
import time

API_BASE_URL = "https://caria-api-418525923468.us-central1.run.app"

def check_secrets_status():
    """Verificar estado de secrets."""
    print("\n" + "="*60)
    print("1. VERIFICANDO SECRETS")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/debug/secrets-status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Secrets Status:")
            print(json.dumps(data, indent=2))
            return data
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_reddit_detailed():
    """Test detallado de Reddit API."""
    print("\n" + "="*60)
    print("2. TESTING REDDIT API (DETALLADO)")
    print("="*60)
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/social/reddit?timeframe=day",
            timeout=20
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body: {response.text[:500]}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Reddit API funciona!")
            if data.get("mock_data"):
                print("⚠️ Usando datos mock (praw no instalado o secrets no configurados)")
            else:
                print(f"✅ Datos reales de Reddit: {len(data.get('stocks', []))} stocks")
            return True
        elif response.status_code == 500:
            error_detail = response.json().get("detail", "")
            print(f"❌ Error 500: {error_detail}")
            if "401" in error_detail:
                print("   → Problema: Reddit rechaza las credenciales")
                print("   → Posibles causas:")
                print("     1. Client ID o Secret incorrectos")
                print("     2. User Agent no aceptado")
                print("     3. Reddit requiere OAuth adicional")
            return False
        else:
            print(f"⚠️ Status inesperado: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_fmp_with_auth():
    """Test FMP con autenticación simulada."""
    print("\n" + "="*60)
    print("3. TESTING FMP API")
    print("="*60)
    
    # Primero intentar sin auth para ver el error
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/prices/realtime",
            json={"tickers": ["AAPL"]},
            timeout=15
        )
        print(f"Status Code (sin auth): {response.status_code}")
        if response.status_code == 403:
            print("⚠️ Requiere autenticación (esperado)")
            print("   → FMP API está configurada, solo necesita login de usuario")
            return True
        elif response.status_code == 500:
            print(f"❌ Error 500: {response.text[:200]}")
            return False
        else:
            print(f"⚠️ Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_fear_greed():
    """Test Fear & Greed Index."""
    print("\n" + "="*60)
    print("4. TESTING FEAR & GREED INDEX")
    print("="*60)
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/market/fear-greed",
            timeout=15
        )
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Fear & Greed funciona!")
            print(f"   Value: {data.get('value', 'N/A')}")
            return True
        elif response.status_code == 403:
            print("⚠️ Requiere autenticación")
            return None
        elif response.status_code == 500:
            print(f"❌ Error 500: {response.text[:200]}")
            return False
        else:
            print(f"⚠️ Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    print("="*60)
    print("VERIFICACIÓN Y DIAGNÓSTICO COMPLETO DE APIs")
    print("="*60)
    print(f"\nBackend URL: {API_BASE_URL}\n")
    
    # 1. Verificar secrets
    secrets_status = check_secrets_status()
    
    # 2. Test Reddit
    reddit_ok = test_reddit_detailed()
    
    # 3. Test FMP
    fmp_ok = test_fmp_with_auth()
    
    # 4. Test Fear & Greed
    fear_greed_ok = test_fear_greed()
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN Y RECOMENDACIONES")
    print("="*60)
    
    if secrets_status:
        all_secrets = secrets_status.get("all_secrets_present", False)
        if not all_secrets:
            print("\n🔴 PROBLEMA: No todos los secrets están configurados")
            print("   Secrets faltantes:")
            for key, value in secrets_status.get("secrets_configured", {}).items():
                if not value:
                    print(f"      - {key}")
        else:
            print("\n✅ Todos los secrets están configurados")
    
    if not reddit_ok:
        print("\n🔴 REDDIT API NO FUNCIONA")
        print("   Soluciones posibles:")
        print("   1. Verificar que las credenciales sean correctas")
        print("   2. Verificar que el User Agent sea aceptado por Reddit")
        print("   3. Reddit puede requerir OAuth flow completo")
        print("   4. Considerar usar Reddit API v2 con diferentes credenciales")
    
    if fmp_ok:
        print("\n✅ FMP API está configurada (requiere autenticación de usuario)")
    
    if fear_greed_ok is None:
        print("\n⚠️ Fear & Greed requiere autenticación (normal)")
    elif fear_greed_ok:
        print("\n✅ Fear & Greed funciona correctamente")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()


"""
Test directo de FMP API para ver qué está pasando.
"""

import requests
import os

# Test FMP API directamente
FMP_API_KEY = "79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq"

def test_fmp_quote():
    """Test FMP quote endpoint directamente."""
    print("="*60)
    print("TEST DIRECTO DE FMP API")
    print("="*60)
    
    # Test con un solo ticker
    ticker = "AAPL"
    url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={FMP_API_KEY}"
    
    print(f"\n1. Probando quote para {ticker}...")
    try:
        response = requests.get(url, timeout=15)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ FMP API responde!")
            print(f"Tipo de datos: {type(data)}")
            if isinstance(data, list) and len(data) > 0:
                print(f"✅ Datos recibidos: {len(data)} items")
                print(f"Primer item: {data[0]}")
            elif isinstance(data, dict):
                print(f"✅ Datos recibidos como dict: {list(data.keys())}")
            else:
                print(f"⚠️ Respuesta vacía o formato inesperado")
        elif response.status_code == 403:
            print("❌ Error 403: API key inválida o sin permisos")
        elif response.status_code == 401:
            print("❌ Error 401: No autorizado")
        else:
            print(f"⚠️ Status inesperado: {response.status_code}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test con múltiples tickers
    print(f"\n2. Probando quote para múltiples tickers (AAPL,MSFT,GOOGL)...")
    tickers_str = "AAPL,MSFT,GOOGL"
    url = f"https://financialmodelingprep.com/api/v3/quote/{tickers_str}?apikey={FMP_API_KEY}"
    
    try:
        response = requests.get(url, timeout=15)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ FMP API responde!")
            if isinstance(data, list):
                print(f"✅ Datos recibidos: {len(data)} items")
                for item in data[:3]:
                    print(f"   {item.get('symbol', 'N/A')}: ${item.get('price', 'N/A')}")
            else:
                print(f"⚠️ Formato inesperado: {type(data)}")
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_fmp_quote()


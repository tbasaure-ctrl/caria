#!/usr/bin/env python3
"""Script para probar la conexión y todos los endpoints de la API de CARIA.

Este script prueba todos los endpoints principales para verificar que la API
está funcionando correctamente.
"""

import os
import sys
import json
import requests
from typing import Dict, Any, Optional

# Colores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

BASE_URL = os.getenv("API_URL", "http://localhost:8000")
TEST_USER = {
    "email": "test@caria.com",
    "username": "testuser",
    "password": "Test123!",  # Contraseña más corta (menos de 72 bytes)
    "full_name": "Test User"
}

class APITester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.access_token: Optional[str] = None
        self.session = requests.Session()
        self.results: Dict[str, Any] = {}
    
    def print_header(self, text: str):
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")
    
    def print_success(self, text: str):
        print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")
    
    def print_error(self, text: str):
        print(f"{Colors.RED}❌ {text}{Colors.RESET}")
    
    def print_warning(self, text: str):
        print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")
    
    def print_info(self, text: str):
        print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")
    
    def test_request(self, name: str, method: str, url: str, **kwargs) -> bool:
        """Ejecuta un request y retorna True si fue exitoso."""
        try:
            if self.access_token and "headers" not in kwargs:
                kwargs["headers"] = {}
            if self.access_token:
                kwargs.setdefault("headers", {})["Authorization"] = f"Bearer {self.access_token}"
            
            response = self.session.request(method, url, **kwargs)
            
            if response.status_code < 400:
                self.print_success(f"{name}: {response.status_code}")
                if response.content:
                    try:
                        data = response.json()
                        self.results[name] = data
                        # Mostrar preview de la respuesta
                        if isinstance(data, dict):
                            preview = {k: str(v)[:50] + "..." if len(str(v)) > 50 else v 
                                      for k, v in list(data.items())[:3]}
                            self.print_info(f"   Response: {json.dumps(preview, indent=2)}")
                    except:
                        self.print_info(f"   Response: {response.text[:100]}")
                return True
            else:
                self.print_error(f"{name}: {response.status_code}")
                try:
                    error_data = response.json()
                    self.print_error(f"   Error: {error_data.get('detail', response.text)}")
                except:
                    self.print_error(f"   Error: {response.text[:200]}")
                return False
        except requests.exceptions.ConnectionError:
            self.print_error(f"{name}: No se pudo conectar a {url}")
            self.print_info("   Asegúrate de que la API esté corriendo en " + self.base_url)
            return False
        except Exception as e:
            self.print_error(f"{name}: {str(e)}")
            return False
    
    def test_health(self) -> bool:
        """Prueba el endpoint de health check."""
        self.print_header("1. HEALTH CHECK")
        return self.test_request(
            "Health Check",
            "GET",
            f"{self.base_url}/health"
        )
    
    def test_register(self) -> bool:
        """Prueba el registro de usuario."""
        self.print_header("2. REGISTRO DE USUARIO")
        success = self.test_request(
            "Register User",
            "POST",
            f"{self.base_url}/api/auth/register",
            json=TEST_USER,
            headers={"Content-Type": "application/json"}
        )
        
        # Si el registro falla con "already exists", no es un error crítico
        if not success and "Register User" in self.results:
            error_msg = str(self.results.get("Register User", {}))
            if "already exists" in error_msg.lower() or "duplicate" in error_msg.lower():
                self.print_warning("Usuario ya existe (esto está bien, puedes hacer login)")
                return True  # No es un error crítico
        
        return success
    
    def test_login(self) -> bool:
        """Prueba el login y guarda el token."""
        self.print_header("3. LOGIN")
        success = self.test_request(
            "Login",
            "POST",
            f"{self.base_url}/api/auth/login",
            data={
                "username": TEST_USER["username"],
                "password": TEST_USER["password"]
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if success and "Login" in self.results:
            token_data = self.results["Login"]
            if "access_token" in token_data:
                self.access_token = token_data["access_token"]
                self.print_success(f"Token obtenido: {self.access_token[:20]}...")
            elif "token" in token_data and "access_token" in token_data["token"]:
                self.access_token = token_data["token"]["access_token"]
                self.print_success(f"Token obtenido: {self.access_token[:20]}...")
        
        return success
    
    def test_get_current_user(self) -> bool:
        """Prueba obtener información del usuario actual."""
        self.print_header("4. OBTENER USUARIO ACTUAL")
        if not self.access_token:
            self.print_warning("No hay token, saltando...")
            return False
        return self.test_request(
            "Get Current User",
            "GET",
            f"{self.base_url}/api/auth/me"
        )
    
    def test_realtime_prices(self) -> bool:
        """Prueba obtener precios en tiempo real."""
        self.print_header("5. PRECIOS EN TIEMPO REAL")
        if not self.access_token:
            self.print_warning("No hay token, saltando...")
            return False
        
        # Batch
        success1 = self.test_request(
            "Get Realtime Prices (Batch)",
            "POST",
            f"{self.base_url}/api/prices/realtime",
            json={"tickers": ["AAPL", "MSFT", "GOOGL", "NVDA"]},
            headers={"Content-Type": "application/json"}
        )
        
        # Single
        success2 = self.test_request(
            "Get Realtime Price (Single)",
            "GET",
            f"{self.base_url}/api/prices/realtime/AAPL"
        )
        
        return success1 and success2
    
    def test_holdings(self) -> bool:
        """Prueba los endpoints de holdings."""
        self.print_header("6. HOLDINGS")
        if not self.access_token:
            self.print_warning("No hay token, saltando...")
            return False
        
        # List
        success1 = self.test_request(
            "List Holdings",
            "GET",
            f"{self.base_url}/api/holdings"
        )
        
        # Create
        success2 = self.test_request(
            "Create Holding",
            "POST",
            f"{self.base_url}/api/holdings",
            json={
                "ticker": "AAPL",
                "quantity": 10,
                "average_cost": 150.0,
                "notes": "Test holding desde script"
            },
            headers={"Content-Type": "application/json"}
        )
        
        # Get with prices
        success3 = self.test_request(
            "Get Holdings with Prices",
            "GET",
            f"{self.base_url}/api/holdings/with-prices"
        )
        
        return success1 and success2 and success3
    
    def test_valuation(self) -> bool:
        """Prueba los endpoints de valuación."""
        self.print_header("7. VALUACIÓN")
        if not self.access_token:
            self.print_warning("No hay token, saltando...")
            return False
        
        # Quick valuation
        success1 = self.test_request(
            "Quick Valuation",
            "POST",
            f"{self.base_url}/api/valuation/AAPL",
            json={"ticker": "AAPL"},
            headers={"Content-Type": "application/json"}
        )
        
        # Monte Carlo (puede tardar)
        self.print_info("Monte Carlo puede tardar unos segundos...")
        success2 = self.test_request(
            "Monte Carlo Valuation",
            "POST",
            f"{self.base_url}/api/valuation/AAPL/monte-carlo",
            json={
                "ticker": "AAPL",
                "n_paths": 1000,  # Menos paths para prueba rápida
                "country_risk": "low"
            },
            headers={"Content-Type": "application/json"},
            timeout=60  # Timeout más largo para Monte Carlo
        )
        
        return success1 and success2
    
    def test_regime(self) -> bool:
        """Prueba el endpoint de régimen."""
        self.print_header("8. RÉGIMEN MACRO")
        return self.test_request(
            "Get Current Regime",
            "GET",
            f"{self.base_url}/api/regime/current"
        )
    
    def test_factors(self) -> bool:
        """Prueba el endpoint de factor screening."""
        self.print_header("9. FACTOR SCREENING")
        if not self.access_token:
            self.print_warning("No hay token, saltando...")
            return False
        
        return self.test_request(
            "Factor Screen",
            "POST",
            f"{self.base_url}/api/factors/screen",
            json={
                "top_n": 5,
                "page": 1,
                "page_size": 5
            },
            headers={"Content-Type": "application/json"}
        )
    
    def test_analysis(self) -> bool:
        """Prueba el endpoint de análisis."""
        self.print_header("10. ANÁLISIS RAG")
        if not self.access_token:
            self.print_warning("No hay token, saltando...")
            return False
        
        return self.test_request(
            "Challenge Thesis",
            "POST",
            f"{self.base_url}/api/analysis/challenge",
            json={
                "thesis": "Creo que NVIDIA va a seguir subiendo porque la IA está en auge",
                "ticker": "NVDA"
            },
            headers={"Content-Type": "application/json"},
            timeout=30  # RAG puede tardar
        )
    
    def run_all_tests(self):
        """Ejecuta todos los tests."""
        print(f"\n{Colors.BOLD}{Colors.BLUE}")
        print("="*70)
        print("  CARIA API - TEST DE CONEXIÓN Y ENDPOINTS")
        print("="*70)
        print(f"{Colors.RESET}")
        print(f"\nBase URL: {self.base_url}\n")
        
        tests = [
            ("Health Check", self.test_health),
            ("Register", self.test_register),
            ("Login", self.test_login),
            ("Get Current User", self.test_get_current_user),
            ("Realtime Prices", self.test_realtime_prices),
            ("Holdings", self.test_holdings),
            ("Valuation", self.test_valuation),
            ("Regime", self.test_regime),
            ("Factors", self.test_factors),
            ("Analysis", self.test_analysis),
        ]
        
        results = {}
        for name, test_func in tests:
            try:
                results[name] = test_func()
            except KeyboardInterrupt:
                self.print_warning("\nTests interrumpidos por el usuario")
                break
            except Exception as e:
                self.print_error(f"Error inesperado en {name}: {e}")
                results[name] = False
        
        # Resumen
        self.print_header("RESUMEN DE TESTS")
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for name, success in results.items():
            status = f"{Colors.GREEN}✅ PASS{Colors.RESET}" if success else f"{Colors.RED}❌ FAIL{Colors.RESET}"
            print(f"  {status} - {name}")
        
        print(f"\n{Colors.BOLD}Total: {passed}/{total} tests pasados{Colors.RESET}\n")
        
        if passed == total:
            self.print_success("¡Todos los tests pasaron! 🎉")
            return 0
        else:
            self.print_warning(f"{total - passed} test(s) fallaron")
            return 1


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Probar conexión y endpoints de CARIA API")
    parser.add_argument(
        "--url",
        default=BASE_URL,
        help=f"URL base de la API (default: {BASE_URL})"
    )
    parser.add_argument(
        "--skip-auth",
        action="store_true",
        help="Saltar tests que requieren autenticación"
    )
    
    args = parser.parse_args()
    
    tester = APITester(base_url=args.url)
    return tester.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())


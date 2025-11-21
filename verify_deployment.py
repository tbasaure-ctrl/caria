#!/usr/bin/env python3
"""
Verification script for Caria deployment on Railway + Neon + Vercel.
Tests all critical endpoints and functionality.
"""

import os
import sys
import requests
import json
from typing import Dict, Any, Optional

# Get API URL from environment or use default
API_BASE_URL = os.getenv("API_BASE_URL", "https://your-railway-url.up.railway.app")

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def test_health_check() -> Dict[str, Any]:
    """Test backend health check endpoint."""
    print_section("1. Health Check")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Database: {data.get('database', 'unknown')}")
            print(f"   RAG: {data.get('rag', 'unknown')}")
            print(f"   Regime: {data.get('regime', 'unknown')}")
            print(f"   Factors: {data.get('factors', 'unknown')}")
            print(f"   Valuation: {data.get('valuation', 'unknown')}")
            return {"status": "success", "data": data}
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return {"status": "error", "code": response.status_code}
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return {"status": "error", "error": str(e)}

def test_auth_flow() -> Optional[str]:
    """Test authentication flow and return access token."""
    print_section("2. Authentication Flow")
    
    test_email = f"test_{os.urandom(4).hex()}@example.com"
    test_password = "TestPassword123!"
    test_username = f"testuser_{os.urandom(4).hex()}"
    
    # Register
    try:
        register_response = requests.post(
            f"{API_BASE_URL}/api/auth/register",
            json={
                "email": test_email,
                "username": test_username,
                "password": test_password,
                "full_name": "Test User"
            },
            timeout=10
        )
        
        if register_response.status_code in [200, 201]:
            print("✅ User registration successful")
            register_data = register_response.json()
            token = register_data.get("access_token") or register_data.get("token", {}).get("access_token")
            if token:
                return token
        elif register_response.status_code == 400:
            print("⚠️  User may already exist, trying login...")
        else:
            print(f"❌ Registration failed: {register_response.status_code}")
            print(f"   Response: {register_response.text[:200]}")
    except Exception as e:
        print(f"❌ Registration error: {e}")
    
    # Login
    try:
        login_response = requests.post(
            f"{API_BASE_URL}/api/auth/login",
            json={
                "email": test_email,
                "password": test_password
            },
            timeout=10
        )
        
        if login_response.status_code == 200:
            print("✅ User login successful")
            login_data = login_response.json()
            token = login_data.get("access_token") or login_data.get("token", {}).get("access_token")
            if token:
                return token
        else:
            print(f"❌ Login failed: {login_response.status_code}")
            print(f"   Response: {login_response.text[:200]}")
    except Exception as e:
        print(f"❌ Login error: {e}")
    
    return None

def test_protected_endpoints(token: str) -> Dict[str, Any]:
    """Test protected endpoints with authentication."""
    print_section("3. Protected Endpoints")
    results = {}
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test holdings endpoint
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/holdings",
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            print("✅ Holdings endpoint working")
            results["holdings"] = "success"
        else:
            print(f"⚠️  Holdings endpoint: {response.status_code}")
            results["holdings"] = f"error_{response.status_code}"
    except Exception as e:
        print(f"❌ Holdings endpoint error: {e}")
        results["holdings"] = "error"
    
    # Test prices endpoint
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/prices/realtime",
            headers=headers,
            json={"tickers": ["AAPL", "MSFT"]},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            prices = data.get("prices", {})
            print(f"✅ Prices endpoint working ({len(prices)} tickers)")
            results["prices"] = "success"
        else:
            print(f"⚠️  Prices endpoint: {response.status_code}")
            results["prices"] = f"error_{response.status_code}"
    except Exception as e:
        print(f"❌ Prices endpoint error: {e}")
        results["prices"] = "error"
    
    # Test analysis endpoint
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/analysis/challenge",
            headers=headers,
            json={
                "thesis": "Apple will outperform the market due to strong iPhone sales",
                "ticker": "AAPL",
                "top_k": 3
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            print("✅ Analysis endpoint working")
            print(f"   Analysis length: {len(data.get('critical_analysis', ''))} chars")
            results["analysis"] = "success"
        else:
            print(f"⚠️  Analysis endpoint: {response.status_code}")
            results["analysis"] = f"error_{response.status_code}"
    except Exception as e:
        print(f"❌ Analysis endpoint error: {e}")
        results["analysis"] = "error"
    
    return results

def test_public_endpoints() -> Dict[str, Any]:
    """Test public endpoints that don't require authentication."""
    print_section("4. Public Endpoints")
    results = {}
    
    # Test fear & greed index
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/market/fear-greed",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Fear & Greed Index working (value: {data.get('value')})")
            results["fear_greed"] = "success"
        else:
            print(f"⚠️  Fear & Greed: {response.status_code}")
            results["fear_greed"] = f"error_{response.status_code}"
    except Exception as e:
        print(f"❌ Fear & Greed error: {e}")
        results["fear_greed"] = "error"
    
    # Test CORS endpoint
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/cors-test",
            timeout=10
        )
        if response.status_code == 200:
            print("✅ CORS test endpoint working")
            results["cors"] = "success"
        else:
            print(f"⚠️  CORS test: {response.status_code}")
            results["cors"] = f"error_{response.status_code}"
    except Exception as e:
        print(f"❌ CORS test error: {e}")
        results["cors"] = "error"
    
    return results

def main():
    """Run all verification tests."""
    print("=" * 80)
    print("  Caria Deployment Verification")
    print("=" * 80)
    print(f"\nAPI Base URL: {API_BASE_URL}")
    print("\nNote: Set API_BASE_URL environment variable to test different deployments")
    
    # Test health check
    health_result = test_health_check()
    
    if health_result.get("status") != "success":
        print("\n❌ Health check failed. Please verify backend is deployed and accessible.")
        sys.exit(1)
    
    # Test authentication
    token = test_auth_flow()
    
    if not token:
        print("\n⚠️  Authentication failed. Some tests will be skipped.")
    else:
        # Test protected endpoints
        protected_results = test_protected_endpoints(token)
    
    # Test public endpoints
    public_results = test_public_endpoints()
    
    # Summary
    print_section("Summary")
    print(f"Health Check: {'✅' if health_result.get('status') == 'success' else '❌'}")
    print(f"Authentication: {'✅' if token else '❌'}")
    
    if token:
        print("\nProtected Endpoints:")
        for endpoint, result in protected_results.items():
            status = "✅" if result == "success" else "⚠️"
            print(f"  {status} {endpoint}: {result}")
    
    print("\nPublic Endpoints:")
    for endpoint, result in public_results.items():
        status = "✅" if result == "success" else "⚠️"
        print(f"  {status} {endpoint}: {result}")
    
    print("\n" + "=" * 80)
    print("Verification complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()


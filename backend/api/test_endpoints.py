#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script simple para testear endpoints de la API."""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("[INFO] Testing /health/live...")
    response = requests.get(f"{BASE_URL}/health/live")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code == 200

def test_cors():
    """Test CORS headers."""
    print("\n[INFO] Testing CORS headers...")
    headers = {"Origin": "http://localhost:3000"}
    response = requests.get(f"{BASE_URL}/health/live", headers=headers)
    print(f"   Status: {response.status_code}")
    print(f"   CORS Headers:")
    for header in ["Access-Control-Allow-Origin", "Access-Control-Allow-Credentials"]:
        value = response.headers.get(header, "NOT PRESENT")
        print(f"      {header}: {value}")
    return "Access-Control-Allow-Origin" in response.headers

def test_register():
    """Test user registration."""
    print("\n[INFO] Testing /api/auth/register...")
    data = {
        "email": "demo3@test.com",
        "username": "demo3",
        "password": "DemoPass123",
        "full_name": "Demo User 3"
    }
    response = requests.post(f"{BASE_URL}/api/auth/register", json=data)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code in [200, 201]

def test_login():
    """Test user login."""
    print("\n[INFO] Testing /api/auth/login...")
    data = {
        "username": "demo3",
        "password": "DemoPass123"
    }
    response = requests.post(f"{BASE_URL}/api/auth/login", json=data)
    print(f"   Status: {response.status_code}")
    result = response.json()
    if response.status_code == 200:
        print(f"   User: {result.get('user', {}).get('username', 'N/A')}")
        print(f"   Response keys: {list(result.keys())}")
        if 'token' in result:
            print(f"   Token keys: {list(result['token'].keys())}")
            print(f"   Access Token: {result['token'].get('access_token', result['token'])[:50]}...")
        return result
    else:
        print(f"   Error: {result}")
        return None

if __name__ == "__main__":
    print("="*60)
    print("TEST DE ENDPOINTS - CARIA API")
    print("="*60)

    # Run tests
    health_ok = test_health()
    cors_ok = test_cors()
    register_ok = test_register()
    login_result = test_login()

    print("\n" + "="*60)
    print("RESULTADOS:")
    print("="*60)
    print(f"   Health endpoint: {'[OK]' if health_ok else '[FAIL]'}")
    print(f"   CORS headers: {'[OK]' if cors_ok else '[FAIL]'}")
    print(f"   Register: {'[OK]' if register_ok else '[FAIL]'}")
    print(f"   Login: {'[OK]' if login_result else '[FAIL]'}")
    print("="*60)

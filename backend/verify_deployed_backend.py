"""
Verify deployed backend on Google Cloud Run.
Tests key endpoints to ensure the backend is accessible and functional.
"""
import requests
import os

# Deployed backend URL
BASE_URL = "https://caria-api-418525923468.us-central1.run.app"

def test_health():
    """Test health/root endpoint."""
    print("\n[INFO] Testing backend health...")
    try:
        resp = requests.get(f"{BASE_URL}/", timeout=10)
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            print("   [OK] Backend is accessible.")
            return True
        else:
            print(f"   [FAIL] Unexpected status: {resp.text[:100]}")
            return False
    except Exception as e:
        print(f"   [FAIL] Exception: {e}")
        return False

def test_login():
    """Test login endpoint."""
    print("\n[INFO] Testing login endpoint...")
    try:
        resp = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={"username": "TBL", "password": "Theolucas7"},
            timeout=10
        )
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            token = data.get("access_token")
            print(f"   [OK] Login successful. Token: {token[:20]}...")
            return token
        else:
            print(f"   [FAIL] Login failed: {resp.text[:100]}")
            return None
    except Exception as e:
        print(f"   [FAIL] Exception: {e}")
        return None

def test_valuation(token):
    """Test valuation endpoint."""
    print("\n[INFO] Testing valuation endpoint...")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(
            f"{BASE_URL}/api/valuation/AAPL",
            headers=headers,
            timeout=30
        )
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   [OK] Valuation retrieved.")
            print(f"   DCF: ${data.get('dcf_value', 'N/A')}")
            print(f"   Graham: ${data.get('graham_value', 'N/A')}")
            print(f"   Lynch: ${data.get('lynch_value', 'N/A')}")
            print(f"   Average: ${data.get('average_value', 'N/A')}")
            return True
        else:
            print(f"   [FAIL] Valuation failed: {resp.text[:100]}")
            return False
    except Exception as e:
        print(f"   [FAIL] Exception: {e}")
        return False

def test_regime(token):
    """Test regime endpoint."""
    print("\n[INFO] Testing regime endpoint...")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(
            f"{BASE_URL}/api/regime/current",
            headers=headers,
            timeout=10
        )
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   [OK] Regime: {data.get('regime', 'N/A')}")
            return True
        else:
            print(f"   [FAIL] Regime failed: {resp.text[:100]}")
            return False
    except Exception as e:
        print(f"   [FAIL] Exception: {e}")
        return False

def test_community_posts(token):
    """Test community posts endpoint."""
    print("\n[INFO] Testing community posts endpoint...")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(
            f"{BASE_URL}/api/community/posts",
            params={"limit": 5},
            headers=headers,
            timeout=10
        )
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   [OK] Retrieved {len(data.get('posts', []))} posts.")
            return True
        else:
            print(f"   [FAIL] Community posts failed: {resp.text[:100]}")
            return False
    except Exception as e:
        print(f"   [FAIL] Exception: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("VERIFYING DEPLOYED BACKEND ON GOOGLE CLOUD RUN")
    print("=" * 60)
    
    results = {}
    
    # Test health
    results['health'] = test_health()
    
    # Test login and get token
    token = test_login()
    results['login'] = token is not None
    
    if token:
        # Test authenticated endpoints
        results['valuation'] = test_valuation(token)
        results['regime'] = test_regime(token)
        results['community'] = test_community_posts(token)
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    for test, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

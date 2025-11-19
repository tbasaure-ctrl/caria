"""
Comprehensive verification of local backend (running on port 8000).
Tests all key endpoints to ensure full functionality.
"""
import requests

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health/root endpoint."""
    print("\n[INFO] Testing backend health...")
    try:
        resp = requests.get(f"{BASE_URL}/", timeout=5)
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
            timeout=5
        )
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            token = data.get("access_token")
            print(f"   [OK] Login successful.")
            return token
        else:
            print(f"   [FAIL] Login failed: {resp.text[:100]}")
            return None
    except Exception as e:
        print(f"   [FAIL] Exception: {e}")
        return None

def test_valuation(token):
    """Test valuation endpoint with combined methods."""
    print("\n[INFO] Testing valuation endpoint (DCF + Graham + Lynch)...")
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
            print(f"       DCF: ${data.get('dcf_value', 'N/A')}")
            print(f"       Graham: ${data.get('graham_value', 'N/A')}")
            print(f"       Lynch: ${data.get('lynch_value', 'N/A')}")
            print(f"       Average: ${data.get('average_value', 'N/A')}")
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

def test_thesis_arena(token):
    """Test thesis arena challenge endpoint."""
    print("\n[INFO] Testing thesis arena challenge...")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.post(
            f"{BASE_URL}/api/thesis/arena/challenge",
            json={
                "thesis": "Apple is undervalued due to strong services growth.",
                "ticker": "AAPL",
                "initial_conviction": 75
            },
            headers=headers,
            timeout=45
        )
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   [OK] Arena challenge created.")
            print(f"       Communities responded: {len(data.get('community_responses', []))}")
            return True
        else:
            print(f"   [FAIL] Arena challenge failed: {resp.text[:100]}")
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
            posts = data.get('posts', [])
            print(f"   [OK] Retrieved {len(posts)} posts.")
            return True
        else:
            print(f"   [FAIL] Community posts failed: {resp.text[:100]}")
            return False
    except Exception as e:
        print(f"   [FAIL] Exception: {e}")
        return False

def test_community_rankings(token):
    """Test community rankings endpoint."""
    print("\n[INFO] Testing community rankings endpoint...")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(
            f"{BASE_URL}/api/community/rankings",
            headers=headers,
            timeout=10
        )
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   [OK] Rankings retrieved.")
            print(f"       Top communities: {len(data.get('top_communities', []))}")
            print(f"       Hot theses: {len(data.get('hot_theses', []))}")
            print(f"       Survivors: {len(data.get('survivors', []))}")
            return True
        else:
            print(f"   [FAIL] Rankings failed: {resp.text[:100]}")
            return False
    except Exception as e:
        print(f"   [FAIL] Exception: {e}")
        return False

def test_reddit_sentiment(token):
    """Test Reddit sentiment endpoint."""
    print("\n[INFO] Testing Reddit sentiment endpoint...")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(
            f"{BASE_URL}/api/reddit/sentiment/AAPL",
            headers=headers,
            timeout=15
        )
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   [OK] Reddit sentiment retrieved.")
            print(f"       Sentiment: {data.get('overall_sentiment', 'N/A')}")
            return True
        else:
            print(f"   [FAIL] Reddit sentiment failed: {resp.text[:100]}")
            return False
    except Exception as e:
        print(f"   [FAIL] Exception: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("COMPREHENSIVE BACKEND VERIFICATION (Local: http://localhost:8000)")
    print("=" * 70)
    
    results = {}
    
    # Test health
    results['Health Check'] = test_health()
    
    # Test login and get token
    token = test_login()
    results['Authentication'] = token is not None
    
    if token:
        # Test all authenticated endpoints
        results['Valuation (DCF+Graham+Lynch)'] = test_valuation(token)
        results['Regime Detection'] = test_regime(token)
        results['Thesis Arena'] = test_thesis_arena(token)
        results['Community Posts'] = test_community_posts(token)
        results['Community Rankings'] = test_community_rankings(token)
        results['Reddit Sentiment'] = test_reddit_sentiment(token)
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\n{'='*70}")
    print(f"TOTAL: {passed}/{total} tests passed ({int(passed/total*100)}%)")
    print(f"{'='*70}")

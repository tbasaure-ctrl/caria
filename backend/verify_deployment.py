import requests
import sys
import time

# Configuration
BASE_URL = "https://caria-production.up.railway.app"  # Update if needed
# BASE_URL = "http://localhost:8000"

def test_endpoint(method, endpoint, data=None, expected_status=[200]):
    url = f"{BASE_URL}{endpoint}"
    print(f"Testing {method} {url}...", end=" ")
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        if response.status_code in expected_status:
            print(f"✅ OK ({response.status_code})")
            return True, response
        else:
            print(f"❌ FAILED ({response.status_code})")
            print(f"   Response: {response.text[:200]}...")
            return False, response
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False, None

def run_verification():
    print(f"=== Caria Backend Verification ({BASE_URL}) ===\n")
    
    # 1. Health Check
    success, _ = test_endpoint("GET", "/health")
    if not success:
        print("⚠️ Critical: Health check failed. Aborting.")
        return

    # 2. Auth Route Check (Expect 401 or 200, NOT 405)
    print("\n--- Authentication ---")
    # Testing with dummy credentials to see if we get 401 (Unauthorized) or 405 (Method Not Allowed)
    success, resp = test_endpoint("POST", "/api/auth/login", 
                                  data={"username": "test", "password": "wrongpassword"},
                                  expected_status=[401])
    
    if resp and resp.status_code == 405:
        print("❌ 405 Method Not Allowed on Login! This confirms the issue.")
    elif resp and resp.status_code == 401:
        print("✅ Login route exists and accepts POST (Got 401 as expected for wrong pwd).")
    
    # 3. Public Endpoints (if any)
    print("\n--- Public Endpoints ---")
    test_endpoint("GET", "/api/market/fear-greed", expected_status=[200, 401]) # Check if protected

    # 4. Protected Endpoints (Check for 401)
    print("\n--- Protected Endpoints (Expect 401) ---")
    test_endpoint("GET", "/api/portfolio/holdings", expected_status=[401])
    test_endpoint("GET", "/api/regime/current", expected_status=[401])

    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1]
    run_verification()


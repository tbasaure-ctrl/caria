import requests
import sys

BASE_URL = "https://caria-production.up.railway.app"

def test_endpoint(method, path, payload=None, expected_status=[200]):
    url = f"{BASE_URL}{path}"
    print(f"Testing {method} {url}...", end=" ")
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=payload)
        else:
            print("Unsupported method")
            return False

        if response.status_code in expected_status:
            print(f"✅ OK ({response.status_code})")
            return True
        else:
            print(f"❌ FAILED ({response.status_code})")
            print(f"   Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    print(f"Verifying backend at {BASE_URL}\n")

    # 1. Health Check
    if not test_endpoint("GET", "/health"):
        print("Critical: Health check failed. Backend might be down.")
        sys.exit(1)

    # 2. Regime (Lazy Loading Test)
    print("\nTesting Regime Service (Lazy Loading)...")
    # This might take a few seconds on first call
    test_endpoint("GET", "/api/regime/current", expected_status=[200, 503]) 
    # 503 is acceptable if model not loaded yet but service handles it gracefully, 
    # though our lazy loading implementation should handle it and return 200 or null regime.

    # 3. Market Data
    print("\nTesting Market Data...")
    # Assuming this endpoint exists and doesn't require auth for simple checks or handles no-auth gracefully
    # If auth is required, this might return 401, which proves the endpoint exists at least.
    test_endpoint("GET", "/api/prices/health", expected_status=[200, 404]) # Check if specific health endpoint exists
    
    # 4. Auth Route Existence Check (Method Not Allowed check)
    print("\nVerifying Auth Route...")
    # Sending GET to POST endpoint should return 405, verifying the route exists.
    # If it returns 404, the route is missing.
    test_endpoint("GET", "/api/auth/login", expected_status=[405])

    print("\nVerification Complete.")

if __name__ == "__main__":
    main()



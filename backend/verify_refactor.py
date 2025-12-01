import requests
import json
import sys

API_BASE_URL = "http://localhost:8000"

def test_endpoint(method, endpoint, payload=None):
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=payload)
        
        print(f"Testing {method} {endpoint}...")
        if response.status_code == 200:
            print(f"✅ Success: {response.status_code}")
            # print(json.dumps(response.json(), indent=2)[:200] + "...") # Print first 200 chars
            return True
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def verify_refactor():
    print("🚀 Starting Refactor Verification...")
    
    # 1. Test Liquidity Status (Hydraulic Stack)
    if not test_endpoint("GET", "/api/liquidity/status"):
        print("⚠️ Liquidity Status check failed. Ensure backend is running.")

    # 2. Test Topology Scan (Cortex)
    if not test_endpoint("GET", "/api/topology/scan"):
        print("⚠️ Topology Scan check failed.")

    # 3. Test Crisis Simulation (Crisis Simulator)
    # Need a valid portfolio payload
    portfolio_payload = {
        "positions": [
            {"ticker": "AAPL", "weight": 0.5, "shares": 100, "cost_basis": 150},
            {"ticker": "GOOGL", "weight": 0.5, "shares": 50, "cost_basis": 2000}
        ]
    }
    # Test with a specific crisis ID
    if not test_endpoint("POST", "/api/simulation/crisis?crisis_id=2008_gfc", portfolio_payload):
        print("⚠️ Crisis Simulation check failed.")

    print("\n🏁 Verification Complete.")

if __name__ == "__main__":
    verify_refactor()

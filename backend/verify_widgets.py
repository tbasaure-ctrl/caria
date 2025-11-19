
import requests
import os
import sys

# Add backend to path
sys.path.append(r"c:\key\wise_adviser_cursor_context\notebooks\backend")

BASE_URL = "http://localhost:8000"

def verify_widgets():
    print("Verifying New Widgets...")
    
    # 1. Login
    print("\n[INFO] Logging in...")
    login_resp = requests.post(
        f"{BASE_URL}/api/auth/login",
        json={"username": "demo3", "password": "DemoPass123"}
    )
    if login_resp.status_code != 200:
        print("   Login failed.")
        return

    token = login_resp.json()["token"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("   [OK] Logged in.")

    # 2. RegimeTestWidget (Using /current as proxy for availability)
    print("\n[INFO] Testing /api/regime/current...")
    try:
        regime_resp = requests.get(
            f"{BASE_URL}/api/regime/current",
            headers=headers
        )
        print(f"   Status: {regime_resp.status_code}")
        if regime_resp.status_code == 200:
            print("   [OK] Regime fetched.")
        else:
            print(f"   [FAIL] Regime failed: {regime_resp.text[:100]}")
    except Exception as e:
        print(f"   [FAIL] Exception: {e}")

    # 3. RankingsWidget
    print("\n[INFO] Testing /api/community/rankings...")
    try:
        rankings_resp = requests.get(
            f"{BASE_URL}/api/community/rankings",
            headers=headers
        )
        print(f"   Status: {rankings_resp.status_code}")
        if rankings_resp.status_code == 200:
            print("   [OK] Rankings fetched.")
        else:
            print(f"   [FAIL] Rankings failed: {rankings_resp.text[:100]}")
    except Exception as e:
        print(f"   [FAIL] Exception: {e}")

    # 4. CommunityFeed
    print("\n[INFO] Testing /api/community/posts (CommunityFeed)...")
    try:
        feed_resp = requests.get(
            f"{BASE_URL}/api/community/posts",
            params={"limit": 5},
            headers=headers
        )
        print(f"   Status: {feed_resp.status_code}")
        if feed_resp.status_code == 200:
            print("   [OK] Feed fetched.")
        else:
            print(f"   [FAIL] Feed failed: {feed_resp.text[:100]}")
    except Exception as e:
        print(f"   [FAIL] Exception: {e}")

    # 5. ModelPortfolioWidget
    print("\n[INFO] Testing /api/portfolio/model/list (ModelPortfolioWidget)...")
    try:
        portfolio_resp = requests.get(
            f"{BASE_URL}/api/portfolio/model/list",
            headers=headers
        )
        print(f"   Status: {portfolio_resp.status_code}")
        if portfolio_resp.status_code == 200:
            print("   [OK] Model portfolio list fetched.")
        else:
            print(f"   [FAIL] Model portfolio list failed: {portfolio_resp.text[:100]}")
    except Exception as e:
        print(f"   [FAIL] Exception: {e}")

if __name__ == "__main__":
    verify_widgets()

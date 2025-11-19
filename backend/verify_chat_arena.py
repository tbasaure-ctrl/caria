
import requests
import os
import sys

# Add backend to path
sys.path.append(r"c:\key\wise_adviser_cursor_context\notebooks\backend")

BASE_URL = "http://localhost:8000"

def verify_chat_arena():
    print("Verifying Chat & Thesis Arena...")
    
    # 1. Login to get token
    print("\n[INFO] Logging in...")
    login_resp = requests.post(
        f"{BASE_URL}/api/auth/login",
        json={"username": "demo3", "password": "DemoPass123"}
    )
    
    if login_resp.status_code != 200:
        # Try registering if login fails
        print("   Login failed, trying to register...")
        requests.post(
            f"{BASE_URL}/api/auth/register",
            json={
                "email": "demo_arena@test.com",
                "username": "demo_arena",
                "password": "DemoPass123",
                "full_name": "Demo Arena"
            }
        )
        login_resp = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={"username": "demo_arena", "password": "DemoPass123"}
        )
        
    if login_resp.status_code != 200:
        print(f"   [FAIL] Login failed: {login_resp.text}")
        return

    token = login_resp.json()["token"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("   [OK] Logged in.")

    # 2. Verify Chat/Challenge (Analysis)
    print("\n[INFO] Testing /api/analysis/challenge...")
    # Note: This might fail if pgvector is not set up, but we check the endpoint existence and basic handling
    try:
        challenge_resp = requests.post(
            f"{BASE_URL}/api/analysis/challenge",
            json={
                "thesis": "Apple will grow due to AI integration.",
                "ticker": "AAPL",
                "top_k": 3
            },
            headers=headers
        )
        print(f"   Status: {challenge_resp.status_code}")
        if challenge_resp.status_code == 200:
            print("   [OK] Challenge endpoint works.")
        elif challenge_resp.status_code == 500 and "pgvector" in challenge_resp.text.lower():
             print("   [WARN] Endpoint reachable but pgvector missing (expected).")
        else:
             print(f"   [FAIL] Unexpected status: {challenge_resp.text[:100]}")
    except Exception as e:
        print(f"   [FAIL] Exception: {e}")

    # 3. Verify Thesis Arena
    print("\n[INFO] Testing /api/thesis/arena/challenge (Create)...")
    try:
        battle_resp = requests.post(
            f"{BASE_URL}/api/thesis/arena/challenge",
            json={
                "thesis": "Tesla is overvalued because competition is increasing.",
                "ticker": "TSLA",
                "initial_conviction": 80
            },
            headers=headers
        )
        print(f"   Status: {battle_resp.status_code}")
        if battle_resp.status_code in [200, 201]:
            print("   [OK] Battle created.")
            battle_data = battle_resp.json()
            battle_id = battle_data.get("arena_id")
            
            # Verify Rounds
            if battle_id:
                print(f"\n[INFO] Testing /api/thesis/arena/respond...")
                round_resp = requests.post(
                    f"{BASE_URL}/api/thesis/arena/respond",
                    json={
                        "thread_id": battle_id,
                        "user_message": "But they have robots."
                    },
                    headers=headers
                )
                print(f"   Status: {round_resp.status_code}")
                if round_resp.status_code == 200:
                    print("   [OK] Round processed.")
                else:
                    print(f"   [FAIL] Round failed: {round_resp.text[:100]}")
        else:
            print(f"   [FAIL] Battle creation failed: {battle_resp.text[:100]}")

    except Exception as e:
        print(f"   [FAIL] Exception: {e}")

if __name__ == "__main__":
    verify_chat_arena()

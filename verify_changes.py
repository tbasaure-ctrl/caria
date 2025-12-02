
import sys
import os
import asyncio
from datetime import datetime

# Add project root and backend to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "backend"))

from api.websocket_chat import _build_socratic_system_prompt
from api.services.alpha_vantage_client import alpha_vantage_client

def test_chat_prompt():
    print("\n--- Testing Chat Prompt ---")
    prompt = _build_socratic_system_prompt("en", ["AAPL"])
    
    checks = [
        "13-Point Framework",
        "Thesis Filter",
        "3-4 Interactions",
        "What They Sell and Who Buys",
        "Catalysts"
    ]
    
    all_passed = True
    for check in checks:
        if check in prompt:
            print(f"[PASS] Found '{check}'")
        else:
            print(f"[FAIL] Missing '{check}'")
            all_passed = False
            
    if all_passed:
        print("[PASS] Chat Prompt Verification Passed")
    else:
        print("[FAIL] Chat Prompt Verification Failed")

def test_alpha_vantage():
    print("\n--- Testing Alpha Vantage Client ---")
    if not alpha_vantage_client.is_available():
        print("[WARN] Alpha Vantage API Key not found. Skipping live test.")
        return

    try:
        overview = alpha_vantage_client.get_company_overview("AAPL")
        if overview and "Symbol" in overview:
            print(f"[PASS] Fetched Overview for {overview['Symbol']}")
            print(f"   P/E: {overview.get('PERatio')}")
            print(f"   EPS: {overview.get('EPS')}")
        else:
            print("[FAIL] Failed to fetch Overview")
            
    except Exception as e:
        print(f"[FAIL] Error testing Alpha Vantage: {e}")

if __name__ == "__main__":
    test_chat_prompt()
    test_alpha_vantage()

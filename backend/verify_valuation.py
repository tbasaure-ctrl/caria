
import sys
import os

# Add backend to path
sys.path.append(r"c:\key\wise_adviser_cursor_context\notebooks\backend")

from api.routes.valuation import _run_simple_dcf, DcfAssumptions
import math

def test_valuation_logic():
    print("Testing Valuation Logic...")

    # Mock Data
    ticker = "TEST"
    price = 150.0
    eps = 5.0
    total_equity = 100000000.0
    shares_outstanding = 1000000.0
    bvps = total_equity / shares_outstanding # 100.0
    fcf = 8000000.0 # 8M
    
    print(f"Inputs: Price=${price}, EPS=${eps}, BVPS=${bvps}, FCF=${fcf}")

    # 1. Test Graham Number
    # Graham = Sqrt(22.5 * EPS * BVPS)
    graham_expected = math.sqrt(22.5 * eps * bvps)
    print(f"Expected Graham: {graham_expected}")
    
    # 2. Test Lynch Value
    # Lynch = EPS * Growth * 100 (PEG=1)
    growth_rate = 0.15 # 15%
    lynch_expected = eps * (growth_rate * 100)
    print(f"Expected Lynch (15% growth): {lynch_expected}")

    # 3. Test DCF
    assumptions = DcfAssumptions(
        discount_rate=0.10,
        high_growth_rate=growth_rate,
        high_growth_years=5,
        fade_years=5,
        terminal_growth_rate=0.03,
        horizon_years=10,
        shares_outstanding=shares_outstanding,
        net_debt=0.0
    )
    
    dcf_result = _run_simple_dcf(fcf, assumptions)
    enterprise_value = dcf_result["enterprise_value"]
    equity_value = enterprise_value - 0.0
    dcf_fair_value = equity_value / shares_outstanding
    print(f"DCF Fair Value: {dcf_fair_value}")

    # 4. Average
    average = (graham_expected + lynch_expected + dcf_fair_value) / 3
    print(f"Average Fair Value: {average}")

    print("\nVerification Complete.")

if __name__ == "__main__":
    test_valuation_logic()

import sys
import os
from pathlib import Path
import pandas as pd
import logging

# Setup paths
current_file = Path(__file__).resolve()
# Script is in notebooks/caria
# Backend is in notebooks/caria/backend
backend_dir = current_file.parent / "backend"
sys.path.insert(0, str(backend_dir))
# Caria lib is in notebooks/caria/caria-lib
caria_lib_dir = current_file.parent / "caria-lib"
sys.path.insert(0, str(caria_lib_dir))

# Mock FactorService
class MockFactorService:
    def screen_companies(self, top_n=50, regime=None, date=None):
        # Return dummy data
        data = []
        for i in range(top_n):
            data.append({
                "ticker": f"TICK{i}",
                "company_name": f"Company {i}",
                "sector": "Technology",
                "factor_scores": {
                    "momentum": 50 + (i % 50), # Varied scores
                    "profitability": 40 + (i % 60),
                    "value": 30 + (i % 70),
                    "growth": 20 + (i % 80)
                }
            })
        return data

# Import AlphaService
try:
    from api.services.alpha_service import AlphaService
    print("Successfully imported AlphaService")
except ImportError as e:
    print(f"Failed to import AlphaService: {e}")
    sys.exit(1)

def test_alpha_picks():
    print("Testing AlphaService...")
    mock_factor_service = MockFactorService()
    service = AlphaService(mock_factor_service)
    
    picks = service.compute_alpha_picks(top_n_candidates=10)
    
    print(f"Generated {len(picks)} picks.")
    
    if len(picks) != 3:
        print("ERROR: Expected 3 picks")
        return False
        
    for pick in picks:
        print(f"Pick: {pick['ticker']} - CAS: {pick['cas_score']}")
        print(f"  Scores: {pick['scores']}")
        print(f"  Explanation: {pick['explanation']}")
        
        if not pick['explanation']:
            print("ERROR: Missing explanation")
            return False
            
    print("SUCCESS: Alpha picks generated correctly.")
    return True

if __name__ == "__main__":
    test_alpha_picks()

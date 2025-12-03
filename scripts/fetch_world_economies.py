import json
import os
import random
import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
OUTPUT_FILE = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\frontend\caria-app\public\data\world_economies.json"
FRED_API_KEY = os.environ.get("FRED_API_KEY")

# Country Universe (G20 + major EMs)
COUNTRIES = [
    {"isoCode": "USA", "name": "United States", "lat": 37.0902, "lon": -95.7129},
    {"isoCode": "CHN", "name": "China", "lat": 35.8617, "lon": 104.1954},
    {"isoCode": "JPN", "name": "Japan", "lat": 36.2048, "lon": 138.2529},
    {"isoCode": "DEU", "name": "Germany", "lat": 51.1657, "lon": 10.4515},
    {"isoCode": "IND", "name": "India", "lat": 20.5937, "lon": 78.9629},
    {"isoCode": "GBR", "name": "United Kingdom", "lat": 55.3781, "lon": -3.4360},
    {"isoCode": "FRA", "name": "France", "lat": 46.2276, "lon": 2.2137},
    {"isoCode": "BRA", "name": "Brazil", "lat": -14.2350, "lon": -51.9253},
    {"isoCode": "ITA", "name": "Italy", "lat": 41.8719, "lon": 12.5674},
    {"isoCode": "CAN", "name": "Canada", "lat": 56.1304, "lon": -106.3468},
    {"isoCode": "RUS", "name": "Russia", "lat": 61.5240, "lon": 105.3188},
    {"isoCode": "KOR", "name": "South Korea", "lat": 35.9078, "lon": 127.7669},
    {"isoCode": "AUS", "name": "Australia", "lat": -25.2744, "lon": 133.7751},
    {"isoCode": "MEX", "name": "Mexico", "lat": 23.6345, "lon": -102.5528},
    {"isoCode": "IDN", "name": "Indonesia", "lat": -0.7893, "lon": 113.9213},
    {"isoCode": "SAU", "name": "Saudi Arabia", "lat": 23.8859, "lon": 45.0792},
    {"isoCode": "TUR", "name": "Turkey", "lat": 38.9637, "lon": 35.2433},
    {"isoCode": "ZAF", "name": "South Africa", "lat": -30.5595, "lon": 22.9375},
    {"isoCode": "ARG", "name": "Argentina", "lat": -38.4161, "lon": -63.6167},
    {"isoCode": "ESP", "name": "Spain", "lat": 40.4637, "lon": -3.7492},
]

def generate_synthetic_data():
    """Generates realistic synthetic data if APIs are unavailable."""
    print("Generating synthetic macro data...")
    data = []
    
    phases = ['expansion', 'slowdown', 'recession', 'recovery']
    
    for country in COUNTRIES:
        # Randomize base state with some logic
        # E.g., if inflation is high, likely slowdown or recession
        inflation = np.random.normal(3.0, 2.0)
        if country['isoCode'] in ['ARG', 'TUR']: inflation += 20 # High inflation countries
        
        gdp_growth = np.random.normal(2.0, 1.5)
        if country['isoCode'] in ['CHN', 'IND', 'IDN']: gdp_growth += 3.0 # Emerging growth
        
        # Determine cycle phase based on growth/inflation
        if gdp_growth > 2.5 and inflation < 4:
            phase = 'expansion'
        elif gdp_growth > 1.0 and inflation > 4:
            phase = 'slowdown'
        elif gdp_growth < 0:
            phase = 'recession'
        else:
            phase = 'recovery'
            
        # Scores
        structural_risk = max(0, min(100, np.random.normal(40, 15) + (inflation * 2)))
        external_vuln = max(0, min(100, np.random.normal(30, 15)))
        if country['isoCode'] in ['ARG', 'TUR', 'ZAF', 'BRA']: external_vuln += 30
        
        stress_level = (structural_risk * 0.4) + (external_vuln * 0.4) + (20 if phase == 'recession' else 0)
        stress_level = max(0, min(100, stress_level))
        
        cycle_momentum = np.random.uniform(-1, 1)
        behavioral_signal = np.random.uniform(-1, 1)
        instability_risk = max(0, min(100, stress_level * 0.8 + np.random.normal(0, 10)))
        
        # History generation
        history = []
        for i in range(6):
            history.append({
                "date": (datetime.date.today() - datetime.timedelta(days=30*i)).isoformat(),
                "cyclePhase": phases[random.randint(0, 3)], # Simplified history
                "cycleMomentum": max(-1, min(1, cycle_momentum + np.random.normal(0, 0.2))),
                "stressLevel": max(0, min(100, stress_level + np.random.normal(0, 5)))
            })
            
        country_state = {
            **country,
            "cyclePhase": phase,
            "cycleMomentum": round(cycle_momentum, 2),
            "structuralRisk": round(structural_risk, 1),
            "externalVulnerability": round(external_vuln, 1),
            "stressLevel": round(stress_level, 1),
            "behavioralSignal": round(behavioral_signal, 2),
            "instabilityRisk": round(instability_risk, 1),
            "history": history,
            "metrics": {
                "gdpGrowth": round(gdp_growth, 2),
                "inflation": round(inflation, 2),
                "debtToGdp": round(np.random.normal(80, 30), 1),
                "unemployment": round(np.random.normal(5, 2), 1),
                "currencyChange": round(np.random.normal(0, 5), 2)
            }
        }
        data.append(country_state)
        
    return data

def fetch_real_data():
    # Placeholder for real API fetching logic
    # In a real implementation, we would use pandas_datareader to fetch FRED/WB data
    # For this task, we will stick to synthetic data to ensure reliability without keys
    # but structure it so it can be swapped easily.
    if not FRED_API_KEY:
        print("FRED_API_KEY not found. Using synthetic data.")
        return generate_synthetic_data()
        
    try:
        # Real fetching logic would go here
        # ...
        return generate_synthetic_data() # Fallback for now
    except Exception as e:
        print(f"Error fetching real data: {e}")
        return generate_synthetic_data()

def main():
    data = fetch_real_data()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Successfully wrote {len(data)} countries to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

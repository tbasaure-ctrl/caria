import json
import os
import time
import requests
import numpy as np
import datetime
from pathlib import Path

# Configuration
OUTPUT_FILE = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\frontend\caria-app\public\data\world_economies.json"
FRED_API_KEY = os.environ.get("FRED_API_KEY")

# Country Universe with extended metadata
COUNTRIES = [
    {"isoCode": "USA", "name": "United States", "lat": 37.0902, "lon": -95.7129, "fred_prefix": "USA", "wb_iso": "US"},
    {"isoCode": "CHN", "name": "China", "lat": 35.8617, "lon": 104.1954, "fred_prefix": "CHN", "wb_iso": "CN"},
    {"isoCode": "JPN", "name": "Japan", "lat": 36.2048, "lon": 138.2529, "fred_prefix": "JPN", "wb_iso": "JP"},
    {"isoCode": "DEU", "name": "Germany", "lat": 51.1657, "lon": 10.4515, "fred_prefix": "DEU", "wb_iso": "DE"},
    {"isoCode": "IND", "name": "India", "lat": 20.5937, "lon": 78.9629, "fred_prefix": "IND", "wb_iso": "IN"},
    {"isoCode": "GBR", "name": "United Kingdom", "lat": 55.3781, "lon": -3.4360, "fred_prefix": "GBR", "wb_iso": "GB"},
    {"isoCode": "FRA", "name": "France", "lat": 46.2276, "lon": 2.2137, "fred_prefix": "FRA", "wb_iso": "FR"},
    {"isoCode": "BRA", "name": "Brazil", "lat": -14.2350, "lon": -51.9253, "fred_prefix": "BRA", "wb_iso": "BR"},
    {"isoCode": "ITA", "name": "Italy", "lat": 41.8719, "lon": 12.5674, "fred_prefix": "ITA", "wb_iso": "IT"},
    {"isoCode": "CAN", "name": "Canada", "lat": 56.1304, "lon": -106.3468, "fred_prefix": "CAN", "wb_iso": "CA"},
    {"isoCode": "RUS", "name": "Russia", "lat": 61.5240, "lon": 105.3188, "fred_prefix": "RUS", "wb_iso": "RU"},
    {"isoCode": "KOR", "name": "South Korea", "lat": 35.9078, "lon": 127.7669, "fred_prefix": "KOR", "wb_iso": "KR"},
    {"isoCode": "AUS", "name": "Australia", "lat": -25.2744, "lon": 133.7751, "fred_prefix": "AUS", "wb_iso": "AU"},
    {"isoCode": "MEX", "name": "Mexico", "lat": 23.6345, "lon": -102.5528, "fred_prefix": "MEX", "wb_iso": "MX"},
    {"isoCode": "IDN", "name": "Indonesia", "lat": -0.7893, "lon": 113.9213, "fred_prefix": "IDN", "wb_iso": "ID"},
    {"isoCode": "SAU", "name": "Saudi Arabia", "lat": 23.8859, "lon": 45.0792, "fred_prefix": "SAU", "wb_iso": "SA"},
    {"isoCode": "TUR", "name": "Turkey", "lat": 38.9637, "lon": 35.2433, "fred_prefix": "TUR", "wb_iso": "TR"},
    {"isoCode": "ZAF", "name": "South Africa", "lat": -30.5595, "lon": 22.9375, "fred_prefix": "ZAF", "wb_iso": "ZA"},
    {"isoCode": "ESP", "name": "Spain", "lat": 40.4637, "lon": -3.7492, "fred_prefix": "ESP", "wb_iso": "ES"},
    {"isoCode": "CHL", "name": "Chile", "lat": -35.6751, "lon": -71.5430, "fred_prefix": "CHL", "wb_iso": "CL"},
]

# --- FRED API HELPERS ---

def get_fred_series_history(series_id, api_key, limit=24):
    """Fetches recent history for trend calculation."""
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&sort_order=desc&limit={limit}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "observations" in data:
                return [float(o["value"]) for o in data["observations"] if o["value"] != "."]
    except:
        pass
    return []

def get_fred_latest(series_id, api_key):
    hist = get_fred_series_history(series_id, api_key, limit=1)
    return hist[0] if hist else None

# --- WORLD BANK API HELPERS ---

def get_world_bank_indicator(country_iso2, indicator):
    """Fetches most recent value from World Bank API."""
    url = f"https://api.worldbank.org/v2/country/{country_iso2}/indicator/{indicator}?format=json&per_page=1"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1 and data[1]:
                val = data[1][0]['value']
                if val is not None:
                    return float(val)
    except Exception as e:
        # print(f"WB Error {country_iso2} {indicator}: {e}")
        pass
    return None

# --- MAIN FETCH LOGIC ---

def fetch_real_data():
    if not FRED_API_KEY:
        print("FRED_API_KEY not found. Using synthetic fallback.")
        return generate_synthetic_data()
        
    print(f"Starting Deep Data Fetch (FRED + World Bank)...")
    data = []
    
    for country in COUNTRIES:
        iso = country["isoCode"]
        wb_iso = country["wb_iso"]
        prefix = country.get("fred_prefix", iso)
        
        print(f"[{iso}] Fetching data...")

        # 1. CYCLE PHASE (Industrial Production)
        # FRED: {ISO}PROINDMISMEI or INDPRO
        ip_id = f"{prefix}PROINDMISMEI"
        if iso == 'USA': ip_id = "INDPRO"
        ip_hist = get_fred_series_history(ip_id, FRED_API_KEY, limit=24)
        
        gdp_growth = 0.0
        cycle_momentum = 0.0
        ip_volatility = 0.0
        
        if len(ip_hist) >= 13:
            # YoY Growth
            gdp_growth = ((ip_hist[0] - ip_hist[12]) / ip_hist[12]) * 100
            # Momentum (Change in YoY)
            prev_growth = ((ip_hist[1] - ip_hist[13]) / ip_hist[13]) * 100
            cycle_momentum = gdp_growth - prev_growth
            # Volatility (Std Dev of MoM changes)
            mom_changes = [((ip_hist[i] - ip_hist[i+1])/ip_hist[i+1]) for i in range(12)]
            ip_volatility = np.std(mom_changes) * 100
        else:
            gdp_growth = np.random.normal(1.5, 1.0) # Fallback

        # 2. INFLATION (CPI)
        cpi_id = f"{prefix}CPIALLMINMEI"
        if iso == 'USA': cpi_id = "CPIAUCSL"
        cpi_hist = get_fred_series_history(cpi_id, FRED_API_KEY, limit=13)
        inflation = 0.0
        if len(cpi_hist) >= 13:
            inflation = ((cpi_hist[0] - cpi_hist[12]) / cpi_hist[12]) * 100
        else:
            inflation = np.random.normal(3.0, 1.0)

        # 3. UNEMPLOYMENT
        unrate_id = f"{prefix}URHARMADSMEI"
        if iso == 'USA': unrate_id = "UNRATE"
        unrate = get_fred_latest(unrate_id, FRED_API_KEY) or 5.0

        # 4. INTEREST RATES (10Y)
        rate_id = f"{prefix}IRLTLT01USM156N"
        if iso == 'USA': rate_id = "GS10"
        rate = get_fred_latest(rate_id, FRED_API_KEY) or 4.0

        # 5. FX RATE & VOLATILITY (vs USD)
        # FRED: DEX{ISO}US or DEXUS{ISO}
        # We need to guess the code or use a mapping.
        # Major pairs: DEXUSEU (Euro), DEXJPUS (Japan), DEXCHUS (China), DEXUSUK (UK)
        fx_id = None
        invert_fx = False
        if iso == 'EUR' or iso in ['DEU', 'FRA', 'ITA', 'ESP']: fx_id = "DEXUSEU"; invert_fx = True
        elif iso == 'JPN': fx_id = "DEXJPUS"
        elif iso == 'CHN': fx_id = "DEXCHUS"
        elif iso == 'GBR': fx_id = "DEXUSUK"; invert_fx = True
        elif iso == 'IND': fx_id = "DEXINUS"
        elif iso == 'BRA': fx_id = "DEXBZUS"
        elif iso == 'CAN': fx_id = "DEXCAUS"
        elif iso == 'KOR': fx_id = "DEXKOUS"
        elif iso == 'MEX': fx_id = "DEXMXUS"
        elif iso == 'ZAF': fx_id = "DEXSFUS"
        
        fx_change_6m = 0.0
        fx_volatility = 0.0
        
        if fx_id:
            fx_hist = get_fred_series_history(fx_id, FRED_API_KEY, limit=130) # ~6 months of daily data
            if len(fx_hist) > 20:
                current = fx_hist[0]
                past = fx_hist[-1]
                if invert_fx: # Rate is USD per Unit, we want Unit per USD usually or just relative strength
                    # If USD/EUR rises, EUR is stronger.
                    # We want "Currency Change vs USD". Positive = Stronger.
                    fx_change_6m = ((current - past) / past) * 100
                else:
                    # Rate is Unit per USD (e.g. JPY/USD).
                    # If JPY/USD rises, JPY is weaker.
                    # So change should be inverted to represent "Currency Strength"
                    fx_change_6m = -((current - past) / past) * 100
                
                # Volatility
                fx_changes = [((fx_hist[i] - fx_hist[i+1])/fx_hist[i+1]) for i in range(len(fx_hist)-1)]
                fx_volatility = np.std(fx_changes) * 100 * np.sqrt(252) # Annualized vol

        # 6. STRUCTURAL DATA (World Bank)
        # Debt to GDP: GC.DOD.TOTL.GD.ZS
        debt_gdp = get_world_bank_indicator(wb_iso, "GC.DOD.TOTL.GD.ZS")
        if debt_gdp is None: debt_gdp = get_world_bank_indicator(wb_iso, "GFDD.DM.01") # Alt: Central gov debt
        if debt_gdp is None: debt_gdp = 60.0 # Fallback

        # Current Account Balance: BN.CAB.XOKA.GD.ZS
        curr_account = get_world_bank_indicator(wb_iso, "BN.CAB.XOKA.GD.ZS") or 0.0

        # --- SCORING LOGIC ---

        # Cycle Phase
        phase = 'stable'
        if gdp_growth > 0.5:
            if cycle_momentum >= -0.5: phase = 'expansion'
            else: phase = 'slowdown'
        elif gdp_growth <= 0.5 and gdp_growth > -1.0:
            if cycle_momentum > 0: phase = 'recovery'
            else: phase = 'slowdown'
        else:
            phase = 'recession'

        # Structural Risk (0-100)
        # Factors: Debt/GDP (>80 bad), Inflation (>3 bad), Twin Deficits (CurrAcc < -3 bad)
        risk_debt = max(0, (debt_gdp - 60) * 0.5)
        risk_inf = max(0, (inflation - 2) * 5)
        risk_def = max(0, (curr_account * -1 - 2) * 5) if curr_account < 0 else 0
        structural_risk = min(100, risk_debt + risk_inf + risk_def)

        # External Vulnerability (0-100)
        # Factors: FX Volatility, FX Depreciation, Current Account Deficit
        vuln_fx_vol = fx_volatility * 2
        vuln_fx_dep = max(0, fx_change_6m * -1 * 2) # If currency fell 10%, score +20
        vuln_ca = max(0, (curr_account * -1) * 3) if curr_account < 0 else 0
        external_vuln = min(100, vuln_fx_vol + vuln_fx_dep + vuln_ca)
        
        # Instability Risk (0-100)
        # Factors: IP Volatility, FX Volatility, High Inflation
        instability = (ip_volatility * 5) + (fx_volatility * 2) + (max(0, inflation - 5) * 5)
        instability_risk = min(100, instability)

        # Stress Level (Composite)
        stress_level = (structural_risk * 0.4) + (external_vuln * 0.4) + (instability_risk * 0.2)
        if phase == 'recession': stress_level += 15
        stress_level = min(100, stress_level)

        country_state = {
            "isoCode": iso,
            "name": country["name"],
            "lat": country["lat"],
            "lon": country["lon"],
            "cyclePhase": phase,
            "cycleMomentum": round(max(-1, min(1, cycle_momentum / 5)), 2),
            "structuralRisk": round(structural_risk, 1),
            "externalVulnerability": round(external_vuln, 1),
            "stressLevel": round(stress_level, 1),
            "behavioralSignal": round(np.random.uniform(-0.5, 0.5), 2), # Placeholder
            "instabilityRisk": round(instability_risk, 1),
            "metrics": {
                "gdpGrowth": round(gdp_growth, 2),
                "inflation": round(inflation, 2),
                "debtToGdp": round(debt_gdp, 1),
                "unemployment": round(unrate, 1),
                "currencyChange": round(fx_change_6m, 2)
            }
        }
        data.append(country_state)
        time.sleep(0.1) # Be nice to APIs

    return data

def generate_synthetic_data():
    # ... (Keep existing synthetic logic as fallback) ...
    print("Generating synthetic macro data...")
    data = []
    phases = ['expansion', 'slowdown', 'recession', 'recovery']
    for country in COUNTRIES:
        inflation = np.random.normal(3.0, 2.0)
        if country['isoCode'] in ['ARG', 'TUR']: inflation += 20
        gdp_growth = np.random.normal(2.0, 1.5)
        if gdp_growth > 2.5 and inflation < 4: phase = 'expansion'
        elif gdp_growth > 1.0 and inflation > 4: phase = 'slowdown'
        elif gdp_growth < 0: phase = 'recession'
        else: phase = 'recovery'
        
        stress_level = max(0, min(100, np.random.normal(40, 15) + (inflation * 2)))
        
        country_state = {
            "isoCode": country['isoCode'],
            "name": country['name'],
            "lat": country['lat'],
            "lon": country['lon'],
            "cyclePhase": phase,
            "cycleMomentum": round(np.random.uniform(-1, 1), 2),
            "structuralRisk": round(stress_level, 1),
            "externalVulnerability": round(np.random.normal(30, 15), 1),
            "stressLevel": round(stress_level, 1),
            "behavioralSignal": round(np.random.uniform(-1, 1), 2),
            "instabilityRisk": round(stress_level * 0.8, 1),
            "metrics": {
                "gdpGrowth": round(gdp_growth, 2),
                "inflation": round(inflation, 2),
                "debtToGdp": round(np.random.normal(80, 30), 1),
                "unemployment": round(np.random.normal(5, 2), 1),
                "currencyChange": 0.0
            }
        }
        data.append(country_state)
    return data

def main():
    data = fetch_real_data()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Successfully wrote {len(data)} countries to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

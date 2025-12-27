import json
import os
import time
import requests
import numpy as np

# --- CONFIGURATION ---
OUTPUT_FILE = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\frontend\caria-app\public\data\world_economies.json"
FRED_API_KEY = os.environ.get("FRED_API_KEY")

# --- EXTENDED COUNTRY UNIVERSE ---
# 'tier': 'DM' (Developed) or 'EM' (Emerging) -> affects growth thresholds
# 'reserve': True/False -> affects debt tolerance
# 'custom_ids': optional dict to override specific FRED series IDs

COUNTRIES = [
    # --- AMERICAS ---
    {
        "isoCode": "USA", "name": "United States",
        "lat": 37.09, "lon": -95.71,
        "fred_prefix": "USA", "wb_iso": "US",
        "tier": "DM", "reserve": True,
        "custom_ids": {
            "ip": "INDPRO",
            "cpi": "CPIAUCSL",
            "rate": "GS10",
            "unrate": "UNRATE",
        },
    },
    {
        "isoCode": "CAN", "name": "Canada",
        "lat": 56.13, "lon": -106.34,
        "fred_prefix": "CAN", "wb_iso": "CA",
        "tier": "DM", "reserve": False,
    },
    {
        "isoCode": "MEX", "name": "Mexico",
        "lat": 23.63, "lon": -102.55,
        "fred_prefix": "MEX", "wb_iso": "MX",
        "tier": "EM", "reserve": False,
    },
    {
        "isoCode": "BRA", "name": "Brazil",
        "lat": -14.23, "lon": -51.92,
        "fred_prefix": "BRA", "wb_iso": "BR",
        "tier": "EM", "reserve": False,
    },
    {
        "isoCode": "CHL", "name": "Chile",
        "lat": -35.67, "lon": -71.54,
        "fred_prefix": "CHL", "wb_iso": "CL",
        "tier": "EM", "reserve": False,
    },

    # --- EUROPE ---
    {
        "isoCode": "DEU", "name": "Germany",
        "lat": 51.16, "lon": 10.45,
        "fred_prefix": "DEU", "wb_iso": "DE",
        "tier": "DM", "reserve": True,
    },
    {
        "isoCode": "GBR", "name": "United Kingdom",
        "lat": 55.37, "lon": -3.43,
        "fred_prefix": "GBR", "wb_iso": "GB",
        "tier": "DM", "reserve": True,
    },
    {
        "isoCode": "FRA", "name": "France",
        "lat": 46.22, "lon": 2.21,
        "fred_prefix": "FRA", "wb_iso": "FR",
        "tier": "DM", "reserve": True,
    },
    {
        "isoCode": "ITA", "name": "Italy",
        "lat": 41.87, "lon": 12.56,
        "fred_prefix": "ITA", "wb_iso": "IT",
        "tier": "DM", "reserve": True,
    },
    {
        "isoCode": "ESP", "name": "Spain",
        "lat": 40.46, "lon": -3.74,
        "fred_prefix": "ESP", "wb_iso": "ES",
        "tier": "DM", "reserve": True,
    },
    {
        "isoCode": "RUS", "name": "Russia",
        "lat": 61.52, "lon": 105.31,
        "fred_prefix": "RUS", "wb_iso": "RU",
        "tier": "EM", "reserve": False,  # API issues & sanctions
    },
    {
        "isoCode": "TUR", "name": "Turkey",
        "lat": 38.96, "lon": 35.24,
        "fred_prefix": "TUR", "wb_iso": "TR",
        "tier": "EM", "reserve": False,
    },

    # --- ASIA / PACIFIC ---
    {
        "isoCode": "CHN", "name": "China",
        "lat": 35.86, "lon": 104.19,
        "fred_prefix": "CHN", "wb_iso": "CN",
        "tier": "EM", "reserve": False,
        # Discount rate as a very rough rate proxy; IP/CPI may still use standard ids
        "custom_ids": {"rate": "INTDSRCNM193N"},
    },
    {
        "isoCode": "JPN", "name": "Japan",
        "lat": 36.20, "lon": 138.25,
        "fred_prefix": "JPN", "wb_iso": "JP",
        "tier": "DM", "reserve": True,
    },
    {
        "isoCode": "IND", "name": "India",
        "lat": 20.59, "lon": 78.96,
        "fred_prefix": "IND", "wb_iso": "IN",
        "tier": "EM", "reserve": False,
    },
    {
        "isoCode": "KOR", "name": "South Korea",
        "lat": 35.90, "lon": 127.76,
        "fred_prefix": "KOR", "wb_iso": "KR",
        "tier": "DM", "reserve": False,
    },
    {
        "isoCode": "AUS", "name": "Australia",
        "lat": -25.27, "lon": 133.77,
        "fred_prefix": "AUS", "wb_iso": "AU",
        "tier": "DM", "reserve": False,
    },
    {
        "isoCode": "IDN", "name": "Indonesia",
        "lat": -0.78, "lon": 113.92,
        "fred_prefix": "IDN", "wb_iso": "ID",
        "tier": "EM", "reserve": False,
    },
    {
        "isoCode": "SAU", "name": "Saudi Arabia",
        "lat": 23.88, "lon": 45.07,
        "fred_prefix": "SAU", "wb_iso": "SA",
        "tier": "EM", "reserve": False,
    },

    # --- AFRICA ---
    {
        "isoCode": "ZAF", "name": "South Africa",
        "lat": -30.55, "lon": 22.93,
        "fred_prefix": "ZAF", "wb_iso": "ZA",
        "tier": "EM", "reserve": False,
    },
]

# --- MANUAL OVERRIDES (Ground Truth Fallbacks) ---
# Used when APIs fail or are known to be unreliable.
MANUAL_OVERRIDES = {
    "CHN": {"metrics": {"gdpGrowth": 4.8, "unemployment": 5.1}},
    "RUS": {"metrics": {"gdpGrowth": 3.0, "inflation": 8.5, "rate": 18.0}},
    "TUR": {"metrics": {"inflation": 45.0, "rate": 50.0}},
    "SAU": {"metrics": {"gdpGrowth": 1.5, "unemployment": 5.5}},
    # "ARG": {"metrics": {"inflation": 150.0, "gdpGrowth": -2.0}},  # if you later add Argentina
}

# --- API HELPERS ---


def get_fred_series_history(series_id, api_key, limit=24):
    """Fetches a recent history of a FRED series. Returns [] on failure."""
    if not api_key or not series_id:
        return []

    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}"
        f"&file_type=json&sort_order=desc&limit={limit}"
    )
    try:
        response = requests.get(url, timeout=4)
        if response.status_code == 200:
            data = response.json()
            if "observations" in data:
                return [
                    float(o["value"])
                    for o in data["observations"]
                    if o.get("value") not in (".", None)
                ]
    except Exception:
        pass
    return []


def get_fred_latest(series_id, api_key):
    hist = get_fred_series_history(series_id, api_key, limit=1)
    return hist[0] if hist else None


def get_world_bank_indicator(country_iso2, indicator):
    """Fetches most recent WB indicator value. Returns None on failure."""
    url = (
        f"https://api.worldbank.org/v2/country/{country_iso2}/indicator/"
        f"{indicator}?format=json&per_page=1"
    )
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1 and data[1] and data[1][0]["value"] is not None:
                return float(data[1][0]["value"])
    except Exception:
        pass
    return None


# --- CORE LOGIC ---


def fetch_real_data():
    if not FRED_API_KEY:
        print("⚠️  FRED_API_KEY not found in environment. Using synthetic data.")
        return generate_synthetic_data()

    print(f"🌍 Starting Macro Data Fetch for {len(COUNTRIES)} economies...")
    data = []

    for country in COUNTRIES:
        iso = country["isoCode"]
        wb_iso = country["wb_iso"]
        prefix = country.get("fred_prefix", iso)
        custom_ids = country.get("custom_ids", {})

        manual_data = MANUAL_OVERRIDES.get(iso, {}).get("metrics", {})
        data_quality = "real"  # will downgrade to "mixed" if we use manual/synthetic

        print(f"   📍 Processing {iso}...", end="\r")

        # ---------------------------
        # 1. Industrial Production -> growth & momentum
        # ---------------------------
        ip_id = custom_ids.get("ip", f"{prefix}PROINDMISMEI")
        ip_hist = get_fred_series_history(ip_id, FRED_API_KEY, limit=24)

        gdp_growth = 0.0
        cycle_momentum = 0.0
        ip_volatility = 0.0

        if len(ip_hist) >= 13 and ip_hist[12] != 0 and ip_hist[13] != 0:
            gdp_growth = ((ip_hist[0] - ip_hist[12]) / ip_hist[12]) * 100.0
            prev_growth = ((ip_hist[1] - ip_hist[13]) / ip_hist[13]) * 100.0
            cycle_momentum = gdp_growth - prev_growth

            mom_changes = [
                (ip_hist[i] - ip_hist[i + 1]) / ip_hist[i + 1]
                for i in range(12)
                if ip_hist[i + 1] != 0
            ]
            if mom_changes:
                ip_volatility = np.std(mom_changes) * 100.0
        else:
            if "gdpGrowth" in manual_data:
                gdp_growth = manual_data["gdpGrowth"]
                cycle_momentum = 0.1
                data_quality = "mixed"
            else:
                # Synthetic fallback by tier
                gdp_growth = np.random.normal(
                    2.0 if country["tier"] == "DM" else 4.0, 1.0
                )
                cycle_momentum = np.random.normal(0.0, 0.5)
                data_quality = "mixed"

        # ---------------------------
        # 2. Inflation (CPI)
        # ---------------------------
        cpi_id = custom_ids.get("cpi", f"{prefix}CPIALLMINMEI")
        cpi_hist = get_fred_series_history(cpi_id, FRED_API_KEY, limit=13)

        if len(cpi_hist) >= 13 and cpi_hist[12] != 0:
            inflation = ((cpi_hist[0] - cpi_hist[12]) / cpi_hist[12]) * 100.0
        elif "inflation" in manual_data:
            inflation = manual_data["inflation"]
            data_quality = "mixed"
        else:
            inflation = np.random.normal(2.5, 0.5)
            data_quality = "mixed"

        # ---------------------------
        # 3. Unemployment & Interest rate
        # ---------------------------
        unrate_id = custom_ids.get("unrate", f"{prefix}URHARMADSMEI")
        unrate = get_fred_latest(unrate_id, FRED_API_KEY)
        if unrate is None:
            unrate = manual_data.get("unemployment", 5.0)
            data_quality = "mixed"

        rate_id = custom_ids.get("rate", f"{prefix}IRLTLT01USM156N")
        rate = get_fred_latest(rate_id, FRED_API_KEY)
        if rate is None:
            rate = manual_data.get("rate", 4.0)
            data_quality = "mixed"

        # ---------------------------
        # 4. FX Market (vs USD) – conservative mapping
        # ---------------------------
        fx_id = None
        invert_fx = False  # False = local per USD, True = USD per local

        if iso in ["DEU", "FRA", "ITA", "ESP"]:
            # Euro area
            fx_id = "DEXUSEU"
            invert_fx = True
        elif iso == "GBR":
            fx_id = "DEXUSUK"
            invert_fx = True
        elif iso == "JPN":
            fx_id = "DEXJPUS"
            invert_fx = False  # JPY per USD
        elif iso == "CHN":
            fx_id = "DEXCHUS"  # CNY per USD
            invert_fx = False
        elif iso == "IND":
            fx_id = "DEXINUS"  # INR per USD
            invert_fx = False
        elif iso == "BRA":
            fx_id = "DEXBZUS"  # BRL per USD
            invert_fx = False
        elif iso == "CAN":
            fx_id = "DEXCAUS"  # CAD per USD
            invert_fx = False
        elif iso == "KOR":
            fx_id = "DEXKOUS"  # KRW per USD
            invert_fx = False
        elif iso == "MEX":
            fx_id = "DEXMXUS"  # MXN per USD
            invert_fx = False
        elif iso == "ZAF":
            fx_id = "DEXSFUS"  # ZAR per USD
            invert_fx = False
        # NOTE: AUS & CHL intentionally left without FX mapping until verified

        fx_change_6m = 0.0  # positive = local stronger vs USD
        fx_volatility = 0.0

        if fx_id:
            fx_hist = get_fred_series_history(fx_id, FRED_API_KEY, limit=130)
            if len(fx_hist) > 20 and fx_hist[-1] != 0:
                current = fx_hist[0]
                past = fx_hist[-1]

                pct_change = ((current - past) / past) * 100.0
                # If series is USD/local, higher value = stronger local
                fx_change_6m = pct_change if invert_fx else -pct_change

                fx_changes = [
                    (fx_hist[i] - fx_hist[i + 1]) / fx_hist[i + 1]
                    for i in range(len(fx_hist) - 1)
                    if fx_hist[i + 1] != 0
                ]
                if fx_changes:
                    # Daily changes → multiply by ~sqrt(252) = ~15.87
                    fx_volatility = np.std(fx_changes) * 100.0 * 15.87

        # ---------------------------
        # 5. Structural (World Bank)
        # ---------------------------
        debt_gdp = manual_data.get("debtToGdp")
        if debt_gdp is None:
            # General gov debt (% of GDP)
            debt_gdp = get_world_bank_indicator(
                wb_iso, "GC.DOD.TOTL.GD.ZS"
            ) or 60.0
            if debt_gdp == 60.0:
                data_quality = "mixed"

        curr_account = get_world_bank_indicator(
            wb_iso, "BN.CAB.XOKA.GD.ZS"
        )
        if curr_account is None:
            curr_account = 0.0
            data_quality = "mixed"

        # ---------------------------
        # 6. Scoring Engine
        # ---------------------------

        # A. Cycle Phase (Adjusted for Tier)
        # EM needs higher growth to be called "expansion"
        growth_threshold = 4.0 if country["tier"] == "EM" else 1.5
        recession_threshold = 1.0 if country["tier"] == "EM" else 0.0

        phase = "slowdown"  # default -> keeps the set within {slowdown,expansion,recession,recovery}
        if gdp_growth < recession_threshold:
            phase = "recession"
        elif gdp_growth > growth_threshold:
            # strong growth, check momentum
            phase = "expansion" if cycle_momentum >= -0.5 else "slowdown"
        elif gdp_growth > recession_threshold:
            # modest positive growth
            phase = "recovery" if cycle_momentum > 0 else "slowdown"

        # B. Structural Risk (0–100) – adjusted for reserve currencies
        debt_limit = 180.0 if country["reserve"] else 70.0
        risk_debt = max(0.0, (debt_gdp - debt_limit) * 0.5)
        risk_inf = max(0.0, (inflation - 3.0) * 5.0)
        risk_def = (
            max(0.0, (curr_account * -1.0 - 3.0) * 5.0)
            if curr_account < 0
            else 0.0
        )

        structural_risk = min(100.0, risk_debt + risk_inf + risk_def)

        # C. External Vulnerability (0–100)
        # EM more sensitive to FX swings
        fx_sensitivity = 2.0 if country["tier"] == "EM" else 1.0
        vuln_fx = (fx_volatility * 2.0 * fx_sensitivity) + max(
            0.0, -fx_change_6m * 3.0
        )
        vuln_ca = max(0.0, curr_account * -1.0 * 2.0)
        external_vuln = min(100.0, vuln_fx + vuln_ca)

        # D. Instability & Stress
        instability = (ip_volatility * 3.0) + max(0.0, inflation - 5.0) * 8.0
        instability_risk = min(100.0, instability)

        stress_level = (
            structural_risk * 0.4
            + external_vuln * 0.3
            + instability_risk * 0.3
        )
        if phase == "recession":
            stress_level += 20.0
        stress_level = min(100.0, stress_level)

        # E. Behavioral Signal (-1 to +1)
        sentiment_score = (
            cycle_momentum * 0.6
            + fx_change_6m * 0.1
            - max(0.0, inflation - 3.0) * 0.1
        )
        behavioral_signal = max(-1.0, min(1.0, sentiment_score))

        # Normalize momentum to [-1, 1]
        norm_momentum = max(-1.0, min(1.0, cycle_momentum / 5.0))

        # Final object
        country_state = {
            "isoCode": iso,
            "name": country["name"],
            "lat": country["lat"],
            "lon": country["lon"],
            "tier": country["tier"],
            "dataQuality": data_quality,  # "real", "mixed", or "synthetic" (in synthetic generator)
            "cyclePhase": phase,
            "cycleMomentum": round(norm_momentum, 2),
            "structuralRisk": round(structural_risk, 1),
            "externalVulnerability": round(external_vuln, 1),
            "stressLevel": round(stress_level, 1),
            "behavioralSignal": round(behavioral_signal, 2),
            "instabilityRisk": round(instability_risk, 1),
            "metrics": {
                "gdpGrowth": round(gdp_growth, 2),
                "inflation": round(inflation, 2),
                "debtToGdp": round(debt_gdp, 1),
                "unemployment": round(unrate, 1),
                "currencyChange": round(fx_change_6m, 2),
                "interestRate": round(rate, 2),
            },
        }

        data.append(country_state)
        time.sleep(0.05)  # avoid hammering APIs

    return data


def generate_synthetic_data():
    """Used if FRED_API_KEY is missing entirely. Generates plausible demo data."""
    print("🧪 Generating synthetic macro data for demo mode...")
    data = []
    for c in COUNTRIES:
        # Simple synthetic engine by tier
        base_growth = 2.0 if c["tier"] == "DM" else 4.0
        gdp = np.random.normal(base_growth, 1.0)
        infl = np.random.normal(3.0, 1.5)
        unemp = np.random.normal(5.0, 2.0)
        debt = np.random.normal(80.0 if c["tier"] == "DM" else 60.0, 25.0)
        rate = np.random.normal(4.0, 1.0)

        # Phase
        if gdp < 0:
            phase = "recession"
        elif gdp > (3.5 if c["tier"] == "EM" else 2.0):
            phase = "expansion"
        elif gdp > 0:
            phase = "recovery"
        else:
            phase = "slowdown"

        structural_risk = max(
            0.0,
            min(
                100.0,
                np.random.normal(40.0, 15.0) + max(0.0, infl - 3.0) * 4.0,
            ),
        )
        external_vuln = max(
            0.0, min(100.0, np.random.normal(30.0, 15.0))
        )
        instability = structural_risk * 0.8
        stress = max(
            0.0, min(100.0, (structural_risk * 0.4) + (external_vuln * 0.3) + (instability * 0.3))
        )

        country_state = {
            "isoCode": c["isoCode"],
            "name": c["name"],
            "lat": c["lat"],
            "lon": c["lon"],
            "tier": c["tier"],
            "dataQuality": "synthetic",
            "cyclePhase": phase,
            "cycleMomentum": round(np.random.uniform(-1, 1), 2),
            "structuralRisk": round(structural_risk, 1),
            "externalVulnerability": round(external_vuln, 1),
            "stressLevel": round(stress, 1),
            "behavioralSignal": round(np.random.uniform(-1, 1), 2),
            "instabilityRisk": round(instability, 1),
            "metrics": {
                "gdpGrowth": round(gdp, 2),
                "inflation": round(infl, 2),
                "debtToGdp": round(debt, 1),
                "unemployment": round(unemp, 1),
                "currencyChange": 0.0,
                "interestRate": round(rate, 2),
            },
        }
        data.append(country_state)

    return data


def main():
    data = fetch_real_data()
    if not data:
        print("No data fetched. Aborting.")
        return

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Successfully wrote {len(data)} economies to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

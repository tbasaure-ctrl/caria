import pandas as pd
import numpy as np
import os
try:
    from fredapi import Fred
except ImportError:
    Fred = None

class LiquidityEngine:
    def __init__(self, api_key=None):
        # Use provided key or hardcoded verified key
        self.api_key = api_key or os.getenv("FRED_API_KEY") or "4b90ca15ff28cfec137179c22fd8246d"
        
        if not self.api_key:
            print("WARNING: No FRED API Key provided. LiquidityEngine will not be able to fetch new data.")
            self.fred = None
        else:
            if Fred:
                try:
                    self.fred = Fred(api_key=self.api_key)
                except Exception as e:
                    print(f"Error initializing Fred client: {e}")
                    self.fred = None
            else:
                print("WARNING: 'fredapi' library not installed.")
                self.fred = None

    def fetch_data(self):
        """Fetches necessary series from FRED."""
        if not self.fred:
            raise RuntimeError("Cannot fetch data: No FRED Client (Key missing or lib missing).")
        
        print("Fetching Liquidity Data from FRED...")
        # WALCL: Fed Total Assets
        # WTREGEN: Treasury General Account
        # RRPONTSYD: Reverse Repo (Overnight)
        # T10Y2Y: 10Y-2Y Yield Spread
        
        series_ids = {
            'WALCL': 'Fed Assets',
            'WTREGEN': 'TGA',
            'RRPONTSYD': 'RRP',
            'T10Y2Y': 'Yield Curve'
        }
        
        data = {}
        for sid, name in series_ids.items():
            try:
                s = self.fred.get_series(sid)
                data[sid] = s
                print(f"  Fetched {name} ({sid}): {len(s)} records")
            except Exception as e:
                print(f"  ERROR fetching {sid}: {e}")
        
        df = pd.DataFrame(data)
        df = df.fillna(method='ffill') # Forward fill daily/weekly mismatches
        return df

    def calculate_signals(self, df):
        """Calculates Net Liquidity and Hydraulic Score using Rule-Based Logic."""
        # Ensure cols exist
        required = ['WALCL', 'WTREGEN', 'RRPONTSYD', 'T10Y2Y']
        if not all(col in df.columns for col in required):
            print(f"Missing columns for Net Liquidity. Have: {df.columns.tolist()}")
            return None

        # 1. Net Liquidity = Fed Assets - TGA - RRP
        # Normalize to Billions (WALCL is Millions)
        fed_assets_bn = df['WALCL'] / 1000
        tga_bn = df['WTREGEN']
        rrp_bn = df['RRPONTSYD']
        
        df['net_liquidity'] = fed_assets_bn - tga_bn - rrp_bn
        
        # 2. Rate of Change (4-week / 20-day)
        df['net_liq_chg_4w'] = df['net_liquidity'].pct_change(periods=20)
        
        # 3. Hydraulic Score Calculation (Rule-Based)
        # Logic provided by user:
        # Base Score: 50
        # Liquidity Factor:
        # > 1% -> +30
        # > 0% -> +10
        # < -1% -> -30
        # < 0% -> -10
        # Yield Curve Factor:
        # < -0.5 -> -20
        # < 0 -> -10
        # > 0.5 -> +10
        
        def calculate_score(row):
            score = 50
            roc = row['net_liq_chg_4w']
            curve = row['T10Y2Y']
            
            if pd.isna(roc) or pd.isna(curve):
                return 50 # Default neutral if missing data
            
            # Liquidity Rules
            if roc > 0.01: score += 30
            elif roc > 0: score += 10
            elif roc < -0.01: score -= 30
            elif roc < 0: score -= 10
            
            # Curve Rules
            if curve < -0.5: score -= 20
            elif curve < 0: score -= 10
            elif curve > 0.5: score += 10
            
            return max(0, min(100, score))

        df['hydraulic_score'] = df.apply(calculate_score, axis=1)
        
        # Determine State
        conditions = [
            (df['hydraulic_score'] >= 60),
            (df['hydraulic_score'] <= 40)
        ]
        choices = ['EXPANSION', 'CONTRACTION']
        df['liquidity_state'] = np.select(conditions, choices, default='NEUTRAL')
        
        return df

    def execute(self):
        """Main execution method."""
        df = self.fetch_data()
        if df is not None:
            df = self.calculate_signals(df)
            current_state = df.iloc[-1]
            return {
                'score': current_state['hydraulic_score'],
                'state': current_state['liquidity_state'],
                'net_liquidity': current_state['net_liquidity'],
                'yield_curve': current_state['T10Y2Y']
            }
        return None

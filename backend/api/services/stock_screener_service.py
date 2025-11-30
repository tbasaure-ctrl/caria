import os
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

LOGGER = logging.getLogger("caria.api.services.stock_screener")

class StockScreenerService:
    def __init__(self):
        self.api_key = os.getenv('FMP_API_KEY', '').strip()
        self.base_url = 'https://financialmodelingprep.com/api/v3'
        self.db_url = os.getenv('DATABASE_URL') or os.getenv('NEON_DB_URL')

    def _get_db_connection(self):
        if not self.db_url:
            password = os.getenv("POSTGRES_PASSWORD")
            if not password:
                raise RuntimeError("DB Connection params missing")
            return psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                user=os.getenv("POSTGRES_USER", "caria_user"),
                password=password,
                database=os.getenv("POSTGRES_DB", "caria"),
            )
        return psycopg2.connect(self.db_url)

    def fetch_fmp(self, endpoint: str, params: Dict = None) -> List[Dict]:
        if params is None:
            params = {}
        params['apikey'] = self.api_key
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            LOGGER.error(f"FMP fetch error {endpoint}: {e}")
            return []

    def get_initial_screeners(self) -> List[str]:
        params = {
            'marketCapMoreThan': 10000000000,
            'volumeMoreThan': 1000000,
            'limit': 200,
            'isActivelyTrading': 'true'
        }
        data = self.fetch_fmp('/stock-screener', params)
        return [item['symbol'] for item in data]

    def get_ratios_ttm(self, symbol: str) -> Dict:
        data = self.fetch_fmp(f'/ratios-ttm/{symbol}')
        return data[0] if data else {}

    def get_ratios_historical(self, symbol: str) -> List[Dict]:
        return self.fetch_fmp(f'/ratios/{symbol}', {'limit': 8})

    def get_key_metrics_ttm(self, symbol: str) -> Dict:
        data = self.fetch_fmp(f'/key-metrics-ttm/{symbol}')
        return data[0] if data else {}

    def get_historical_price(self, symbol: str, days: int = 180) -> List[Dict]:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        data = self.fetch_fmp(f'/historical-price-full/{symbol}', {'from': start_date, 'to': end_date})
        if isinstance(data, dict):
            return data.get('historical', [])
        return []

    def get_insider_trading(self, symbol: str) -> List[Dict]:
        three_months_ago = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        data = self.fetch_fmp('/insider-trading', {'symbol': symbol, 'limit': 50}) 
        buys = []
        for t in data:
            t_date = t.get('transactionDate', '')
            if t_date >= three_months_ago and t.get('transactionType') == 'P-Purchase':
                 buys.append(t)
        return buys

    def get_sec_filings(self, cik: str) -> List[Dict]:
        if not cik: return []
        data = self.fetch_fmp(f'/sec_filings/{cik}', {'type': '8-K', 'limit': 20})
        return data

    def get_profile(self, symbol: str) -> Dict:
        data = self.fetch_fmp(f'/profile/{symbol}')
        return data[0] if data else {}

    def calculate_quality_score(self, symbol: str) -> float:
        ratios_ttm = self.get_ratios_ttm(symbol)
        ratios_hist = self.get_ratios_historical(symbol)
        key_metrics = self.get_key_metrics_ttm(symbol)
        
        if not ratios_ttm or len(ratios_hist) < 2:
            return 0.0
        
        roic_current = ratios_ttm.get('returnOnInvestedCapitalTTM', 0) or 0
        if roic_current is None: roic_current = 0

        roic_1y_ago = roic_current
        if len(ratios_hist) > 4:
             val = ratios_hist[4].get('returnOnInvestedCapital')
             if val is not None: roic_1y_ago = val
        
        roic_delta = roic_current - roic_1y_ago
        
        gross_margin = ratios_ttm.get('grossProfitMarginTTM', 0) or 0
        sga = ratios_ttm.get('sellingGeneralAndAdminExpensesTTM', 0) or ratios_ttm.get('sellingGeneralAndAdministrativeExpensesTTM', 0) 
        if not sga: sga = 1
        
        eficiencia = gross_margin / sga
        roic_adj = roic_delta * eficiencia
        
        fcf_current = key_metrics.get('freeCashFlowTTM', 0) or 0
        fcf_per_share_hist = key_metrics.get('freeCashFlowPerShareTTM', 0) or 0
        shares = key_metrics.get('weightedAverageShsOutTTM', 1) or 1
        fcf_1y_ago_est = fcf_per_share_hist * shares
        
        fcf_growth = 0
        if fcf_1y_ago_est != 0:
            fcf_growth = (fcf_current - fcf_1y_ago_est) / abs(fcf_1y_ago_est)
            
        quality_raw = roic_adj + (roic_delta if roic_delta else 0) + fcf_growth
        return quality_raw

    def calculate_valuation_score(self, symbol: str) -> float:
        ratios_ttm = self.get_ratios_ttm(symbol)
        if not ratios_ttm:
            return 0.0
        
        score = 0
        pe = ratios_ttm.get('priceEarningsRatioTTM', 50) or 50
        if pe > 0: score += (100 / pe)
        
        pfcf = ratios_ttm.get('priceToFreeCashFlowsRatioTTM', 50) or 50
        if pfcf > 0: score += (100 / pfcf)
        
        return score

    def calculate_momentum_score(self, symbol: str) -> float:
        hist = self.get_historical_price(symbol, 60)
        if not hist or len(hist) < 30: return 0.0
        
        df = pd.DataFrame(hist)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        recent_vol = df['volume'].iloc[-5:].mean()
        avg_vol = df['volume'].mean()
        vol_spike = recent_vol / avg_vol if avg_vol > 0 else 1
        
        price_now = df['close'].iloc[-1]
        price_start = df['close'].iloc[0]
        ret = (price_now - price_start) / price_start
        
        return (vol_spike * 20) + (ret * 100)

    def calculate_catalyst_score(self, symbol: str) -> float:
        insiders = self.get_insider_trading(symbol)
        return len(insiders) * 10.0

    def calculate_risk_penalty(self, symbol: str) -> float:
        ratios = self.get_ratios_ttm(symbol)
        debt_equity = ratios.get('debtEquityRatioTTM', 0) or 0
        if debt_equity > 2.0: return -20.0
        if debt_equity > 1.0: return -10.0
        return 0.0

    def run_screen(self):
        # Increase candidate pool from 5 to 30 for better Hidden Gems discovery
        candidates = self.get_initial_screeners()[:30]
        
        if not candidates:
            LOGGER.warning("No candidates found from initial screeners")
            return [{
                "symbol": "N/A",
                "name": "No stocks found",
                "sector": "N/A",
                "quality": 0, "valuation": 0, "momentum": 0, "catalyst": 0, "risk": 0,
                "c_score": 0,
                "error": "No candidates matched screening criteria. Try adjusting filters or check FMP API."
            }]
        
        results = []
        
        for ticker in candidates:
            try:
                q = self.calculate_quality_score(ticker)
                v = self.calculate_valuation_score(ticker)
                m = self.calculate_momentum_score(ticker)
                c = self.calculate_catalyst_score(ticker)
                r = self.calculate_risk_penalty(ticker)
                
                profile = self.get_profile(ticker)
                name = profile.get('companyName', ticker)
                sector = profile.get('sector', 'Unknown')
                
                total = (q * 40) + (v * 25) + (m * 20) + (c * 15) + r
                results.append({
                    "symbol": ticker,
                    "name": name,
                    "sector": sector,
                    "quality": q, "valuation": v, "momentum": m, "catalyst": c, "risk": r,
                    "c_score": total
                })
            except Exception as e:
                LOGGER.error(f"Error screening {ticker}: {e}")
                continue
                
        if not results:
            LOGGER.warning("No valid results after screening all candidates")
            return [{
                "symbol": "N/A",
                "name": "No valid stocks found",
                "sector": "N/A",
                "quality": 0, "valuation": 0, "momentum": 0, "catalyst": 0, "risk": 0,
                "c_score": 0,
                "error": "All candidates failed screening. Check API connectivity or data quality."
            }]
                
        results.sort(key=lambda x: x['c_score'], reverse=True)
        top_results = results[:10]
        
        self._save_results(top_results)
        return top_results

    def _save_results(self, results):
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS screening_results (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    symbol VARCHAR(10),
                    quality_score DECIMAL,
                    valuation_score DECIMAL,
                    momentum_score DECIMAL,
                    catalyst_score DECIMAL,
                    risk_penalty DECIMAL,
                    c_score DECIMAL,
                    rank INTEGER
                );
            """)
            
            timestamp = datetime.now()
            for i, res in enumerate(results):
                rank = i + 1
                cur.execute("""
                    INSERT INTO screening_results 
                    (timestamp, symbol, quality_score, valuation_score, momentum_score, catalyst_score, risk_penalty, c_score, rank)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (timestamp, res['symbol'], res['quality'], res['valuation'], res['momentum'], res['catalyst'], res['risk'], res['c_score'], rank))
            
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            LOGGER.error(f"DB Save failed: {e}")

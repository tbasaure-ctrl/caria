import requests
import pandas as pd
import numpy as np

BASE_URL = "https://financialmodelingprep.com/api/v3"

class CariaScoreEngine:
    def __init__(self, api_key, symbols):
        self.api_key = api_key
        self.symbols = symbols

    def _fetch(self, endpoint, symbol, params=""):
        url = f"{BASE_URL}/{endpoint}/{symbol}?apikey={self.api_key}{params}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data if data else []
        except Exception:
            return []

    def _normalize(self, value, min_val, max_val):
        if value is None: return 50
        clamped = max(min_val, min(value, max_val))
        return ((clamped - min_val) / (max_val - min_val)) * 100

    def get_financial_data(self, symbol):
        # Descarga optimizada
        ratios_ttm = self._fetch("ratios-ttm", symbol)
        key_metrics = self._fetch("key-metrics-ttm", symbol)
        income_stmt = self._fetch("income-statement", symbol, "&limit=2")
        ratios_hist = self._fetch("ratios", symbol, "&limit=5")
        quote = self._fetch("quote", symbol)
        price_change = self._fetch("stock-price-change", symbol)
        insiders = self._fetch("insider-trading", symbol, "&limit=10")
        # AGREGAR: Descargar perfil para tener Sector y Nombre correcto
        profile = self._fetch("profile", symbol)

        return {
            "ratios_ttm": ratios_ttm[0] if ratios_ttm else {},
            "key_metrics_ttm": key_metrics[0] if key_metrics else {},
            "income_stmt": income_stmt,
            "ratios_hist": ratios_hist,
            "quote": quote[0] if quote else {},
            "price_change": price_change[0] if price_change else {},
            "insider_trading": insiders,
            "profile": profile[0] if profile else {}  # Nuevo campo
        }

    # --- MOTORES DE CÁLCULO (Resumidos para no ocupar tanto espacio, usan tu lógica) ---
    
    def analyze_quality(self, data):
        stmt = data.get("income_stmt", [{}, {}])
        metrics = data.get("key_metrics_ttm", {})
        
        try:
            gp = stmt[0].get("grossProfit", 0)
            sga = stmt[0].get("sellingGeneralAndAdministrativeExpenses", 1)
            eff = gp / sga if sga != 0 else 0
        except: eff = 1
        
        roic = metrics.get("roicTTM", 0) * 100
        return (self._normalize(roic, 0, 30) * 0.6) + (self._normalize(eff, 1, 4) * 0.4)

    def analyze_valuation(self, data):
        r_ttm = data.get("ratios_ttm", {})
        r_hist = data.get("ratios_hist", [])
        ev_sales = r_ttm.get("enterpriseValueMultipleTTM", 0)
        
        avg_5y = np.mean([x.get("enterpriseValueMultiple", 0) for x in r_hist]) if r_hist else ev_sales
        ratio = ev_sales / avg_5y if avg_5y > 0 else 1
        return self._normalize(2.0 - ratio, 0.5, 1.5)

    def analyze_momentum(self, data):
        quote = data.get("quote", {})
        pch = data.get("price_change", {})
        
        vol_ratio = quote.get("volume", 0) / quote.get("avgVolume", 1) if quote.get("avgVolume") else 1
        perf_6m = pch.get("6M", 0)
        return (self._normalize(vol_ratio, 0.8, 2.0) * 0.4) + (self._normalize(perf_6m, -10, 30) * 0.6)

    def analyze_catalysts(self, data):
        insiders = data.get("insider_trading", [])
        points = 0
        for tx in insiders[:5]:
            t = tx.get("transactionType", "").lower()
            if "buy" in t or "purchase" in t: points += 20
            elif "sale" in t: points -= 10
        return self._normalize(points, 0, 100)

    def analyze_risk(self, data):
        de = data.get("ratios_ttm", {}).get("debtEquityRatioTTM", 0)
        score = 100
        if de > 2.5: score -= 50
        elif de > 1.5: score -= 20
        return score

    def calculate_scores(self):
        results = []
        for sym in self.symbols:
            d = self.get_financial_data(sym)
            
            # Extraer Datos de Identidad
            profile_data = d.get("profile", {})
            quote_data = d.get("quote", {})
            
            # Usamos 'get' con un valor por defecto por si falla la API
            company_name = profile_data.get("companyName", quote_data.get("name", sym))
            sector = profile_data.get("sector", "Unknown")  # Si no hay sector, ponemos Unknown
            
            q = self.analyze_quality(d)
            v = self.analyze_valuation(d)
            m = self.analyze_momentum(d)
            c = self.analyze_catalysts(d)
            r = self.analyze_risk(d)
            
            final = (q*0.4) + (v*0.25) + (m*0.2) + (c*0.15)
            if r < 50: final -= 10
            
            results.append({
                "Ticker": sym,
                "company_name": company_name,  # ¡ESTO FALTABA!
                "sector": sector,              # ¡ESTO FALTABA!
                "C_Score": round(final, 1),
                "Desglose": {"Quality": round(q,1), "Valuation": round(v,1), "Momentum": round(m,1), "Catalysts": round(c,1), "Risk_Safety": round(r,1)}
            })
        return pd.DataFrame(results).sort_values(by="C_Score", ascending=False)

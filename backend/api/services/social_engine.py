"""
Social Radar Engine - Detección avanzada de anomalías en social sentiment.
Implementa Velocity Spike, Rumble Score y Tiny Titan Ratio.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

LOGGER = logging.getLogger("caria.api.social_engine")


class SocialRadarEngine:
    """
    Motor de análisis social que detecta:
    - Velocity Spike: Aceleración de menciones vs promedio histórico
    - Rumble Score: Polémica (sentimiento neutral con alto volumen)
    - Tiny Titan Ratio: Densidad de hype relativo al tamaño
    """
    
    def __init__(self, fmp_client=None):
        """
        Args:
            fmp_client: Cliente FMP para obtener datos de market cap y perfil
        """
        self.fmp_client = fmp_client
        # Cache para datos históricos (en producción, usar Redis o DB)
        self._historical_cache: Dict[str, List[Dict]] = {}
    
    def _get_market_cap(self, ticker: str) -> float:
        """Obtiene market cap en billones desde FMP."""
        if not self.fmp_client:
            return 0.0
        
        try:
            profile = self.fmp_client._get(f"profile/{ticker}")
            if profile and isinstance(profile, list) and len(profile) > 0:
                mcap = profile[0].get("mktCap", 0)
                return mcap / 1e9  # Convertir a billones
        except Exception as e:
            LOGGER.debug(f"Could not fetch market cap for {ticker}: {e}")
        
        return 0.0
    
    def _normalize_sentiment(self, sentiment: str) -> float:
        """Convierte sentimiento de string a número (-1 a +1)."""
        sentiment_lower = sentiment.lower() if isinstance(sentiment, str) else "neutral"
        if "bull" in sentiment_lower or "positive" in sentiment_lower:
            return 0.7
        elif "bear" in sentiment_lower or "negative" in sentiment_lower:
            return -0.7
        else:
            return 0.0
    
    def analyze_under_radar(
        self,
        reddit_data: List[Dict[str, Any]],
        stocktwits_data: List[Dict[str, Any]],
        historical_data: Optional[Dict[str, List[Dict]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detecta anomalías, gemas ocultas y zonas de guerra.
        
        Args:
            reddit_data: Lista de stocks con datos de Reddit
            stocktwits_data: Lista de stocks con datos de StockTwits
            historical_data: Dict con datos históricos por ticker (opcional)
        
        Returns:
            Lista de stocks con anomalías detectadas
        """
        results = []
        
        # Combinar datos de Reddit y StockTwits
        combined_data: Dict[str, Dict] = {}
        
        # Procesar Reddit
        for item in reddit_data:
            ticker = item.get("ticker", "").upper()
            if not ticker:
                continue
            
            if ticker not in combined_data:
                combined_data[ticker] = {
                    "ticker": ticker,
                    "reddit_mentions": 0,
                    "stocktwits_mentions": 0,
                    "reddit_sentiment": 0.0,
                    "stocktwits_sentiment": 0.0,
                    "reddit_posts": [],
                    "stocktwits_posts": []
                }
            
            combined_data[ticker]["reddit_mentions"] = item.get("mentions", 0)
            combined_data[ticker]["reddit_sentiment"] = self._normalize_sentiment(
                item.get("sentiment", "neutral")
            )
            if item.get("top_post_title"):
                combined_data[ticker]["reddit_posts"].append(item.get("top_post_title"))
        
        # Procesar StockTwits
        for item in stocktwits_data:
            ticker = item.get("ticker", "").upper()
            if not ticker:
                continue
            
            if ticker not in combined_data:
                combined_data[ticker] = {
                    "ticker": ticker,
                    "reddit_mentions": 0,
                    "stocktwits_mentions": 0,
                    "reddit_sentiment": 0.0,
                    "stocktwits_sentiment": 0.0,
                    "reddit_posts": [],
                    "stocktwits_posts": []
                }
            
            combined_data[ticker]["stocktwits_mentions"] = item.get("mentions", 0)
            combined_data[ticker]["stocktwits_sentiment"] = self._normalize_sentiment(
                item.get("sentiment", "neutral")
            )
            if item.get("top_message"):
                combined_data[ticker]["stocktwits_posts"].append(item.get("top_message"))
        
        # Analizar cada ticker
        for ticker, data in combined_data.items():
            curr_mentions = data["reddit_mentions"] + data["stocktwits_mentions"]
            
            # Obtener market cap
            mcap_billions = self._get_market_cap(ticker)
            
            # --- 1. VELOCITY SPIKE (Aceleración de Menciones) ---
            # Si tenemos datos históricos, calcular spike
            avg_mentions = 0
            velocity_score = 0
            
            if historical_data and ticker in historical_data:
                hist = historical_data[ticker]
                if len(hist) >= 7:
                    # Promedio de últimos 7 días
                    avg_mentions = np.mean([
                        h.get("reddit_mentions", 0) + h.get("stocktwits_mentions", 0)
                        for h in hist[:7]
                    ])
                    
                    if avg_mentions >= 10 and curr_mentions >= 50:
                        velocity_score = (curr_mentions - avg_mentions) / avg_mentions
            else:
                # Sin datos históricos, usar heurística basada en menciones absolutas
                # Si tiene muchas menciones pero no es mega-cap, podría ser spike
                if curr_mentions > 200 and mcap_billions < 50:
                    velocity_score = 1.5  # Estimación conservadora
            
            # --- 2. RUMBLE SCORE (Polémica) ---
            # Sentimiento promedio ponderado
            total_mentions = curr_mentions
            if total_mentions > 0:
                avg_sentiment = (
                    (data["reddit_sentiment"] * data["reddit_mentions"] +
                     data["stocktwits_sentiment"] * data["stocktwits_mentions"]) /
                    total_mentions
                )
            else:
                avg_sentiment = 0.0
            
            sentiment_abs = abs(avg_sentiment)
            is_controversial = (sentiment_abs < 0.2) and (curr_mentions > 200)
            
            # --- 3. TINY TITAN (Hype relativo al tamaño) ---
            if mcap_billions > 0:
                hype_density = curr_mentions / mcap_billions
            else:
                hype_density = 0
            
            # --- CLASIFICACIÓN ---
            categories = []
            insight_parts = []
            
            # Criterio: Under the Radar (Spike fuerte en empresa mediana/pequeña)
            if velocity_score > 2.0 and mcap_billions < 50:  # +200% menciones
                categories.append("🚀 Under the Radar Spike")
                insight_parts.append(f"Menciones subieron un {int(velocity_score * 100)}% vs promedio semanal")
            
            # Criterio: War Zone (Alto volumen, sentimiento dividido)
            if is_controversial:
                categories.append("⚔️ Bull/Bear War")
                insight_parts.append(f"Sentimiento neutral ({avg_sentiment:.2f}) con volumen extremo. Alta indecisión")
            
            # Criterio: Viral Small Cap
            if hype_density > 500 and mcap_billions < 2:
                categories.append("💎 Viral Micro-Cap")
                insight_parts.append(f"Volumen social {int(hype_density)}x superior al tamaño de mercado")
            
            # Solo guardamos si tiene algo interesante
            if categories:
                # Obtener precio si está disponible
                price = None
                if self.fmp_client:
                    try:
                        quote = self.fmp_client.get_realtime_price(ticker)
                        if quote:
                            price = quote.get("price")
                    except:
                        pass
                
                results.append({
                    "ticker": ticker,
                    "price": price,
                    "mentions_today": int(curr_mentions),
                    "spike_pct": round(velocity_score * 100, 1) if velocity_score > 0 else 0,
                    "sentiment_score": round(avg_sentiment, 2),
                    "hype_density": round(hype_density, 1),
                    "market_cap_billions": round(mcap_billions, 2),
                    "tags": categories,
                    "insight": " | ".join(insight_parts) if insight_parts else "Anomalía detectada en volumen social",
                    "reddit_mentions": data["reddit_mentions"],
                    "stocktwits_mentions": data["stocktwits_mentions"],
                    "top_post": data["reddit_posts"][0] if data["reddit_posts"] else (
                        data["stocktwits_posts"][0] if data["stocktwits_posts"] else None
                    )
                })
        
        # Ordenar por el Spike más fuerte o hype density
        results.sort(key=lambda x: max(x.get("spike_pct", 0), x.get("hype_density", 0) / 10), reverse=True)
        
        return results[:10]  # Top 10

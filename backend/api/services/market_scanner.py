"""
Market Scanner Service - Event-Driven Social Screener

Estrategia invertida: Busca anomalías de precio primero, valida con ruido social después.
Evita el problema de "pescar en el mismo estanque" (NVDA, TSLA, AAPL siempre ganan por volumen absoluto).

Devuelve dos tipos de señales:
1. momentum_signals: Precio subiendo fuerte (>5%) con volumen alto (>1.2x RVol)
2. accumulation_signals: Volumen muy alto (>1.8x RVol) pero precio comprimido (-1.5% a +2.5%)
"""

from __future__ import annotations

import logging
from typing import Any

from caria.ingestion.clients.fmp_client import FMPClient

LOGGER = logging.getLogger("caria.api.market_scanner")

# Límites de Market Cap para filtrar mega-caps
MEGA_CAP_THRESHOLD_MOMENTUM = 100_000_000_000  # 100B para momentum
MEGA_CAP_THRESHOLD_ACCUMULATION = 50_000_000_000  # 50B para accumulation


class MarketScannerService:
    """Scanner profesional que busca oportunidades basadas en anomalías de precio y volumen."""

    def __init__(self, fmp_client: FMPClient | None = None):
        self.fmp = fmp_client or FMPClient()

    def get_professional_opportunities(self) -> dict[str, list[dict[str, Any]]]:
        """
        Lógica unificada que devuelve dos arrays de señales.
        
        Returns:
            {
                "momentum_signals": [...],      # Precio + Social (High Velocity)
                "accumulation_signals": [...]   # Volumen + Precio Estable (High RVol)
            }
        """
        results = {
            "momentum_signals": [],
            "accumulation_signals": []
        }

        try:
            # 1. OBTENER RAW DATA (Screener masivo)
            # Filtros: Vol > 300k, Precio > $5, Excluir penny stocks
            params = {
                "volumeMoreThan": 300000,
                "priceMoreThan": 5,
                "isEtf": "false",
                "exchange": "NASDAQ,NYSE",
                "limit": 500
            }
            
            candidates = self.fmp._get("stock-screener", params)
            
            if not candidates or len(candidates) == 0:
                LOGGER.info("No candidates found from screener")
                return results

            # 2. PROCESAMIENTO (Iteramos una sola vez para eficiencia)
            # Necesitamos datos detallados (quote) para calcular RVol exacto
            symbols = [c.get('symbol') for c in candidates[:100] if c.get('symbol')]  # Limitamos a 100 para demo
            
            if not symbols:
                return results

            # Obtener quotes en batch
            quotes = {}
            chunk_size = 3  # FMP permite hasta 3 tickers por request
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i:i + chunk_size]
                try:
                    batch_data = self.fmp._get(f"quote/{','.join(chunk)}", None)
                    if isinstance(batch_data, list):
                        for q in batch_data:
                            ticker = q.get('symbol')
                            if ticker:
                                quotes[ticker] = q
                    elif isinstance(batch_data, dict):
                        ticker = batch_data.get('symbol')
                        if ticker:
                            quotes[ticker] = batch_data
                except Exception as e:
                    LOGGER.debug(f"Error fetching batch quotes: {e}")
                    continue

            LOGGER.info(f"Processing {len(quotes)} quotes for analysis")

            # 3. ANÁLISIS DE CADA CANDIDATO
            for ticker, q in quotes.items():
                try:
                    price = q.get('price', 0) or 0
                    change_pct = q.get('changesPercentage', 0) or 0
                    vol_today = q.get('volume', 0) or 0
                    vol_avg = q.get('avgVolume', 1) or 1
                    mcap = q.get('marketCap', 0) or 0

                    if not vol_avg or vol_avg == 0:
                        continue

                    # Métrica Clave: Relative Volume (RVol)
                    r_vol = vol_today / vol_avg if vol_avg > 0 else 0

                    # --- ESTRATEGIA A: MOMENTUM & VELOCITY (Alpha Generation) ---
                    # Criterio: Precio subiendo fuerte (>5%) CON soporte de volumen (>1.2x)
                    # Excluimos Mega Caps (>100B) para evitar ruido de índices
                    if change_pct > 5.0 and r_vol > 1.2 and mcap < MEGA_CAP_THRESHOLD_MOMENTUM:
                        # Validar con spike social (opcional pero mejora la señal)
                        # No bloqueamos la señal si falla el check social
                        try:
                            social_spike = self._check_social_spike(ticker)
                        except Exception:
                            social_spike = None
                        
                        results["momentum_signals"].append({
                            "ticker": ticker,
                            "price": round(price, 2),
                            "change": round(change_pct, 2),
                            "rvol": round(r_vol, 2),
                            "market_cap": mcap,
                            "signal_strength": "High",
                            "tag": "PRICE VELOCITY",
                            "desc": f"Price outlier (+{round(change_pct, 2)}%) on expanded volume ({round(r_vol, 1)}x avg).",
                            "social_spike": social_spike
                        })

                    # --- ESTRATEGIA B: INSTITUTIONAL ACCUMULATION (Divergence) ---
                    # Criterio: Volumen muy alto (>1.8x) PERO precio comprimido (-1.5% a +2.5%)
                    # Esto indica absorción de oferta.
                    elif r_vol > 1.8 and -1.5 < change_pct < 2.5 and mcap < MEGA_CAP_THRESHOLD_ACCUMULATION:
                        # Validar con spike social (opcional)
                        try:
                            social_spike = self._check_social_spike(ticker)
                        except Exception:
                            social_spike = None
                        
                        results["accumulation_signals"].append({
                            "ticker": ticker,
                            "price": round(price, 2),
                            "change": round(change_pct, 2),
                            "rvol": round(r_vol, 2),
                            "market_cap": mcap,
                            "signal_strength": "Critical",
                            "tag": "VOL DIVERGENCE",
                            "desc": f"Volume anomaly ({round(r_vol, 1)}x avg) with flat price action ({round(change_pct, 2)}%).",
                            "social_spike": social_spike
                        })

                except Exception as e:
                    LOGGER.debug(f"Error processing {ticker}: {e}")
                    continue

            # Ordenar resultados
            # Momentum: Por % de cambio
            results["momentum_signals"] = sorted(
                results["momentum_signals"], 
                key=lambda x: x['change'], 
                reverse=True
            )[:5]
            
            # Accumulation: Por RVol (lo más anómalo arriba)
            results["accumulation_signals"] = sorted(
                results["accumulation_signals"], 
                key=lambda x: x['rvol'], 
                reverse=True
            )[:5]

            LOGGER.info(
                f"Scanner complete: {len(results['momentum_signals'])} momentum signals, "
                f"{len(results['accumulation_signals'])} accumulation signals"
            )

        except Exception as e:
            LOGGER.error(f"Error in market scanner: {e}", exc_info=True)
            return results

        return results

    def _check_social_spike(self, ticker: str) -> dict[str, Any] | None:
        """
        Valida con ruido social usando historical-social-sentiment de FMP.
        Calcula spike ratio (aumento relativo de menciones).
        
        Returns:
            Dict con spike_ratio y métricas sociales, o None si no hay datos suficientes
        """
        try:
            # Usamos 'historical-social-sentiment' para ver la tendencia
            social_data = self.fmp._get(
                "historical-social-sentiment",
                {"symbol": ticker, "page": 0, "limit": 10}
            )
            
            if not social_data or len(social_data) < 2:
                return None

            today = social_data[0]
            yesterday = social_data[1]

            # --- LA MATEMÁTICA DEL HYPE ---
            mentions_today = (
                (today.get('stocktwitsPosts', 0) or 0) + 
                (today.get('redditPosts', 0) or 0)
            )
            mentions_yesterday = (
                (yesterday.get('stocktwitsPosts', 0) or 0) + 
                (yesterday.get('redditPosts', 0) or 0)
            )

            # FILTRO CRÍTICO:
            # 1. Debe tener un mínimo de actividad (para no traer basura con 1 post)
            if mentions_today < 50:
                return None

            # 2. VELOCIDAD: ¿Se duplicó el ruido respecto a ayer?
            spike_ratio = mentions_today / (mentions_yesterday + 1)  # +1 evita div por 0

            # 3. SENTIMIENTO: ¿Es positivo?
            sentiment = today.get('stocktwitsSentiment', 0) or 0

            # Si pasa los filtros, retornamos métricas
            if spike_ratio > 1.5:  # 50% de aumento en ruido
                return {
                    "spike_ratio": round(spike_ratio, 1),
                    "mentions_today": mentions_today,
                    "mentions_yesterday": mentions_yesterday,
                    "sentiment": round(sentiment, 2),
                    "has_spike": True
                }

            return {
                "spike_ratio": round(spike_ratio, 1),
                "mentions_today": mentions_today,
                "mentions_yesterday": mentions_yesterday,
                "sentiment": round(sentiment, 2),
                "has_spike": False
            }

        except Exception as e:
            LOGGER.debug(f"Error checking social spike for {ticker}: {e}")
            return None

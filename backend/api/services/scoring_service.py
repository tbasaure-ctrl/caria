import logging
import statistics
from typing import Any, Dict, Optional

from caria.ingestion.clients.fmp_client import FMPClient

from .simple_valuation import SimpleValuationService

LOGGER = logging.getLogger("caria.api.scoring")


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


class ScoringService:
    """Calcula puntajes Quality / Valuation / Momentum para un ticker."""

    def __init__(self, fmp_client: Optional[FMPClient] = None) -> None:
        self.fmp_client = fmp_client or FMPClient()
        self.valuation_service = SimpleValuationService(self.fmp_client)

    def get_scores(self, ticker: str) -> Dict[str, Any]:
        ticker = ticker.upper()
        metrics = self._fetch_metrics(ticker)
        current_price = self._get_current_price(ticker)
        valuation = self.valuation_service.get_valuation(ticker, current_price)

        quality_score, quality_factors = self._calculate_quality_score(metrics)
        valuation_score, valuation_details = self._calculate_valuation_score(valuation, current_price)
        momentum_score, momentum_details = self._calculate_momentum_score(ticker)

        composite = round(
            (quality_score + valuation_score + momentum_score) / 3,
            2,
        )

        return {
            "ticker": ticker,
            "qualityScore": round(quality_score, 2),
            "valuationScore": round(valuation_score, 2),
            "momentumScore": round(momentum_score, 2),
            "compositeScore": composite,
            "current_price": current_price,
            "fair_value": valuation_details.get("fair_value"),
            "valuation_upside_pct": valuation_details.get("upside_pct"),
            "details": {
                "quality": quality_factors,
                "valuation": valuation_details,
                "momentum": momentum_details,
            },
        }

    def _fetch_metrics(self, ticker: str) -> Dict[str, Any]:
        key_metrics = self.fmp_client.get_key_metrics(ticker, period="quarter")
        ratios = self.fmp_client.get_financial_ratios(ticker, period="quarter")
        growth = self.fmp_client.get_financial_growth(ticker, period="quarter")

        return {
            **(key_metrics[0] if key_metrics else {}),
            **(ratios[0] if ratios else {}),
            "growth": growth[0] if growth else {},
        }

    def _get_current_price(self, ticker: str) -> float:
        quote = self.fmp_client.get_realtime_price(ticker)
        if quote and quote.get("price"):
            return float(quote["price"])
        raise RuntimeError(f"No se pudo obtener precio actual para {ticker}")

    def _normalize(self, value: Optional[float], min_value: float, max_value: float) -> float:
        if value is None:
            return 0.5
        if max_value == min_value:
            return 0.5
        clipped = _clamp(value, min_value, max_value)
        return (clipped - min_value) / (max_value - min_value)

    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        roic = metrics.get("returnOnInvestedCapitalTTM") or metrics.get("returnOnInvestedCapital")
        gross_margin = metrics.get("grossProfitMarginTTM") or metrics.get("grossProfitMargin")
        revenue_growth = metrics.get("growth", {}).get("revenueGrowth")

        components = []
        if roic is not None:
            components.append(self._normalize(roic, 0.0, 0.25))
        if gross_margin is not None:
            components.append(self._normalize(gross_margin, 0.2, 0.6))
        if revenue_growth is not None:
            components.append(self._normalize(revenue_growth, -0.05, 0.3))

        score = sum(components) / len(components) if components else 0.5
        return score * 100, {
            "roic": roic,
            "gross_margin": gross_margin,
            "revenue_growth": revenue_growth,
        }

    def _calculate_valuation_score(
        self,
        valuation: Dict[str, Any],
        current_price: float,
    ) -> tuple[float, Dict[str, Any]]:
        fair_value = valuation.get("multiples_valuation", {}).get("fair_value")
        if not fair_value:
            fair_value = valuation.get("dcf", {}).get("fair_value_per_share")

        if not fair_value:
            return 50.0, {"fair_value": None, "upside_pct": None}

        upside = (fair_value - current_price) / current_price
        valuation_score = self._normalize(upside, -0.3, 0.3) * 100
        return valuation_score, {
            "fair_value": round(fair_value, 2),
            "upside_pct": round(upside * 100, 2),
            "ev_sales_median": valuation.get("multiples_valuation", {}).get("ev_sales_median"),
            "ev_ebitda_median": valuation.get("multiples_valuation", {}).get("ev_ebitda_median"),
        }

    def _calculate_momentum_score(self, ticker: str) -> tuple[float, Dict[str, Any]]:
        history = self.fmp_client.get_price_history(ticker)
        if not history:
            return 50.0, {"short_return": None, "long_return": None, "volatility": None}

        history_sorted = sorted(history, key=lambda x: x.get("date"))
        closes = [float(item["close"]) for item in history_sorted if item.get("close")]
        if len(closes) < 30:
            return 50.0, {"short_return": None, "long_return": None, "volatility": None}

        latest = closes[-1]
        short_idx = -20 if len(closes) >= 20 else 0
        long_idx = -90 if len(closes) >= 90 else 0

        short_price = closes[short_idx]
        long_price = closes[long_idx]

        short_return = (latest - short_price) / short_price if short_price else 0.0
        long_return = (latest - long_price) / long_price if long_price else 0.0

        daily_returns = []
        for prev, curr in zip(closes[-120:-1], closes[-119:]):
            if prev:
                daily_returns.append((curr - prev) / prev)

        volatility = statistics.pstdev(daily_returns[-60:]) if len(daily_returns) >= 10 else 0.02

        short_component = self._normalize(short_return, -0.2, 0.2)
        long_component = self._normalize(long_return, -0.4, 0.4)
        volatility_component = 1 - self._normalize(volatility, 0.0, 0.4)

        score = (short_component + long_component + volatility_component) / 3

        return score * 100, {
            "short_return": round(short_return * 100, 2),
            "long_return": round(long_return * 100, 2),
            "volatility": round(volatility * 100, 2),
        }


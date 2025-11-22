import logging
import math
import statistics
from typing import Any, Dict, List, Optional

from api.services.openbb_client import openbb_client
from api.services.simple_valuation import SimpleValuationService

LOGGER = logging.getLogger("caria.api.scoring")


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


class ScoringService:
    """Calculates Quality / Valuation / Momentum and composite C-Score."""

    def __init__(self) -> None:
        self.openbb = openbb_client
        self.valuation_service = SimpleValuationService()

    def get_scores(self, ticker: str) -> Dict[str, Any]:
        ticker = ticker.upper()
        dataset = self.openbb.get_ticker_data(ticker)
        if not dataset:
            raise RuntimeError(f"Dataset missing for {ticker}")

        price_history = dataset.get("price_history", [])
        latest_price = dataset.get("latest_price")
        if not latest_price:
            raise RuntimeError(f"Price history unavailable for {ticker}")

        valuation = self.valuation_service.get_valuation(ticker, latest_price)

        quality_score, quality_details = self._quality_model(dataset)
        valuation_score, valuation_details = self._valuation_model(valuation, latest_price, dataset)
        momentum_score, momentum_details = self._momentum_model(price_history)
        qualitative_moat = self._qualitative_moat_score(dataset, quality_details)

        c_score = round(
            0.35 * quality_score
            + 0.25 * valuation_score
            + 0.20 * momentum_score
            + 0.20 * qualitative_moat,
            2,
        )

        return {
            "ticker": ticker,
            "qualityScore": round(quality_score, 2),
            "valuationScore": round(valuation_score, 2),
            "momentumScore": round(momentum_score, 2),
            "qualitativeMoatScore": round(qualitative_moat, 2),
            "cScore": c_score,
            "classification": self._classify_c_score(c_score),
            "current_price": latest_price,
            "fair_value": valuation_details.get("fair_value"),
            "valuation_upside_pct": valuation_details.get("upside_pct"),
            "details": {
                "quality": quality_details,
                "valuation": valuation_details,
                "momentum": momentum_details,
            },
        }

    # ---------- Quality Model ----------
    def _quality_model(self, dataset: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        income = dataset["financials"].get("income_statement", [])
        cash = dataset["financials"].get("cash_flow", [])
        multiples = dataset.get("multiples", [])

        income_sorted = sorted(income, key=lambda x: x.get("date") or "", reverse=True)
        cash_sorted = sorted(cash, key=lambda x: x.get("date") or "", reverse=True)

        latest_income = income_sorted[0] if income_sorted else {}
        latest_cash = cash_sorted[0] if cash_sorted else {}
        latest_multiples = multiples[0] if multiples else {}

        roic = self._pick(latest_multiples, ["roic", "returnOnInvestedCapital"])
        roe = self._pick(latest_multiples, ["roe", "returnOnEquity"])
        gross_margins = self._series_ratio(income_sorted[:6], "grossProfit", "totalRevenue")
        operating_margins = self._series_ratio(income_sorted[:6], "operatingIncome", "totalRevenue")
        revenue_growth = self._series_growth(income_sorted[:6], "totalRevenue")
        ebitda_growth = self._series_growth(income_sorted[:6], "ebitda")
        fcf_volatility = self._volatility(cash_sorted[:8], "freeCashFlow")
        leverage = self._pick(latest_multiples, ["debtToEquity", "ltDebtToEquity"])

        components: List[float] = []

        if roic is not None:
            components.append(self._normalize(roic, 0.05, 0.25))
        if roe is not None:
            components.append(self._normalize(roe, 0.08, 0.30))
        if gross_margins:
            margin_avg = sum(gross_margins) / len(gross_margins)
            margin_stability = 1 - (statistics.pstdev(gross_margins) if len(gross_margins) > 1 else 0)
            components.append(self._normalize(margin_avg, 0.2, 0.7))
            components.append(self._normalize(margin_stability, 0.0, 0.4))
        if revenue_growth is not None:
            components.append(self._normalize(revenue_growth, -0.05, 0.25))
        if ebitda_growth is not None:
            components.append(self._normalize(ebitda_growth, -0.05, 0.25))
        if fcf_volatility is not None:
            components.append(1 - self._normalize(fcf_volatility, 0.0, 0.5))
        if leverage is not None:
            components.append(1 - self._normalize(leverage, 0.0, 2.0))

        score = (sum(components) / len(components)) if components else 0.5
        return score * 100, {
            "roic": roic,
            "roe": roe,
            "gross_margin_avg": gross_margins[0] if gross_margins else None,
            "gross_margin_stability": margin_stability if gross_margins else None,
            "revenue_cagr": revenue_growth,
            "ebitda_cagr": ebitda_growth,
            "fcf_volatility": fcf_volatility,
            "leverage": leverage,
        }

    # ---------- Valuation Model ----------
    def _valuation_model(
        self,
        valuation: Dict[str, Any],
        current_price: float,
        dataset: Dict[str, Any],
    ) -> tuple[float, Dict[str, Any]]:
        fair_value = valuation.get("multiples_valuation", {}).get("fair_value")
        if not fair_value:
            fair_value = valuation.get("dcf", {}).get("fair_value_per_share")

        upside = (fair_value - current_price) / current_price if fair_value else 0

        ev_sales_hist = [
            self._pick(m, ["ev_to_revenue", "enterpriseValueRevenueMultiple", "evSales"])
            for m in dataset.get("multiples", [])[:8]
            if self._pick(m, ["ev_to_revenue", "enterpriseValueRevenueMultiple", "evSales"]) is not None
        ]
        ev_ebitda_hist = [
            self._pick(m, ["ev_to_ebitda", "enterpriseValueToEbitda", "evEbitda"])
            for m in dataset.get("multiples", [])[:8]
            if self._pick(m, ["ev_to_ebitda", "enterpriseValueToEbitda", "evEbitda"]) is not None
        ]
        ev_sales_current = ev_sales_hist[0] if ev_sales_hist else None
        ev_ebitda_current = ev_ebitda_hist[0] if ev_ebitda_hist else None
        hist_sales_median = statistics.median(ev_sales_hist[1:]) if len(ev_sales_hist) > 2 else None
        hist_ebitda_median = statistics.median(ev_ebitda_hist[1:]) if len(ev_ebitda_hist) > 2 else None

        margin_of_safety = 0
        if ev_sales_current and hist_sales_median:
            margin_of_safety += (hist_sales_median - ev_sales_current) / hist_sales_median
        if ev_ebitda_current and hist_ebitda_median:
            margin_of_safety += (hist_ebitda_median - ev_ebitda_current) / hist_ebitda_median

        score_components = [
            self._normalize(upside, -0.3, 0.3),
            self._normalize(margin_of_safety, -0.5, 0.5),
        ]
        score = sum(score_components) / len(score_components)

        return score * 100, {
            "fair_value": round(fair_value, 2) if fair_value else None,
            "upside_pct": round(upside * 100, 2) if fair_value else None,
            "current_ev_sales": ev_sales_current,
            "median_ev_sales": hist_sales_median,
            "current_ev_ebitda": ev_ebitda_current,
            "median_ev_ebitda": hist_ebitda_median,
        }

    # ---------- Momentum Model ----------
    def _momentum_model(self, history: List[Dict[str, Any]]) -> tuple[float, Dict[str, Any]]:
        if not history or len(history) < 30:
            return 50.0, {"returns": None, "volatility": None, "drawdown": None}

        closes = [float(item["close"]) for item in history if item.get("close")]
        if len(closes) < 30:
            return 50.0, {"returns": None, "volatility": None, "drawdown": None}

        returns_map = {
            "1m": self._return_over(closes, 21),
            "3m": self._return_over(closes, 63),
            "6m": self._return_over(closes, 126),
            "12m": self._return_over(closes, 252),
        }

        vol = self._daily_volatility(closes[-126:])
        drawdown = self._max_drawdown(closes[-252:])
        rsi = self._rsi(closes[-90:])

        components = []
        for period in ("1m", "3m", "6m", "12m"):
            if returns_map[period] is not None:
                components.append(self._normalize(returns_map[period], -0.2, 0.4))
        if vol is not None:
            components.append(1 - self._normalize(vol, 0.0, 0.6))
        if drawdown is not None:
            components.append(1 - self._normalize(drawdown, 0.0, 0.6))
        if rsi is not None:
            components.append(self._normalize(rsi, 30, 70))

        score = (sum(components) / len(components)) if components else 0.5

        return score * 100, {
            "returns": {k: round(v * 100, 2) if v is not None else None for k, v in returns_map.items()},
            "volatility": vol,
            "drawdown": drawdown,
            "rsi": rsi,
        }

    # ---------- Helpers ----------
    def _qualitative_moat_score(self, dataset: Dict[str, Any], quality: Dict[str, Any]) -> float:
        gross_margin_stability = quality.get("gross_margin_stability") or 0.2
        revenue_trend = quality.get("revenue_cagr") or 0.05
        return _clamp((gross_margin_stability * 0.6 + revenue_trend * 0.4) * 100, 0, 100)

    def _classify_c_score(self, score: float) -> str:
        if score >= 80:
            return "Probable Outlier"
        if score >= 60:
            return "High-Quality Compounder"
        return "Standard / Non-Outlier"

    def _normalize(self, value: Optional[float], min_value: float, max_value: float) -> float:
        if value is None or math.isnan(value):
            return 0.5
        if max_value == min_value:
            return 0.5
        clipped = _clamp(value, min_value, max_value)
        return (clipped - min_value) / (max_value - min_value)

    def _series_ratio(self, statements: List[Dict[str, Any]], num_key: str, den_key: str) -> List[float]:
        ratios: List[float] = []
        for stmt in statements:
            numerator = stmt.get(num_key)
            denominator = stmt.get(den_key)
            if numerator is not None and denominator not in (None, 0):
                ratios.append(numerator / denominator)
        return ratios

    def _series_growth(self, statements: List[Dict[str, Any]], key: str) -> Optional[float]:
        values = [stmt.get(key) for stmt in statements if stmt.get(key) not in (None, 0)]
        if len(values) < 2:
            return None
        latest = values[0]
        older = values[-1]
        if older == 0:
            return None
        periods = len(values) - 1
        return (latest / older) ** (1 / periods) - 1

    def _volatility(self, statements: List[Dict[str, Any]], key: str) -> Optional[float]:
        values = [stmt.get(key) for stmt in statements if stmt.get(key) is not None]
        if len(values) < 3:
            return None
        return statistics.pstdev(values) / (abs(statistics.mean(values)) + 1e-9)

    def _daily_volatility(self, closes: List[float]) -> Optional[float]:
        if len(closes) < 10:
            return None
        returns = [(curr - prev) / prev for prev, curr in zip(closes[:-1], closes[1:]) if prev != 0]
        if not returns:
            return None
        return statistics.pstdev(returns)

    def _max_drawdown(self, closes: List[float]) -> Optional[float]:
        if not closes:
            return None
        max_price = closes[0]
        max_drawdown = 0.0
        for price in closes:
            max_price = max(max_price, price)
            drawdown = (price - max_price) / max_price
            max_drawdown = min(max_drawdown, drawdown)
        return abs(max_drawdown)

    def _return_over(self, closes: List[float], window: int) -> Optional[float]:
        if len(closes) < window + 1:
            return None
        latest = closes[-1]
        base = closes[-window - 1]
        if base == 0:
            return None
        return (latest - base) / base

    def _rsi(self, closes: List[float], period: int = 14) -> Optional[float]:
        if len(closes) <= period:
            return None
        gains = []
        losses = []
        for prev, curr in zip(closes[:-1], closes[1:]):
            change = curr - prev
            if change >= 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        avg_gain = statistics.mean(gains[-period:]) if gains else 0
        avg_loss = statistics.mean(losses[-period:]) if losses else 0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _pick(self, record: Dict[str, Any], keys: List[str]) -> Optional[float]:
        for key in keys:
            if record.get(key) not in (None, "", "NA"):
                return record[key]
        return None


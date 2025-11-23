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
        qualitative_moat, moat_details = self._qualitative_moat_score(dataset, quality_details)

        # Composite Score Formula (C-Score)
        # Target weights: Quality (35%), Valuation (25%), Momentum (20%), Moat (20%)
        # Since Moat is missing, we re-normalize: 35+25+20 = 80
        # Quality: 0.35 / 0.8 = 0.4375
        # Valuation: 0.25 / 0.8 = 0.3125
        # Momentum: 0.20 / 0.8 = 0.25
        composite = round(
            (quality_score * 0.4375) + (valuation_score * 0.3125) + (momentum_score * 0.25),
            0, # Round to integer for cleaner UI
        )

        explanations = self._build_explanations(quality_details, valuation_details, momentum_details, moat_details)
        factor_attribution = {
            "quality": quality_details.get("drivers"),
            "valuation": valuation_details.get("drivers"),
            "momentum": momentum_details.get("drivers"),
            "moat": moat_details.get("drivers"),
        }

        return {
            "ticker": ticker,
            "qualityScore": round(quality_score, 0),
            "valuationScore": round(valuation_score, 0),
            "momentumScore": round(momentum_score, 0),
            "compositeScore": composite,
            "current_price": latest_price,
            "fair_value": valuation_details.get("fair_value"),
            "valuation_upside_pct": valuation_details.get("upside_pct"),
            "details": {
                "quality": quality_details,
                "valuation": valuation_details,
                "momentum": momentum_details,
                "moat": moat_details,
            },
            "factorAttribution": factor_attribution,
            "explanations": explanations,
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
        drivers: Dict[str, float] = {}
        margin_stability = None

        if roic is not None:
            roic_norm = self._normalize(roic, 0.05, 0.25)
            components.append(roic_norm)
            drivers["ROIC"] = round(roic_norm * 100, 1)
        if roe is not None:
            roe_norm = self._normalize(roe, 0.08, 0.30)
            components.append(roe_norm)
            drivers["ROE"] = round(roe_norm * 100, 1)
        if gross_margins:
            margin_avg = sum(gross_margins) / len(gross_margins)
            margin_stability = 1 - (statistics.pstdev(gross_margins) if len(gross_margins) > 1 else 0)
            margin_avg_norm = self._normalize(margin_avg, 0.2, 0.7)
            margin_stability_norm = self._normalize(margin_stability, 0.0, 0.4)
            components.extend([margin_avg_norm, margin_stability_norm])
            drivers["Gross margin level"] = round(margin_avg_norm * 100, 1)
            drivers["Margin stability"] = round(margin_stability_norm * 100, 1)
        if revenue_growth is not None:
            rev_norm = self._normalize(revenue_growth, -0.05, 0.25)
            components.append(rev_norm)
            drivers["Revenue CAGR"] = round(rev_norm * 100, 1)
        if ebitda_growth is not None:
            ebitda_norm = self._normalize(ebitda_growth, -0.05, 0.25)
            components.append(ebitda_norm)
            drivers["EBITDA CAGR"] = round(ebitda_norm * 100, 1)
        if fcf_volatility is not None:
            fcf_norm = 1 - self._normalize(fcf_volatility, 0.0, 0.5)
            components.append(fcf_norm)
            drivers["FCF stability"] = round(fcf_norm * 100, 1)
        if leverage is not None:
            leverage_norm = 1 - self._normalize(leverage, 0.0, 2.0)
            components.append(leverage_norm)
            drivers["Leverage safety"] = round(leverage_norm * 100, 1)

        rule_score = (sum(components) / len(components)) if components else 0.5
        ensemble_score = rule_score
        if revenue_growth is not None:
            growth_trend = self._normalize(revenue_growth, -0.05, 0.25)
            ensemble_score = 0.75 * rule_score + 0.25 * growth_trend
        score = self._bayesian_blend(ensemble_score)

        return score * 100, {
            "roic": roic,
            "roe": roe,
            "gross_margin_avg": gross_margins[0] if gross_margins else None,
            "gross_margin_stability": margin_stability if gross_margins else None,
            "revenue_cagr": revenue_growth,
            "ebitda_cagr": ebitda_growth,
            "fcf_volatility": fcf_volatility,
            "leverage": leverage,
            "drivers": drivers,
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

        upside_norm = self._normalize(upside, -0.3, 0.3)
        mos_norm = self._normalize(margin_of_safety, -0.5, 0.5)
        score = (upside_norm + mos_norm) / 2
        score = self._bayesian_blend(score, prior=0.55, weight=0.25)

        return score * 100, {
            "fair_value": round(fair_value, 2) if fair_value else None,
            "upside_pct": round(upside * 100, 2) if fair_value else None,
            "current_ev_sales": ev_sales_current,
            "median_ev_sales": hist_sales_median,
            "current_ev_ebitda": ev_ebitda_current,
            "median_ev_ebitda": hist_ebitda_median,
            "drivers": {
                "DCF / Upside": round(upside_norm * 100, 1),
                "Multiple re-rating": round(mos_norm * 100, 1),
            },
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
        drivers: Dict[str, float] = {}
        for period in ("1m", "3m", "6m", "12m"):
            if returns_map[period] is not None:
                norm_value = self._normalize(returns_map[period], -0.2, 0.4)
                components.append(norm_value)
                drivers[f"{period} return"] = round(norm_value * 100, 1)
        if vol is not None:
            vol_norm = 1 - self._normalize(vol, 0.0, 0.6)
            components.append(vol_norm)
            drivers["Volatility"] = round(vol_norm * 100, 1)
        if drawdown is not None:
            dd_norm = 1 - self._normalize(drawdown, 0.0, 0.6)
            components.append(dd_norm)
            drivers["Drawdown resilience"] = round(dd_norm * 100, 1)
        if rsi is not None:
            rsi_norm = self._normalize(rsi, 30, 70)
            components.append(rsi_norm)
            drivers["RSI balance"] = round(rsi_norm * 100, 1)

        rule_score = (sum(components) / len(components)) if components else 0.5
        secondary = self._normalize(returns_map.get("3m") or 0, -0.2, 0.4)
        score = self._bayesian_blend(0.7 * rule_score + 0.3 * secondary, prior=0.5, weight=0.2)

        return score * 100, {
            "returns": {k: round(v * 100, 2) if v is not None else None for k, v in returns_map.items()},
            "volatility": vol,
            "drawdown": drawdown,
            "rsi": rsi,
            "drivers": drivers,
        }

    # ---------- Helpers ----------
    def _qualitative_moat_score(self, dataset: Dict[str, Any], quality: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        profile = dataset.get("profile") or {}
        insider_pct = self._pick(profile, ["insiderOwnership", "insider_ownership"])
        institutional_pct = self._pick(profile, ["institutionOwnership", "institutionalOwnership"])
        employees = profile.get("fullTimeEmployees") or profile.get("employees")
        innovation_ratio = self._research_intensity(dataset["financials"].get("income_statement", []))

        insider_pct = (insider_pct / 100) if insider_pct and insider_pct > 1 else insider_pct or 0.01
        institutional_pct = (institutional_pct / 100) if institutional_pct and institutional_pct > 1 else institutional_pct or 0.5

        gross_margin_stability = quality.get("gross_margin_stability") or 0.2
        revenue_trend = quality.get("revenue_cagr") or 0.05

        components = [
            ("Margin stability", self._normalize(gross_margin_stability, 0.0, 0.4)),
            ("Revenue trajectory", self._normalize(revenue_trend, -0.03, 0.2)),
            ("Insider alignment", self._normalize(insider_pct, 0.01, 0.18)),
            ("Institutional sponsorship", 1 - self._normalize(institutional_pct, 0.2, 0.95)),
            ("Innovation intensity", self._normalize(innovation_ratio or 0.02, 0.0, 0.15)),
        ]
        if employees:
            scale_score = self._normalize(math.log(employees + 1), 8, 12)
            components.append(("Scale moat", scale_score))

        score = (sum(value for _, value in components) / len(components)) if components else 0.5
        score = self._bayesian_blend(score, prior=0.6, weight=0.25)
        drivers = {label: round(value * 100, 1) for label, value in components}

        return _clamp(score * 100, 0, 100), {
            "insider_ownership_pct": round(insider_pct * 100, 2),
            "institutional_ownership_pct": round(institutional_pct * 100, 2),
            "full_time_employees": employees,
            "innovation_ratio": innovation_ratio,
            "drivers": drivers,
        }

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

    def _research_intensity(self, income_statements: List[Dict[str, Any]]) -> Optional[float]:
        ratios: List[float] = []
        for stmt in income_statements[:4]:
            rnd = stmt.get("researchAndDevelopment") or stmt.get("researchDevelopment")
            revenue = stmt.get("totalRevenue")
            if rnd and revenue:
                ratios.append(rnd / revenue)
        if not ratios:
            return None
        return statistics.mean(ratios)

    def _bayesian_blend(self, score: float, prior: float = 0.5, weight: float = 0.2) -> float:
        return (score * (1 - weight)) + (prior * weight)

    def _build_explanations(
        self,
        quality: Dict[str, Any],
        valuation: Dict[str, Any],
        momentum: Dict[str, Any],
        moat: Dict[str, Any],
    ) -> Dict[str, str]:
        explanations: Dict[str, str] = {}

        def summarize(drivers: Optional[Dict[str, float]], label: str) -> Optional[str]:
            if not drivers:
                return None
            best = max(drivers, key=drivers.get)
            worst = min(drivers, key=drivers.get)
            return f"{label} liderado por {best.lower()} ({drivers[best]:.0f}/100). Vigila {worst.lower()} ({drivers[worst]:.0f}/100)."

        q_msg = summarize(quality.get("drivers"), "Calidad")
        if q_msg:
            explanations["quality"] = q_msg

        v_drivers = valuation.get("drivers")
        if v_drivers:
            best = max(v_drivers, key=v_drivers.get)
            explanations["valuation"] = (
                f"Valoración apoyada en {best.lower()} ({v_drivers[best]:.0f}/100) "
                f"con upside estimado de {valuation.get('upside_pct', 0) or 0:.1f}%."
            )

        m_msg = summarize(momentum.get("drivers"), "Momentum")
        if m_msg:
            explanations["momentum"] = m_msg

        moat_msg = summarize(moat.get("drivers"), "Moat cualitativo")
        if moat_msg:
            explanations["moat"] = moat_msg

        return explanations

    def _pick(self, record: Dict[str, Any], keys: List[str]) -> Optional[float]:
        for key in keys:
            if record.get(key) not in (None, "", "NA"):
                return record[key]
        return None

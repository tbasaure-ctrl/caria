"""Monte Carlo Valuation para empresas usando DCF + múltiplos + escenarios macro."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import base64
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI
import matplotlib.pyplot as plt

from caria.models.valuation.monte_carlo_presets import BASE_PRESET, get_preset

LOGGER = logging.getLogger("caria.models.valuation.monte_carlo")


@dataclass
class MonteCarloValuation:
    """Resultado de valuación Monte Carlo."""
    ticker: str
    percentiles: Dict[str, float]  # P5, P10, P25, P50, P75, P90, P95
    mean: float
    median: float
    current_price: float
    paths_df: pd.DataFrame  # DataFrame con todos los paths
    visualization_histogram: str  # Base64 de histograma
    visualization_paths: str  # Base64 de paths muestrales
    configuration_used: Dict[str, Any]  # Configuración aplicada
    methods_used: Dict[str, float]  # Métodos y sus pesos
    explanation: str


class MonteCarloValuator:
    """Valuador Monte Carlo que combina DCF, múltiplos y escenarios macro."""
    
    def __init__(self, storage_path: Optional[Path] = None) -> None:
        """Inicializa el valuador.
        
        Args:
            storage_path: Directorio para guardar visualizaciones (opcional)
        """
        self.storage_path = storage_path or Path("data/artifacts/monte_carlo")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._global_state: Dict[str, np.ndarray] = {}
    
    @staticmethod
    def _clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Limita valores entre lo y hi."""
        return np.minimum(np.maximum(x, lo), hi)
    
    @staticmethod
    def _safe_positive(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        """Asegura valores positivos."""
        return np.where(x > eps, x, eps)
    
    def _draw_macro(
        self, cfg: Dict[str, Any], size: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Dibuja escenarios macro."""
        probs = np.array(cfg["macro"]["probs"], dtype=float)
        idx = rng.choice(len(probs), p=probs / probs.sum(), size=size)
        shocks = {
            "rev": np.take(np.array(cfg["macro"]["rev_shock"], dtype=float), idx),
            "margin": np.take(np.array(cfg["macro"]["margin_shock"], dtype=float), idx),
            "wacc": np.take(np.array(cfg["macro"]["wacc_shock"], dtype=float), idx),
            "fcf": np.take(np.array(cfg["macro"].get("fcf_shock", [0, 0, 0]), dtype=float), idx),
            "buybacks": np.take(np.array(cfg["macro"]["buyback_rate"], dtype=float), idx),
            "fx": {}
        }
        fx_cfg = cfg["macro"].get("fx", {})
        for ccy, arr in fx_cfg.items():
            shocks["fx"][ccy] = np.take(np.array(arr, dtype=float), idx)
        return idx, shocks
    
    def _draw_correlated_growth_margin(
        self, n: int, sd_g: float, sd_m: float, rho: float, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dibuja crecimiento y margen correlacionados."""
        z1 = rng.normal(0.0, 1.0, size=n)
        z2 = rng.normal(0.0, 1.0, size=n)
        g = sd_g * z1
        m = sd_m * (rho * z1 + math.sqrt(max(0.0, 1 - rho**2)) * z2)
        return g, m
    
    def _price_from_ev(self, ev: np.ndarray, net_debt: float, shares: np.ndarray) -> np.ndarray:
        """Convierte EV a precio por acción."""
        eq = ev - net_debt
        return eq / self._safe_positive(shares)
    
    def _ev_sales_valuation(
        self, revenue_12m: np.ndarray, mu: float, sd: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Valuación por EV/Sales."""
        mult = rng.normal(mu, sd, size=revenue_12m.shape[0])
        return mult * revenue_12m
    
    def _ev_ebitda_valuation(
        self, ebitda_12m: np.ndarray, mu: float, sd: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Valuación por EV/EBITDA."""
        mult = rng.normal(mu, sd, size=ebitda_12m.shape[0])
        return mult * self._safe_positive(ebitda_12m)
    
    def _pe_valuation(
        self, eps_12m: np.ndarray, mu: float, sd: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Valuación por P/E."""
        pe = rng.normal(mu, sd, size=eps_12m.shape[0])
        return pe * self._safe_positive(eps_12m)
    
    def _pbv_valuation(
        self, tbv_ps: np.ndarray, mu: float, sd: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Valuación por P/TBV."""
        pb = rng.normal(mu, sd, size=tbv_ps.shape[0])
        return pb * self._safe_positive(tbv_ps)
    
    def _pnav_valuation(
        self, nav_ps: np.ndarray, mu: float, sd: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Valuación por P/NAV."""
        pn = rng.normal(mu, sd, size=nav_ps.shape[0])
        return pn * self._safe_positive(nav_ps)
    
    def _dcf_valuation(
        self, cfg: Dict[str, Any], macro_fcf_shock: np.ndarray,
        wacc_shift: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Valuación DCF."""
        d = cfg["dcf"]
        n = macro_fcf_shock.shape[0]
        
        wacc = rng.normal(d["wacc_mean"], d["wacc_sd"], size=n) + wacc_shift
        tg = rng.normal(d["tg_mean"], d["tg_sd"], size=n)
        g = rng.normal(d["fcf_growth_mean"], d["fcf_growth_sd"], size=n)
        wacc = np.maximum(wacc, tg + 0.002)
        wacc = np.maximum(wacc, g + 0.002)
        
        base = cfg["base"]
        use_direct_fcf = base.get("fcf_start") is not None
        
        if use_direct_fcf:
            fcf1 = np.maximum(0.0, float(base["fcf_start"]) + macro_fcf_shock)
        else:
            rev = self._global_state["revenue_12m"]
            ebitda = self._global_state["ebitda_12m"]
            tax = base.get("tax_rate", 0.25)
            da = base.get("da_pct_sales", 0.03) * rev
            ebit = ebitda - da
            nopat = np.maximum(0.0, ebit) * (1.0 - tax)
            capex = base.get("capex_pct_sales", 0.04) * rev
            nwc_now = base.get("nwc_pct_sales", 0.10) * rev
            nwc_tgt = base.get("nwc_target_pct_sales", base.get("nwc_pct_sales", 0.10)) * rev
            d_nwc = nwc_tgt - nwc_now
            fcf1 = nopat + da - capex - d_nwc
            fcf1 = np.maximum(0.0, fcf1)
        
        years = int(d["years"])
        fcf_series = np.empty((n, years))
        fcf_series[:, 0] = fcf1
        for t in range(1, years):
            fcf_series[:, t] = fcf_series[:, t-1] * (1.0 + g)
        
        disc = np.array([1.0 / (1.0 + wacc)**(t+1) for t in range(years)])
        pv_years = (fcf_series * disc.T).sum(axis=1)
        fcf_term = fcf_series[:, -1] * (1.0 + g)
        ev_term = fcf_term / np.maximum(wacc - tg, 0.001)
        pv_term = ev_term / (1.0 + wacc)**years
        ev = pv_years + pv_term
        return ev
    
    def _generate_visualizations(
        self, blended_price: np.ndarray, paths_sample: np.ndarray, ticker: str
    ) -> Tuple[str, str]:
        """Genera visualizaciones y retorna como base64.
        
        Returns:
            Tuple de (histogram_base64, paths_base64)
        """
        # Histograma
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.hist(blended_price, bins=60, edgecolor='black', alpha=0.7)
        ax1.axvline(np.median(blended_price), color='red', linestyle='--', label=f'Mediana: ${np.median(blended_price):.2f}')
        ax1.axvline(np.mean(blended_price), color='green', linestyle='--', label=f'Media: ${np.mean(blended_price):.2f}')
        ax1.set_xlabel("Precio por acción ($)")
        ax1.set_ylabel("Frecuencia")
        ax1.set_title(f"Distribución de Precios - {ticker} (Monte Carlo)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png', bbox_inches='tight', dpi=100)
        buf1.seek(0)
        hist_base64 = base64.b64encode(buf1.read()).decode('utf-8')
        plt.close(fig1)
        
        # Paths muestrales
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        n_paths_show = min(50, paths_sample.shape[0])
        for i in range(n_paths_show):
            ax2.plot(paths_sample[i, :], alpha=0.3, linewidth=0.5)
        ax2.set_xlabel("Paso de simulación")
        ax2.set_ylabel("Precio ($)")
        ax2.set_title(f"Paths Muestrales - {ticker} (Monte Carlo)")
        ax2.grid(True, alpha=0.3)
        
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', bbox_inches='tight', dpi=100)
        buf2.seek(0)
        paths_base64 = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close(fig2)
        
        return hist_base64, paths_base64
    
    def value(
        self,
        ticker: str,
        cfg: Dict[str, Any],
        n_paths: int = 10_000,
        seed: Optional[int] = 42,
    ) -> MonteCarloValuation:
        """Ejecuta simulación Monte Carlo.
        
        Args:
            ticker: Símbolo de la empresa
            cfg: Configuración completa (puede venir de get_preset o ser personalizada)
            n_paths: Número de simulaciones
            seed: Semilla para reproducibilidad
            
        Returns:
            MonteCarloValuation con resultados completos
        """
        rng = np.random.default_rng(seed)
        self._global_state = {}
        
        # 1. Macro
        _, shocks = self._draw_macro(cfg["macro"], size=n_paths, rng=rng)
        
        # 2. Crecimiento y margen con correlación + FX
        base_rev = cfg["base"]["revenue"] * (1.0 + cfg["base"]["revenue_growth_next"])
        g_noise, m_noise = self._draw_correlated_growth_margin(
            n_paths,
            cfg["noise"]["rev_growth_sd"],
            cfg["noise"]["margin_sd"],
            cfg["noise"]["rho_growth_margin"],
            rng
        )
        
        # FX
        rev_mix = cfg.get("revenue_ccy_mix", {})
        fx_adj = np.zeros(n_paths)
        for ccy, w in rev_mix.items():
            fx_adj += w * np.array(shocks["fx"].get(ccy, np.zeros(n_paths)))
        rev_growth = shocks["rev"] + g_noise + fx_adj
        revenue_12m = base_rev * (1.0 + rev_growth)
        
        base_margin = cfg["base"]["ebitda_margin"]
        margin_12m = self._clamp(base_margin + shocks["margin"] + m_noise, 0.0, 0.6)
        ebitda_12m = revenue_12m * margin_12m
        
        self._global_state["revenue_12m"] = revenue_12m
        self._global_state["ebitda_12m"] = ebitda_12m
        
        # 3. Acciones después de recompras
        shares_after = cfg["shares_out"] * (1.0 - shocks["buybacks"])
        
        # 4. Valuaciones por método
        m = cfg["multiples"]
        ev_sales = self._ev_sales_valuation(revenue_12m, m["ev_sales_mean"], m["ev_sales_sd"], rng)
        ev_ebitda = self._ev_ebitda_valuation(ebitda_12m, m["ev_ebitda_mean"], m["ev_ebitda_sd"], rng)
        
        # EPS
        da = cfg["base"].get("da_pct_sales", 0.03) * revenue_12m
        ebit = ebitda_12m - da
        ni = np.maximum(0.0, ebit) * (1.0 - cfg["base"].get("tax_rate", 0.25))
        eps = ni / self._safe_positive(shares_after)
        
        # TBVps y NAVps
        tbv_ps = cfg.get("tbv_ps", None)
        if tbv_ps is None:
            tbv_ps = np.full(n_paths, np.nan)
        else:
            tbv_ps = np.full(n_paths, float(tbv_ps))
        
        nav_ps = cfg.get("nav_ps", None)
        if nav_ps is None:
            nav_ps = np.full(n_paths, np.nan)
        else:
            nav_ps = np.full(n_paths, float(nav_ps))
        
        price_pe = self._pe_valuation(eps, m["pe_mean"], m["pe_sd"], rng)
        price_pbv = self._pbv_valuation(tbv_ps, m["pbv_mean"], m["pbv_sd"], rng)
        price_pnav = self._pnav_valuation(nav_ps, m["p_nav_mean"], m["p_nav_sd"], rng)
        
        price_sales = self._price_from_ev(ev_sales, cfg["net_debt"], shares_after)
        price_ebitda = self._price_from_ev(ev_ebitda, cfg["net_debt"], shares_after)
        
        ev_dcf = self._dcf_valuation(cfg, shocks["fcf"], shocks["wacc"], rng)
        price_dcf = self._price_from_ev(ev_dcf, cfg["net_debt"], shares_after)
        
        # 5. Mezcla ponderada
        w = cfg["weights"]
        stack = []
        wts = []
        methods_dict = {
            "dcf": price_dcf,
            "ev_sales": price_sales,
            "ev_ebitda": price_ebitda,
            "pe": price_pe,
            "pbv": price_pbv,
            "p_nav": price_pnav
        }
        
        for key, arr in methods_dict.items():
            wt = w.get(key, 0.0)
            if wt > 0:
                stack.append(np.nan_to_num(arr, nan=0.0))
                wts.append(wt)
        
        if len(stack) == 0:
            raise ValueError("No hay métodos activos (pesos>0)")
        
        wts = np.array(wts)
        wts = wts / wts.sum()
        blended_price = np.average(np.vstack(stack), axis=0, weights=wts)
        
        # 6. Total return
        div_next = cfg["current_price"] * cfg.get("dividend_yield", 0.0) * (1.0 + cfg.get("dividend_growth", 0.0))
        tr_price = blended_price + div_next
        
        # 7. Resultados
        pct = np.percentile(blended_price, [5, 10, 25, 50, 75, 90, 95])
        
        # Crear DataFrame con paths (muestra de primeros 1000 para no hacerlo muy pesado)
        sample_size = min(1000, n_paths)
        sample_indices = np.random.choice(n_paths, size=sample_size, replace=False)
        
        paths_df = pd.DataFrame({
            "price": blended_price[sample_indices],
            "price_TR": tr_price[sample_indices],
            "rev_12m": revenue_12m[sample_indices],
            "ebitda_12m": ebitda_12m[sample_indices],
            "margin_12m": margin_12m[sample_indices],
            "eps_12m": eps[sample_indices],
            "shares_after": shares_after[sample_indices],
        })
        
        # Generar visualizaciones
        # Para paths muestrales, crear una matriz simple con los primeros 50 paths
        # (en una simulación real podríamos tener paths temporales, aquí solo mostramos distribución final)
        paths_sample = blended_price[:50].reshape(50, 1)
        
        hist_base64, paths_base64 = self._generate_visualizations(blended_price, paths_sample, ticker)
        
        # Métodos usados
        methods_used = {k: v for k, v in w.items() if v > 0}
        
        # Explicación
        explanation = self._generate_explanation(
            ticker, blended_price, cfg["current_price"], pct, methods_used
        )
        
        return MonteCarloValuation(
            ticker=ticker,
            percentiles={
                "p5": float(pct[0]),
                "p10": float(pct[1]),
                "p25": float(pct[2]),
                "p50": float(pct[3]),
                "p75": float(pct[4]),
                "p90": float(pct[5]),
                "p95": float(pct[6]),
            },
            mean=float(np.nanmean(blended_price)),
            median=float(np.median(blended_price)),
            current_price=cfg["current_price"],
            paths_df=paths_df,
            visualization_histogram=hist_base64,
            visualization_paths=paths_base64,
            configuration_used=cfg,
            methods_used=methods_used,
            explanation=explanation,
        )
    
    def _generate_explanation(
        self,
        ticker: str,
        blended_price: np.ndarray,
        current_price: float,
        percentiles: np.ndarray,
        methods_used: Dict[str, float],
    ) -> str:
        """Genera explicación de los resultados."""
        median = float(np.median(blended_price))
        upside_downside = ((median - current_price) / current_price) * 100
        
        if upside_downside > 20:
            valuation_status = "significativamente infravalorada"
        elif upside_downside > 5:
            valuation_status = "ligeramente infravalorada"
        elif upside_downside < -20:
            valuation_status = "significativamente sobrevalorada"
        elif upside_downside < -5:
            valuation_status = "ligeramente sobrevalorada"
        else:
            valuation_status = "razonablemente valorada"
        
        methods_text = ", ".join([f"{k.upper()}" for k in methods_used.keys()])
        
        explanation = (
            f"{ticker} está {valuation_status} según simulación Monte Carlo. "
            f"Precio actual: ${current_price:.2f} vs valor mediano estimado: ${median:.2f} "
            f"(diferencia: {upside_downside:+.1f}%). "
            f"Rango percentil 5-95: ${percentiles[0]:.2f} - ${percentiles[6]:.2f}. "
            f"Métodos utilizados: {methods_text}."
        )
        
        return explanation


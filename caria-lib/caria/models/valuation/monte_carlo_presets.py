"""Presets de configuración para Monte Carlo Valuation por industria y etapa."""

from __future__ import annotations

from typing import Dict, Any
from copy import deepcopy

# Base preset del template
BASE_PRESET: Dict[str, Any] = {
    "as_of": "YYYY-MM-DD",
    "ticker": "TICK",
    "industry": "generic",
    "current_price": 50.0,
    "shares_out": 1_000_000_000,
    "net_debt": 0.0,
    "report_ccy": "USD",
    "revenue_ccy_mix": {"USD": 1.0},
    "base": {
        "revenue": 10e9,
        "revenue_growth_next": 0.03,
        "ebitda_margin": 0.18,
        "tax_rate": 0.23,
        "da_pct_sales": 0.035,
        "capex_pct_sales": 0.04,
        "nwc_pct_sales": 0.10,
        "nwc_target_pct_sales": 0.10,
        "fcf_start": None,
    },
    "macro": {
        "states": ["recession", "normal", "boom"],
        "probs": [0.15, 0.60, 0.25],
        "rev_shock": [-0.03, 0.00, 0.02],
        "margin_shock": [-0.01, 0.00, 0.01],
        "wacc_shock": [0.01, 0.00, -0.005],
        "fcf_shock": [-0.4e9, 0.0, 0.8e9],
        "buyback_rate": [0.00, 0.02, 0.05],
        "fx": {
            "USD": [0.00, 0.00, 0.00],
        }
    },
    "noise": {
        "rev_growth_sd": 0.01,
        "margin_sd": 0.01,
        "rho_growth_margin": 0.35
    },
    "multiples": {
        "ev_sales_mean": 2.0, "ev_sales_sd": 0.3,
        "ev_ebitda_mean": 12.0, "ev_ebitda_sd": 3.0,
        "pe_mean": 20.0, "pe_sd": 4.0,
        "pbv_mean": 1.8, "pbv_sd": 0.4,
        "p_nav_mean": 0.9, "p_nav_sd": 0.15,
    },
    "dcf": {
        "years": 5,
        "wacc_mean": 0.09, "wacc_sd": 0.01,
        "tg_mean": 0.02, "tg_sd": 0.005,
        "fcf_growth_mean": 0.03, "fcf_growth_sd": 0.01,
    },
    "weights": {"dcf": 0.20, "ev_sales": 0.35, "ev_ebitda": 0.35, "pe": 0.05, "pbv": 0.00, "p_nav": 0.05},
    "dividend_yield": 0.02,
    "dividend_growth": 0.06,
    "fund_buybacks_with_fcf": False
}


# Presets por industria
INDUSTRY_PRESETS: Dict[str, Dict[str, Any]] = {
    "healthcare": {
        **deepcopy(BASE_PRESET),
        "industry": "healthcare",
        "base": {
            **BASE_PRESET["base"],
            "revenue_growth_next": 0.05,
            "ebitda_margin": 0.20,
            "tax_rate": 0.21,
        },
        "multiples": {
            **BASE_PRESET["multiples"],
            "ev_sales_mean": 3.5, "ev_sales_sd": 0.5,
            "ev_ebitda_mean": 15.0, "ev_ebitda_sd": 3.0,
            "pe_mean": 25.0, "pe_sd": 5.0,
        },
        "dcf": {
            **BASE_PRESET["dcf"],
            "wacc_mean": 0.085, "wacc_sd": 0.01,
            "fcf_growth_mean": 0.05, "fcf_growth_sd": 0.015,
        },
        "weights": {"dcf": 0.25, "ev_sales": 0.30, "ev_ebitda": 0.35, "pe": 0.10, "pbv": 0.00, "p_nav": 0.00},
    },
    "software": {
        **deepcopy(BASE_PRESET),
        "industry": "software",
        "base": {
            **BASE_PRESET["base"],
            "revenue_growth_next": 0.15,
            "ebitda_margin": 0.25,
            "tax_rate": 0.21,
            "capex_pct_sales": 0.02,
            "nwc_pct_sales": 0.05,
        },
        "multiples": {
            **BASE_PRESET["multiples"],
            "ev_sales_mean": 8.0, "ev_sales_sd": 2.0,
            "ev_ebitda_mean": 30.0, "ev_ebitda_sd": 8.0,
            "pe_mean": 35.0, "pe_sd": 10.0,
        },
        "dcf": {
            **BASE_PRESET["dcf"],
            "wacc_mean": 0.10, "wacc_sd": 0.015,
            "fcf_growth_mean": 0.15, "fcf_growth_sd": 0.05,
        },
        "weights": {"dcf": 0.20, "ev_sales": 0.50, "ev_ebitda": 0.25, "pe": 0.05, "pbv": 0.00, "p_nav": 0.00},
    },
    "bank": {
        **deepcopy(BASE_PRESET),
        "industry": "bank",
        "base": {
            **BASE_PRESET["base"],
            "revenue_growth_next": 0.03,
            "ebitda_margin": 0.45,
            "tax_rate": 0.21,
        },
        "multiples": {
            **BASE_PRESET["multiples"],
            "pe_mean": 11.0, "pe_sd": 2.0,
            "pbv_mean": 1.2, "pbv_sd": 0.2,
            "ev_sales_mean": 0.0, "ev_sales_sd": 0.0,
            "ev_ebitda_mean": 0.0, "ev_ebitda_sd": 0.0,
        },
        "dcf": {
            **BASE_PRESET["dcf"],
            "wacc_mean": 0.08, "wacc_sd": 0.01,
        },
        "weights": {"dcf": 0.10, "ev_sales": 0.00, "ev_ebitda": 0.00, "pe": 0.60, "pbv": 0.30, "p_nav": 0.00},
    },
    "mining": {
        **deepcopy(BASE_PRESET),
        "industry": "mining",
        "base": {
            **BASE_PRESET["base"],
            "revenue_growth_next": 0.00,
            "ebitda_margin": 0.32,
            "tax_rate": 0.25,
        },
        "multiples": {
            **BASE_PRESET["multiples"],
            "ev_ebitda_mean": 6.0, "ev_ebitda_sd": 1.5,
            "p_nav_mean": 0.9, "p_nav_sd": 0.1,
            "ev_sales_mean": 0.0, "ev_sales_sd": 0.0,
            "pe_mean": 0.0, "pe_sd": 0.0,
        },
        "dcf": {
            **BASE_PRESET["dcf"],
            "wacc_mean": 0.10, "wacc_sd": 0.015,
        },
        "weights": {"dcf": 0.25, "ev_sales": 0.00, "ev_ebitda": 0.25, "pe": 0.00, "pbv": 0.00, "p_nav": 0.50},
    },
    "consumer": {
        **deepcopy(BASE_PRESET),
        "industry": "consumer",
        "base": {
            **BASE_PRESET["base"],
            "revenue_growth_next": 0.04,
            "ebitda_margin": 0.15,
            "tax_rate": 0.23,
        },
        "multiples": {
            **BASE_PRESET["multiples"],
            "ev_sales_mean": 2.0, "ev_sales_sd": 0.4,
            "ev_ebitda_mean": 12.0, "ev_ebitda_sd": 3.0,
            "pe_mean": 20.0, "pe_sd": 4.0,
        },
        "dcf": {
            **BASE_PRESET["dcf"],
            "wacc_mean": 0.09, "wacc_sd": 0.01,
        },
        "weights": {"dcf": 0.20, "ev_sales": 0.40, "ev_ebitda": 0.35, "pe": 0.05, "pbv": 0.00, "p_nav": 0.00},
    },
    "technology": {
        **deepcopy(BASE_PRESET),
        "industry": "technology",
        "base": {
            **BASE_PRESET["base"],
            "revenue_growth_next": 0.10,
            "ebitda_margin": 0.22,
            "tax_rate": 0.21,
        },
        "multiples": {
            **BASE_PRESET["multiples"],
            "ev_sales_mean": 4.0, "ev_sales_sd": 1.0,
            "ev_ebitda_mean": 18.0, "ev_ebitda_sd": 5.0,
            "pe_mean": 25.0, "pe_sd": 6.0,
        },
        "dcf": {
            **BASE_PRESET["dcf"],
            "wacc_mean": 0.095, "wacc_sd": 0.012,
            "fcf_growth_mean": 0.10, "fcf_growth_sd": 0.03,
        },
        "weights": {"dcf": 0.25, "ev_sales": 0.35, "ev_ebitda": 0.30, "pe": 0.10, "pbv": 0.00, "p_nav": 0.00},
    },
}


# Ajustes por riesgo geopolítico del país
COUNTRY_RISK_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    "low": {  # USA, Canadá, países desarrollados estables
        "wacc_adjustment": 0.0,
        "volatility_adjustment": 0.0,
    },
    "medium": {  # Países emergentes estables
        "wacc_adjustment": 0.01,
        "volatility_adjustment": 0.005,
    },
    "high": {  # Países con alto riesgo geopolítico
        "wacc_adjustment": 0.02,
        "volatility_adjustment": 0.01,
    },
}


def map_sector_to_industry(sector: str | None) -> str:
    """Mapea sector de CARIA a industria del preset.
    
    Args:
        sector: Sector de la empresa (puede ser None)
        
    Returns:
        Nombre de industria para usar en presets
    """
    if not sector:
        return "generic"
    
    sector_lower = sector.lower()
    
    # Mapeo de sectores comunes
    if any(x in sector_lower for x in ["healthcare", "pharma", "biotech", "medical"]):
        return "healthcare"
    elif any(x in sector_lower for x in ["software", "saas", "cloud", "tech services"]):
        return "software"
    elif any(x in sector_lower for x in ["bank", "financial", "finance"]):
        return "bank"
    elif any(x in sector_lower for x in ["mining", "metals", "minerals"]):
        return "mining"
    elif any(x in sector_lower for x in ["consumer", "retail", "apparel", "consumer goods"]):
        return "consumer"
    elif any(x in sector_lower for x in ["technology", "tech", "semiconductor", "hardware"]):
        return "technology"
    else:
        return "generic"


def get_preset(
    industry: str | None = None,
    stage: str = "consolidated",
    country_risk: str = "low",
    sector: str | None = None,
) -> Dict[str, Any]:
    """Obtiene preset de configuración según industria, etapa y riesgo geopolítico.
    
    Args:
        industry: Industria específica (si None, se infiere de sector)
        stage: Etapa de la empresa ("consolidated" o "pre_revenue")
        country_risk: Nivel de riesgo geopolítico ("low", "medium", "high")
        sector: Sector de la empresa (para mapear a industria si industry es None)
        
    Returns:
        Diccionario de configuración para Monte Carlo
    """
    # Determinar industria
    if not industry and sector:
        industry = map_sector_to_industry(sector)
    elif not industry:
        industry = "generic"
    
    # Obtener preset base de industria
    preset = INDUSTRY_PRESETS.get(industry, deepcopy(BASE_PRESET))
    preset = deepcopy(preset)
    
    # Ajustar por etapa
    if stage == "pre_revenue":
        # Para pre-revenue, ajustar crecimiento y márgenes más conservadores
        preset["base"]["revenue_growth_next"] = preset["base"].get("revenue_growth_next", 0.03) * 1.5
        preset["base"]["ebitda_margin"] = max(0.0, preset["base"].get("ebitda_margin", 0.18) - 0.05)
        # Aumentar volatilidad
        preset["noise"]["rev_growth_sd"] = preset["noise"].get("rev_growth_sd", 0.01) * 2
        preset["noise"]["margin_sd"] = preset["noise"].get("margin_sd", 0.01) * 1.5
    
    # Aplicar ajustes por riesgo geopolítico
    risk_adj = COUNTRY_RISK_ADJUSTMENTS.get(country_risk, COUNTRY_RISK_ADJUSTMENTS["low"])
    preset["dcf"]["wacc_mean"] += risk_adj["wacc_adjustment"]
    preset["dcf"]["wacc_sd"] += risk_adj["volatility_adjustment"]
    
    # Ajustar volatilidad de múltiplos según riesgo
    for key in ["ev_sales_sd", "ev_ebitda_sd", "pe_sd", "pbv_sd", "p_nav_sd"]:
        if key in preset["multiples"]:
            preset["multiples"][key] *= (1.0 + risk_adj["volatility_adjustment"] * 10)
    
    return preset


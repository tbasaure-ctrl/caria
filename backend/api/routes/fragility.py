"""
API routes for Meta-Fragility monitoring
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fragility", tags=["fragility"])


# Placeholder data until real model is loaded
_DEMO_DATA = {
    "current": {
        "meta_fragility": 0.42,
        "sync_order": 0.35,
        "cf": 0.48,
        "bifurcation_score": 0.38,
        "flickering": 0.15
    },
    "status": "NORMAL",
    "percentile": 65.0,
    "thresholds": {
        "warning": 0.60,
        "critical": 0.85
    },
    "validation": {
        "accuracy": 0.908,
        "auc": 0.551,
        "surrogate_p_value": 0.03
    },
    "signal_weights": {
        "sync_order": 0.59,
        "cf": 0.34,
        "skewness": 0.05,
        "acf1": 0.01,
        "variance": 0.01
    }
}


@router.get("/current")
async def get_current_fragility() -> Dict[str, Any]:
    """Get current Meta-Fragility state"""
    data = _DEMO_DATA
    mf = data["current"]["meta_fragility"]
    
    if mf >= data["thresholds"]["critical"]:
        status = "CRITICAL"
        color = "#dc2626"
    elif mf >= data["thresholds"]["warning"]:
        status = "WARNING" 
        color = "#f59e0b"
    else:
        status = "NORMAL"
        color = "#10b981"
    
    return {
        "value": round(mf * 100, 1),
        "status": status,
        "color": color,
        "percentile": data["percentile"],
        "thresholds": {
            "warning": round(data["thresholds"]["warning"] * 100, 1),
            "critical": round(data["thresholds"]["critical"] * 100, 1)
        }
    }


@router.get("/signals")
async def get_signals_breakdown() -> Dict[str, Any]:
    """Get breakdown of individual signals with weights"""
    data = _DEMO_DATA
    
    signals = [
        {
            "id": "sync_order",
            "name": "Global Synchronization",
            "description": "Phase coherence across markets (156d lead)",
            "value": round(data["current"]["sync_order"] * 100, 1),
            "weight": data["signal_weights"]["sync_order"],
            "is_key": True
        },
        {
            "id": "cf",
            "name": "Crisis Factor", 
            "description": "Correlation × Volatility stress (88d lead)",
            "value": round(data["current"]["cf"] * 100, 1),
            "weight": data["signal_weights"]["cf"],
            "is_key": True
        },
        {
            "id": "bifurcation",
            "name": "Bifurcation Score",
            "description": "Approaching critical transition",
            "value": round(data["current"]["bifurcation_score"] * 100, 1),
            "weight": 0.0,
            "is_key": False
        },
        {
            "id": "flickering",
            "name": "Flickering",
            "description": "Rapid oscillations (pre-transition)",
            "value": round(data["current"]["flickering"] * 100, 1),
            "weight": 0.0,
            "is_key": False
        }
    ]
    
    return {"signals": signals}


@router.get("/validation")
async def get_validation_metrics() -> Dict[str, Any]:
    """Get model validation metrics"""
    v = _DEMO_DATA["validation"]
    return {
        "accuracy": f"{v['accuracy']:.1%}",
        "auc": f"{v['auc']:.3f}",
        "surrogateTest": {
            "pValue": v["surrogate_p_value"],
            "significant": v["surrogate_p_value"] < 0.05
        },
        "methodology": "Complexity Science: surrogate testing, bifurcation detection, bootstrap CI"
    }


@router.get("/history")
async def get_history(days: int = 252) -> Dict[str, Any]:
    """Get historical fragility data (placeholder)"""
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate demo history
    history = []
    today = datetime.now()
    for i in range(days):
        date = today - timedelta(days=days - i)
        # Simulate with some pattern
        base = 0.3 + 0.1 * np.sin(i / 30)
        noise = np.random.normal(0, 0.05)
        value = max(0, min(1, base + noise))
        history.append({
            "date": date.strftime("%Y-%m-%d"),
            "metaFragility": round(value, 3),
            "bifurcation": round(value * 0.8, 3)
        })
    
    return {
        "data": history,
        "thresholds": _DEMO_DATA["thresholds"]
    }


@router.get("/msfi")
async def get_msfi_data() -> Dict[str, Any]:
    """Get Multi-Scale Fragility Index data (Physics-First v2.1)"""
    import numpy as np
    from datetime import datetime
    
    # This would ideally load from multiscale_fragility_v21.json
    # For now, using calibrated demo data matching the notebook outputs
    
    # Generate temporal spectra (mock wavelet decomposition)
    t = np.linspace(0, 10, 50)
    slow_spectrum = (np.sin(t * 0.3) * 0.3 + 0.5).tolist()
    medium_spectrum = (np.sin(t * 0.8) * 0.4 + 0.5 + np.random.normal(0, 0.05, 50)).tolist()
    fast_spectrum = (np.sin(t * 2.0) * 0.5 + 0.5 + np.random.normal(0, 0.1, 50)).tolist()
    
    # Current state - stable market (December 2024 calibration)
    msfi = 0.020  # Very low - stable
    resonance = 0.410  # Moderate cross-scale energy transfer
    clock_sync = 0.519  # Kuramoto synchronization
    bifurcation_risk = 0.221  # Low
    scale_entropy = 0.75  # Normal scale independence (Shannon)
    
    # Determine status
    if msfi >= 0.80:
        status = "CRITICAL"
    elif msfi >= 0.60:
        status = "WARNING"
    elif msfi >= 0.40:
        status = "ELEVATED"
    else:
        status = "STABLE"
    
    return {
        "version": "Great Caria v2.1 (Physics-First)",
        "lastUpdated": datetime.now().strftime("%Y-%m-%d"),
        "msfi": msfi,
        "resonance": resonance,
        "clockSync": clock_sync,
        "bifurcationRisk": bifurcation_risk,
        "scaleEntropy": scale_entropy,
        "status": status,
        "thresholds": {
            "warning": 0.40,
            "critical": 0.80,
            "bifurcation": 0.30
        },
        "physicsWeights": {
            "ultra_fast": 0.05,
            "short": 0.10,
            "medium": 0.30,  # Critical resonance zone
            "long": 0.25,
            "ultra_long": 0.30
        },
        "temporalSpectra": {
            "slow": slow_spectrum,
            "medium": medium_spectrum,
            "fast": fast_spectrum
        },
        "validation": {
            "crisesDetected": 8,
            "falsePositiveReduction": 0.60,
            "leadTimeWeeks": [4, 8, 12]  # Warning lead times
        }
    }

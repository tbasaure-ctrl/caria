"""
Great Caria Meta-Fragility Service
Serves the Meta-Fragility Index with Sync (59%) + CF (34%) as main signals
Based on Complexity Science validation with surrogate testing
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# Signal weights based on lead time analysis (156d Sync, 88d CF)
SIGNAL_WEIGHTS = {
    "sync_order": 0.59,  # 156 day lead time - strongest early warning
    "cf": 0.34,          # 88 day lead time - second strongest
    "skewness": 0.05,    # 14 day lead time
    "acf1": 0.01,        # 3 day lead time
    "variance": 0.01     # minimal contribution
}


class FragilityService:
    """Service for systemic fragility monitoring"""
    
    def __init__(self, signals_path: Optional[str] = None):
        self.signals_path = signals_path or self._find_signals_file()
        self._cache = None
        self._cache_time = None
        self._cache_ttl = 3600  # 1 hour
    
    def _find_signals_file(self) -> str:
        """Find the fragility signals file"""
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / "models" / "fragility_signals.json",
            Path("models/fragility_signals.json"),
            Path("/app/models/fragility_signals.json"),
        ]
        for p in possible_paths:
            if p.exists():
                return str(p)
        return str(possible_paths[0])
    
    def _load_signals(self) -> Dict[str, Any]:
        """Load signals from JSON file"""
        try:
            with open(self.signals_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Fragility signals file not found: {self.signals_path}")
            return self._get_demo_data()
        except Exception as e:
            logger.error(f"Error loading fragility signals: {e}")
            return self._get_demo_data()
    
    def _get_demo_data(self) -> Dict[str, Any]:
        """Return demo data when real data unavailable"""
        return {
            "version": "Demo",
            "generated": datetime.now().isoformat(),
            "current_fragility": 0.45,
            "fragility_percentile": 65.0,
            "thresholds": {"warning": 0.60, "critical": 0.85},
            "signals": {
                "sync_order": 0.35,
                "curvature": 0.42,
                "acf1": 0.78,
                "variance": 0.25,
                "cf": 0.38
            },
            "history": []
        }
    
    def get_current_fragility(self) -> Dict[str, Any]:
        """Get current fragility state"""
        data = self._load_signals()
        
        # Determine status
        current = data["current_fragility"]
        thresholds = data["thresholds"]
        
        if current >= thresholds["critical"]:
            status = "CRITICAL"
            color = "#dc2626"  # red
        elif current >= thresholds["warning"]:
            status = "WARNING"
            color = "#f59e0b"  # amber
        else:
            status = "NORMAL"
            color = "#10b981"  # green
        
        return {
            "value": round(current * 100, 1),
            "percentile": round(data["fragility_percentile"], 1),
            "status": status,
            "color": color,
            "thresholds": {
                "warning": round(thresholds["warning"] * 100, 1),
                "critical": round(thresholds["critical"] * 100, 1)
            },
            "lastUpdated": data.get("generated", datetime.now().isoformat())
        }
    
    def get_signals_breakdown(self) -> Dict[str, Any]:
        """Get breakdown of individual signals"""
        data = self._load_signals()
        signals = data.get("signals", {})
        
        # Normalize and interpret each signal
        breakdown = []
        
        signal_meta = {
            "sync_order": {"name": "Global Synchronization", "desc": "Phase coherence across markets"},
            "curvature": {"name": "Network Resilience", "desc": "Topological stability of market network"},
            "acf1": {"name": "Critical Slowing", "desc": "Autocorrelation indicating loss of resilience"},
            "variance": {"name": "Volatility Regime", "desc": "System-wide variance"},
            "cf": {"name": "Crisis Factor", "desc": "Combined correlation × volatility stress"}
        }
        
        for key, value in signals.items():
            meta = signal_meta.get(key, {"name": key, "desc": ""})
            breakdown.append({
                "id": key,
                "name": meta["name"],
                "description": meta["desc"],
                "value": round(value * 100, 1) if value <= 1 else round(value, 2),
                "contribution": round(value * 20, 1)  # Approximate contribution
            })
        
        return {"signals": breakdown}
    
    def get_history(self, days: int = 252) -> Dict[str, Any]:
        """Get historical fragility data"""
        data = self._load_signals()
        history = data.get("history", [])
        
        # Limit to requested days
        history = history[-days:] if len(history) > days else history
        
        return {
            "data": history,
            "thresholds": data.get("thresholds", {"warning": 0.6, "critical": 0.85})
        }


# Singleton instance
_fragility_service = None

def get_fragility_service() -> FragilityService:
    global _fragility_service
    if _fragility_service is None:
        _fragility_service = FragilityService()
    return _fragility_service

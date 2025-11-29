import logging
from typing import Dict, Any
from caria.services.liquidity_service import LiquidityService

LOGGER = logging.getLogger(__name__)

class ExecutionAgentService:
    """
    The Sniper: RL-inspired Execution Agent (Position Sizing).
    
    Learns (via rules for MVP) optimal position size based on:
    - Hydraulic Score (Liquidity Regime)
    - Volatility (from VolDetector)
    - Asset Trend
    
    Future: Replace with actual RL agent (PPO/SAC).
    """
    
    def __init__(self):
        self.liquidity_service = LiquidityService()
        
    def get_position_size(self, 
                         volatility_ratio: float = 1.0, 
                         trend_strength: float = 0.5) -> Dict[str, Any]:
        """
        Determines optimal position size based on market conditions.
        
        Args:
            volatility_ratio: Current vol / Avg vol (from VolDetector)
            trend_strength: 0-1 score indicating trend confidence
            
        Returns:
            Dict with 'position_pct', 'reasoning', 'risk_level'
        """
        try:
            # Get current liquidity state
            status = self.liquidity_service.get_current_status()
            score = status.get("score", 50)
            state = status.get("state", "NEUTRAL")
            
            # Base position size rules
            if score < 40:  # CONTRACTION
                if trend_strength > 0.6:
                    # Up trend during contraction = Bear Market Rally
                    position_pct = 0.25
                    reasoning = "🔴 Liquidity contracting but trend up. Small size (bear rally trap risk)."
                    risk_level = "HIGH"
                else:
                    # No trend during contraction = Stay out
                    position_pct = 0.0
                    reasoning = "🔴 Liquidity contracting, no clear trend. No position."
                    risk_level = "EXTREME"
                    
            elif score > 60:  # EXPANSION
                if volatility_ratio < 1.5:
                    # Low vol + expansion = Max size
                    position_pct = 1.0
                    reasoning = "🟢 Liquidity expanding, low volatility. Max size."
                    risk_level = "LOW"
                else:
                    # High vol during expansion = Reduce size
                    position_pct = 0.5
                    reasoning = "🟡 Liquidity expanding but volatility elevated. Half size."
                    risk_level = "MEDIUM"
                    
            else:  # NEUTRAL (40-60)
                if volatility_ratio < 1.3:
                    position_pct = 0.5
                    reasoning = "🟡 Neutral liquidity, normal volatility. Standard size."
                    risk_level = "MEDIUM"
                else:
                    position_pct = 0.25
                    reasoning = "🟡 Neutral liquidity, elevated volatility. Small size."
                    risk_level = "MEDIUM-HIGH"
            
            # Volatility override: If vol spike (>2x), cut position in half
            if volatility_ratio >= 2.0:
                position_pct = position_pct * 0.5
                reasoning += " ⚠️ Vol spike detected - position halved."
                risk_level = "HIGH"
            
            return {
                "position_pct": position_pct,
                "position_description": self._get_position_label(position_pct),
                "reasoning": reasoning,
                "risk_level": risk_level,
                "hydraulic_score": score,
                "liquidity_state": state,
                "volatility_ratio": volatility_ratio
            }
            
        except Exception as e:
            LOGGER.error(f"Error in ExecutionAgent: {e}")
            return {
                "position_pct": 0.5,
                "position_description": "Standard",
                "reasoning": "Error in calculation, using fallback 50%",
                "risk_level": "UNKNOWN",
                "error": str(e)
            }
    
    def _get_position_label(self, pct: float) -> str:
        """Convert position % to label."""
        if pct == 0:
            return "No Position"
        elif pct <= 0.25:
            return "Small (25%)"
        elif pct <= 0.5:
            return "Half (50%)"
        elif pct <= 0.75:
            return "Large (75%)"
        else:
            return "Max (100%)"

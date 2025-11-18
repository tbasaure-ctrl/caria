"""
Conviction Service.
Calculates conviction impact from community responses in Thesis Arena.
"""

from __future__ import annotations

import logging
from typing import Dict, List

LOGGER = logging.getLogger("caria.api.conviction")


class ConvictionService:
    """Service to calculate conviction impact from community responses."""

    def __init__(self):
        """Initialize the service."""
        pass

    def calculate_conviction_impact(
        self,
        responses: Dict[str, str],
        initial_conviction: float,
    ) -> Dict[str, any]:
        """
        Calculate conviction impact from community responses.
        
        Args:
            responses: Dict mapping community name to their response text
            initial_conviction: Initial conviction level (0-100)
            
        Returns:
            Dict with:
            - conviction_change: Change in conviction (-100 to +100)
            - new_conviction: Updated conviction level (0-100)
            - community_impacts: Dict mapping community to their impact score
        """
        # Analyze each response for sentiment and strength
        community_impacts = {}
        total_impact = 0.0
        
        # Keywords that indicate positive/negative sentiment
        positive_keywords = [
            "good", "great", "excellent", "strong", "buy", "bullish",
            "opportunity", "potential", "recommend", "agree", "support",
            "bueno", "excelente", "fuerte", "comprar", "oportunidad",
            "recomiendo", "de acuerdo", "apoyo"
        ]
        
        negative_keywords = [
            "bad", "weak", "sell", "bearish", "risk", "concern", "skeptical",
            "disagree", "caution", "warning", "malo", "débil", "vender",
            "riesgo", "preocupación", "esceptico", "desacuerdo", "precaución"
        ]
        
        strong_modifiers = [
            "very", "extremely", "highly", "strongly", "definitely",
            "muy", "extremadamente", "altamente", "definitivamente"
        ]
        
        for community, response in responses.items():
            response_lower = response.lower()
            
            # Count positive and negative keywords
            positive_count = sum(1 for kw in positive_keywords if kw in response_lower)
            negative_count = sum(1 for kw in negative_keywords if kw in response_lower)
            
            # Check for strong modifiers
            has_strong_modifier = any(mod in response_lower for mod in strong_modifiers)
            
            # Calculate impact score (-1 to +1)
            if positive_count > negative_count:
                impact = 0.3 + (0.2 if has_strong_modifier else 0.0)
                impact = min(impact, 1.0)
            elif negative_count > positive_count:
                impact = -0.3 - (0.2 if has_strong_modifier else 0.0)
                impact = max(impact, -1.0)
            else:
                # Neutral or mixed
                impact = 0.0
            
            # Weight by community type (some communities have more influence)
            community_weights = {
                "value_investor": 1.2,  # Value investors are more conservative
                "crypto_bro": 0.8,  # Crypto bros are more volatile
                "growth_investor": 1.0,
                "contrarian": 1.1,  # Contrarians can be influential
            }
            
            weight = community_weights.get(community, 1.0)
            weighted_impact = impact * weight
            
            community_impacts[community] = {
                "impact": impact,
                "weighted_impact": weighted_impact,
                "sentiment": "positive" if impact > 0 else "negative" if impact < 0 else "neutral",
            }
            
            total_impact += weighted_impact
        
        # Normalize total impact to -100 to +100 range
        # Average of 4 communities with max weight 1.2 = max total of 4.8
        # Normalize to -100 to +100
        max_possible_impact = 4.8
        normalized_impact = (total_impact / max_possible_impact) * 100
        normalized_impact = max(-100, min(100, normalized_impact))
        
        # Calculate new conviction
        new_conviction = initial_conviction + normalized_impact
        new_conviction = max(0, min(100, new_conviction))
        
        conviction_change = new_conviction - initial_conviction
        
        return {
            "conviction_change": conviction_change,
            "new_conviction": new_conviction,
            "initial_conviction": initial_conviction,
            "community_impacts": community_impacts,
            "total_impact": total_impact,
        }


def get_conviction_service() -> ConvictionService:
    """Get singleton instance of ConvictionService."""
    if not hasattr(get_conviction_service, "_instance"):
        get_conviction_service._instance = ConvictionService()
    return get_conviction_service._instance


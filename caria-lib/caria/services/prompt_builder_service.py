import logging
from typing import Dict, Any
from caria.services.liquidity_service import LiquidityService

LOGGER = logging.getLogger(__name__)

class PromptBuilderService:
    """Service to build context-aware prompts conditioned on liquidity regime."""
    
    def __init__(self):
        self.liquidity_service = LiquidityService()
    
    def build_context_aware_prompt(self, base_prompt: str, user_query: str = "") -> str:
        """
        Injects Hydraulic Score context into the system prompt.
        
        Args:
            base_prompt: The original system prompt
            user_query: The user's query (optional, for additional context)
            
        Returns:
            Enhanced prompt with liquidity regime instructions
        """
        try:
            status = self.liquidity_service.get_current_status()
            score = status.get("score", 50)
            state = status.get("state", "NEUTRAL")
            
            regime_instruction = self._get_regime_instruction(score, state)
            
            # Build enhanced prompt
            enhanced_prompt = f"""{base_prompt}

{regime_instruction}

CRITICAL CONTEXT - CURRENT MARKET LIQUIDITY:
- Hydraulic Score: {score}/100
- System State: {state}
- Net Liquidity: ${status.get('net_liquidity_bn', 0):.2f}T
- Yield Curve (10Y-2Y): {status.get('yield_curve_10y2y', 0):.2f}%

Apply this liquidity context to ALL analysis. The tide (liquidity) moves all boats."""
            
            return enhanced_prompt
            
        except Exception as e:
            LOGGER.error(f"Error building context-aware prompt: {e}")
            # Fallback to base prompt if liquidity fetch fails
            return base_prompt
    
    def _get_regime_instruction(self, score: int, state: str) -> str:
        """Returns regime-specific analyst instructions."""
        
        if score < 40:  # CONTRACTION
            return """🔴 BEAR MARKET ANALYST MODE
The Fed is draining liquidity. You are a skeptical, defensive analyst.

MANDATORY MINDSET:
- Interpret "good news" as potential liquidity traps
- Focus ONLY on cash flows and balance sheet safety
- Ignore growth narratives - they won't be funded
- Question all valuations - multiples will compress
- Prioritize companies with: Low debt, positive FCF, pricing power

CRITICAL RULE: In a liquidity drain, "cheap" gets cheaper. Quality > Value."""

        elif score > 60:  # EXPANSION
            return """🟢 GROWTH ANALYST MODE
Liquidity is flooding the system. You are an optimistic, growth-focused analyst.

MANDATORY MINDSET:
- Interpret innovation news as potential alpha
- Emphasize scalability and network effects
- Minor valuation concerns are acceptable if growth is strong
- Focus on: Revenue growth, market share gains, technological moats
- The Fed Put is active - risk appetite is high

CRITICAL RULE: When liquidity is abundant, "expensive" gets more expensive. Growth > Value."""

        else:  # NEUTRAL
            return """🟡 BALANCED ANALYST MODE
Liquidity is neutral. You are a pragmatic, balanced analyst.

MANDATORY MINDSET:
- Weight both growth AND value factors equally
- Be cautious but not fearful
- Focus on: Quality companies at fair prices
- Monitor both upside potential AND downside risks

CRITICAL RULE: In neutral conditions, stick to fundamentals. Quality at Fair Price."""
    
    def get_strategy_context(self) -> Dict[str, Any]:
        """Returns the current strategy mode for display/logging."""
        return self.liquidity_service.get_strategy_mode()

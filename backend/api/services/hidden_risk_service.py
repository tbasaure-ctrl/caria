"""
Hidden Risk Service.
Generates AI-driven risk reports combining Hydraulic Stack, Caria Cortex, and News Sentiment.
"""

import logging
from typing import Any, Dict, List
from uuid import UUID

from api.services.portfolio_analytics import get_portfolio_analytics_service
from api.services.llm_service import LLMService
from api.services.alpha_vantage_client import alpha_vantage_client

# Attempt to import specialized services, handling potential missing dependencies
try:
    from caria.services.liquidity_service import LiquidityService
    LIQUIDITY_AVAILABLE = True
except ImportError:
    LIQUIDITY_AVAILABLE = False

try:
    from caria.services.topology_service import TopologyService
    from caria.services.fmp_service import FMPDataService
    TOPOLOGY_AVAILABLE = True
except ImportError:
    TOPOLOGY_AVAILABLE = False

LOGGER = logging.getLogger("caria.api.services.hidden_risk")

class HiddenRiskService:
    def __init__(self):
        self.analytics_service = get_portfolio_analytics_service()
        self.llm_service = LLMService()
        
        if LIQUIDITY_AVAILABLE:
            self.liquidity_service = LiquidityService()
        else:
            self.liquidity_service = None
            
        if TOPOLOGY_AVAILABLE:
            self.topology_service = TopologyService()
            self.data_service = FMPDataService()
        else:
            self.topology_service = None
            self.data_service = None

    async def generate_report(self, user_id: UUID, db_connection) -> Dict[str, Any]:
        """
        Generates a comprehensive hidden risk report.
        """
        # 1. Get Holdings
        try:
            holdings, _ = self.analytics_service.get_user_holdings_with_prices(user_id, db_connection)
        except Exception as e:
            LOGGER.error(f"Failed to fetch holdings: {e}")
            holdings = []

        if not holdings:
            return {
                "status": "no_holdings",
                "message": "No holdings found to analyze."
            }

        # 2. Get Hydraulic Status (Liquidity)
        liquidity_status = {"score": 50, "state": "NEUTRAL"}
        if self.liquidity_service:
            try:
                liquidity_status = self.liquidity_service.get_current_status()
            except Exception as e:
                LOGGER.warning(f"Failed to fetch liquidity status: {e}")

        # 3. Get Cortex Status (Topology)
        topology_status = {"status": "UNKNOWN", "aliens": []}
        if self.topology_service and self.data_service:
            try:
                market_data = self.data_service.fetch_market_pulse(lookback_days=60)
                if not market_data.empty:
                    topology_status = self.topology_service.scan_market_topology(market_data)
            except Exception as e:
                LOGGER.warning(f"Failed to fetch topology status: {e}")

        # 4. Get News Sentiment (Top 5 holdings by value if possible, or just first 5)
        # Sort by value if available, or quantity
        # holdings is a list of dicts from get_user_holdings_with_prices
        tickers = [h["ticker"] for h in holdings[:5]]
        news_summary = "No recent news available."
        try:
            sentiment_data = alpha_vantage_client.get_news_sentiment(tickers=",".join(tickers), limit=5)
            if sentiment_data:
                headlines = [f"- {item.get('title')} (Sentiment: {item.get('overall_sentiment_label')})" for item in sentiment_data]
                news_summary = "\n".join(headlines)
        except Exception as e:
            LOGGER.warning(f"Failed to fetch news sentiment: {e}")

        # 5. Construct Prompt
        holdings_str = "\n".join([f"- {h['ticker']}: {h['quantity']} units @ ${h.get('current_price', 'N/A')}" for h in holdings])
        
        prompt = f"""
        ANALYZE HIDDEN RISKS FOR THIS PORTFOLIO.

        **PORTFOLIO CONTEXT:**
        {holdings_str}

        **MARKET REGIME (Hydraulic Stack):**
        - Liquidity Score: {liquidity_status.get('score', 50)}/100
        - State: {liquidity_status.get('state', 'NEUTRAL')}
        
        **MARKET TOPOLOGY (Caria Cortex):**
        - Structural Anomalies (Aliens): {[a.get('ticker') for a in topology_status.get('aliens', [])]}
        - Fragility Status: {topology_status.get('status', 'UNKNOWN')}

        **RECENT NEWS & SENTIMENT:**
        {news_summary}

        **TASK:**
        Identify 4-5 HIDDEN RISKS that the user might not be aware of. 
        Focus on:
        1. Sector concentration relative to the current Liquidity Regime.
        2. Exposure to "Alien" stocks (disconnected from market physics).
        3. Sentiment headwinds vs Price action (Divergences).
        4. Macro-economic vulnerabilities (Rates, Inflation) given the specific holdings.

        **OUTPUT FORMAT:**
        Return a JSON object with a "risks" key containing a list of objects, each with "title", "severity" (High/Medium/Low), and "description".
        """

        system_prompt = "You are Caria's Risk Engine. You are a cynical, paranoid institutional risk manager. You look for what could go wrong. Be concise, specific, and actionable."

        # 6. Call LLM
        response_text = self.llm_service.call_llm(prompt, system_prompt=system_prompt)
        
        # 7. Parse and Return
        try:
            report = self.llm_service.parse_json_response(response_text)
            if not report or "risks" not in report:
                # Fallback if JSON parsing fails or format is wrong
                return {
                    "status": "success",
                    "risks": [
                        {"title": "Analysis Format Error", "severity": "Low", "description": "Raw analysis generated but format was unexpected. Please try again."}
                    ],
                    "raw_text": response_text
                }
            return {"status": "success", **report}
        except Exception as e:
            LOGGER.error(f"Failed to parse LLM response: {e}")
            return {
                "status": "error", 
                "message": "Failed to generate risk report.",
                "raw_text": response_text
            }

hidden_risk_service = HiddenRiskService()



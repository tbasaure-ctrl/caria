import logging
import pandas as pd
from typing import List, Dict, Any
from caria.services.factor_service import FactorService

LOGGER = logging.getLogger("caria.api.services.alpha")

class AlphaService:
    """Service for generating Alpha Stock Picks using Composite Alpha Score (CAS)."""

    def __init__(self, factor_service: FactorService):
        self.factor_service = factor_service

    def compute_alpha_picks(self, top_n_candidates: int = 100) -> List[Dict[str, Any]]:
        """
        Computes the top 3 stock picks based on CAS.
        
        CAS = 0.32*Quality + 0.28*Momentum + 0.22*Valuation + 0.18*Catalyst
        """
        # 1. Fetch universe data (using factor service to get raw scores)
        # We ask for more candidates to ensure we have a good pool for ranking
        try:
            candidates = self.factor_service.screen_companies(top_n=top_n_candidates)
        except Exception as e:
            LOGGER.warning(f"FactorService failed: {e}. Trying fallback to fundamentals cache...")
            # Fallback: Use fundamentals cache service directly
            candidates = self._get_candidates_from_cache(top_n_candidates)
        
        if not candidates:
            LOGGER.error("No candidates found for alpha picks")
            return []

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(candidates)

        # 2. Map existing factors to our model's inputs
        # Note: 'catalyst_score' is proxied by 'growth_score' as per plan
        # factor_scores is a dict in the list of dicts, so we need to extract it
        # The FactorService returns a list of dicts where 'factor_scores' is a nested dict.
        # Let's normalize it.
        
        # Extract factor scores into columns
        factor_data = df['factor_scores'].apply(pd.Series)
        df = pd.concat([df.drop(['factor_scores'], axis=1), factor_data], axis=1)

        # Map to required columns
        # momentum_score <- momentum
        # quality_score <- profitability
        # valuation_score <- value
        # catalyst_score <- growth (proxy)
        
        df["momentum_score"] = df.get("momentum", 50.0)
        df["quality_score"] = df.get("profitability", 50.0)
        df["valuation_score"] = df.get("value", 50.0)
        df["catalyst_score"] = df.get("growth", 50.0)

        # 3. Calculate Percentile Ranks
        df["mom_r"] = df["momentum_score"].rank(pct=True)
        df["qual_r"] = df["quality_score"].rank(pct=True)
        # For valuation, typically lower (cheaper) is better if it's a multiple like PE, 
        # but here 'value_score' from FactorService is likely already a "higher is better" score 
        # (e.g. 100 = very undervalued/good). 
        # Let's assume FactorService returns "scores" where higher is better.
        # If FactorService 'value' is a score (0-100), then higher is better.
        # Checking FactorService implementation... it returns 'value_score'.
        # Assuming standard score behavior: Higher = Better Value.
        df["val_r"] = df["valuation_score"].rank(pct=True) 
        df["cat_r"] = df["catalyst_score"].rank(pct=True)

        # 4. Composite Alpha Score (CAS)
        df["CAS"] = (
            0.32 * df["qual_r"] +
            0.28 * df["mom_r"] +
            0.22 * df["val_r"] +
            0.18 * df["cat_r"]
        )

        # 5. Sort and take top 3
        top_picks = df.sort_values("CAS", ascending=False).head(3)

        # 6. Format output
        results = []
        for _, row in top_picks.iterrows():
            explanation = self._generate_explanation(row)
            results.append({
                "ticker": row["ticker"],
                "company_name": row.get("company_name", row["ticker"]), # Fallback
                "sector": row.get("sector", "Unknown"),
                "cas_score": round(row["CAS"] * 100, 1), # 0-100 scale
                "scores": {
                    "momentum": round(row["momentum_score"], 1),
                    "quality": round(row["quality_score"], 1),
                    "valuation": round(row["valuation_score"], 1),
                    "catalyst": round(row["catalyst_score"], 1),
                },
                "explanation": explanation
            })
            
        return results

    def _generate_explanation(self, row: pd.Series) -> str:
        """Generates a dynamic explanation based on factor ranks."""
        qual_r = row["qual_r"]
        val_r = row["val_r"]
        mom_r = row["mom_r"]
        cat_r = row["cat_r"]
        
        parts = []
        
        # Primary driver
        if qual_r > 0.8:
            parts.append("This stock stands out for its exceptional quality and profitability")
        elif val_r > 0.8:
            parts.append("This pick is trading at a highly attractive valuation relative to peers")
        elif mom_r > 0.8:
            parts.append("Strong price momentum drives this selection")
        else:
            parts.append("This stock offers a balanced profile across all key factors")
            
        # Secondary context
        if val_r > 0.7 and qual_r > 0.7:
            parts.append("combining solid fundamentals with a discount price.")
        elif mom_r > 0.7:
            parts.append("with confirmed trend strength.")
        elif cat_r > 0.7:
            parts.append("supported by positive near-term growth catalysts.")
        else:
            parts.append(".")
            
        # Catalyst specific
        if cat_r > 0.8:
            parts.append("Our catalyst model detects significant upcoming drivers.")
        elif cat_r < 0.3:
            parts.append("While catalysts are quiet, the core thesis remains strong.")
            
        return " ".join(parts)
    
    def _get_candidates_from_cache(self, top_n: int = 100) -> List[Dict[str, Any]]:
        """
        Fallback method: Get candidates directly from fundamentals cache service.
        This works when parquet files are missing.
        Calculates scores directly from cached fundamentals data.
        """
        try:
            from api.services.fundamentals_cache_service import get_fundamentals_cache_service
            from api.services.openbb_client import OpenBBClient
            
            cache_service = get_fundamentals_cache_service()
            obb_client = OpenBBClient()
            
            # Get all cached tickers
            cached_tickers = cache_service.get_all_cached_tickers()
            
            if not cached_tickers:
                LOGGER.error("No tickers in cache. Run weekly screening first via POST /api/screening/weekly/yahoo-finance")
                return []
            
            LOGGER.info(f"Using {len(cached_tickers)} tickers from cache for alpha picks")
            
            # Score each ticker
            candidates = []
            for ticker in cached_tickers[:min(top_n * 3, 200)]:  # Limit to 200 for performance
                try:
                    # Get fundamentals from cache
                    cached = cache_service.get_fundamentals(ticker)
                    if not cached or not cached.get('data'):
                        continue
                    
                    data = cached['data']
                    
                    # Get current price for valuation
                    current_price = obb_client.get_current_price(ticker)
                    if not current_price or current_price <= 0:
                        continue
                    
                    # Calculate scores directly from fundamentals
                    quality_score = self._calculate_quality_from_data(data)
                    valuation_score = self._calculate_valuation_from_data(data, current_price)
                    momentum_score = 50.0  # Default, would need price history for real momentum
                    growth_score = self._calculate_growth_from_data(data)
                    
                    # Composite score (same weights as CAS)
                    composite = (
                        0.32 * quality_score +
                        0.28 * momentum_score +
                        0.22 * valuation_score +
                        0.18 * growth_score
                    )
                    
                    candidates.append({
                        "ticker": ticker,
                        "company_name": data.get("company_name", ticker),
                        "sector": data.get("sector", "Unknown"),
                        "factor_scores": {
                            "value": valuation_score,
                            "profitability": quality_score,
                            "growth": growth_score,
                            "momentum": momentum_score,
                        },
                        "composite_score": composite,
                    })
                except Exception as e:
                    LOGGER.debug(f"Error scoring {ticker}: {e}")
                    continue
            
            # Sort by composite score and return top N
            candidates.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
            LOGGER.info(f"✅ Generated {len(candidates)} candidates from cache, returning top {top_n}")
            return candidates[:top_n]
            
        except Exception as e:
            LOGGER.error(f"Error getting candidates from cache: {e}")
            return []
    
    def _calculate_quality_from_data(self, data: Dict[str, Any]) -> float:
        """Calculate quality score from fundamentals data."""
        score = 50.0  # Base score
        
        # ROIC (Return on Invested Capital)
        roic = data.get('roic') or data.get('returnOnEquity')
        if roic:
            score += min(roic * 200, 30)  # Max 30 points
        
        # Profit margins
        net_margin = data.get('netProfitMargin')
        if net_margin:
            score += min(net_margin * 100, 20)  # Max 20 points
        
        # Free cash flow yield
        fcf_yield = data.get('freeCashFlowYield')
        if fcf_yield:
            score += min(fcf_yield * 200, 20)  # Max 20 points
        
        return min(score, 100.0)
    
    def _calculate_valuation_from_data(self, data: Dict[str, Any], current_price: float) -> float:
        """Calculate valuation score from fundamentals data."""
        score = 50.0  # Base score
        
        # Price to Book
        pb = data.get('priceToBookRatio')
        if pb and pb > 0:
            # Lower P/B is better (cheaper)
            if pb < 1.0:
                score += 30  # Very cheap
            elif pb < 2.0:
                score += 15  # Reasonable
            elif pb > 5.0:
                score -= 20  # Expensive
        
        # Price to Sales
        ps = data.get('priceToSalesRatio')
        if ps and ps > 0:
            # Lower P/S is better
            if ps < 1.0:
                score += 20
            elif ps < 3.0:
                score += 10
            elif ps > 10.0:
                score -= 15
        
        return min(max(score, 0.0), 100.0)
    
    def _calculate_growth_from_data(self, data: Dict[str, Any]) -> float:
        """Calculate growth score from fundamentals data."""
        score = 50.0  # Base score
        
        # Revenue growth
        rev_growth = data.get('revenueGrowth')
        if rev_growth:
            score += min(rev_growth * 100, 25)  # Max 25 points
        
        # Net income growth
        ni_growth = data.get('netIncomeGrowth')
        if ni_growth:
            score += min(ni_growth * 100, 25)  # Max 25 points
        
        return min(score, 100.0)

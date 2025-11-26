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
        
        # Map factor scores to required columns
        # Use direct column access - create columns if they don't exist
        df["momentum_score"] = df["momentum"] if "momentum" in df.columns else pd.Series(dtype=float, index=df.index)
        df["quality_score"] = df["profitability"] if "profitability" in df.columns else pd.Series(dtype=float, index=df.index)
        df["valuation_score"] = df["value"] if "value" in df.columns else pd.Series(dtype=float, index=df.index)
        df["catalyst_score"] = df["growth"] if "growth" in df.columns else pd.Series(dtype=float, index=df.index)
        
        # Filter out rows where all scores are missing (don't use defaults that make everything equal)
        required_cols = ["momentum_score", "quality_score", "valuation_score", "catalyst_score"]
        df = df.dropna(subset=required_cols, how='all')  # Keep rows with at least some data
        
        # For rows with partial data, fill only missing individual scores with median of that column
        # This preserves differentiation between stocks
        for col in required_cols:
            if col in df.columns:
                median_val = df[col].median()
                if pd.notna(median_val):
                    df[col] = df[col].fillna(median_val)
                else:
                    # If no valid data at all, use 0 instead of 50 to flag as incomplete
                    df[col] = df[col].fillna(0)
        
        # Final check: remove stocks with all zeros (no valid data)
        df = df[~((df["momentum_score"] == 0) & (df["quality_score"] == 0) & 
                  (df["valuation_score"] == 0) & (df["catalyst_score"] == 0))]
        
        if len(df) == 0:
            LOGGER.error("No valid candidates after filtering - all stocks have missing data")
            return []

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
        Fallback method: Get candidates directly from FMP screener or fundamentals cache.
        This works when parquet files are missing.
        Calculates scores directly from fundamentals data.
        """
        try:
            from api.services.fundamentals_cache_service import get_fundamentals_cache_service
            from api.services.openbb_client import OpenBBClient
            from api.services.scoring_service import ScoringService
            
            cache_service = get_fundamentals_cache_service()
            obb_client = OpenBBClient()
            scoring_service = ScoringService()
            
            # Get all cached tickers
            cached_tickers = cache_service.get_all_cached_tickers()
            
            # If no cache, use FMP screener to get candidates
            if not cached_tickers:
                LOGGER.info("No tickers in cache, using FMP screener to get candidates")
                try:
                    # Use FMP screener to get a pool of stocks
                    fmp_params = {
                        "marketCapMoreThan": 100_000_000,  # $100M+
                        "marketCapLowerThan": 50_000_000_000,  # < $50B
                        "isActivelyTrading": "true",
                        "isEtf": "false",
                        "isFund": "false",
                        "limit": min(top_n * 5, 200),  # Get more candidates
                        "exchange": "NASDAQ,NYSE,AMEX"
                    }
                    screener_results = scoring_service.fmp.get_stock_screener(fmp_params)
                    if screener_results and len(screener_results) > 0:
                        # Extract tickers from screener results
                        cached_tickers = [r.get("symbol") for r in screener_results if r.get("symbol")]
                        LOGGER.info(f"FMP screener returned {len(cached_tickers)} candidates")
                    else:
                        # Ultimate fallback: use a predefined list
                        cached_tickers = [
                            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
                            "UNH", "JNJ", "V", "WMT", "PG", "JPM", "MA", "HD", "DIS", "BAC",
                            "ADBE", "NFLX", "CRM", "PYPL", "INTC", "CMCSA", "PEP", "COST", "AVGO",
                            "TMO", "ABT", "NKE", "MRK", "ACN", "CSCO", "TXN", "QCOM", "DHR"
                        ]
                        LOGGER.info(f"Using fallback ticker list: {len(cached_tickers)} stocks")
                except Exception as e:
                    LOGGER.warning(f"FMP screener failed: {e}, using fallback list")
                    cached_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
            
            if not cached_tickers:
                LOGGER.error("No tickers available for alpha picks")
                return []
            
            LOGGER.info(f"Using {len(cached_tickers)} tickers from cache for alpha picks")
            
            # Score each ticker
            candidates = []
            for ticker in cached_tickers[:min(top_n * 3, 200)]:  # Limit to 200 for performance
                try:
                    data = None
                    
                    # Try to get fundamentals from cache first
                    cached = cache_service.get_fundamentals(ticker)
                    if cached and cached.get('data'):
                        data = cached['data']
                    else:
                        # If not in cache, fetch from OpenBB/FMP using scoring service
                        try:
                            from api.services.scoring_service import ScoringService
                            scoring = ScoringService()
                            # Use scoring service to get scores which will fetch data
                            score_result = scoring.get_scores(ticker)
                            if score_result and score_result.get("details"):
                                # Extract data from score details
                                quality_details = score_result.get("details", {}).get("quality", {})
                                valuation_details = score_result.get("details", {}).get("valuation", {})
                                momentum_details = score_result.get("details", {}).get("momentum", {})
                                
                                # Build data dict from score details
                                data = {
                                    **quality_details,
                                    **valuation_details,
                                    **momentum_details,
                                    "company_name": ticker,
                                    "sector": "Unknown"
                                }
                        except Exception as fetch_error:
                            LOGGER.debug(f"Could not fetch data for {ticker}: {fetch_error}")
                            continue
                    
                    if not data:
                        continue
                    
                    # Get current price for valuation
                    current_price = obb_client.get_current_price(ticker)
                    if not current_price or current_price <= 0:
                        continue
                    
                    # Calculate scores directly from fundamentals
                    quality_score = self._calculate_quality_from_data(data)
                    valuation_score = self._calculate_valuation_from_data(data, current_price)
                    
                    # Try to calculate real momentum from price history
                    momentum_score = None
                    try:
                        price_history = obb_client.get_price_history(ticker, start_date=None, period="1y")
                        if price_history:
                            # Calculate momentum as 1-year return
                            if hasattr(price_history, 'to_df'):
                                df_prices = price_history.to_df()
                                if not df_prices.empty and 'close' in df_prices.columns:
                                    closes = df_prices['close'].dropna()
                                    if len(closes) > 20:  # Need at least 20 days
                                        start_price = closes.iloc[0]
                                        end_price = closes.iloc[-1]
                                        if start_price > 0:
                                            momentum_score = min(max((end_price / start_price - 1) * 100, 0), 100)
                    except Exception as e:
                        LOGGER.debug(f"Could not calculate momentum for {ticker}: {e}")
                    
                    # Calculate growth score
                    growth_score = self._calculate_growth_from_data(data)
                    
                    # If momentum calculation failed, use a proxy based on other factors
                    if momentum_score is None:
                        # Use growth as proxy for momentum, or average of quality/valuation if growth unavailable
                        if growth_score > 0:
                            momentum_score = min(growth_score * 0.8, 100)  # Momentum typically lower than growth
                        else:
                            momentum_score = (quality_score + valuation_score) / 2 * 0.6  # Conservative proxy
                    
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

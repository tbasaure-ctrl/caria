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
        candidates = self.factor_service.screen_companies(top_n=top_n_candidates)
        
        if not candidates:
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

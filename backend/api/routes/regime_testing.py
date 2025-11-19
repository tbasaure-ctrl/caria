"""
Regime Testing endpoints.
Test portfolio exposure and protection against different economic regimes.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_current_user
from api.services.asset_regime_service import get_asset_regime_service
from api.services.monte_carlo_service import get_monte_carlo_service
from caria.models.auth import UserInDB

LOGGER = logging.getLogger("caria.api.regime_testing")

router = APIRouter(prefix="/api/portfolio", tags=["Regime Testing"])


class HoldingInput(BaseModel):
    """Input model for a holding."""
    ticker: str = Field(..., description="Stock ticker symbol")
    allocation: float = Field(..., ge=0, le=100, description="Allocation percentage")


class RegimeTestRequest(BaseModel):
    """Request model for regime testing."""
    regime: str = Field(
        ...,
        description="Target regime to test: expansion, recession, slowdown, stress"
    )
    holdings: Optional[List[HoldingInput]] = Field(
        None,
        description="Holdings to test. If None, uses current user's holdings."
    )


class RegimeTestResponse(BaseModel):
    """Response model for regime testing."""
    regime: str
    exposure_score: float = Field(..., ge=0, le=100, description="Exposure score (0-100)")
    protection_level: str = Field(..., description="high, medium, or low")
    drawdown_estimate: dict = Field(..., description="Drawdown estimates")
    monte_carlo_results: dict = Field(..., description="Monte Carlo simulation results")
    recommendations: List[str] = Field(..., description="Recommendations to improve protection")


@router.post("/regime-test", response_model=RegimeTestResponse)
async def test_regime_exposure(
    request: RegimeTestRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> RegimeTestResponse:
    """
    Test portfolio exposure and protection against a specific economic regime.
    
    Process:
    1. Get user's current holdings (or use provided holdings)
    2. Classify each holding by regime suitability
    3. Calculate exposure score vs target regime
    4. Run Monte Carlo simulation for 12 months
    5. Calculate drawdown estimates
    6. Generate recommendations
    
    Args:
        request: Regime test request with regime and optional holdings
        current_user: Current authenticated user
        
    Returns:
        Regime test results with exposure, protection level, and recommendations
    """
    # Validate regime
    valid_regimes = ["expansion", "recession", "slowdown", "stress"]
    if request.regime.lower() not in valid_regimes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid regime. Must be one of: {', '.join(valid_regimes)}"
        )
    
    target_regime = request.regime.lower()
    
    try:
        # 1. Get holdings
        holdings = request.holdings
        if holdings is None:
            # Fetch user's current holdings from database
            holdings = await _get_user_holdings(current_user.id)
        
        if not holdings:
            raise HTTPException(
                status_code=400,
                detail="No holdings provided and user has no holdings in portfolio"
            )
        
        # 2. Classify assets by regime suitability
        asset_service = get_asset_regime_service()
        regime_scores = {}
        total_allocation = sum(h.allocation for h in holdings)
        
        if total_allocation == 0:
            raise HTTPException(status_code=400, detail="Total allocation must be > 0")
        
        for holding in holdings:
            scores = asset_service.classify_asset(holding.ticker)
            regime_scores[holding.ticker] = scores
        
        # 3. Calculate exposure score
        # Weighted average of suitability scores
        weighted_score = 0.0
        for holding in holdings:
            weight = holding.allocation / total_allocation
            suitability = regime_scores[holding.ticker].get(target_regime, 0.5)
            weighted_score += weight * suitability
        
        exposure_score = weighted_score * 100  # Convert to 0-100 scale
        
        # 4. Determine protection level
        if exposure_score >= 70:
            protection_level = "high"
        elif exposure_score >= 40:
            protection_level = "medium"
        else:
            protection_level = "low"
        
        # 5. Run Monte Carlo simulation for 12 months
        monte_carlo_service = get_monte_carlo_service()
        
        # Estimate portfolio parameters based on holdings
        # For simplicity, use average of individual asset parameters
        # In production, could use portfolio-level calculations
        portfolio_mu = 0.10  # 10% expected return (could be calculated from holdings)
        portfolio_sigma = 0.20  # 20% volatility (could be calculated from holdings)
        
        # Adjust based on regime
        regime_adjustments = {
            "expansion": {"mu": 0.15, "sigma": 0.18},  # Higher returns, lower vol
            "recession": {"mu": -0.10, "sigma": 0.30},  # Negative returns, high vol
            "slowdown": {"mu": 0.05, "sigma": 0.22},  # Lower returns, moderate vol
            "stress": {"mu": -0.20, "sigma": 0.35},  # Very negative, very high vol
        }
        
        adjustment = regime_adjustments.get(target_regime, {"mu": 0.10, "sigma": 0.20})
        adjusted_mu = adjustment["mu"]
        adjusted_sigma = adjustment["sigma"]
        
        # Run simulation for 12 months (1 year)
        mc_result = monte_carlo_service.run_portfolio_simulation(
            initial_value=100000,  # Normalized to $100k for calculation
            mu=adjusted_mu,
            sigma=adjusted_sigma,
            years=1,  # 12 months
            simulations=10000,
        )
        
        # 6. Calculate drawdown estimates
        final_values = mc_result["final_values"]
        percentiles = mc_result["percentiles"]
        
        # Calculate max drawdown estimate
        # Use 5th percentile as worst-case scenario
        worst_case = percentiles.get("p5", final_values[0] if final_values else 100000)
        max_drawdown_pct = ((worst_case - 100000) / 100000) * 100
        
        drawdown_estimate = {
            "worst_case_p5": worst_case,
            "max_drawdown_pct": max_drawdown_pct,
            "median": percentiles.get("p50", 100000),
            "best_case_p95": percentiles.get("p95", 100000),
        }
        
        # 7. Generate recommendations
        recommendations = _generate_recommendations(
            protection_level, exposure_score, target_regime, holdings, regime_scores
        )
        
        return RegimeTestResponse(
            regime=target_regime,
            exposure_score=exposure_score,
            protection_level=protection_level,
            drawdown_estimate=drawdown_estimate,
            monte_carlo_results=mc_result,
            recommendations=recommendations,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception(f"Error testing regime exposure: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error testing regime exposure: {str(e)}"
        ) from e


async def _get_user_holdings(user_id: str) -> List[HoldingInput]:
    """Get user's current holdings from database."""
    import psycopg2
    import os
    from urllib.parse import urlparse, parse_qs
    
    # Use same DB connection logic as other endpoints
    database_url = os.getenv("DATABASE_URL")
    conn = None
    
    if database_url:
        try:
            parsed = urlparse(database_url)
            query_params = parse_qs(parsed.query)
            unix_socket_host = query_params.get('host', [None])[0]
            
            if unix_socket_host:
                conn = psycopg2.connect(
                    host=unix_socket_host,
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path.lstrip('/'),
                )
            elif parsed.hostname:
                conn = psycopg2.connect(
                    host=parsed.hostname,
                    port=parsed.port or 5432,
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path.lstrip('/'),
                )
        except Exception as e:
            LOGGER.warning(f"Error using DATABASE_URL: {e}")
    
    if conn is None:
        password = os.getenv("POSTGRES_PASSWORD")
        if not password:
            return []
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "caria_user"),
            password=password,
            database=os.getenv("POSTGRES_DB", "caria"),
        )
    
    try:
        with conn.cursor() as cur:
            # Get holdings with quantity and average_cost, calculate allocation percentage
            cur.execute(
                """
                SELECT ticker, quantity, average_cost
                FROM holdings
                WHERE user_id = %s AND quantity > 0
                ORDER BY ticker
                """,
                (user_id,)
            )
            rows = cur.fetchall()
            if not rows:
                return []
            
            # Calculate total portfolio value
            total_value = sum(float(row[1]) * float(row[2]) for row in rows)  # quantity * average_cost
            
            if total_value == 0:
                return []
            
            # Calculate allocation percentage for each holding
            holdings = []
            for row in rows:
                ticker = row[0]
                quantity = float(row[1])
                average_cost = float(row[2])
                holding_value = quantity * average_cost
                allocation_pct = (holding_value / total_value) * 100
                holdings.append(HoldingInput(ticker=ticker, allocation=allocation_pct))
            
            return holdings
    except Exception as e:
        LOGGER.exception(f"Error fetching user holdings: {e}")
        return []
    finally:
        conn.close()


def _generate_recommendations(
    protection_level: str,
    exposure_score: float,
    target_regime: str,
    holdings: List[HoldingInput],
    regime_scores: dict,
) -> List[str]:
    """Generate recommendations to improve protection."""
    recommendations = []
    
    if protection_level == "low":
        recommendations.append(
            f"Tu portfolio tiene baja protección contra {target_regime}. "
            "Considera agregar activos más defensivos."
        )
        
        # Find worst-performing holdings for this regime
        holding_scores = [
            (h.ticker, h.allocation, regime_scores[h.ticker].get(target_regime, 0.5))
            for h in holdings
        ]
        holding_scores.sort(key=lambda x: x[2])  # Sort by suitability (lowest first)
        
        worst_holdings = holding_scores[:3]
        if worst_holdings:
            worst_tickers = [t[0] for t in worst_holdings]
            recommendations.append(
                f"Considera reducir exposición a: {', '.join(worst_tickers)} "
                f"en escenarios de {target_regime}."
            )
    
    elif protection_level == "medium":
        recommendations.append(
            f"Protección moderada contra {target_regime}. "
            "Puedes mejorar diversificando más."
        )
    
    # Regime-specific recommendations
    if target_regime == "recession":
        recommendations.append(
            "Para protegerte en recesión, considera aumentar exposición a: "
            "bonos (TLT), oro (GLD), o efectivo."
        )
    elif target_regime == "stress":
        recommendations.append(
            "En períodos de estrés, considera aumentar efectivo y reducir "
            "exposición a activos de riesgo."
        )
    elif target_regime == "expansion":
        recommendations.append(
            "En expansión, puedes aumentar exposición a acciones de crecimiento "
            "y reducir efectivo."
        )
    
    return recommendations


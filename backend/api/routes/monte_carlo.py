"""
Monte Carlo endpoints per audit document (3.1).
Optimized visualization with Plotly and Scattergl.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.dependencies import get_current_user
from api.services.monte_carlo_service import get_monte_carlo_service
from caria.models.auth import UserInDB

router = APIRouter(prefix="/api/montecarlo", tags=["Monte Carlo"])


class MonteCarloRequest(BaseModel):
    """Request model for Monte Carlo simulation."""
    initial_value: float = Field(..., gt=0, description="Initial portfolio value")
    mu: float = Field(..., description="Expected annual return (e.g., 0.10 for 10%)")
    sigma: float = Field(..., gt=0, description="Annual volatility (e.g., 0.25 for 25%)")
    years: int = Field(5, ge=1, le=30, description="Investment horizon in years")
    simulations: int = Field(10000, ge=1000, le=50000, description="Number of simulations")
    contributions_per_year: float = Field(0.0, ge=0, description="Annual contributions")
    annual_fee: float = Field(0.0, ge=0, le=0.1, description="Annual fee rate")
    regime: str = Field(None, description="Optional: Economic regime (expansion, recession, slowdown, stress) to adjust parameters")


class StockForecastRequest(BaseModel):
    """Request model for stock forecast."""
    ticker: str = Field(..., min_length=1, max_length=10)
    horizon_years: int = Field(2, ge=1, le=10)
    simulations: int = Field(10000, ge=1000, le=50000)


@router.post("/simulate")
async def run_monte_carlo_simulation(
    request: MonteCarloRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> dict:
    """
    Run Monte Carlo simulation for portfolio.
    
    Per audit document (3.1): Returns optimized Plotly data using technique
    of concatenating all simulations with np.nan separators for efficient rendering.
    Frontend should use Scattergl (WebGL) for visualization.
    
    If regime is specified, adjusts mu and sigma based on regime characteristics.
    """
    try:
        service = get_monte_carlo_service()
        
        # Adjust parameters based on regime if provided
        mu = request.mu
        sigma = request.sigma
        
        if request.regime:
            regime_adjustments = {
                "expansion": {"mu_mult": 1.2, "sigma_mult": 0.9},  # Higher returns, lower vol
                "recession": {"mu_mult": -0.5, "sigma_mult": 1.5},  # Negative returns, high vol
                "slowdown": {"mu_mult": 0.6, "sigma_mult": 1.1},  # Lower returns, moderate vol
                "stress": {"mu_mult": -1.0, "sigma_mult": 1.8},  # Very negative, very high vol
            }
            
            adjustment = regime_adjustments.get(request.regime.lower())
            if adjustment:
                mu = request.mu * adjustment["mu_mult"]
                sigma = request.sigma * adjustment["sigma_mult"]
                # Ensure sigma stays positive
                sigma = max(0.01, sigma)
        
        result = service.run_portfolio_simulation(
            initial_value=request.initial_value,
            mu=mu,
            sigma=sigma,
            years=request.years,
            simulations=request.simulations,
            contributions_per_year=request.contributions_per_year,
            annual_fee=request.annual_fee,
        )
        
        # Add regime info if provided
        if request.regime:
            result["regime"] = request.regime.lower()
            result["adjusted_mu"] = mu
            result["adjusted_sigma"] = sigma
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}") from e


@router.post("/forecast/stock")
async def forecast_stock(
    request: StockForecastRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> dict:
    """
    Run Monte Carlo forecast for a stock ticker.
    
    Uses historical data to estimate parameters and runs simulation.
    Returns histogram data and optimized Plotly visualization data.
    """
    try:
        service = get_monte_carlo_service()
        result = service.run_stock_forecast(
            ticker=request.ticker.upper(),
            horizon_years=request.horizon_years,
            simulations=request.simulations,
        )
        
        # Generate visualization data with exact percentiles
        visualization_data = service.run_visualization_data(
            ticker=request.ticker.upper(),
            final_values=result["final_values"]
        )
        result["visualization_data"] = visualization_data
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}") from e


@router.get("/forecast/stock/{ticker}")
async def forecast_stock_get(
    ticker: str,
    horizon_years: int = Query(2, ge=1, le=10),
    simulations: int = Query(10000, ge=1000, le=50000),
    current_user: UserInDB = Depends(get_current_user),
) -> dict:
    """
    GET endpoint for stock forecast (convenience).
    """
    try:
        service = get_monte_carlo_service()
        result = service.run_stock_forecast(
            ticker=ticker.upper(),
            horizon_years=horizon_years,
            simulations=simulations,
        )
        
        # Generate visualization data with exact percentiles
        visualization_data = service.run_visualization_data(
            ticker=ticker.upper(),
            final_values=result["final_values"]
        )
        result["visualization_data"] = visualization_data
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}") from e


"""
Enhanced Monte Carlo Service

Provides more reliable Monte Carlo simulations by:
1. Using fundamental data to adjust drift (mu) and volatility (sigma)
2. Incorporating financial health metrics
3. Adjusting for sector/industry risk
4. Using historical volatility with fundamental adjustments
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from api.services.openbb_client import OpenBBClient

LOGGER = logging.getLogger("caria.services.enhanced_monte_carlo")

class EnhancedMonteCarloService:
    """
    Enhanced Monte Carlo service with fundamental adjustments.
    """
    
    def __init__(self):
        self.obb_client = OpenBBClient()
        self.trading_days_per_year = 252
        self.default_simulations = 10000
    
    def run_enhanced_forecast(
        self,
        ticker: str,
        horizon_years: int = 2,
        simulations: int = 10000,
        intrinsic_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run enhanced Monte Carlo forecast with fundamental adjustments.
        
        Args:
            ticker: Stock ticker symbol
            horizon_years: Forecast horizon in years
            simulations: Number of simulations
            intrinsic_value: Optional intrinsic value to use for mean reversion
        """
        LOGGER.info(f"Running enhanced Monte Carlo forecast for {ticker}")
        
        try:
            # 1. Get historical price data
            history = self.obb_client.get_price_history(ticker, start_date="2018-01-01")
            
            closes = self._extract_closes(history)
            if len(closes) < 30:
                raise ValueError(f"Insufficient historical data for {ticker} (got {len(closes)} points)")
            
            current_price = float(closes[-1])
            
            # 2. Calculate historical returns and volatility
            returns = np.diff(closes) / np.array(closes[:-1])
            historical_mu = float(np.mean(returns) * self.trading_days_per_year)
            historical_sigma = float(np.std(returns) * np.sqrt(self.trading_days_per_year))
            
            # 3. Get fundamental data for adjustments
            fundamental_data = self._get_fundamental_adjustments(ticker)
            
            # 4. Adjust mu and sigma based on fundamentals
            adjusted_mu, adjusted_sigma = self._adjust_parameters(
                historical_mu,
                historical_sigma,
                fundamental_data,
                current_price,
                intrinsic_value
            )
            
            # 5. Run simulation
            result = self._run_simulation(
                current_price,
                adjusted_mu,
                adjusted_sigma,
                horizon_years,
                simulations
            )
            
            # 6. Add analysis
            result["ticker"] = ticker
            result["current_price"] = current_price
            result["intrinsic_value"] = intrinsic_value
            result["estimated_mu"] = adjusted_mu
            result["estimated_sigma"] = adjusted_sigma
            result["historical_mu"] = historical_mu
            result["historical_sigma"] = historical_sigma
            result["fundamental_adjustments"] = fundamental_data.get("adjustments", {})
            
            # 7. Calculate percentiles and probabilities
            final_prices = np.array(result["final_values"])
            returns_pct = (final_prices / current_price - 1.0) * 100
            
            result["percentiles"] = {
                "5th": float(np.percentile(final_prices, 5)),
                "10th": float(np.percentile(final_prices, 10)),
                "25th": float(np.percentile(final_prices, 25)),
                "50th": float(np.percentile(final_prices, 50)),
                "75th": float(np.percentile(final_prices, 75)),
                "90th": float(np.percentile(final_prices, 90)),
                "95th": float(np.percentile(final_prices, 95))
            }
            
            result["expected_value"] = float(np.mean(final_prices))
            result["prob_positive"] = float(np.mean(returns_pct > 0))
            result["prob_above_intrinsic"] = (
                float(np.mean(final_prices > intrinsic_value)) if intrinsic_value else None
            )
            
            # Probability of significant moves
            result["probabilities"] = {
                "loss_20pct": float(np.mean(returns_pct <= -20)),
                "loss_10pct": float(np.mean(returns_pct <= -10)),
                "gain_10pct": float(np.mean(returns_pct >= 10)),
                "gain_20pct": float(np.mean(returns_pct >= 20)),
                "gain_50pct": float(np.mean(returns_pct >= 50))
            }
            
            LOGGER.info(f"✅ Enhanced Monte Carlo completed for {ticker}")
            return result
            
        except Exception as e:
            LOGGER.error(f"Enhanced Monte Carlo failed for {ticker}: {e}", exc_info=True)
            raise
    
    def _extract_closes(self, history) -> list:
        """Extract closing prices from history data."""
        closes = []
        
        if history and hasattr(history, 'to_df'):
            df = history.to_df()
            if not df.empty and 'close' in df.columns:
                closes = df['close'].dropna().tolist()
        elif isinstance(history, list):
            closes = [float(item.get("close")) for item in history if item.get("close") is not None]
        elif hasattr(history, '__iter__'):
            for item in history:
                if isinstance(item, dict):
                    close_val = item.get("close")
                elif hasattr(item, 'close'):
                    close_val = item.close
                else:
                    continue
                if close_val is not None:
                    closes.append(float(close_val))
        
        return closes
    
    def _get_fundamental_adjustments(self, ticker: str) -> Dict[str, Any]:
        """Get fundamental data and calculate adjustment factors."""
        try:
            data = self.obb_client.get_ticker_data(ticker)
            key_metrics = data.get("key_metrics", {})
            if isinstance(key_metrics, list) and key_metrics:
                key_metrics = key_metrics[0]
            
            profile = data.get("profile", {})
            
            adjustments = {}
            
            # Financial health score (0.5 to 1.5)
            health_score = 1.0
            
            # ROE adjustment
            roe = key_metrics.get("roe") or key_metrics.get("returnOnEquity")
            if roe:
                if roe > 0.20:
                    health_score += 0.15
                elif roe > 0.15:
                    health_score += 0.10
                elif roe < 0:
                    health_score -= 0.20
            
            # Profit margin adjustment
            margin = key_metrics.get("netProfitMargin") or key_metrics.get("profitMargin")
            if margin:
                if margin > 0.20:
                    health_score += 0.10
                elif margin > 0.10:
                    health_score += 0.05
                elif margin < 0:
                    health_score -= 0.15
            
            # Debt/Equity adjustment
            debt_equity = key_metrics.get("debtEquity") or key_metrics.get("debtToEquity")
            if debt_equity:
                if debt_equity < 0.5:
                    health_score += 0.10
                elif debt_equity > 2.0:
                    health_score -= 0.15
            
            # Current ratio adjustment
            current_ratio = key_metrics.get("currentRatio")
            if current_ratio:
                if current_ratio > 2.0:
                    health_score += 0.05
                elif current_ratio < 1.0:
                    health_score -= 0.10
            
            health_score = max(0.5, min(1.5, health_score))
            adjustments["health_score"] = health_score
            
            # Sector/industry risk multiplier
            sector = profile.get("sector", "").lower()
            industry = profile.get("industry", "").lower()
            
            sector_risk = 1.0
            if "technology" in sector or "tech" in sector:
                sector_risk = 1.2  # Higher volatility
            elif "utilities" in sector or "consumer staples" in sector:
                sector_risk = 0.8  # Lower volatility
            elif "biotechnology" in industry or "biotech" in industry:
                sector_risk = 1.4  # Very high volatility
            
            adjustments["sector_risk"] = sector_risk
            
            # Growth prospects (from revenue growth or FCF growth)
            revenue_growth = key_metrics.get("revenueGrowth") or key_metrics.get("revenueGrowthTTM")
            if revenue_growth:
                if revenue_growth > 0.20:
                    adjustments["growth_multiplier"] = 1.15
                elif revenue_growth > 0.10:
                    adjustments["growth_multiplier"] = 1.05
                elif revenue_growth < 0:
                    adjustments["growth_multiplier"] = 0.90
                else:
                    adjustments["growth_multiplier"] = 1.0
            else:
                adjustments["growth_multiplier"] = 1.0
            
            return {
                "key_metrics": key_metrics,
                "profile": profile,
                "adjustments": adjustments
            }
            
        except Exception as e:
            LOGGER.warning(f"Could not fetch fundamental adjustments for {ticker}: {e}")
            return {
                "adjustments": {
                    "health_score": 1.0,
                    "sector_risk": 1.0,
                    "growth_multiplier": 1.0
                }
            }
    
    def _adjust_parameters(
        self,
        historical_mu: float,
        historical_sigma: float,
        fundamental_data: Dict[str, Any],
        current_price: float,
        intrinsic_value: Optional[float]
    ) -> tuple:
        """
        Adjust mu and sigma based on fundamental data.
        
        Returns:
            (adjusted_mu, adjusted_sigma)
        """
        adjustments = fundamental_data.get("adjustments", {})
        health_score = adjustments.get("health_score", 1.0)
        sector_risk = adjustments.get("sector_risk", 1.0)
        growth_multiplier = adjustments.get("growth_multiplier", 1.0)
        
        # Adjust mu (drift) based on fundamentals
        # Base market return assumption
        base_market_return = 0.08  # 8% annual
        
        # Adjust for financial health
        health_adjusted_return = base_market_return * health_score
        
        # Adjust for growth prospects
        growth_adjusted_return = health_adjusted_return * growth_multiplier
        
        # Blend historical with fundamental (60% fundamental, 40% historical)
        # But cap historical influence if it's extreme
        if abs(historical_mu) > 0.50:  # If historical return is >50% or <-50%, reduce weight
            historical_weight = 0.2
        else:
            historical_weight = 0.4
        
        fundamental_weight = 1.0 - historical_weight
        
        adjusted_mu = (
            fundamental_weight * growth_adjusted_return +
            historical_weight * historical_mu
        )
        
        # Mean reversion to intrinsic value (if provided)
        if intrinsic_value and intrinsic_value > 0:
            # If current price is below intrinsic value, add upward drift
            # If above, reduce drift
            price_to_intrinsic = current_price / intrinsic_value
            if price_to_intrinsic < 0.8:  # Undervalued by >20%
                adjusted_mu += 0.02  # Add 2% annual return from mean reversion
            elif price_to_intrinsic > 1.2:  # Overvalued by >20%
                adjusted_mu -= 0.02  # Reduce by 2%
        
        # Adjust sigma (volatility) based on sector risk and health
        adjusted_sigma = historical_sigma * sector_risk
        
        # Reduce volatility for healthier companies
        if health_score > 1.2:
            adjusted_sigma *= 0.9  # 10% reduction
        elif health_score < 0.8:
            adjusted_sigma *= 1.1  # 10% increase
        
        # Ensure reasonable bounds
        adjusted_mu = np.clip(adjusted_mu, -0.30, 0.50)  # Between -30% and +50%
        adjusted_sigma = np.clip(adjusted_sigma, 0.10, 1.00)  # Between 10% and 100%
        
        LOGGER.info(
            f"Parameter adjustments: mu {historical_mu:.2%} → {adjusted_mu:.2%}, "
            f"sigma {historical_sigma:.2%} → {adjusted_sigma:.2%}"
        )
        
        return adjusted_mu, adjusted_sigma
    
    def _run_simulation(
        self,
        initial_price: float,
        mu: float,
        sigma: float,
        years: int,
        simulations: int
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation using Geometric Brownian Motion."""
        np.random.seed(42)
        
        steps_per_year = 12  # Monthly steps
        dt = 1.0 / steps_per_year
        n_steps = int(years * steps_per_year)
        
        # GBM parameters
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Generate random shocks
        Z = np.random.standard_normal((simulations, n_steps))
        
        # Initialize paths
        paths = np.empty((simulations, n_steps + 1), dtype=float)
        paths[:, 0] = initial_price
        
        # Simulate paths
        for t in range(n_steps):
            paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion * Z[:, t])
        
        # Ensure prices don't go negative
        paths = np.maximum(paths, initial_price * 0.01)
        
        final_values = paths[:, -1]
        
        # Generate plotly data (optimized)
        time_years = np.arange(n_steps + 1) / steps_per_year
        x_data = []
        y_data = []
        
        for i in range(min(100, simulations)):  # Limit to 100 paths for visualization
            x_data.extend(time_years)
            y_data.extend(paths[i, :])
            if i < min(100, simulations) - 1:
                x_data.append(np.nan)
                y_data.append(np.nan)
        
        return {
            "paths": paths.tolist(),
            "final_values": final_values.tolist(),
            "plotly_data": {
                "x": [float(x) if not np.isnan(x) else None for x in x_data],
                "y": [float(y) if not np.isnan(y) else None for y in y_data],
                "type": "scattergl",
                "mode": "lines",
                "line": {"width": 1, "color": "rgba(100, 100, 200, 0.1)"},
                "name": "Monte Carlo Simulations"
            },
            "simulation_params": {
                "initial_price": initial_price,
                "mu": mu,
                "sigma": sigma,
                "years": years,
                "simulations": simulations
            }
        }

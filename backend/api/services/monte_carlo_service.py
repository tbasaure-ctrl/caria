"""
Monte Carlo Service with optimized Plotly visualization per audit document (3.1).
Uses technique: concatenate all simulations in single trace with np.nan as separator.
Frontend will use Scattergl (WebGL) for rendering thousands of lines.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from api.services.openbb_client import OpenBBClient

# Instantiate OpenBBClient
openbb_client = OpenBBClient()


LOGGER = logging.getLogger("caria.api.monte_carlo")

# Import Monte Carlo scripts from Montecarlo_feature
import sys
from pathlib import Path

# Add Montecarlo_feature to path
montecarlo_dir = Path(__file__).parent.parent.parent.parent / "Montecarlo_feature"
if str(montecarlo_dir) not in sys.path:
    sys.path.insert(0, str(montecarlo_dir))


class MonteCarloService:
    """Monte Carlo simulation service with optimized visualization."""

    def __init__(self):
        self.trading_days_per_year = 252
        self.default_simulations = 10000
        self.default_horizon_years = 5

    def run_portfolio_simulation(
        self,
        initial_value: float,
        mu: float,
        sigma: float,
        years: int = 5,
        simulations: int = 10000,
        contributions_per_year: float = 0.0,
        annual_fee: float = 0.0,
        seed: int = 42,
    ) -> dict:
        """
        Run Monte Carlo simulation for portfolio.
        Based on montecarlo.py script.
        
        Returns:
            dict with paths, final_values, percentiles, and plotly_data
        """
        np.random.seed(seed)
        
        steps_per_year = 12  # Monthly steps
        dt = 1.0 / steps_per_year
        n = int(years * steps_per_year)
        mu_eff = mu - annual_fee
        drift = (mu_eff - 0.5 * sigma**2) * dt
        diff = sigma * np.sqrt(dt)

        # Generate random shocks
        Z = np.random.standard_normal((simulations, n))
        
        # Initialize paths
        paths = np.empty((simulations, n + 1), dtype=float)
        paths[:, 0] = initial_value

        # Simulate paths
        for t in range(n):
            paths[:, t + 1] = (
                paths[:, t] * np.exp(drift + diff * Z[:, t]) + contributions_per_year
            )

        final_values = paths[:, -1]
        total_invested = initial_value + contributions_per_year * n
        moic = final_values / max(total_invested, 1e-9)

        # Calculate percentiles
        def pctl(x, q):
            return float(np.percentile(x, q))

        percentiles = {
            "p5": pctl(final_values, 5),
            "p10": pctl(final_values, 10),
            "p25": pctl(final_values, 25),
            "p50": pctl(final_values, 50),
            "p75": pctl(final_values, 75),
            "p90": pctl(final_values, 90),
            "p95": pctl(final_values, 95),
        }

        # Risk metrics
        var5 = pctl(final_values - total_invested, 5)
        cvar5 = float(
            (final_values - total_invested)[(final_values - total_invested) <= var5].mean()
        )

        # Generate optimized Plotly data per audit document (3.1)
        # Technique: concatenate all simulations with np.nan as separator
        plotly_data = self._generate_optimized_plotly_data(paths, years)

        # Generate histogram data
        histogram_data = self.generate_histogram_data(final_values)

        return {
            "paths": paths.tolist(),  # Full paths for detailed analysis
            "final_values": final_values.tolist(),
            "total_invested": float(total_invested),
            "moic": moic.tolist(),
            "percentiles": percentiles,
            "metrics": {
                "mean": float(np.mean(final_values)),
                "median": float(np.median(final_values)),
                "std": float(np.std(final_values)),
                "var_5pct": var5,
                "cvar_5pct": cvar5,
                "prob_final_less_invested": float(np.mean(final_values < total_invested)),
                "moic_median": float(np.median(moic)),
            },
            "plotly_data": plotly_data,  # Optimized for Scattergl
            "histogram": histogram_data,  # Histogram for distribution
            "simulation_params": {
                "initial_value": initial_value,
                "mu": mu,
                "sigma": sigma,
                "years": years,
                "simulations": simulations,
                "contributions_per_year": contributions_per_year,
                "annual_fee": annual_fee,
            },
        }

    def _generate_optimized_plotly_data(
        self, paths: np.ndarray, years: int
    ) -> dict:
        """
        Generate optimized Plotly data per audit document (3.1).
        
        Technique: Concatenate all simulations in a single trace with np.nan as separator.
        This allows rendering thousands of lines efficiently with Scattergl (WebGL).
        """
        n_sims, n_steps = paths.shape
        
        # Create time axis (in years)
        steps_per_year = 12  # Monthly
        time_years = np.arange(n_steps) / steps_per_year

        # Per audit document: concatenate all paths with np.nan separators
        # This creates a single continuous trace that Plotly can render efficiently
        x_data = []
        y_data = []

        for i in range(n_sims):
            # Add path data
            x_data.extend(time_years)
            y_data.extend(paths[i, :])

            # Add separator (np.nan) between paths (except for last one)
            if i < n_sims - 1:
                x_data.append(np.nan)
                y_data.append(np.nan)

        return {
            "x": [float(x) if not np.isnan(x) else None for x in x_data],
            "y": [float(y) if not np.isnan(y) else None for y in y_data],
            "type": "scattergl",  # WebGL for performance
            "mode": "lines",
            "line": {"width": 1, "color": "rgba(100, 100, 200, 0.1)"},  # Semi-transparent
            "name": "Monte Carlo Simulations",
        }

    def run_stock_forecast(
        self,
        ticker: str,
        horizon_years: int = 2,
        simulations: int = 10000,
    ) -> dict:
        """
        Run Monte Carlo forecast for a stock ticker.
        Based on monte_carlo_forecast.py script.
        
        Uses historical data to estimate mu and sigma.
        """
        LOGGER.info(f"Running Monte Carlo forecast for {ticker}")

        try:
            history = openbb_client.get_price_history(ticker, start_date="2018-01-01")
            closes = [float(item["close"]) for item in history if item.get("close") is not None]
            if len(closes) < 30:
                raise ValueError(f"No historical data available for {ticker}")

            returns = np.diff(closes) / np.array(closes[:-1])
            mu_annual = float(np.mean(returns) * self.trading_days_per_year)
            sigma_annual = float(np.std(returns) * np.sqrt(self.trading_days_per_year))
            current_price = float(closes[-1])
        except Exception as e:
            LOGGER.error(f"Error fetching OpenBB data for {ticker}: {e}")
            raise ValueError(f"Could not fetch data for {ticker}: {e}") from e

        # Run simulation
        result = self.run_portfolio_simulation(
            initial_value=current_price,
            mu=mu_annual,
            sigma=sigma_annual,
            years=horizon_years,
            simulations=simulations,
        )

        # Add ticker-specific info
        result["ticker"] = ticker
        result["current_price"] = current_price
        result["estimated_mu"] = mu_annual
        result["estimated_sigma"] = sigma_annual

        return result

    def generate_histogram_data(self, final_values: np.ndarray, bins: int = 60) -> dict:
        """
        Generate histogram data for final values distribution.
        Returns data optimized for Plotly histogram.
        """
        hist, bin_edges = np.histogram(final_values, bins=bins)
        
        return {
            "x": final_values.tolist(),
            "type": "histogram",
            "nbinsx": bins,
            "marker": {
                "color": "rgba(100, 150, 200, 0.7)",
                "line": {"color": "rgba(0, 0, 0, 0.5)", "width": 1},
            },
            "name": "Distribution",
        }


def get_monte_carlo_service() -> MonteCarloService:
    """Get singleton instance."""
    return MonteCarloService()


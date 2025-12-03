"""
Economic Indicators Processing Service.

Provides data processing functions:
- HP Filter for trend/cycle decomposition
- Z-score normalization
- Business cycle phase classification
- Indicator aggregation
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    from statsmodels.tsa.filters.hpfilter import hpfilter
except ImportError:
    hpfilter = None

from api.models.economic_monitor import (
    BusinessCyclePhase, EconomicIndicator, BusinessCyclePoint,
    HeatmapCell, TemporalClass
)

LOGGER = logging.getLogger("caria.services.economic_indicators_service")


class EconomicIndicatorsService:
    """Service for processing economic indicators."""

    def __init__(self):
        if hpfilter is None:
            LOGGER.warning("statsmodels not available. HP filter will use fallback implementation.")

    def apply_hp_filter(self, series: pd.Series, lambda_param: float = 14400) -> Tuple[pd.Series, pd.Series]:
        """
        Apply Hodrick-Prescott filter to separate trend from cycle.
        
        Args:
            series: Time series data
            lambda_param: HP filter parameter (14400 for monthly, 1600 for quarterly)
        
        Returns:
            Tuple of (trend, cycle)
        """
        if len(series) < 3:
            # Not enough data for HP filter
            return series, pd.Series([0] * len(series), index=series.index)
        
        try:
            if hpfilter is not None:
                cycle, trend = hpfilter(series, lamb=lambda_param)
                return trend, cycle
            else:
                # Simple fallback: use moving average as trend
                window = min(12, len(series) // 2)
                if window < 2:
                    window = 2
                trend = series.rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                cycle = series - trend
                return trend, cycle
        except Exception as e:
            LOGGER.error(f"Error applying HP filter: {e}")
            # Fallback to simple detrending
            trend = series.rolling(window=min(12, len(series) // 2), center=True).mean()
            cycle = series - trend
            return trend.fillna(0), cycle.fillna(0)

    def calculate_z_score(self, value: float, mean: float, std: float) -> float:
        """Calculate Z-score."""
        if std == 0:
            return 0.0
        return (value - mean) / std

    def normalize_indicator(self, value: float, mean: float, std: float, 
                           invert: bool = False) -> float:
        """
        Normalize indicator value to 0-1 range for visualization.
        
        Args:
            value: Current value
            mean: Historical mean
            std: Historical standard deviation
            invert: If True, invert so high values = bad (e.g., inflation)
        
        Returns:
            Normalized value between 0 and 1
        """
        z_score = self.calculate_z_score(value, mean, std)
        
        # Clamp Z-score to reasonable range (-3 to +3)
        z_score = max(-3, min(3, z_score))
        
        # Normalize to 0-1 range
        # Z-score of -3 -> 0, Z-score of +3 -> 1
        normalized = (z_score + 3) / 6
        
        if invert:
            normalized = 1 - normalized
        
        return max(0.0, min(1.0, normalized))

    def classify_business_cycle_phase(self, y: float, x: float) -> BusinessCyclePhase:
        """
        Classify business cycle phase based on clock coordinates.
        
        Args:
            y: Deviation from trend (Y-axis)
            x: Momentum/rate of change (X-axis)
        
        Returns:
            Business cycle phase
        """
        if y > 0 and x > 0:
            return BusinessCyclePhase.EXPANSION
        elif y > 0 and x <= 0:
            return BusinessCyclePhase.SLOWDOWN
        elif y <= 0 and x <= 0:
            return BusinessCyclePhase.RECESSION
        else:  # y <= 0 and x > 0
            return BusinessCyclePhase.RECOVERY

    def calculate_business_cycle_point(self, series: pd.Series, 
                                      country_code: str, country_name: str) -> Optional[BusinessCyclePoint]:
        """
        Calculate business cycle clock coordinates for a country.
        
        Args:
            series: Time series (e.g., Industrial Production)
            country_code: ISO country code
            country_name: Country name
        
        Returns:
            BusinessCyclePoint with x, y coordinates and phase
        """
        if len(series) < 12:
            return None
        
        try:
            # Apply HP filter
            trend, cycle = self.apply_hp_filter(series)
            
            if len(cycle) < 2:
                return None
            
            # Y-axis: Current deviation from trend (standardized)
            current_cycle = cycle.iloc[-1]
            cycle_std = cycle.std()
            if cycle_std == 0:
                y = 0.0
            else:
                y = current_cycle / cycle_std
            
            # X-axis: Momentum (rate of change of cycle)
            if len(cycle) >= 2:
                momentum = cycle.iloc[-1] - cycle.iloc[-2]
                if cycle_std == 0:
                    x = 0.0
                else:
                    x = momentum / cycle_std
            else:
                x = 0.0
            
            # Classify phase
            phase = self.classify_business_cycle_phase(y, x)
            
            # Calculate trajectory (last 6-12 months)
            trajectory_points = []
            for i in range(max(0, len(cycle) - 12), len(cycle)):
                if i >= 1:
                    traj_y = cycle.iloc[i] / cycle_std if cycle_std > 0 else 0
                    traj_x = (cycle.iloc[i] - cycle.iloc[i-1]) / cycle_std if cycle_std > 0 else 0
                    trajectory_points.append({"x": float(traj_x), "y": float(traj_y)})
            
            return BusinessCyclePoint(
                country_code=country_code,
                country_name=country_name,
                x=float(x),
                y=float(y),
                phase=phase,
                trajectory=trajectory_points[-6:] if trajectory_points else None
            )
        except Exception as e:
            LOGGER.error(f"Error calculating business cycle point for {country_code}: {e}")
            return None

    def calculate_rolling_stats(self, series: pd.Series, window: int = 60) -> Tuple[float, float]:
        """
        Calculate rolling mean and standard deviation.
        
        Args:
            series: Time series data
            window: Rolling window size (default 60 months = 5 years)
        
        Returns:
            Tuple of (mean, std)
        """
        if len(series) == 0:
            return 0.0, 1.0
        
        # Use available data, minimum window of 12
        actual_window = min(window, len(series))
        if actual_window < 12:
            actual_window = len(series)
        
        rolling = series.rolling(window=actual_window, min_periods=1)
        mean = rolling.mean().iloc[-1]
        std = rolling.std().iloc[-1]
        
        # Avoid division by zero
        if std == 0 or pd.isna(std):
            std = 1.0
        
        return float(mean), float(std)

    def create_heatmap_cell(self, country_code: str, country_name: str,
                           indicator_name: str, indicator_category: str,
                           value: float, historical_series: pd.Series,
                           invert: bool = False) -> HeatmapCell:
        """
        Create a heatmap cell with normalized values.
        
        Args:
            country_code: ISO country code
            country_name: Country name
            indicator_name: Indicator name
            indicator_category: Category (Growth, External, Fiscal, Prices)
            value: Current indicator value
            historical_series: Historical time series for normalization
            invert: Whether to invert normalization (high = bad)
        
        Returns:
            HeatmapCell with Z-score and normalized value
        """
        mean, std = self.calculate_rolling_stats(historical_series)
        z_score = self.calculate_z_score(value, mean, std)
        normalized_value = self.normalize_indicator(value, mean, std, invert=invert)
        
        # Determine status
        if normalized_value > 0.7:
            status = "health" if not invert else "deterioration"
        elif normalized_value < 0.3:
            status = "deterioration" if not invert else "health"
        else:
            status = "warning"
        
        return HeatmapCell(
            country_code=country_code,
            country_name=country_name,
            indicator_name=indicator_name,
            indicator_category=indicator_category,
            value=value,
            z_score=z_score,
            normalized_value=normalized_value,
            status=status
        )


"""
Model Validation Service per audit document (2.1).
Implements backtesting, statistical metrics (P-value, R²), and benchmarking.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf

LOGGER = logging.getLogger("caria.api.model_validation")

# Import regime service to get historical predictions
from caria.services.regime_service import RegimeService


class ModelValidationService:
    """Service for validating quantitative model predictions."""

    def __init__(self):
        self.trading_days_per_year = 252

    def get_historical_regime_predictions(
        self, start_date: str, end_date: str, db_connection
    ) -> pd.DataFrame:
        """
        Get historical regime predictions from database.
        Returns DataFrame with columns: date, predicted_regime, confidence, predicted_return
        """
        from psycopg2.extras import RealDictCursor

        try:
            with db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT date, predicted_regime, predicted_confidence, predicted_return
                    FROM model_predictions
                    WHERE date >= %s AND date <= %s
                    ORDER BY date
                """, (start_date, end_date))
                
                rows = cursor.fetchall()
                
                if not rows:
                    LOGGER.warning(f"No predictions found for {start_date} to {end_date}")
                    return pd.DataFrame(columns=["date", "predicted_regime", "predicted_confidence", "predicted_return"])
                
                df = pd.DataFrame(rows)
                if len(df) > 0:
                    df["date"] = pd.to_datetime(df["date"])
                    # Convert numeric columns to float (DECIMAL from PostgreSQL may be Decimal objects)
                    if "predicted_return" in df.columns:
                        df["predicted_return"] = pd.to_numeric(df["predicted_return"], errors='coerce')
                    if "predicted_confidence" in df.columns:
                        df["predicted_confidence"] = pd.to_numeric(df["predicted_confidence"], errors='coerce')
                return df
                
        except Exception as e:
            LOGGER.error(f"Error fetching historical predictions: {e}")
            # Check if table exists
            try:
                with db_connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'model_predictions'
                        )
                    """)
                    table_exists = cursor.fetchone()[0]
                    if not table_exists:
                        LOGGER.warning("model_predictions table does not exist. Run generate_validation_data.py first.")
            except:
                pass
            return pd.DataFrame()

    def get_actual_regime_data(self, start_date: str, end_date: str, db_connection=None) -> pd.DataFrame:
        """
        Get actual regime data from database (preferred) or economic indicators (fallback).
        Returns DataFrame with columns: date, actual_regime, actual_return
        """
        # First try to get from database
        if db_connection:
            try:
                from psycopg2.extras import RealDictCursor
                with db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT date, actual_regime, actual_return
                        FROM model_predictions
                        WHERE date >= %s AND date <= %s
                        ORDER BY date
                    """, (start_date, end_date))
                    
                    rows = cursor.fetchall()
                    if rows:
                        df = pd.DataFrame(rows)
                        if len(df) > 0:
                            df["date"] = pd.to_datetime(df["date"])
                            # Convert numeric columns to float (DECIMAL from PostgreSQL may be Decimal objects)
                            if "actual_return" in df.columns:
                                df["actual_return"] = pd.to_numeric(df["actual_return"], errors='coerce')
                        return df
            except Exception as e:
                LOGGER.warning(f"Could not fetch from database: {e}, falling back to yfinance")
        
        # Fallback: Fetch economic indicators that proxy for regime
        # VIX for stress, SPY returns for expansion/recession
        try:
            spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
            vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)

            if spy.empty or vix.empty:
                return pd.DataFrame()

            # Classify regimes based on market behavior
            spy_returns = spy["Close"].pct_change()
            vix_levels = vix["Close"]

            regimes = []
            for date in spy_returns.index:
                ret_val = spy_returns[date]
                vol_val = vix_levels.get(date, 20)
                
                # Convert to scalar if Series
                if isinstance(ret_val, pd.Series):
                    ret_val = ret_val.iloc[0] if len(ret_val) > 0 else np.nan
                if isinstance(vol_val, pd.Series):
                    vol_val = vol_val.iloc[0] if len(vol_val) > 0 else 20
                
                if pd.isna(ret_val) or pd.isna(vol_val):
                    continue

                # Simple classification
                if vol_val > 25:
                    regime = "stress"
                elif ret_val < -0.02:
                    regime = "recession"
                elif ret_val > 0.01:
                    regime = "expansion"
                else:
                    regime = "slowdown"

                regimes.append({"date": date, "actual_regime": regime, "actual_return": float(ret_val)})

            return pd.DataFrame(regimes)

        except Exception as e:
            LOGGER.error(f"Error fetching actual regime data: {e}")
            return pd.DataFrame()

    def calculate_regime_accuracy(
        self, predictions: pd.DataFrame, actual: pd.DataFrame
    ) -> Dict:
        """
        Calculate accuracy metrics for regime predictions.
        """
        if len(predictions) == 0 or len(actual) == 0:
            return {"error": "Insufficient data"}

        # Merge on date
        merged = pd.merge(
            predictions, actual, on="date", how="inner", suffixes=("_pred", "_actual")
        )

        if merged.empty:
            return {"error": "No overlapping dates"}

        # Calculate accuracy
        correct = (merged["predicted_regime"] == merged["actual_regime"]).sum()
        total = len(merged)
        accuracy = correct / total if total > 0 else 0.0

        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report

        regimes = ["expansion", "slowdown", "recession", "stress"]
        cm = confusion_matrix(
            merged["actual_regime"], merged["predicted_regime"], labels=regimes
        )

        return {
            "accuracy": float(accuracy),
            "total_predictions": int(total),
            "correct_predictions": int(correct),
            "confusion_matrix": [[int(x) for x in row] for row in cm.tolist()],
            "regime_labels": [str(r) for r in regimes],
        }

    def calculate_statistical_metrics(
        self, predicted_values: np.ndarray, actual_values: np.ndarray
    ) -> Dict:
        """
        Calculate P-value and R² per audit document (2.1).
        
        Returns metrics with proper interpretation:
        - "significant relationship but noisy predictions" if R² low but P-value < 0.05
        """
        if len(predicted_values) != len(actual_values) or len(predicted_values) < 30:
            return {
                "error": f"Insufficient data points: {len(predicted_values)} (minimum 30 required)"
            }

        # Convert to float64 arrays to ensure numeric types
        pred_array = np.array(predicted_values, dtype=np.float64)
        actual_array = np.array(actual_values, dtype=np.float64)

        # Remove NaN values
        mask = ~(np.isnan(pred_array) | np.isnan(actual_array))
        pred_clean = pred_array[mask]
        actual_clean = actual_array[mask]

        if len(pred_clean) < 30:
            return {"error": "Insufficient valid data points after cleaning"}

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            pred_clean, actual_clean
        )

        r_squared = r_value ** 2

        # Interpretation per audit document
        is_significant = p_value < 0.05
        is_strong_relationship = r_squared > 0.5

        if is_significant and not is_strong_relationship:
            interpretation = (
                "Significant relationship detected (p < 0.05), but predictions are noisy "
                f"(R² = {r_squared:.3f}). The model captures some signal but with high variance."
            )
        elif is_significant and is_strong_relationship:
            interpretation = (
                f"Strong and significant relationship (R² = {r_squared:.3f}, p = {p_value:.4f}). "
                "Model predictions are reliable."
            )
        elif not is_significant:
            interpretation = (
                f"No significant relationship detected (p = {p_value:.4f}). "
                "Model may not be capturing underlying patterns."
            )
        else:
            interpretation = "Relationship detected but requires further analysis."

        return {
            "r_squared": float(r_squared),
            "p_value": float(p_value),
            "slope": float(slope),
            "intercept": float(intercept),
            "std_error": float(std_err),
            "n_observations": int(len(pred_clean)),
            "is_significant": bool(is_significant),
            "is_strong_relationship": bool(is_strong_relationship),
            "interpretation": str(interpretation),
        }

    def benchmark_vs_simple_strategies(
        self,
        model_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Benchmark model performance vs simple strategies per audit document (2.1).
        Compares vs: buy-and-hold, moving average crossover.
        """
        if len(model_returns) == 0:
            return {"error": "No model returns data"}

        # Fetch benchmark if not provided
        if benchmark_returns is None or len(benchmark_returns) == 0:
            try:
                spy = yf.download(
                    "SPY",
                    start=model_returns.index[0],
                    end=model_returns.index[-1],
                    progress=False,
                )
                if len(spy) > 0:
                    benchmark_returns = spy["Close"].pct_change().dropna()
            except Exception as e:
                LOGGER.warning(f"Could not fetch benchmark: {e}")

        results = {}

        # 1. Buy-and-Hold Strategy
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Align dates
            common_dates = model_returns.index.intersection(benchmark_returns.index)
            model_aligned = model_returns.loc[common_dates]
            bench_aligned = benchmark_returns.loc[common_dates]

            if len(model_aligned) > 0 and len(bench_aligned) > 0:
                model_cumulative_val = (1 + model_aligned).cumprod().iloc[-1] - 1
                bench_cumulative_val = (1 + bench_aligned).cumprod().iloc[-1] - 1
                
                # Convert to scalar if Series
                if isinstance(model_cumulative_val, pd.Series):
                    model_cumulative_val = model_cumulative_val.iloc[0]
                if isinstance(bench_cumulative_val, pd.Series):
                    bench_cumulative_val = bench_cumulative_val.iloc[0]

                results["buy_and_hold"] = {
                    "model_return": float(model_cumulative_val),
                    "benchmark_return": float(bench_cumulative_val),
                    "excess_return": float(model_cumulative_val - bench_cumulative_val),
                    "outperformed": bool(model_cumulative_val > bench_cumulative_val),
                }

        # 2. Moving Average Crossover Strategy
        # Simple MA(20) vs MA(50) crossover
        if len(model_returns) > 50:
            ma_short = model_returns.rolling(20).mean()
            ma_long = model_returns.rolling(50).mean()

            # Generate signals
            signals = pd.Series(0, index=model_returns.index)
            signals[ma_short > ma_long] = 1  # Buy signal
            signals[ma_short < ma_long] = -1  # Sell signal (go to cash)

            # Calculate strategy returns (simplified - assume cash return is 0)
            strategy_returns = signals.shift(1) * model_returns
            strategy_returns = strategy_returns.fillna(0)

            strategy_cumulative_val = (1 + strategy_returns).cumprod().iloc[-1] - 1
            model_cumulative_val = (1 + model_returns).cumprod().iloc[-1] - 1
            
            # Convert to scalar if Series
            if isinstance(strategy_cumulative_val, pd.Series):
                strategy_cumulative_val = strategy_cumulative_val.iloc[0]
            if isinstance(model_cumulative_val, pd.Series):
                model_cumulative_val = model_cumulative_val.iloc[0]

            results["moving_average"] = {
                "strategy_return": float(strategy_cumulative_val),
                "model_return": float(model_cumulative_val),
                "excess_return": float(model_cumulative_val - strategy_cumulative_val),
                "outperformed": bool(model_cumulative_val > strategy_cumulative_val),
            }

        return results

    def run_full_validation(
        self, start_date: str, end_date: str, db_connection
    ) -> Dict:
        """
        Run full model validation per audit document (2.1).
        Combines backtesting, statistical metrics, and benchmarking.
        """
        LOGGER.info(f"Running model validation from {start_date} to {end_date}")

        # Get predictions and actuals
        try:
            predictions = self.get_historical_regime_predictions(
                start_date, end_date, db_connection
            )
            LOGGER.info(f"Retrieved {len(predictions)} predictions")
        except Exception as e:
            LOGGER.error(f"Error getting predictions: {e}", exc_info=True)
            predictions = pd.DataFrame()
        
        try:
            actuals = self.get_actual_regime_data(start_date, end_date, db_connection)
            LOGGER.info(f"Retrieved {len(actuals)} actuals")
        except Exception as e:
            LOGGER.error(f"Error getting actuals: {e}", exc_info=True)
            actuals = pd.DataFrame()

        validation_results = {
            "start_date": start_date,
            "end_date": end_date,
            "validation_date": datetime.utcnow().isoformat(),
            "n_predictions": len(predictions) if isinstance(predictions, pd.DataFrame) else 0,
            "n_actuals": len(actuals) if isinstance(actuals, pd.DataFrame) else 0,
        }

        # Backtesting - regime accuracy
        try:
            if isinstance(predictions, pd.DataFrame) and isinstance(actuals, pd.DataFrame):
                pred_len = len(predictions)
                actual_len = len(actuals)
                if pred_len > 0 and actual_len > 0:
                    accuracy_metrics = self.calculate_regime_accuracy(predictions, actuals)
                    validation_results["backtesting"] = accuracy_metrics
        except Exception as e:
            LOGGER.error(f"Error in backtesting: {e}", exc_info=True)
            validation_results["backtesting"] = {"error": str(e)}

        # Statistical metrics - return predictions vs actuals
        try:
            if isinstance(predictions, pd.DataFrame) and isinstance(actuals, pd.DataFrame):
                pred_len = len(predictions)
                actual_len = len(actuals)
                has_pred_return = pred_len > 0 and "predicted_return" in predictions.columns
                has_actual_return = actual_len > 0 and "actual_return" in actuals.columns
                if has_pred_return and has_actual_return:
                    merged = pd.merge(
                        predictions, actuals, on="date", how="inner", suffixes=("_pred", "_actual")
                    )
                    merged_len = len(merged)
                    if merged_len >= 30:
                        # Ensure numeric types and handle any non-numeric values
                        pred_returns = pd.to_numeric(merged["predicted_return"], errors='coerce').values
                        actual_returns = pd.to_numeric(merged["actual_return"], errors='coerce').values
                        stats = self.calculate_statistical_metrics(pred_returns, actual_returns)
                        if "error" not in stats:
                            validation_results["statistical_metrics"] = stats
                        else:
                            validation_results["statistical_metrics"] = {"error": stats["error"]}
                    else:
                        validation_results["statistical_metrics"] = {
                            "error": f"Insufficient data for statistics: {merged_len} points (need 30+)"
                        }
        except Exception as e:
            LOGGER.error(f"Error in statistical metrics: {e}", exc_info=True)
            validation_results["statistical_metrics"] = {"error": str(e)}

        # Benchmarking - convert returns to cumulative performance
        try:
            if isinstance(predictions, pd.DataFrame):
                pred_len = len(predictions)
                if pred_len > 0 and "predicted_return" in predictions.columns:
                    # Use predicted returns as model strategy returns
                    model_returns = pd.Series(
                        predictions["predicted_return"].values,
                        index=pd.to_datetime(predictions["date"])
                    )
                    benchmark_results = self.benchmark_vs_simple_strategies(model_returns)
                    if "error" not in benchmark_results:
                        validation_results["benchmarking"] = benchmark_results
                    else:
                        validation_results["benchmarking"] = {"error": benchmark_results["error"]}
        except Exception as e:
            LOGGER.error(f"Error in benchmarking: {e}", exc_info=True)
            validation_results["benchmarking"] = {"error": str(e)}

        return validation_results


def get_model_validation_service() -> ModelValidationService:
    """Get singleton instance."""
    return ModelValidationService()


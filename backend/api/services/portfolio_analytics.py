"""
Portfolio Analytics Service using quantstats per audit document (3.2).
Generates professional HTML tearsheet with metrics: Sharpe, Sortino, Alpha, Beta, Max Drawdown, CAGR.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np
import pandas as pd
import quantstats as qs
import yfinance as yf

LOGGER = logging.getLogger("caria.api.portfolio_analytics")

# Configure quantstats
qs.extend_pandas()


class PortfolioAnalyticsService:
    """Professional portfolio analytics using quantstats."""

    def __init__(self):
        self.trading_days_per_year = 252
        # en el futuro puedes leer esto de un env var (ej: 0.02 = 2% anual)
        self.risk_free_rate = 0.0

    # ---------------------------------------------------------------------
    # Helpers de datos
    # ---------------------------------------------------------------------

    def get_user_holdings_with_prices(
        self, user_id: UUID, db_connection
    ) -> tuple[list[dict], pd.DataFrame]:
        """
        Get user holdings and fetch historical prices.
        Returns: (holdings_list, price_dataframe)
        """
        import psycopg2
        from psycopg2.extras import RealDictCursor

        with db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT h.id, h.ticker, h.quantity, h.average_cost, h.created_at
                FROM holdings h
                WHERE h.user_id = %s
                ORDER BY h.created_at
                """,
                (str(user_id),),
            )
            holdings = [dict(row) for row in cursor.fetchall()]

        if not holdings:
            return [], pd.DataFrame()

        tickers = [h["ticker"] for h in holdings]
        unique_tickers = list(set(tickers))

        try:
            hist_data = yf.download(
                unique_tickers,
                period="2y",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )

            # MultiIndex → nos quedamos con Close
            if isinstance(hist_data.columns, pd.MultiIndex):
                close_data: dict[str, pd.Series] = {}
                for ticker in unique_tickers:
                    if ticker in hist_data.columns.levels[0]:
                        ticker_data = hist_data[ticker]
                        if "Close" in ticker_data.columns:
                            close_data[ticker] = ticker_data["Close"]
                price_df = pd.DataFrame(close_data)
            else:
                # caso single-ticker
                if "Close" in hist_data.columns:
                    price_df = hist_data["Close"].to_frame()
                    price_df.columns = unique_tickers
                else:
                    price_df = hist_data

            # precios actuales
            for holding in holdings:
                ticker = holding["ticker"]
                if ticker in price_df.columns and not price_df[ticker].empty:
                    holding["current_price"] = float(price_df[ticker].iloc[-1])
                else:
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        info = getattr(ticker_obj, "fast_info", None)
                        if info is not None and getattr(info, "last_price", None):
                            holding["current_price"] = float(info.last_price)
                        else:
                            hist = ticker_obj.history(period="1d")
                            if not hist.empty:
                                holding["current_price"] = float(hist["Close"].iloc[-1])
                    except Exception as e:
                        LOGGER.warning(f"Could not fetch price for {ticker}: {e}")
                        holding["current_price"] = None

        except Exception as e:
            LOGGER.error(f"Error fetching historical data: {e}")
            return holdings, pd.DataFrame()

        return holdings, price_df

    def calculate_portfolio_returns(
        self, holdings: list[dict], price_df: pd.DataFrame
    ) -> Optional[pd.Series]:
        """Build portfolio equity curve and compute returns."""
        if price_df.empty or not holdings:
            return None

        portfolio_values = pd.Series(index=price_df.index, dtype=float)

        for holding in holdings:
            ticker = holding["ticker"]
            quantity = float(holding["quantity"])
            if ticker not in price_df.columns:
                continue

            ticker_prices = price_df[ticker].dropna()
            if ticker_prices.empty:
                continue

            position_values = ticker_prices * quantity
            portfolio_values = portfolio_values.add(position_values, fill_value=0.0)

        if portfolio_values.empty:
            return None

        returns = portfolio_values.pct_change().dropna()
        return returns

    # ---------------------------------------------------------------------
    # Core: análisis de portfolio
    # ---------------------------------------------------------------------

    def analyze_portfolio(
        self, user_id: UUID, db_connection, benchmark: str = "SPY"
    ) -> dict:
        """
        Analyze user portfolio and generate quantstats report.
        Returns metrics dict and (opcional) HTML tearsheet path.
        """
        LOGGER.info(f"Analyzing portfolio for user {user_id}")

        try:
            holdings, price_df = self.get_user_holdings_with_prices(
                user_id, db_connection
            )
            LOGGER.info(
                f"Retrieved {len(holdings)} holdings, "
                f"price_df shape: {price_df.shape if not price_df.empty else 'empty'}"
            )

            if not holdings:
                raise ValueError("No holdings available")

            if price_df.empty:
                raise ValueError("No price data available")

            portfolio_returns = self.calculate_portfolio_returns(holdings, price_df)
            LOGGER.info(
                f"Portfolio returns calculated: "
                f"{len(portfolio_returns) if portfolio_returns is not None else 'None'} values"
            )

            if portfolio_returns is None or portfolio_returns.empty:
                raise ValueError("Could not calculate portfolio returns")

            # Benchmark
            try:
                bench_data = yf.download(
                    benchmark, period="2y", auto_adjust=True, progress=False
                )
                if isinstance(bench_data, pd.DataFrame) and "Close" in bench_data.columns:
                    benchmark_prices = bench_data["Close"]
                else:
                    benchmark_prices = bench_data
                benchmark_returns = benchmark_prices.pct_change().dropna()
            except Exception as e:
                LOGGER.warning(f"Could not fetch benchmark {benchmark}: {e}")
                benchmark_returns = None

            # Alineamos fechas
            if benchmark_returns is not None and not benchmark_returns.empty:
                common_dates = portfolio_returns.index.intersection(
                    benchmark_returns.index
                )
                portfolio_returns = portfolio_returns.loc[common_dates]
                benchmark_returns = benchmark_returns.loc[common_dates]
            else:
                benchmark_returns = None

            LOGGER.info("Calculating metrics with quantstats / custom fallbacks...")
            metrics = self._calculate_metrics(portfolio_returns, benchmark_returns)
            LOGGER.info(f"Metrics calculated: {len(metrics)} metrics")

            # De momento omitimos el HTML para evitar problemas de compatibilidad
            html_path = None

            return {
                "metrics": metrics,
                "html_report_path": str(html_path) if html_path else None,
                "holdings_count": len(holdings),
                "analysis_date": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            LOGGER.exception(f"Error in analyze_portfolio for user {user_id}: {e}")
            raise

    # ---------------------------------------------------------------------
    # Métricas: blindadas contra AttributeError
    # ---------------------------------------------------------------------

    def _qs_or_default(self, name: str, *args, default: float = 0.0, **kwargs) -> float:
        """Llama a qs.stats.<name> si existe; si no, devuelve default."""
        fn = getattr(qs.stats, name, None)
        if fn is None:
            return default
        try:
            return float(fn(*args, **kwargs))
        except Exception as e:
            LOGGER.warning(f"quantstats.stats.{name} failed: {e}")
            return default

    def _compute_beta(self, pr: pd.Series, br: pd.Series) -> float:
        """Beta CAPM calculada a mano (por si no existe qs.stats.beta)."""
        try:
            # Align series by index to ensure same length
            aligned = pd.DataFrame({'portfolio': pr, 'benchmark': br}).dropna()
            if len(aligned) < 2:
                LOGGER.warning("Insufficient aligned data for beta calculation")
                return 0.0
            
            pr_aligned = aligned['portfolio'].values
            br_aligned = aligned['benchmark'].values
            
            # Ensure both arrays have same length and are 1D
            if len(pr_aligned) != len(br_aligned):
                min_len = min(len(pr_aligned), len(br_aligned))
                pr_aligned = pr_aligned[:min_len]
                br_aligned = br_aligned[:min_len]
            
            if len(pr_aligned) < 2:
                return 0.0
            
            # Ensure arrays are 1D
            pr_aligned = pr_aligned.flatten()
            br_aligned = br_aligned.flatten()
            
            cov = np.cov(pr_aligned, br_aligned)[0, 1]
            var_b = np.var(br_aligned)
            if var_b <= 0:
                return 0.0
            return float(cov / var_b)
        except Exception as e:
            LOGGER.warning(f"Manual beta computation failed: {e}")
            return 0.0

    def _compute_alpha(self, pr: pd.Series, br: pd.Series, beta: float) -> float:
        """Alpha CAPM diaria (no anualizada) como fallback."""
        try:
            daily_rf = self.risk_free_rate / self.trading_days_per_year
            
            # Align series and ensure they're Series, not single values
            aligned = pd.DataFrame({'portfolio': pr, 'benchmark': br}).dropna()
            if len(aligned) < 1:
                return 0.0
            
            pr_aligned = aligned['portfolio']
            br_aligned = aligned['benchmark']
            
            # Calculate means - handle both Series and scalar cases
            if isinstance(pr_aligned, pd.Series):
                mean_p = pr_aligned.mean()
                # Handle case where mean() returns a Series (shouldn't happen, but be safe)
                if isinstance(mean_p, pd.Series):
                    excess_p = float(mean_p.iloc[0]) - daily_rf
                else:
                    excess_p = float(mean_p) - daily_rf
            else:
                excess_p = float(pr_aligned) - daily_rf
                
            if isinstance(br_aligned, pd.Series):
                mean_b = br_aligned.mean()
                # Handle case where mean() returns a Series (shouldn't happen, but be safe)
                if isinstance(mean_b, pd.Series):
                    excess_b = float(mean_b.iloc[0]) - daily_rf
                else:
                    excess_b = float(mean_b) - daily_rf
            else:
                excess_b = float(br_aligned) - daily_rf
            
            # Calculate result - ensure we're working with scalars
            result = excess_p - beta * excess_b
            if isinstance(result, pd.Series):
                return float(result.iloc[0])
            return float(result)
        except Exception as e:
            LOGGER.warning(f"Manual alpha computation failed: {e}")
            return 0.0

    def _calculate_metrics(
        self, portfolio_returns: pd.Series, benchmark_returns: Optional[pd.Series]
    ) -> dict:
        """Calculate professional metrics using quantstats + fallbacks."""
        pr = portfolio_returns
        br = benchmark_returns

        metrics: dict[str, float | None] = {}

        # Básicos
        metrics["sharpe_ratio"] = self._qs_or_default("sharpe", pr)
        metrics["sortino_ratio"] = self._qs_or_default("sortino", pr)
        metrics["max_drawdown"] = self._qs_or_default("max_drawdown", pr)
        metrics["cagr"] = self._qs_or_default("cagr", pr)

        # Volatilidad
        metrics["volatility_annual"] = self._qs_or_default(
            "volatility", pr, annualize=True
        )
        metrics["volatility_monthly"] = self._qs_or_default(
            "volatility", pr, annualize=False
        )

        # Retornos
        metrics["total_return"] = self._qs_or_default("comp", pr)
        metrics["avg_return_annual"] = self._qs_or_default(
            "avg_return", pr, aggregate="A"
        )

        # Riesgo (VaR / CVaR) – con fallback manual
        try:
            metrics["var_95"] = float(
                qs.stats.value_at_risk(pr, confidence=0.05)
            )
        except Exception:
            metrics["var_95"] = float(pr.quantile(0.05))

        try:
            metrics["cvar_95"] = float(
                qs.stats.conditional_value_at_risk(pr, confidence=0.05)
            )
        except Exception:
            var_95 = pr.quantile(0.05)
            cvar = pr[pr <= var_95].mean()
            metrics["cvar_95"] = float(cvar) if not pd.isna(cvar) else 0.0

        # Comparación con benchmark
        if br is not None and not br.empty:
            # Beta: prioridad a quantstats, luego fallback manual
            beta_qs = None
            fn_beta = getattr(qs.stats, "beta", None)
            if fn_beta is not None:
                try:
                    beta_qs = float(fn_beta(pr, br))
                except Exception as e:
                    LOGGER.warning(f"quantstats.stats.beta failed: {e}")
                    beta_qs = None

            beta = beta_qs if beta_qs is not None else self._compute_beta(pr, br)
            metrics["beta"] = beta

            # Alpha: prioridad a quantstats, luego fallback manual
            alpha_qs = None
            fn_alpha = getattr(qs.stats, "alpha", None)
            if fn_alpha is not None:
                try:
                    alpha_qs = float(fn_alpha(pr, br))
                except Exception as e:
                    LOGGER.warning(f"quantstats.stats.alpha failed: {e}")
                    alpha_qs = None

            metrics["alpha"] = (
                alpha_qs if alpha_qs is not None else self._compute_alpha(pr, br, beta)
            )

            # r²
            r2_qs = None
            fn_r2 = getattr(qs.stats, "r_squared", None)
            if fn_r2 is not None:
                try:
                    r2_qs = float(fn_r2(pr, br))
                except Exception as e:
                    LOGGER.warning(f"quantstats.stats.r_squared failed: {e}")
                    r2_qs = None

            if r2_qs is not None:
                metrics["r_squared"] = r2_qs
            else:
                try:
                    corr = np.corrcoef(pr, br)[0, 1]
                    metrics["r_squared"] = float(corr**2)
                except Exception:
                    metrics["r_squared"] = None

            # Information ratio & tracking error (si existen; si no, 0)
            metrics["information_ratio"] = self._qs_or_default(
                "information_ratio", pr, br, default=0.0
            )
            metrics["tracking_error"] = self._qs_or_default(
                "tracking_error", pr, br, default=0.0
            )
        else:
            metrics["beta"] = None
            metrics["alpha"] = None
            metrics["r_squared"] = None
            metrics["information_ratio"] = None
            metrics["tracking_error"] = None

        # Otras
        metrics["skewness"] = self._qs_or_default("skew", pr)
        metrics["kurtosis"] = self._qs_or_default("kurtosis", pr)
        metrics["calmar_ratio"] = self._qs_or_default("calmar", pr)
        metrics["win_rate"] = self._qs_or_default("win_rate", pr)
        metrics["win_loss_ratio"] = self._qs_or_default("win_loss_ratio", pr)

        return metrics

    # ---------------------------------------------------------------------

    def _generate_tearsheet(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series],
        user_id: UUID,
        benchmark: str,
    ) -> Path:
        """Generate HTML tearsheet using quantstats (no se está usando ahora)."""
        output_dir = Path("/tmp/portfolio_reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        html_path = output_dir / f"portfolio_{user_id}_{datetime.utcnow().strftime('%Y%m%d')}.html"

        if benchmark_returns is not None and not benchmark_returns.empty:
            qs.reports.html(
                portfolio_returns,
                benchmark=benchmark_returns,
                output=str(html_path),
                title=f"Portfolio Analysis - {benchmark}",
                download_filename=str(html_path),
            )
        else:
            qs.reports.html(
                portfolio_returns,
                output=str(html_path),
                title="Portfolio Analysis",
                download_filename=str(html_path),
            )

        LOGGER.info(f"Generated tearsheet: {html_path}")
        return html_path


def get_portfolio_analytics_service() -> PortfolioAnalyticsService:
    """Get singleton instance."""
    return PortfolioAnalyticsService()

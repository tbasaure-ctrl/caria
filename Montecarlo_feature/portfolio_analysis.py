"""
Professional Portfolio Analytics System
Integrates with Google Sheets and provides institutional-grade risk metrics
"""

from __future__ import annotations

import os
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import SpreadsheetNotFound

# Optional: your MC engine
try:
    from . import mc_engine  # type: ignore
    HAS_MC_ENGINE = True
except Exception:
    HAS_MC_ENGINE = False
    print("[INFO] Monte Carlo engine not available")

# ===================== CONFIGURATION =====================

@dataclass
class Config:
    """System configuration"""
    finnhub_token: str = os.getenv("FINNHUB_TOKEN", "d1v5qgpr01qo0ln2ma3gd1v5qgpr01qo0ln2ma40")
    sheet_id: str = os.getenv("PORTFOLIO_SHEET_ID", "1T7kBp2o69rFkKKyHGnuqmNh8L4Cb_ueqxnVRwxH99_c")
    worksheet_name: str = os.getenv("PORTFOLIO_SHEET_TAB", "Portafolio")
    service_account_json: str = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "C:/Users/tomas/Downloads/dcfs-461616-8194af158ba5.json"
    )

    # Portfolio settings
    portfolio_currency: str = "USD"
    quote_currency: str = "USD"

    # Column headers (Spanish)
    headers: Dict[str, str] = field(default_factory=lambda: {
        "ticker": "Stonk",
        "qty": "Cantidad total",
        "avg_cost": "Avg buy (CLP)",
        "current_price": "Precio actual",
        "return_pct": "%Retorno",
        "pnl": "P/L",
        "value": "Valor actual",
        "mc_p10": "MC P10",
        "mc_median": "MC Median",
        "mc_p90": "MC P90"
    })

    # Benchmarks
    primary_benchmark: str = "SPY"
    benchmarks: Dict[str, str] = field(default_factory=lambda: {
        "SPY": "SPY",
        "IPSA": "^IPSA",
        "ACWI": "ACWI"
    })

    # Macro indicators
    macro_tickers: Dict[str, str] = field(default_factory=lambda: {
        "DXY": "DX-Y.NYB",
        "EURUSD": "EURUSD=X",
        "USDCLP": "USDCLP=X",
        "UST10Y": "^TNX",
        "UST3M": "^IRX",
        "VIX": "^VIX",
        "HY Credit": "HYG",
        "Gold": "GC=F",
        "Oil": "CL=F",
        "Copper": "HG=F",
    })

    # Output settings
    trading_days_per_year: int = 252
    chart_dir: Path = Path(__file__).parent / "charts"

CONFIG = Config()

# ===================== DATA MODELS =====================

@dataclass
class Position:
    """Portfolio position"""
    ticker: str
    quantity: float
    avg_cost: Optional[float] = None
    current_price: Optional[float] = None

    @property
    def market_value(self) -> float:
        if self.current_price is None:
            return 0.0
        return float(self.quantity) * float(self.current_price)

    @property
    def pnl(self) -> Optional[float]:
        if self.avg_cost is None or self.current_price is None:
            return None
        return (float(self.current_price) - float(self.avg_cost)) * float(self.quantity)

    @property
    def return_pct(self) -> Optional[float]:
        if self.avg_cost is None or self.avg_cost == 0:
            return None
        return (float(self.current_price) / float(self.avg_cost) - 1) if self.current_price else None


@dataclass
class PortfolioMetrics:
    """Comprehensive portfolio analytics"""
    # Returns
    total_return_ann: float
    volatility_ann: float
    sharpe_ratio: float

    # Risk
    max_drawdown: float
    var_95: float
    cvar_95: float

    # vs Benchmark
    beta: float
    alpha_ann: float
    r_squared: float
    tracking_error: float
    information_ratio: float

    # Advanced
    downside_beta: float
    ulcer_index: float
    ulcer_performance_index: float
    sortino_ratio: float

    # Distribution
    skewness: float
    kurtosis: float

    # Concentration
    herfindahl_index: float

    # Drawdown
    max_drawdown_duration_days: int

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}

# ===================== UTILITIES =====================

class NumberFormatter:
    """Format numbers for Spanish locale (comma decimal, dot thousands)"""

    @staticmethod
    def parse(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        s = str(value).strip().replace('$', '').replace('CLP', '').replace('%', '').replace(' ', '')
        try:
            has_dot = '.' in s
            has_comma = ',' in s
            if has_dot and has_comma:
                last_dot = s.rfind('.')
                last_comma = s.rfind(',')
                if last_comma > last_dot:
                    return float(s.replace('.', '').replace(',', '.'))
                return float(s.replace(',', ''))
            if has_comma and not has_dot:
                parts = s.split(',')
                if len(parts) == 2 and len(parts[1]) <= 2:
                    return float(s.replace(',', '.'))
                return float(s.replace(',', ''))
            return float(s)
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def format(value: Optional[float], decimals: int = 2) -> str:
        if value is None or np.isnan(value):
            return ""
        try:
            formatted = f"{float(value):,.{decimals}f}"
            return formatted.replace(',', 'TEMP').replace('.', ',').replace('TEMP', '.')
        except (ValueError, TypeError):
            return ""

# ===================== DATA PROVIDERS =====================

class PriceProvider:
    """Fetch market prices from multiple sources"""

    def __init__(self, config: Config):
        self.config = config

    def get_fx_rate(self, base: str, quote: str) -> float:
        """Return FX rate base->quote with fallbacks."""
        if base == quote:
            return 1.0
        # Finnhub
        try:
            url = "https://finnhub.io/api/v1/forex/rates"
            params = {"base": base, "token": self.config.finnhub_token}
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json() or {}
            # Finnhub returns {"base":"USD","quote":{"EUR":0.93,...}}
            rate = (data.get("quote") or {}).get(quote)
            if rate:
                return float(rate)
        except Exception:
            pass
        # yfinance fallback
        try:
            pair = f"{base}{quote}=X"
            t = yf.Ticker(pair)
            hist = t.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception as e:
            print(f"[WARN] yfinance FX rate failed for {base}/{quote}: {e}")
        # Last resort
        fallback_rates = {("USD", "CLP"): 950.0, ("CLP", "USD"): 1/950.0}
        return float(fallback_rates.get((base, quote), 1.0))

    def get_price(self, ticker: str) -> Optional[float]:
        """Get current price, try Finnhub then yfinance"""
        try:
            url = "https://finnhub.io/api/v1/quote"
            params = {"symbol": ticker, "token": self.config.finnhub_token}
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json() or {}
            price = data.get("c") or data.get("pc")
            if price and price > 0:
                return float(price)
        except Exception:
            pass
        try:
            stock = yf.Ticker(ticker)
            info = getattr(stock, "fast_info", None)
            if info and hasattr(info, "last_price") and info.last_price:
                return float(info.last_price)
            hist = stock.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception as e:
            print(f"[WARN] Price fetch failed for {ticker}: {e}")
        return None

class HistoricalDataProvider:
    """Download historical closes via yfinance"""

    @staticmethod
    def fetch(tickers: List[str], period: str = "2y") -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()
        try:
            data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
            if data.empty:
                print(f"[WARN] No data returned for tickers: {tickers}")
                return pd.DataFrame()
            if "Close" in data:
                data = data["Close"]
            if isinstance(data, pd.Series):
                data = data.to_frame(name=tickers[0])
            return data
        except Exception as e:
            print(f"[WARN] Historical data fetch failed: {e}")
            print("[INFO] Attempting individual ticker downloads...")
            dfs = []
            for tk in tickers:
                try:
                    df = yf.download(tk, period=period, auto_adjust=True, progress=False)
                    if not df.empty and "Close" in df:
                        dfs.append(df["Close"].rename(tk))
                except Exception as te:
                    print(f"[WARN] Failed to download {tk}: {te}")
            return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

# ===================== GOOGLE SHEETS CLIENT =====================

class SheetsClient:
    """Handle Google Sheets operations"""

    def __init__(self, config: Config):
        self.config = config
        self._worksheet = None

    def connect(self):
        """Connect to Google Sheets"""
        creds_path = Path(self.config.service_account_json)
        if not creds_path.exists():
            raise FileNotFoundError(f"Service account JSON not found: {creds_path}")

        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(str(creds_path), scopes=scopes)
        gc = gspread.authorize(creds)

        # Extract sheet ID from URL if needed
        sheet_id = self.config.sheet_id
        if "/d/" in sheet_id:
            sheet_id = sheet_id.split("/d/")[1].split("/")[0]

        try:
            spreadsheet = gc.open_by_key(sheet_id)
            self._worksheet = spreadsheet.worksheet(self.config.worksheet_name)
            return self._worksheet
        except SpreadsheetNotFound as e:
            try:
                with open(creds_path) as f:
                    email = json.load(f).get("client_email", "<unknown>")
            except Exception:
                email = "<unknown>"
            raise RuntimeError(
                f"Sheet not found. Please:\n"
                f"1. Verify sheet ID: {sheet_id}\n"
                f"2. Share sheet with: {email}\n"
                f"3. Check worksheet tab name: {self.config.worksheet_name}"
            ) from e

    def read_positions(self) -> List[Position]:
        if not self._worksheet:
            self.connect()
        rows = self._worksheet.get_all_values()
        if not rows:
            return []
        header_row = rows[0]
        headers = {name.strip(): idx for idx, name in enumerate(header_row)}

        required = [
            self.config.headers["ticker"],
            self.config.headers["qty"],
            self.config.headers["avg_cost"],
        ]
        for col in required:
            if col not in headers:
                raise ValueError(f"Missing required column: {col}")

        positions: List[Position] = []
        formatter = NumberFormatter()
        for row in rows[1:]:
            ticker = row[headers[self.config.headers["ticker"]]].strip()
            if not ticker or ticker.lower() in {"total", "cash", "sgov", "subtotal"}:
                continue
            qty = formatter.parse(row[headers[self.config.headers["qty"]]])
            avg_cost = formatter.parse(row[headers[self.config.headers["avg_cost"]]])
            if qty and qty > 0:
                positions.append(Position(ticker=ticker, quantity=qty, avg_cost=avg_cost))
        return positions

    def write_batch(self, updates: List[Tuple[int, int, Any]]):
        if not updates or not self._worksheet:
            return
        cells = [gspread.Cell(row, col, str(val)) for row, col, val in updates]
        self._worksheet.update_cells(cells, value_input_option="USER_ENTERED")

    def ensure_columns(self, columns: List[str]) -> Dict[str, int]:
        if not self._worksheet:
            self.connect()
        header_row = self._worksheet.row_values(1)
        header_map = {name.strip(): idx + 1 for idx, name in enumerate(header_row)}
        max_col = len(header_row)
        for col_name in columns:
            if col_name not in header_map:
                max_col += 1
                self._worksheet.update_cell(1, max_col, col_name)
                header_map[col_name] = max_col
        return header_map

# ===================== ANALYTICS ENGINE =====================

class RiskMetrics:
    """Calculate portfolio risk metrics"""

    def __init__(self, trading_days: int = 252):
        self.trading_days = trading_days

    def annualized_return(self, returns: pd.Series) -> float:
        if returns.empty:
            return np.nan
        mean_return = returns.mean()
        return float((1 + mean_return) ** self.trading_days - 1)

    def annualized_volatility(self, returns: pd.Series) -> float:
        if returns.empty:
            return np.nan
        return float(returns.std() * np.sqrt(self.trading_days))

    def sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        ann_return = self.annualized_return(returns)
        ann_vol = self.annualized_volatility(returns)
        if ann_vol == 0 or np.isnan(ann_vol):
            return np.nan
        return float((ann_return - risk_free_rate) / ann_vol)

    def sortino_ratio(self, returns: pd.Series, target_return: float = 0.0) -> float:
        if returns.empty:
            return np.nan
        downside = returns[returns < target_return]
        if downside.empty:
            return np.nan
        downside_std = downside.std() * np.sqrt(self.trading_days)
        ann_return = self.annualized_return(returns)
        return float((ann_return - target_return) / downside_std) if downside_std > 0 else np.nan

    def max_drawdown(self, returns: pd.Series) -> float:
        if returns.empty:
            return np.nan
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max - 1)
        return float(drawdown.min())

    def value_at_risk(self, returns: pd.Series, confidence: float = 0.95) -> float:
        if returns.empty:
            return np.nan
        return float(np.percentile(returns, (1 - confidence) * 100))

    def conditional_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        if returns.empty:
            return np.nan
        var = self.value_at_risk(returns, confidence)
        tail = returns[returns <= var]
        return float(tail.mean()) if not tail.empty else np.nan

    def beta(self, asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        df = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
        if df.empty or len(df) < 2:
            return np.nan
        covariance = np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1]
        benchmark_var = np.var(df.iloc[:, 1])
        return float(covariance / benchmark_var) if benchmark_var > 0 else np.nan

    def alpha(self, asset_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        df = pd.concat([benchmark_returns, asset_returns], axis=1).dropna()
        if df.empty or len(df) < 2:
            return {"alpha_ann": np.nan, "beta": np.nan, "r_squared": np.nan}
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        beta_coef, alpha_daily = np.polyfit(x, y, 1)  # y = beta*x + alpha
        alpha_ann = (1 + alpha_daily) ** self.trading_days - 1
        correlation = np.corrcoef(x, y)[0, 1]
        r_squared = correlation ** 2
        return {"alpha_ann": float(alpha_ann), "beta": float(beta_coef), "r_squared": float(r_squared)}

    def tracking_error(self, asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        df = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
        if df.empty:
            return np.nan
        diff = df.iloc[:, 0] - df.iloc[:, 1]
        return float(diff.std() * np.sqrt(self.trading_days))

    def information_ratio(self, asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        te = self.tracking_error(asset_returns, benchmark_returns)
        if np.isnan(te) or te == 0:
            return np.nan
        asset_ann = self.annualized_return(asset_returns)
        bench_ann = self.annualized_return(benchmark_returns)
        return float((asset_ann - bench_ann) / te)

    def ulcer_index(self, returns: pd.Series) -> Tuple[float, float]:
        if returns.empty:
            return np.nan, np.nan
        cumulative = (1 + returns).cumprod()
        drawdown_pct = 100 * (1 - cumulative / cumulative.cummax())
        ulcer = np.sqrt(np.mean(drawdown_pct ** 2))
        ann_return = self.annualized_return(returns)
        upi = (ann_return / (ulcer / 100)) if ulcer > 0 else np.nan
        return float(ulcer), float(upi)

    def downside_beta(self, asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        df = pd.concat([benchmark_returns, asset_returns], axis=1).dropna()
        df_down = df[df.iloc[:, 0] < 0]
        if df_down.empty or len(df_down) < 2:
            return np.nan
        x = df_down.iloc[:, 0].values
        y = df_down.iloc[:, 1].values
        beta_coef, _ = np.polyfit(x, y, 1)
        return float(beta_coef)

    def drawdown_duration(self, returns: pd.Series) -> int:
        if returns.empty:
            return 0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        durations = []
        current_duration = 0
        for val, peak in zip(cumulative, running_max):
            if val < peak:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        if current_duration > 0:
            durations.append(current_duration)
        return max(durations) if durations else 0

    def herfindahl_index(self, weights: np.ndarray) -> float:
        return float(np.sum(np.square(weights)))

# ===================== ANALYZER =====================

class PortfolioAnalyzer:
    """Main analytics engine"""

    def __init__(self, config: Config):
        self.config = config
        self.price_provider = PriceProvider(config)
        self.hist_provider = HistoricalDataProvider()
        self.risk_metrics = RiskMetrics(config.trading_days_per_year)
        self.sheets_client = SheetsClient(config)

    def analyze(self) -> Tuple[List[Position], PortfolioMetrics, pd.Series]:
        print("[INFO] Connecting to Google Sheets...")
        self.sheets_client.connect()

        print("[INFO] Reading positions...")
        positions = self.sheets_client.read_positions()
        if not positions:
            raise ValueError("No positions found in sheet")
        print(f"[INFO] Found {len(positions)} positions")

        # Fetch current prices
        print("[INFO] Fetching current prices...")
        fx_rate = 1.0
        if self.config.portfolio_currency != self.config.quote_currency:
            fx_rate = self.price_provider.get_fx_rate(
                self.config.quote_currency, self.config.portfolio_currency
            )
        print(f"[INFO] Using FX rate: 1 {self.config.quote_currency} = {fx_rate:.2f} {self.config.portfolio_currency}")

        for pos in positions:
            price_quote = self.price_provider.get_price(pos.ticker)
            if price_quote:
                pos.current_price = price_quote * fx_rate
            time.sleep(0.15)  # Rate limiting

        # Fetch historical data
        print("[INFO] Fetching historical data...")
        tickers = [p.ticker for p in positions]
        bench_tickers = list(self.config.benchmarks.values())
        hist_data = self.hist_provider.fetch(tickers + bench_tickers, period="2y")
        if hist_data.empty:
            print("[WARN] Failed to fetch all data, trying portfolio tickers only...")
            hist_data = self.hist_provider.fetch(tickers, period="2y")
        if hist_data.empty:
            print("[WARN] Trying shorter period (1y)...")
            hist_data = self.hist_provider.fetch(tickers, period="1y")
        if hist_data.empty:
            raise ValueError(
                "No historical data available. Please check:\n"
                "1. Internet connection\n"
                "2. Ticker symbols are correct\n"
                "3. yfinance is working (try: yf.download('SPY', period='5d'))"
            )
        print(f"[INFO] Successfully fetched data for {len(hist_data.columns)} tickers")

        # Calculate portfolio weights and returns
        valid_positions = [p for p in positions if p.current_price and p.ticker in hist_data.columns]
        if not valid_positions:
            raise ValueError("No positions with valid price and history data")
        print(f"[INFO] Analyzing {len(valid_positions)} positions with complete data")

        total_value = sum(p.market_value for p in valid_positions)
        weights = np.array(
            [p.market_value / total_value if total_value > 0 else 0 for p in valid_positions],
            dtype=float,
        )

        valid_tickers = [p.ticker for p in valid_positions]
        returns_df = hist_data[valid_tickers].pct_change().dropna()
        if returns_df.empty:
            raise ValueError("No return data available after processing")

        portfolio_returns = pd.Series(returns_df.values @ weights, index=returns_df.index)

        # Benchmark returns
        bench_ticker = self.config.benchmarks[self.config.primary_benchmark]
        if bench_ticker not in hist_data.columns:
            print(f"[WARN] Benchmark {bench_ticker} not in data, fetching separately...")
            bench_data = self.hist_provider.fetch([bench_ticker], period="2y")
            if not bench_data.empty:
                hist_data = pd.concat([hist_data, bench_data], axis=1)

        if bench_ticker not in hist_data.columns:
            print(f"[WARN] Could not fetch {bench_ticker}, using portfolio as its own benchmark")
            benchmark_returns = portfolio_returns.copy()
        else:
            benchmark_returns = hist_data[bench_ticker].pct_change().dropna()

        # Align dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_returns = portfolio_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]

        # Calculate metrics
        print("[INFO] Calculating metrics...")
        metrics = self._calculate_metrics(portfolio_returns, benchmark_returns, weights)
        return positions, metrics, portfolio_returns

    def _calculate_metrics(
        self, port_returns: pd.Series, bench_returns: pd.Series, weights: np.ndarray
    ) -> PortfolioMetrics:
        rm = self.risk_metrics
        ann_return = rm.annualized_return(port_returns)
        ann_vol = rm.annualized_volatility(port_returns)
        sharpe = rm.sharpe_ratio(port_returns)
        sortino = rm.sortino_ratio(port_returns)
        mdd = rm.max_drawdown(port_returns)
        var95 = rm.value_at_risk(port_returns)
        cvar95 = rm.conditional_var(port_returns)
        beta = rm.beta(port_returns, bench_returns)
        alpha_stats = rm.alpha(port_returns, bench_returns)
        te = rm.tracking_error(port_returns, bench_returns)
        ir = rm.information_ratio(port_returns, bench_returns)
        downside_b = rm.downside_beta(port_returns, bench_returns)
        ulcer, upi = rm.ulcer_index(port_returns)
        dd_duration = rm.drawdown_duration(port_returns)
        hhi = rm.herfindahl_index(weights)
        skew = float(port_returns.skew())
        kurt = float(port_returns.kurtosis())
        return PortfolioMetrics(
            total_return_ann=ann_return,
            volatility_ann=ann_vol,
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            var_95=var95,
            cvar_95=cvar95,
            beta=beta,
            alpha_ann=alpha_stats["alpha_ann"],
            r_squared=alpha_stats["r_squared"],
            tracking_error=te,
            information_ratio=ir,
            downside_beta=downside_b,
            ulcer_index=ulcer,
            ulcer_performance_index=upi,
            sortino_ratio=sortino,
            skewness=skew,
            kurtosis=kurt,
            herfindahl_index=hhi,
            max_drawdown_duration_days=dd_duration,
        )

    def update_sheet(self, positions: List[Position], metrics: PortfolioMetrics):
        print("[INFO] Updating Google Sheet...]")

        required_cols = [
            self.config.headers[k]
            for k in ["current_price", "return_pct", "pnl", "value", "mc_p10", "mc_median", "mc_p90"]
        ]
        header_map = self.sheets_client.ensure_columns(required_cols)

        updates: List[Tuple[int, int, Any]] = []
        formatter = NumberFormatter()

        for row_idx, pos in enumerate(positions, start=2):
            if pos.current_price:
                updates.append(
                    (row_idx, header_map[self.config.headers["current_price"]], formatter.format(pos.current_price, 2))
                )
            if pos.return_pct is not None:
                updates.append(
                    (row_idx, header_map[self.config.headers["return_pct"]], formatter.format(pos.return_pct * 100, 2) + "%")
                )
            if pos.pnl is not None:
                updates.append((row_idx, header_map[self.config.headers["pnl"]], formatter.format(pos.pnl, 2)))
            updates.append((row_idx, header_map[self.config.headers["value"]], formatter.format(pos.market_value, 2)))

        self.sheets_client.write_batch(updates)
        self._write_analytics_tab(metrics)
        print(f"[INFO] Updated {len(updates)} cells")

    def _write_analytics_tab(self, metrics: PortfolioMetrics):
        try:
            worksheet = self.sheets_client._worksheet
            book = worksheet.spreadsheet
            analytics_ws = next((w for w in book.worksheets() if w.title == "Analytics"), None)
            if not analytics_ws:
                analytics_ws = book.add_worksheet(title="Analytics", rows=50, cols=3)

            formatter = NumberFormatter()
            rows = [
                ["Metric", "Value"],
                ["As of", datetime.now().strftime("%Y-%m-%d %H:%M")],
                [""],
                ["=== RETURNS ===", ""],
                ["Annual Return", formatter.format(metrics.total_return_ann * 100, 2) + "%"],
                ["Annual Volatility", formatter.format(metrics.volatility_ann * 100, 2) + "%"],
                ["Sharpe Ratio", formatter.format(metrics.sharpe_ratio, 3)],
                ["Sortino Ratio", formatter.format(metrics.sortino_ratio, 3)],
                [""],
                ["=== RISK ===", ""],
                ["Max Drawdown", formatter.format(metrics.max_drawdown * 100, 2) + "%"],
                ["VaR 95% (1d)", formatter.format(metrics.var_95 * 100, 2) + "%"],
                ["CVaR 95% (1d)", formatter.format(metrics.cvar_95 * 100, 2) + "%"],
                ["Ulcer Index", formatter.format(metrics.ulcer_index, 2)],
                [""],
                ["=== vs BENCHMARK ===", ""],
                ["Beta", formatter.format(metrics.beta, 3)],
                ["Alpha (annual)", formatter.format(metrics.alpha_ann * 100, 2) + "%"],
                ["R²", formatter.format(metrics.r_squared, 3)],
                ["Tracking Error", formatter.format(metrics.tracking_error * 100, 2) + "%"],
                ["Information Ratio", formatter.format(metrics.information_ratio, 3)],
                ["Downside Beta", formatter.format(metrics.downside_beta, 3)],
                [""],
                ["=== DISTRIBUTION ===", ""],
                ["Skewness", formatter.format(metrics.skewness, 3)],
                ["Kurtosis", formatter.format(metrics.kurtosis, 3)],
                [""],
                ["=== CONCENTRATION ===", ""],
                ["Herfindahl Index", formatter.format(metrics.herfindahl_index, 4)],
                ["Max DD Duration (days)", str(metrics.max_drawdown_duration_days)],
            ]
            analytics_ws.clear()
            analytics_ws.update("A1", rows, value_input_option="USER_ENTERED")
        except Exception as e:
            print(f"[WARN] Failed to write analytics tab: {e}")

    def generate_charts(
        self, positions: List[Position], portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ):
        print("[INFO] Generating charts...]")
        self.config.chart_dir.mkdir(exist_ok=True)
        sns.set_style("whitegrid")

        # 1. Allocation pie chart
        plt.figure(figsize=(8, 8))
        weights = [p.market_value for p in positions if p.current_price]
        labels = [p.ticker for p in positions if p.current_price]
        if sum(weights) > 0:
            plt.pie(weights, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title("Portfolio Allocation", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.config.chart_dir / "allocation.png", dpi=150)
        plt.close()

        # 2. Cumulative returns
        plt.figure(figsize=(10, 6))
        port_cum = (1 + portfolio_returns).cumprod()
        bench_cum = (1 + benchmark_returns).cumprod()
        port_cum.plot(label="Portfolio", linewidth=2)
        if not bench_cum.empty:
            bench_cum.plot(label=self.config.primary_benchmark, linewidth=2, alpha=0.7)
        plt.title("Cumulative Returns", fontsize=14, fontweight='bold')
        plt.xlabel("Date")
        plt.ylabel("Growth of $1")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.config.chart_dir / "cumulative_returns.png", dpi=150)
        plt.close()

        # 3. Drawdown
        plt.figure(figsize=(10, 4))
        cumulative = (1 + portfolio_returns).cumprod()
        drawdown = (cumulative / cumulative.cummax() - 1) * 100
        drawdown.plot(color='darkred', linewidth=1.5)
        plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        plt.title(f"Drawdown (Max: {drawdown.min():.1f}%)", fontsize=14, fontweight='bold')
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.config.chart_dir / "drawdown.png", dpi=150)
        plt.close()

        # 4. Rolling volatility
        plt.figure(figsize=(10, 5))
        rolling_vol = portfolio_returns.rolling(60).std() * np.sqrt(252) * 100
        rolling_vol.plot(color='purple', linewidth=2)
        plt.title("Rolling 60-Day Volatility (Annualized)", fontsize=14, fontweight='bold')
        plt.xlabel("Date")
        plt.ylabel("Volatility (%)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.config.chart_dir / "rolling_volatility.png", dpi=150)
        plt.close()

        print(f"[INFO] Charts saved to {self.config.chart_dir}")

# ===================== MAIN =====================

def main():
    try:
        analyzer = PortfolioAnalyzer(CONFIG)
        positions, metrics, portfolio_returns = analyzer.analyze()

        print("\n" + "="*70)
        print("PORTFOLIO ANALYSIS SUMMARY")
        print("="*70)
        print(f"Total Positions: {len(positions)}")
        print(f"Total Value: {sum(p.market_value for p in positions):,.2f} {CONFIG.portfolio_currency}")
        print("\n--- Key Metrics ---")
        print(f"Annual Return:     {metrics.total_return_ann*100:+.2f}%")
        print(f"Annual Volatility: {metrics.volatility_ann*100:.2f}%")
        print(f"Sharpe Ratio:      {metrics.sharpe_ratio:.3f}")
        print(f"Max Drawdown:      {metrics.max_drawdown*100:.2f}%")
        print(f"Beta vs {CONFIG.primary_benchmark}: {metrics.beta:.3f}")
        print(f"Alpha (annual):    {metrics.alpha_ann*100:+.2f}%")
        print(f"Information Ratio: {metrics.information_ratio:.3f}")
        print("="*70)

        analyzer.update_sheet(positions, metrics)

        bench_ticker = CONFIG.benchmarks[CONFIG.primary_benchmark]
        hist = analyzer.hist_provider.fetch([bench_ticker], period="2y")
        bench_returns = hist[bench_ticker].pct_change().dropna() if not hist.empty else pd.Series(dtype=float)
        analyzer.generate_charts(positions, portfolio_returns, bench_returns)

        output_file = CONFIG.chart_dir / "metrics.json"
        CONFIG.chart_dir.mkdir(exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"\n[INFO] Metrics exported to {output_file}")
        print("\n✅ Analysis complete!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise

if __name__ == "__main__":
    main()

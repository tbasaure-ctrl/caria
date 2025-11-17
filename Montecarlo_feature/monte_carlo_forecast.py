"""
Monte Carlo Stock Portfolio Forecasting with Multi-Factor Risk Analysis
========================================================================

CUSTOMIZATION INSTRUCTIONS:
---------------------------
To modify the portfolio or add new tickers:
1. Update the PORTFOLIO list below (line ~50) with your desired ticker symbols
2. Ensure tickers are valid US stock symbols available on Finnhub
3. Run the script - it will automatically analyze all tickers in the list

The script will:
- Fetch real data from Finnhub for each ticker
- Apply appropriate valuation methods based on company stage
- Run 10,000 Monte Carlo simulations per ticker
- Generate visualizations and reports for each stock
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE VARIABLES
# ============================================================================

# Finnhub API Key
API_KEY = "d1v5qgpr01qo0ln2ma3gd1v5qgpr01qo0ln2ma40"

# Portfolio Configuration - ADD OR REMOVE TICKERS HERE
PORTFOLIO = [
    "UNH", "BABA", "OSCR", "ASML", "ASTS", "UBER", "HIMS", "DLO",
    "JD", "NU", "MELI", "PEP", "BFLY", "ZETA", "BIDU", "KVUE",
    "GNRC", "AXON", "NVO", "ELF", "UUUU", "SMR", "PATH"
]

# Simulation Parameters
NUM_SIMULATIONS = 10000
FORECAST_HORIZON_YEARS = 2
TRADING_DAYS_PER_YEAR = 252
TOTAL_TRADING_DAYS = FORECAST_HORIZON_YEARS * TRADING_DAYS_PER_YEAR
RANDOM_SEED = 42

# API Configuration
BASE_URL = "https://finnhub.io/api/v1"
API_DELAY = 0.15  # Delay between API calls to respect rate limits

# Country Risk Scores (higher = more risk)
COUNTRY_RISK = {
    "US": 1.0,
    "CN": 1.4,  # China - regulatory, geopolitical risks
    "NL": 1.05,  # Netherlands
    "DK": 1.0,   # Denmark
    "BR": 1.3,   # Brazil - emerging market volatility
    "AR": 1.5,   # Argentina - high inflation, currency risk
    "DEFAULT": 1.2
}

# Industry Risk Multipliers
INDUSTRY_RISK = {
    "Healthcare": 1.1,
    "Technology": 1.3,
    "Financial Services": 1.2,
    "Consumer": 1.15,
    "Energy": 1.4,
    "Industrials": 1.2,
    "Biotechnology": 1.5,
    "E-commerce": 1.3,
    "Telecommunications": 1.25,
    "DEFAULT": 1.2
}

# ============================================================================
# API DATA COLLECTION FUNCTIONS
# ============================================================================
try:
    import yfinance as yf
    _HAS_YF = True
except Exception:
    _HAS_YF = False

def api_call(endpoint, params=None):
    """Make API call with error handling and rate limiting."""
    if params is None:
        params = {}
    params['token'] = API_KEY
    try:
        time.sleep(API_DELAY)
        response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Error for {endpoint}: {e}")
        return None

def get_company_profile(ticker):
    """Fetch company profile including country and industry."""
    return api_call("stock/profile2", {"symbol": ticker})

def get_current_quote(ticker):
    """Fetch current price data."""
    return api_call("quote", {"symbol": ticker})

def get_financial_metrics(ticker):
    """Fetch comprehensive financial metrics."""
    return api_call("stock/metric", {"symbol": ticker, "metric": "all"})

def get_historical_data(ticker, days_back=730):
    """Try Finnhub first. If 403/any error, fall back to yfinance."""
    end_date = int(datetime.now().timestamp())
    start_date = int((datetime.now() - timedelta(days=days_back)).timestamp())

    data = api_call("stock/candle", {
        "symbol": ticker, "resolution": "D",
        "from": start_date, "to": end_date
    })

    if data and isinstance(data, dict) and data.get('s') == 'ok' and data.get('c'):
        return pd.DataFrame({
            'close': data['c'],
            'timestamp': pd.to_datetime(data['t'], unit='s')
        })

    # Fallback
    if _HAS_YF:
        try:
            df = yf.Ticker(ticker).history(period=f"{days_back}d", interval="1d")
            if not df.empty and 'Close' in df.columns:
                return pd.DataFrame({
                    'close': df['Close'].values,
                    'timestamp': df.index.tz_localize(None)
                })
        except Exception:
            pass

    # Final fallback
    return None

# ============================================================================
# RISK ANALYSIS FUNCTIONS
# ============================================================================

def calculate_historical_volatility(price_data):
    """Annualized volatility from daily log-returns."""
    if price_data is None or len(price_data) < 40:
        return 0.4  # default
    log_ret = np.diff(np.log(np.asarray(price_data['close'], dtype=float)))
    daily_vol = np.std(log_ret, ddof=1)
    annual_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return float(np.clip(annual_vol, 0.10, 1.80))

def estimate_mu_from_history(price_data):
    """Annualized drift μ from daily log-returns."""
    if price_data is None or len(price_data) < 40:
        return 0.08
    log_ret = np.diff(np.log(np.asarray(price_data['close'], dtype=float)))
    mu_daily = np.mean(log_ret)
    return float(mu_daily * TRADING_DAYS_PER_YEAR)

def assess_country_risk(country):
    """Get country risk multiplier."""
    return COUNTRY_RISK.get(country, COUNTRY_RISK["DEFAULT"])

def assess_industry_risk(industry):
    """Get industry risk multiplier."""
    for key in INDUSTRY_RISK:
        if key.lower() in (industry or "").lower():
            return INDUSTRY_RISK[key]
    return INDUSTRY_RISK["DEFAULT"]

def calculate_financial_health_score(metrics):
    """
    Calculate financial health score from actual metrics.
    Score ranges from 0.5 (weak) to 1.5 (strong).
    """
    if not metrics or 'metric' not in metrics:
        return 1.0  # Neutral score if no data
    metric_data = metrics['metric']
    score = 1.0

    debt_to_equity = metric_data.get('debtEquityAnnual')
    if debt_to_equity is not None:
        if debt_to_equity < 0.5:
            score += 0.2
        elif debt_to_equity > 2.0:
            score -= 0.2

    current_ratio = metric_data.get('currentRatioAnnual')
    if current_ratio is not None:
        if current_ratio > 2.0:
            score += 0.15
        elif current_ratio < 1.0:
            score -= 0.15

    roe = metric_data.get('roeRfy')
    if roe is not None:
        if roe > 0.15:
            score += 0.15
        elif roe < 0:
            score -= 0.2

    profit_margin = metric_data.get('netProfitMarginAnnual')
    if profit_margin is not None:
        if profit_margin > 0.15:
            score += 0.1
        elif profit_margin < 0:
            score -= 0.15

    beta = metric_data.get('beta')
    if beta is not None:
        if beta > 1.5:
            score -= 0.1
        elif beta < 0.8:
            score += 0.05

    return max(0.5, min(score, 1.5))

def determine_business_stage(metrics):
    """
    Determine business stage and appropriate valuation metric.
    Returns: (stage, valuation_ratio, is_profitable)
    """
    if not metrics or 'metric' not in metrics:
        return ("unknown", None, False)

    metric_data = metrics['metric']
    pe_ratio = metric_data.get('peAnnual')
    ps_ratio = metric_data.get('psAnnual')
    revenue = metric_data.get('revenuePerShareAnnual')
    net_margin = metric_data.get('netProfitMarginAnnual')

    if revenue is None or revenue <= 0:
        return ("pre-revenue", None, False)

    if pe_ratio is not None and pe_ratio > 0 and net_margin and net_margin > 0:
        return ("profitable", pe_ratio, True)

    if ps_ratio is not None and ps_ratio > 0:
        return ("growth", ps_ratio, False)

    return ("unknown", None, False)

def calculate_black_swan_probability():
    """Annual probability of rare crash."""
    return 0.08  # 8% per year

# ============================================================================
# MONTE CARLO SIMULATION ENGINE
# ============================================================================

class StockSimulator:
    """Monte Carlo simulator with multi-factor risk modeling."""

    def __init__(self, ticker):
        self.ticker = ticker
        self.current_price = None
        self.volatility = None
        self.drift = None
        self.risk_factors = {}
        self.metrics_used = []

    def collect_data(self):
        """Collect all necessary data from Finnhub."""
        print(f"\n{'='*60}")
        print(f"Collecting data for {self.ticker}...")
        print(f"{'='*60}")

        # Company profile
        profile = get_company_profile(self.ticker)
        if not profile:
            print(f"Warning: Could not fetch profile for {self.ticker}")
            return False

        # Quote
        quote = get_current_quote(self.ticker)
        if not quote or quote.get('c') is None:
            print(f"Error: Could not fetch current price for {self.ticker}")
            return False
        self.current_price = quote['c']
        self.metrics_used.append(f"Current Price: ${self.current_price:.2f}")

        # Historical prices → vol and μ_hist
        historical = get_historical_data(self.ticker)
        self.volatility = calculate_historical_volatility(historical)
        self.metrics_used.append(f"Historical Volatility (Annual): {self.volatility:.2%}")
        mu_hist = estimate_mu_from_history(historical)

        # Financial metrics
        metrics = get_financial_metrics(self.ticker)

        # Country/industry and risk factors (always set keys)
        country = profile.get('country', 'US')
        industry = profile.get('finnhubIndustry', 'Unknown')
        self.risk_factors = {
            'country_risk': assess_country_risk(country or 'US'),
            'industry_risk': assess_industry_risk(industry or 'Unknown'),
            'financial_health': calculate_financial_health_score(metrics)
        }

        self.metrics_used.extend([
            f"Company: {profile.get('name', self.ticker)}",
            f"Country: {country} (Risk Multiplier: {self.risk_factors['country_risk']:.2f})",
            f"Industry: {industry} (Risk Multiplier: {self.risk_factors['industry_risk']:.2f})",
            f"Financial Health Score: {self.risk_factors['financial_health']:.2f}",
            f"Historical μ (log-annual): {mu_hist:.2%}"
        ])

        # Business stage analysis (needs metrics)
        stage, valuation_ratio, is_profitable = determine_business_stage(metrics)
        self.risk_factors['business_stage'] = stage
        self.risk_factors['valuation_ratio'] = valuation_ratio
        self.risk_factors['is_profitable'] = is_profitable
        stage_info = f"Business Stage: {stage.title()}"
        if valuation_ratio:
            ratio_type = "P/E" if is_profitable else "P/S"
            stage_info += f" ({ratio_type} Ratio: {valuation_ratio:.2f})"
        self.metrics_used.append(stage_info)

        # Add a few metrics to the report
        if metrics and 'metric' in metrics:
            m = metrics['metric']
            if m.get('debtEquityAnnual') is not None:
                self.metrics_used.append(f"Debt/Equity: {m['debtEquityAnnual']:.2f}")
            if m.get('currentRatioAnnual') is not None:
                self.metrics_used.append(f"Current Ratio: {m['currentRatioAnnual']:.2f}")
            if m.get('roeRfy') is not None:
                self.metrics_used.append(f"ROE: {m['roeRfy']:.2%}")
            if m.get('netProfitMarginAnnual') is not None:
                self.metrics_used.append(f"Net Profit Margin: {m['netProfitMarginAnnual']:.2%}")
            if m.get('beta') is not None:
                self.metrics_used.append(f"Beta: {m['beta']:.2f}")

        # Build drift from market+risk anchor, then blend with μ_hist
        base_market_return = 0.08
        risk_adjusted = base_market_return
        risk_adjusted *= (2.0 - self.risk_factors['financial_health'])
        risk_adjusted *= (2.0 - self.risk_factors['country_risk'])

        # Stage tweaks
        if stage == "pre-revenue":
            risk_adjusted *= 0.6
            self.volatility *= 1.5
        elif stage == "growth" and not is_profitable:
            risk_adjusted *= 0.85
            self.volatility *= 1.2

        # Keep your later references that expect risk_adjusted_return
        risk_adjusted_return = risk_adjusted

        # Final drift
        self.drift = 0.4 * mu_hist + 0.6 * risk_adjusted_return
        self.metrics_used.append(f"Expected Annual Return (Drift, blended): {self.drift:.2%}")
        self.metrics_used.append(f"Adjusted Volatility: {self.volatility:.2%}")

        print(f"Data collection complete for {self.ticker}")
        return True

    def run_simulation(self):
        """
        Vectorized GBM in log space with optional fat tails & rare jumps.
        S_{t+1} = S_t * exp[(μ - 0.5σ²)Δt + σ√Δt * Z_t + J_t]
        """
        rng = np.random.default_rng(RANDOM_SEED)
        n_sims = NUM_SIMULATIONS
        steps_per_year = TRADING_DAYS_PER_YEAR
        T_years = FORECAST_HORIZON_YEARS
        n_steps = int(steps_per_year * T_years)
        dt = 1.0 / steps_per_year

        vol_multiplier = np.sqrt(
            self.risk_factors['country_risk'] *
            self.risk_factors['industry_risk'] *
            (2.0 - self.risk_factors['financial_health'])
        )
        sigma = float(self.volatility * vol_multiplier)
        mu = float(self.drift)

        # Fat tails
        df = 6
        shocks = rng.standard_t(df, size=(n_steps, n_sims)) / np.sqrt(df/(df-2))

        # Jumps
        jump_lambda = calculate_black_swan_probability()
        jump_mu, jump_sigma = -0.12, 0.20
        N = rng.poisson(jump_lambda * dt, size=(n_steps, n_sims))
        J = N * (jump_mu + jump_sigma * rng.standard_normal((n_steps, n_sims)))

        drift_term = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * shocks
        log_increments = drift_term + diffusion + J

        log_paths = np.vstack([np.zeros((1, n_sims)), np.cumsum(log_increments, axis=0)])
        price_paths = self.current_price * np.exp(log_paths)

        # Soft floor
        floor = self.current_price * 0.01
        price_paths = np.maximum(price_paths, floor)

        print(f"\nRan {NUM_SIMULATIONS:,} Monte Carlo simulations (vectorized).")
        return price_paths.T  # shape: [sims, steps+1]

    def analyze_results(self, price_paths):
        final_prices = price_paths[:, -1]
        returns = final_prices / self.current_price - 1.0

        p10, p50, p90 = np.quantile(final_prices, [0.10, 0.50, 0.90])
        exp_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1))
        var_95 = float(np.quantile(returns, 0.05))
        es_95 = float(returns[returns <= np.quantile(returns, 0.05)].mean())

        targets = [0.10, 0.20, 0.50, 1.00]
        p_ge = {t: float(np.mean(returns >= t)) for t in targets}

        probs = {
            'prob_loss_20': float(np.mean(returns <= -0.20)) * 100.0,
            'prob_loss_50': float(np.mean(returns <= -0.50)) * 100.0,
            'prob_gain_20': float(np.mean(returns >= 0.20)) * 100.0,
            'prob_gain_50': float(np.mean(returns >= 0.50)) * 100.0,
            'prob_gain_100': float(np.mean(returns >= 1.00)) * 100.0,
        }

        results = {
            'final_prices': final_prices,
            'returns': returns * 100.0,  # percent for your printouts
            'p10': p10, 'p50': p50, 'p90': p90,
            'expected_return': exp_ret * 100.0,
            'std_return': std_ret * 100.0,
            'var_95': var_95 * 100.0,
            'es_95': es_95 * 100.0,
            **probs,
            'p_ge_targets': p_ge
        }
        return results

    def plot_results(self, results, price_paths=None):
        fig = plt.figure(figsize=(15, 12))

        # (A) Terminal price histogram
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.hist(results['final_prices'], bins=100, alpha=0.7, edgecolor='black', linewidth=0.3)
        ax1.axvline(results['p10'], color='red', linestyle='--', linewidth=2, label=f"P10: ${results['p10']:.2f}")
        ax1.axvline(results['p50'], color='green', linestyle='--', linewidth=2, label=f"P50: ${results['p50']:.2f}")
        ax1.axvline(results['p90'], color='blue', linestyle='--', linewidth=2, label=f"P90: ${results['p90']:.2f}")
        ax1.axvline(self.current_price, color='black', linestyle='-', linewidth=2, label=f"Current: ${self.current_price:.2f}")
        box = (f"Exp. Return: {results['expected_return']:.1f}%\n"
               f"StdDev: {results['std_return']:.1f}%\n"
               f"VaR5%: {results['var_95']:.1f}%  |  ES5%: {results['es_95']:.1f}%")
        ax1.text(0.02, 0.98, box, transform=ax1.transAxes, va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax1.set_title(f"{self.ticker} – {FORECAST_HORIZON_YEARS}-Year Monte Carlo (N={NUM_SIMULATIONS:,})")
        ax1.set_xlabel("Terminal Price (USD)")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(alpha=0.3, linestyle='--')

        # (B) CDF of terminal returns
        ax2 = fig.add_subplot(3, 1, 2)
        r_sorted = np.sort(results['returns'])
        cdf = np.linspace(0, 1, len(r_sorted), endpoint=True)
        ax2.plot(r_sorted, cdf, linewidth=2)
        ax2.axvline(0, color='black', linestyle='--', linewidth=1)
        for t in [10, 20, 50, 100]:
            p_ge = 1.0 - np.interp(t, r_sorted, cdf)
            ax2.axvline(t, linestyle=':', linewidth=1, label=f"P(Return ≥ {t}%) ~ {p_ge*100:.1f}%")
        ax2.set_title("CDF of Terminal Returns")
        ax2.set_xlabel("Return (%)")
        ax2.set_ylabel("Cumulative Probability")
        ax2.legend()
        ax2.grid(alpha=0.3, linestyle='--')

        # (C) Fan chart
        if price_paths is not None:
            ax3 = fig.add_subplot(3, 1, 3)
            Q = np.quantile(price_paths, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
            t = np.arange(Q.shape[1])
            ax3.fill_between(t, Q[0], Q[4], alpha=0.2, label='P5–P95')
            ax3.fill_between(t, Q[1], Q[3], alpha=0.4, label='P25–P75')
            ax3.plot(t, Q[2], linewidth=2, label='Median')
            ax3.axhline(self.current_price, color='black', linestyle='--', linewidth=1)
            ax3.set_title("Price Fan Chart")
            ax3.set_xlabel("Trading Days")
            ax3.set_ylabel("Price (USD)")
            ax3.legend()
            ax3.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        fname = f"{self.ticker}_monte_carlo_forecast.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as: {fname}")
        plt.show()

    def generate_report(self, results):
        """Generate comprehensive text report."""
        print(f"\n{'='*70}")
        print(f"MONTE CARLO SIMULATION REPORT: {self.ticker}")
        print(f"{'='*70}")

        print(f"\nSIMULATION PARAMETERS:")
        print(f"  Number of Simulations: {NUM_SIMULATIONS:,}")
        print(f"  Forecast Horizon: {FORECAST_HORIZON_YEARS} years ({TOTAL_TRADING_DAYS} trading days)")
        print(f"  Current Price: ${self.current_price:.2f}")

        print(f"\nDATA SOURCES & METRICS USED:")
        for metric in self.metrics_used:
            print(f"  • {metric}")

        print(f"\nRISK FACTORS INCORPORATED:")
        print(f"  1. Macroeconomic Risk (Country): {self.risk_factors['country_risk']:.2f}x multiplier")
        print(f"  2. Industry Risk: {self.risk_factors['industry_risk']:.2f}x multiplier")
        print(f"  3. Financial Health Score: {self.risk_factors['financial_health']:.2f}")
        print(f"  4. Black Swan Events: Modeled with {calculate_black_swan_probability():.1%} annual probability")
        print(f"  5. Business Stage: {self.risk_factors['business_stage'].title()}")

        print(f"\nFORECAST RESULTS (2-Year Horizon):")
        print(f"  Percentile Analysis:")
        print(f"    P10 (Pessimistic): ${results['p10']:.2f} ({(results['p10']/self.current_price - 1)*100:+.1f}%)")
        print(f"    P50 (Median):      ${results['p50']:.2f} ({(results['p50']/self.current_price - 1)*100:+.1f}%)")
        print(f"    P90 (Optimistic):  ${results['p90']:.2f} ({(results['p90']/self.current_price - 1)*100:+.1f}%)")

        print(f"\nPROBABILITY ANALYSIS:")
        print(f"  Probability of Loss > 20%:  {results['prob_loss_20']:.1f}%")
        print(f"  Probability of Loss > 50%:  {results['prob_loss_50']:.1f}%")
        print(f"  Probability of Gain > 20%:  {results['prob_gain_20']:.1f}%")
        print(f"  Probability of Gain > 50%:  {results['prob_gain_50']:.1f}%")
        print(f"  Probability of Gain > 100%: {results['prob_gain_100']:.1f}%")

        print(f"\nSTATISTICAL SUMMARY:")
        print(f"  Expected Return: {results['expected_return']:.2f}%")
        print(f"  Standard Deviation: {results['std_return']:.2f}%")
        print(f"  Value at Risk (VaR) at 95% confidence: {results['var_95']:.2f}%")
        print(f"    (There is a 5% chance of losing more than {abs(results['var_95']):.1f}%)")

        print(f"\n{'='*70}\n")

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def analyze_ticker(ticker):
    """
    Main function to analyze a single ticker.
    """
    try:
        simulator = StockSimulator(ticker)

        if not simulator.collect_data():
            print(f"Failed to collect data for {ticker}. Skipping...")
            return False

        price_paths = simulator.run_simulation()
        results = simulator.analyze_results(price_paths)

        simulator.generate_report(results)
        simulator.plot_results(results, price_paths)
        return True

    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Analyze all tickers in the portfolio."""
    print("="*70)
    print("MONTE CARLO PORTFOLIO FORECASTING SYSTEM")
    print("="*70)
    print(f"\nAnalyzing {len(PORTFOLIO)} stocks from portfolio...")
    print(f"Simulation parameters: {NUM_SIMULATIONS:,} paths, {FORECAST_HORIZON_YEARS}-year horizon")
    print("\nPress Ctrl+C to stop the analysis at any time.\n")

    successful, failed = [], []

    for i, ticker in enumerate(PORTFOLIO, 1):
        print(f"\n[{i}/{len(PORTFOLIO)}] Processing {ticker}...")
        try:
            if analyze_ticker(ticker):
                successful.append(ticker)
                print(f"✓ {ticker} analysis complete")
            else:
                failed.append(ticker)
                print(f"✗ {ticker} analysis failed")
        except KeyboardInterrupt:
            print("\n\nAnalysis interrupted by user.")
            break
        except Exception as e:
            print(f"✗ {ticker} encountered an error: {e}")
            failed.append(ticker)

        if i < len(PORTFOLIO):
            time.sleep(1)

    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print(f"Successfully analyzed: {len(successful)} stocks")
    if successful:
        print(f"  {', '.join(successful)}")
    if failed:
        print(f"\nFailed to analyze: {len(failed)} stocks")
        print(f"  {', '.join(failed)}")
    print("\nAll visualizations have been saved as PNG files in the current directory.")
    print("="*70)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    TO USE THIS SCRIPT:

    1. Ensure you have the required libraries installed:
       pip install numpy pandas matplotlib requests scipy

    2. Modify the PORTFOLIO list at the top of the script (line ~50) with your desired tickers

    3. Run the script:
       python monte_carlo_forecast.py

    4. The script will:
       - Fetch real-time data from Finnhub for each ticker
       - Run 10,000 Monte Carlo simulations per ticker
       - Generate a histogram visualization (saved as PNG)
       - Print a comprehensive analysis report

    5. To analyze a single ticker instead of the whole portfolio:
       Uncomment the following line and replace 'TICKER' with your symbol:
    """
    # analyze_ticker('AAPL')
    main()

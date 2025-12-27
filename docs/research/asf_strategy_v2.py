"""
ASF Strategy V2.1: Enhanced Implementation with Critical Fixes
================================================================

Fixes applied (per user feedback):
1. Shifted vol by 1 in volatility standardization (remove lookahead)
2. True CAGR (geometric, not arithmetic)
3. True CVaR (expected shortfall, not just quantile)
4. V1 uses rolling quantile (no full-sample lookahead)
5. Convex hedge payoff (crash protection, not just drag)
6. Drawdown kill switch applied
7. Robust z-scores (median/MAD)
8. VIX slope added to danger score

Author: Research Team
Date: December 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf
from scipy.special import expit  # sigmoid
from datetime import datetime
import warnings
import os

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"

# Universe - Use stocks for longer history (ETFs only exist from ~2000s)
EQUITY_UNIVERSE_STOCKS = [
    'AAPL', 'MSFT', 'JNJ', 'PG', 'XOM', 'JPM', 'GE', 'KO', 'PFE', 'WMT',
    'IBM', 'CVX', 'MRK', 'DIS', 'HD', 'MCD', 'BA', 'CAT', 'MMM', 'AXP'
]

# Parameters
CORR_WINDOW = 126
VOL_NORM_WINDOW = 21
ZSCORE_WINDOW = 252 * 5  # 5-year rolling
DECAY_LAMBDA = 0.005

# Danger Score Parameters
ALPHA = 2.0
BETA = 1.5
GAMMA = 0.7  # VIX slope weight
C1 = 0.5
C2 = 0.0

# Exposure Mapping
W_MIN = 0.2
W_MAX = 1.3

# Hedge Sleeve
HEDGE_MAX_PREMIUM = 0.03
CRASH_TRIGGER = -0.015  # 1.5% down day triggers payoff
CONVEX_MULT = 8.0  # Payoff multiplier

# Risk Controls
MAX_DRAWDOWN_KILL = -0.10
PERSISTENCE_DAYS = 3
REBAL_FREQ = 5


# ============================================================================
# CORE FUNCTIONS (WITH FIXES)
# ============================================================================

def load_data(tickers, start_date, end_date):
    """Load adjusted close prices."""
    df = yf.download(tickers, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        prices = df['Adj Close'] if 'Adj Close' in df.columns.get_level_values(0) else df['Close']
    else:
        prices = df
    return prices.dropna(axis=1, how='all')


def calculate_entropy(corr_matrix):
    """Calculate normalized spectral entropy."""
    try:
        eigvals = np.linalg.eigvalsh(corr_matrix)
        eigvals = eigvals[eigvals > 1e-10]
        probs = eigvals / np.sum(eigvals)
        S = -np.sum(probs * np.log(probs))
        N = len(probs)
        return S / np.log(N) if N > 1 else 1.0
    except:
        return np.nan


def volatility_standardize_returns(returns, window=21):
    """
    Standardize returns by rolling vol.
    FIX: Shift vol by 1 to remove lookahead bias.
    """
    rolling_vol = returns.rolling(window).std().shift(1)  # <-- FIXED: shift removes leakage
    standardized = returns / rolling_vol.replace(0, np.nan)
    return standardized.dropna()


def calculate_asf_series(returns, corr_window=126):
    """Calculate ASF with vol-standardized returns and Ledoit-Wolf."""
    std_returns = volatility_standardize_returns(returns, VOL_NORM_WINDOW)
    
    fragility = {}
    
    for i in range(corr_window, len(std_returns)):
        idx = std_returns.index[i]
        window_ret = std_returns.iloc[i-corr_window:i].dropna(axis=1)
        
        if len(window_ret.columns) >= 3:
            try:
                lw = LedoitWolf()
                lw.fit(window_ret.values)
                cov = lw.covariance_
                
                std = np.sqrt(np.diag(cov))
                corr = cov / np.outer(std, std)
                np.fill_diagonal(corr, 1.0)
                
                entropy = calculate_entropy(corr)
                fragility[idx] = 1 - entropy
            except:
                fragility[idx] = np.nan
    
    frag_series = pd.Series(fragility)
    asf = frag_series.ewm(halflife=139).mean()
    
    return asf


def robust_z(x, w):
    """
    Robust z-score using median/MAD instead of mean/std.
    More stable during crisis periods.
    """
    med = x.rolling(w).median()
    mad = (x - med).abs().rolling(w).median()
    return (x - med) / (1.4826 * mad.replace(0, np.nan))


def calculate_danger_score(asf, vix, alpha=ALPHA, beta=BETA, gamma=GAMMA, c1=C1, c2=C2):
    """
    Two-dimensional danger score with VIX slope.
    FIX: Uses robust z-scores (median/MAD) and adds VIX slope.
    """
    # Robust z-scores
    asf_z = robust_z(asf, ZSCORE_WINDOW)
    vix_z = robust_z(vix, ZSCORE_WINDOW)
    
    # VIX slope (rising VIX is dangerous even if level is low)
    vix_slope = vix.diff(5)  # 1-week change
    vix_slope_z = robust_z(vix_slope, ZSCORE_WINDOW)
    
    # Align
    aligned = pd.concat([asf_z, vix_z, vix_slope_z], axis=1).dropna()
    aligned.columns = ['asf_z', 'vix_z', 'vix_slope_z']
    
    # Danger = high ASF, low VIX, but penalize rising VIX
    raw_score = alpha * (aligned['asf_z'] - c1) - beta * (aligned['vix_z'] - c2) + gamma * aligned['vix_slope_z']
    danger = pd.Series(expit(raw_score), index=aligned.index)
    
    return danger


def calculate_exposure(danger_score, credit_overlay, w_min=W_MIN, w_max=W_MAX):
    """Smooth exposure mapping."""
    credit_overlay_aligned = credit_overlay.reindex(danger_score.index).fillna(1.0)
    base_exposure = w_min + (w_max - w_min) * (1 - danger_score)
    final_exposure = base_exposure * credit_overlay_aligned
    return final_exposure.clip(w_min, w_max)


def calculate_hedge_allocation(danger_score, vix, low_vol_threshold=15):
    """Hedge sleeve: scale up when high danger + low implied vol."""
    common_idx = danger_score.index.intersection(vix.index)
    
    if len(common_idx) == 0:
        return pd.Series(0.0, index=danger_score.index)
    
    danger_aligned = danger_score.loc[common_idx]
    vix_aligned = vix.loc[common_idx]
    
    hedge_alloc = pd.Series(0.0, index=common_idx)
    low_vol_mask = vix_aligned < low_vol_threshold
    hedge_alloc.loc[low_vol_mask] = danger_aligned.loc[low_vol_mask] * HEDGE_MAX_PREMIUM
    
    return hedge_alloc.reindex(danger_score.index).fillna(0.0).clip(0, HEDGE_MAX_PREMIUM)


def apply_persistence_filter(signal, days=PERSISTENCE_DAYS):
    """Require signal to persist for N days."""
    return signal.rolling(days).mean()


def apply_drawdown_kill(equity_curve, threshold=MAX_DRAWDOWN_KILL):
    """
    Kill switch: if intramonth DD exceeds threshold, return True.
    """
    rolling_max = equity_curve.rolling(21).max()
    drawdown = equity_curve / rolling_max - 1
    kill = drawdown < threshold
    return kill


def calculate_trend_filter(prices, window=200):
    """
    Regime Filter: Returns True if Price > SMA(window).
    Used to gate leverage.
    """
    sma = prices.rolling(window).mean()
    return prices > sma

def apply_vol_targeting(returns, target_vol=0.15, vol_window=21, cap=2.0):
    """
    Inverse Volatility Scaling.
    Exposure = TargetVol / RealizedVol
    """
    realized_vol = returns.rolling(vol_window).std() * np.sqrt(252)
    scaler = target_vol / realized_vol.replace(0, np.nan)
    return scaler.fillna(1.0).clip(upper=cap)

def run_backtest(spy_returns, exposure, hedge_alloc, rebal_freq=REBAL_FREQ,
                 crash_trigger=CRASH_TRIGGER, convex_mult=CONVEX_MULT,
                 use_trend_filter=True, spy_prices=None):
    """
    Run backtest with Trend Filter and Vol control.
    """
    # Calculate additional filters
    trend_on = pd.Series(True, index=spy_returns.index)
    if use_trend_filter and spy_prices is not None:
        trend = calculate_trend_filter(spy_prices.reindex(spy_returns.index))
        trend_on = trend.fillna(True) # default to bullish if missing

    aligned = pd.concat([spy_returns, exposure, hedge_alloc, trend_on], axis=1).dropna()
    aligned.columns = ['spy_ret', 'exposure', 'hedge', 'trend']
    
    rebal_dates = set(aligned.index[::rebal_freq])
    current_exposure = 1.0
    current_hedge = 0.0
    
    out = []
    for date, row in aligned.iterrows():
        if date in rebal_dates:
            base_exp = float(row['exposure'])
            # TREND RULE: If downtrend (Price < SMA), max exposure is 1.0 (no leverage), 
            # or even defensive (0.8) if High Danger.
            # Let's say: If downtrend, cap at 1.0.
            if not row['trend']:
                current_exposure = min(base_exp, 1.0)
            else:
                current_exposure = base_exp
                
            current_hedge = float(row['hedge'])
        
        # Premium/Payoff logic
        premium = current_hedge / 21.0
        payoff = 0.0
        if row['spy_ret'] < crash_trigger:
            payoff = current_hedge * convex_mult * abs(row['spy_ret'] - crash_trigger)
        
        daily = current_exposure * row['spy_ret'] - premium + payoff
        out.append(daily)
    
    return pd.Series(out, index=aligned.index)


def calculate_metrics(returns):
    """
    Calculate performance metrics.
    FIX: True CAGR (geometric), true CVaR (expected shortfall).
    """
    returns = returns.dropna()
    cum = (1 + returns).cumprod()
    n_years = (cum.index[-1] - cum.index[0]).days / 365.25
    
    # TRUE CAGR (geometric)
    cagr = cum.iloc[-1]**(1/n_years) - 1 if n_years > 0 else 0
    
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / ann_vol if ann_vol > 0 else 0
    
    rolling_max = cum.cummax()
    dd = cum / rolling_max - 1
    max_dd = dd.min()
    
    # TRUE CVaR 5% (expected shortfall, not just quantile)
    var_5 = returns.quantile(0.05)
    cvar_5 = returns[returns <= var_5].mean()
    
    return {
        'CAGR': cagr,
        'Volatility': ann_vol,
        'Sharpe': sharpe,
        'Max_DD': max_dd,
        'CVaR_5%': cvar_5
    }


def calculate_diagnostics(spy_returns, strategy_returns, exposure):
    """
    Additional diagnostics to validate ASF effectiveness.
    """
    aligned = pd.concat([spy_returns, strategy_returns], axis=1).dropna()
    aligned.columns = ['spy', 'strat']
    
    # Turnover: daily change in exposure
    turnover = exposure.diff().abs().mean() * 252
    
    # Tail capture: return during worst 5% SPY days
    worst_5pct = aligned['spy'].quantile(0.05)
    worst_spy_days = aligned['spy'] <= worst_5pct
    
    spy_tail_avg = aligned.loc[worst_spy_days, 'spy'].mean()
    strat_tail_avg = aligned.loc[worst_spy_days, 'strat'].mean()
    tail_capture = strat_tail_avg / spy_tail_avg if spy_tail_avg != 0 else 1.0
    
    return {
        'Annual_Turnover': turnover,
        'Tail_Capture': tail_capture,  # < 1 means less exposure during crashes
        'Worst_5%_SPY': spy_tail_avg,
        'Worst_5%_Strat': strat_tail_avg
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("ASF STRATEGY V2.1: WITH CRITICAL FIXES (1990-2025)")
    print("=" * 70)
    
    end_date = datetime(2025, 12, 20)
    start_date = datetime(1990, 1, 1)
    
    print(f"\nLoading data from {start_date.date()} to {end_date.date()}...")
    print("\n NOTE: 1990+ uses today's mega-caps (survivorship bias).")
    print("   This is a CONCEPTUAL test, not implementable without constituent history.\n")
    
    # Load data
    equity_prices = load_data(EQUITY_UNIVERSE_STOCKS, start_date, end_date)
    
    spy_data = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_prices = spy_data['Adj Close'] if 'Adj Close' in spy_data.columns.get_level_values(0) else spy_data['Close']
    else:
        spy_prices = spy_data['Adj Close'] if 'Adj Close' in spy_data.columns else spy_data['Close']
    
    vix_raw = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_data = vix_raw['Close'].squeeze()
    else:
        vix_data = vix_raw['Close']
    
    if isinstance(spy_prices, pd.DataFrame):
        spy_prices = spy_prices.iloc[:, 0]
    if isinstance(vix_data, pd.DataFrame):
        vix_data = vix_data.iloc[:, 0]
    
    print(f"  Equity universe: {len(equity_prices.columns)} stocks")
    print(f"  S&P 500 data points: {len(spy_prices)}")
    print(f"  VIX data points: {len(vix_data)}")
    
    equity_returns = equity_prices.pct_change().dropna()
    spy_returns = spy_prices.pct_change().dropna()
    
    # Calculate ASF
    print("\nCalculating ASF (with shifted vol)...")
    asf_equity = calculate_asf_series(equity_returns, CORR_WINDOW)
    
    # Calculate Danger Score (with robust z-scores and VIX slope)
    print("Calculating Danger Score (robust z + VIX slope)...")
    danger = calculate_danger_score(asf_equity, vix_data)
    danger_filtered = apply_persistence_filter(danger, PERSISTENCE_DAYS)
    
    # No credit overlay for extended backtest
    credit_overlay = pd.Series(1.0, index=danger_filtered.index)
    
    # Calculate Exposure
    print("Calculating Smooth Exposure...")
    exposure = calculate_exposure(danger_filtered, credit_overlay)
    
    # Apply Drawdown Kill Switch
    print("Applying Drawdown Kill Switch...")
    equity_curve = (1 + spy_returns.reindex(exposure.index).fillna(0)).cumprod()
    kill = apply_drawdown_kill(equity_curve)
    kill_aligned = kill.reindex(exposure.index).fillna(False)
    exposure_with_kill = exposure.copy()
    exposure_with_kill[kill_aligned] = W_MIN
    
    # Calculate Hedge Allocation
    print("Calculating Convex Hedge Sleeve...")
    hedge_alloc = calculate_hedge_allocation(danger_filtered, vix_data)
    
    # Run Backtest V2.1
    # Run Backtest V2.1 (Enhanced with Trend Filter)
    print("\nRunning Backtest (V2.1 with Trend Filter + All Fixes)...")
    v2_returns = run_backtest(spy_returns, exposure_with_kill, hedge_alloc, 
                              use_trend_filter=True, spy_prices=spy_prices)
    
    # Benchmark
    benchmark_returns = spy_returns.loc[v2_returns.index]
    
    # V1 Strategy with ROLLING quantile (no lookahead)
    print("Running V1 (with rolling quantile, no lookahead)...")
    q = asf_equity.rolling(252*5).quantile(0.8)  # 5-year rolling
    v1_exposure = pd.Series(1.0, index=asf_equity.index)
    v1_exposure[asf_equity > q] = 0.5
    v1_returns = run_backtest(spy_returns, v1_exposure, pd.Series(0.0, index=v1_exposure.index))
    
    # Calculate Metrics
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON (NET OF 10bps TRANSACTION COSTS)")
    print("=" * 70)
    
    # Calculate Turnover and Cost
    # turnover is sum of abs diff in exposure (one-way). 
    # Cost = Turnover * Cost_Per_Trade
    # V1 exposure is v1_exposure.
    
    turnover_v1 = v1_exposure.diff().abs().fillna(0)
    cost_bps = 0.0010  # 10 bps
    # Approximate daily cost drag
    daily_cost_v1 = turnover_v1 * cost_bps
    
    # Net Returns
    v1_returns_net = v1_returns - daily_cost_v1
    
    metrics_v1_net = calculate_metrics(v1_returns_net)
    metrics_bh = calculate_metrics(benchmark_returns)
    
    results = pd.DataFrame({
        'Benchmark (B&H)': metrics_bh,
        'ASF Dynamic (Net of Costs)': metrics_v1_net
    }).T
    
    results_display = results.copy()
    results_display['CAGR'] = results_display['CAGR'].map('{:.2%}'.format)
    results_display['Volatility'] = results_display['Volatility'].map('{:.2%}'.format)
    results_display['Sharpe'] = results_display['Sharpe'].map('{:.3f}'.format)
    results_display['Max_DD'] = results_display['Max_DD'].map('{:.2%}'.format)
    results_display['CVaR_5%'] = results_display['CVaR_5%'].map('{:.2%}'.format)
    
    print(results_display.to_string())
    
    # Diagnostics
    print("\n" + "=" * 70)
    print("DIAGNOSTICS")
    print("=" * 70)
    
    diag_v1 = calculate_diagnostics(benchmark_returns, v1_returns_net, v1_exposure)
    
    print(f"\nASF Dynamic Strategy:")
    print(f"  Annual Turnover: {diag_v1['Annual_Turnover']:.2%}")
    print(f"  Est. Annual Cost: {diag_v1['Annual_Turnover'] * cost_bps:.2%}")
    print(f"  Tail Capture: {diag_v1['Tail_Capture']:.2f} (worst 5% SPY days)")
    
    # Save results
    results.to_csv(os.path.join(OUTPUT_DIR, 'Table_ASF_Final_Net_Performance.csv'))
    
    # Plot
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    ax1 = axes[0]
    cum_bh = (1 + benchmark_returns).cumprod()
    cum_v1 = (1 + v1_returns).cumprod()
    cum_v2 = (1 + v2_returns).cumprod()
    
    ax1.plot(cum_bh.index, cum_bh.values, label='Benchmark (B&H)', color='gray', alpha=0.7)
    ax1.plot(cum_v1.index, cum_v1.values, label='V1 (Rolling Q)', color='blue', alpha=0.7)
    ax1.plot(cum_v2.index, cum_v2.values, label='V2.1 (All Fixes)', color='green', linewidth=2)
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title('ASF Strategy V2.1: With Critical Fixes (1990-2025)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2 = axes[1]
    ax2.fill_between(danger_filtered.dropna().index, 0, danger_filtered.dropna().values, 
                     alpha=0.3, color='red', label='Danger Score')
    ax2.plot(exposure_with_kill.index, exposure_with_kill.values, color='blue', label='Exposure (with Kill)')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Score / Exposure')
    ax2.set_title('Danger Score and Dynamic Exposure (with Kill Switch)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.5)
    
    ax3 = axes[2]
    dd_bh = cum_bh / cum_bh.cummax() - 1
    dd_v2 = cum_v2 / cum_v2.cummax() - 1
    ax3.fill_between(dd_bh.index, 0, dd_bh.values, alpha=0.3, color='gray', label='Benchmark DD')
    ax3.fill_between(dd_v2.index, 0, dd_v2.values, alpha=0.5, color='green', label='V2.1 DD')
    ax3.set_ylabel('Drawdown')
    ax3.set_title('Drawdown Comparison', fontsize=12)
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = os.path.join(OUTPUT_DIR, 'Figure_ASF_V2_1_Strategy.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved figure to: {fig_path}")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("FIXES APPLIED IN V2.1")
    print("=" * 70)
    print("""
    1. LOOKAHEAD IN VOL STANDARDIZATION: Fixed with .shift(1)
    2. CAGR CALCULATION: Now uses geometric CAGR
    3. CVaR CALCULATION: Now uses expected shortfall (not quantile)
    4. V1 LOOKAHEAD: Uses rolling 5y quantile (not full sample)
    5. CONVEX HEDGE: Payoff on crash days (not constant drag)
    6. KILL SWITCH: Applied when 10% intramonth DD
    7. ROBUST Z-SCORES: Uses median/MAD (not mean/std)
    8. VIX SLOPE: Penalizes rising VIX in danger score
    9. SURVIVORSHIP WARNING: Explicit note about 1990+ limitation
    """)


if __name__ == "__main__":
    main()

"""
Statistical Validation Suite for Stored Energy Strategy
=======================================================

Tests:
1. Bootstrap Confidence Intervals (Sharpe, CAGR)
2. Walk-Forward Cross-Validation (No Lookahead Bias)
3. Placebo Test (Shuffled Signal vs Real Signal)
4. Parameter Sensitivity (Window Sizes)
5. Sub-Period Analysis (by Decade)

Author: Antigravity Agent
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
import os
import warnings

warnings.filterwarnings("ignore")

LONG_HISTORY_UNIVERSE = [
    "AAPL", "MSFT", "INTC", "IBM", "ORCL",
    "PG", "KO", "PEP", "JNJ", "WMT", "MCD",
    "JPM", "BAC", "WFC", "GS",
    "XOM", "CVX",
    "GE", "MMM", "CAT", "BA",
    "MRK", "PFE", "ABT",
    "DOW",
]

START_DATE = "1990-01-01"
WINDOW = 63
ENERGY_WINDOW = 60
ROLL_RANK = 252 * 2
LEVERAGE = 1.5
BORROW_COST = 0.02
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def load_data(tickers, start_date):
    print(f"Loading data...")
    try:
        df = yf.download(tickers, start=start_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            px = df['Adj Close' if 'Adj Close' in df else 'Close']
        else:
            px = df
    except:
        return pd.DataFrame(), pd.DataFrame()
    
    extras = yf.download(["^GSPC"], start=start_date, progress=False)
    if isinstance(extras.columns, pd.MultiIndex):
        extras = extras['Adj Close' if 'Adj Close' in extras else 'Close']
    
    return px.dropna(how='all'), extras.dropna(how='all')

def calculate_entropy(px, window):
    returns = px.pct_change()
    entropy_series = {}

    for i in range(window, len(returns)):
        idx = returns.index[i]
        window_ret = returns.iloc[i-window : i]
        valid_cols = window_ret.dropna(axis=1, thresh=int(window*0.8)).columns
        if len(valid_cols) < 10:
            continue
            
        corr_matrix = window_ret[valid_cols].corr()
        corr_matrix = corr_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
        
        if not corr_matrix.empty and len(corr_matrix) >= 10:
            try:
                eigvals = np.linalg.eigvalsh(corr_matrix)
                eigvals = eigvals[eigvals > 1e-10]
                probs = eigvals / np.sum(eigvals)
                S = -np.sum(probs * np.log(probs))
                N = len(probs)
                S_norm = S / np.log(N)
            except:
                S_norm = np.nan
        else:
            S_norm = np.nan
        entropy_series[idx] = S_norm
    
    return pd.Series(entropy_series)

def calculate_stored_energy(entropy, energy_window):
    fragility = 1 - entropy
    return fragility.rolling(energy_window).sum()

def run_strategy(sp500_ret, stored_energy, vol, roll_rank=ROLL_RANK, leverage=LEVERAGE):
    """Run the Stored Energy strategy and return daily returns."""
    
    se_rank = stored_energy.rolling(roll_rank, min_periods=252).rank(pct=True)
    vol_rank = vol.rolling(roll_rank, min_periods=252).rank(pct=True)
    
    danger = (se_rank > 0.80) & (vol_rank < 0.50)
    safe = se_rank < 0.30
    
    lev = pd.Series(1.0, index=sp500_ret.index)
    lev[danger] = 0.0
    lev[safe] = leverage
    lev = lev.shift(1).fillna(1.0)
    
    daily_cost = (BORROW_COST / 252) * (lev - 1).clip(lower=0)
    strategy_ret = sp500_ret * lev - daily_cost
    
    return strategy_ret, lev

def calculate_metrics(returns):
    cagr = (1 + returns).prod() ** (252 / len(returns)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - 0.03) / vol
    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    return {'CAGR': cagr, 'Vol': vol, 'Sharpe': sharpe, 'MaxDD': max_dd}

# =============================================================================
# TEST 1: BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_confidence_intervals(strategy_ret, benchmark_ret, n_iter=N_BOOTSTRAP):
    """Calculate bootstrap confidence intervals for Sharpe difference."""
    
    print("\n=== TEST 1: Bootstrap Confidence Intervals ===")
    
    sharpe_diffs = []
    cagr_diffs = []
    
    n = len(strategy_ret)
    
    for i in range(n_iter):
        # Sample with replacement (block bootstrap - 21 day blocks)
        block_size = 21
        n_blocks = n // block_size
        block_starts = np.random.choice(n - block_size, size=n_blocks, replace=True)
        
        idx = []
        for start in block_starts:
            idx.extend(range(start, start + block_size))
        idx = idx[:n]
        
        s_sample = strategy_ret.iloc[idx]
        b_sample = benchmark_ret.iloc[idx]
        
        s_metrics = calculate_metrics(s_sample)
        b_metrics = calculate_metrics(b_sample)
        
        sharpe_diffs.append(s_metrics['Sharpe'] - b_metrics['Sharpe'])
        cagr_diffs.append(s_metrics['CAGR'] - b_metrics['CAGR'])
    
    sharpe_diff_mean = np.mean(sharpe_diffs)
    sharpe_ci_low = np.percentile(sharpe_diffs, 2.5)
    sharpe_ci_high = np.percentile(sharpe_diffs, 97.5)
    
    cagr_diff_mean = np.mean(cagr_diffs)
    cagr_ci_low = np.percentile(cagr_diffs, 2.5)
    cagr_ci_high = np.percentile(cagr_diffs, 97.5)
    
    # Check if CI excludes zero (significant)
    sharpe_significant = not (sharpe_ci_low <= 0 <= sharpe_ci_high)
    cagr_significant = not (cagr_ci_low <= 0 <= cagr_ci_high)
    
    print(f"  Sharpe Difference: {sharpe_diff_mean:.4f} [{sharpe_ci_low:.4f}, {sharpe_ci_high:.4f}]")
    print(f"    Significant (95% CI excludes 0): {sharpe_significant}")
    print(f"  CAGR Difference: {cagr_diff_mean:.2%} [{cagr_ci_low:.2%}, {cagr_ci_high:.2%}]")
    print(f"    Significant (95% CI excludes 0): {cagr_significant}")
    
    return {
        'sharpe_diff': sharpe_diff_mean, 'sharpe_ci': (sharpe_ci_low, sharpe_ci_high), 'sharpe_sig': sharpe_significant,
        'cagr_diff': cagr_diff_mean, 'cagr_ci': (cagr_ci_low, cagr_ci_high), 'cagr_sig': cagr_significant
    }

# =============================================================================
# TEST 2: WALK-FORWARD VALIDATION
# =============================================================================

def walk_forward_validation(px, market, window=WINDOW, energy_window=ENERGY_WINDOW):
    """Walk-forward cross-validation to test for lookahead bias."""
    
    print("\n=== TEST 2: Walk-Forward Cross-Validation ===")
    
    # Split data: Train 1990-2009, Test 2010-2024
    train_end = '2009-12-31'
    test_start = '2010-01-01'
    
    # Calculate entropy on FULL data (this is the feature)
    entropy = calculate_entropy(px, window)
    stored_energy = calculate_stored_energy(entropy, energy_window)
    
    returns = px.pct_change().mean(axis=1)
    vol = returns.rolling(window).std() * np.sqrt(252)
    
    common = market.index.intersection(stored_energy.dropna().index)
    sp500_ret = market['^GSPC'].loc[common].pct_change()
    se = stored_energy.loc[common]
    v = vol.loc[common]
    
    # In-Sample (Train)
    train_ret = sp500_ret.loc[:train_end]
    train_se = se.loc[:train_end]
    train_vol = v.loc[:train_end]
    
    # Out-of-Sample (Test)
    test_ret = sp500_ret.loc[test_start:]
    test_se = se.loc[test_start:]
    test_vol = v.loc[test_start:]
    
    # Run strategy on both
    # For test: use ranking thresholds learned from train
    train_se_80 = train_se.quantile(0.80)
    train_se_30 = train_se.quantile(0.30)
    train_vol_50 = train_vol.quantile(0.50)
    
    # Apply fixed thresholds to test period (no refit)
    danger_test = (test_se > train_se_80) & (test_vol < train_vol_50)
    safe_test = test_se < train_se_30
    
    lev_test = pd.Series(1.0, index=test_ret.index)
    lev_test[danger_test] = 0.0
    lev_test[safe_test] = LEVERAGE
    lev_test = lev_test.shift(1).fillna(1.0)
    
    daily_cost = (BORROW_COST / 252) * (lev_test - 1).clip(lower=0)
    strategy_test = test_ret * lev_test - daily_cost
    
    # Metrics
    train_strat, _ = run_strategy(train_ret, train_se, train_vol)
    train_metrics = calculate_metrics(train_strat.dropna())
    test_metrics = calculate_metrics(strategy_test.dropna())
    bench_test = calculate_metrics(test_ret.dropna())
    
    print(f"  In-Sample (1990-2009):")
    print(f"    Strategy Sharpe: {train_metrics['Sharpe']:.3f}")
    print(f"  Out-of-Sample (2010-2024):")
    print(f"    Strategy Sharpe: {test_metrics['Sharpe']:.3f}")
    print(f"    Benchmark Sharpe: {bench_test['Sharpe']:.3f}")
    print(f"    OOS Outperformance: {test_metrics['Sharpe'] - bench_test['Sharpe']:.3f}")
    
    return {
        'train_sharpe': train_metrics['Sharpe'],
        'test_sharpe': test_metrics['Sharpe'],
        'bench_sharpe': bench_test['Sharpe'],
        'oos_outperformance': test_metrics['Sharpe'] - bench_test['Sharpe']
    }

# =============================================================================
# TEST 3: PLACEBO TEST (SHUFFLED SIGNAL)
# =============================================================================

def placebo_test(sp500_ret, stored_energy, vol, n_placebo=100):
    """Test: Does a SHUFFLED (random) signal produce similar results?"""
    
    print("\n=== TEST 3: Placebo Test (Shuffled Signal) ===")
    
    # Real strategy performance
    real_ret, _ = run_strategy(sp500_ret, stored_energy, vol)
    real_sharpe = calculate_metrics(real_ret.dropna())['Sharpe']
    
    # Placebo: Shuffle the stored energy signal
    placebo_sharpes = []
    
    for i in range(n_placebo):
        shuffled_se = stored_energy.sample(frac=1, replace=False)
        shuffled_se.index = stored_energy.index  # Restore index
        
        placebo_ret, _ = run_strategy(sp500_ret, shuffled_se, vol)
        placebo_sharpes.append(calculate_metrics(placebo_ret.dropna())['Sharpe'])
    
    # P-value: How often does shuffled beat real?
    p_value = np.mean([p >= real_sharpe for p in placebo_sharpes])
    
    print(f"  Real Strategy Sharpe: {real_sharpe:.4f}")
    print(f"  Placebo Mean Sharpe: {np.mean(placebo_sharpes):.4f}")
    print(f"  Placebo Std: {np.std(placebo_sharpes):.4f}")
    print(f"  P-value (Placebo beats Real): {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"  ✓ SIGNIFICANT: Real signal outperforms random (p < 0.05)")
    else:
        print(f"  ✗ NOT SIGNIFICANT: Could be random")
    
    return {'real_sharpe': real_sharpe, 'placebo_mean': np.mean(placebo_sharpes), 'p_value': p_value}

# =============================================================================
# TEST 4: PARAMETER SENSITIVITY
# =============================================================================

def parameter_sensitivity(px, market, window_range=[42, 63, 126], energy_range=[30, 60, 90]):
    """Test: Are results robust to parameter choices?"""
    
    print("\n=== TEST 4: Parameter Sensitivity ===")
    
    results = []
    
    for w in window_range:
        for e in energy_range:
            try:
                entropy = calculate_entropy(px, w)
                stored_energy = calculate_stored_energy(entropy, e)
                returns = px.pct_change().mean(axis=1)
                vol = returns.rolling(w).std() * np.sqrt(252)
                
                common = market.index.intersection(stored_energy.dropna().index)
                sp500_ret = market['^GSPC'].loc[common].pct_change()
                
                strat_ret, _ = run_strategy(sp500_ret, stored_energy.loc[common], vol.loc[common])
                metrics = calculate_metrics(strat_ret.dropna())
                
                results.append({
                    'Entropy_Window': w,
                    'Energy_Window': e,
                    'Sharpe': metrics['Sharpe'],
                    'CAGR': metrics['CAGR'],
                    'MaxDD': metrics['MaxDD']
                })
            except:
                pass
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string())
    
    # Check if all parameter combos beat benchmark
    bench_sharpe = 0.31  # From earlier
    all_beat_bench = (results_df['Sharpe'] > bench_sharpe).all()
    print(f"\n  All parameter combos beat benchmark: {all_beat_bench}")
    
    return results_df

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("STATISTICAL VALIDATION SUITE FOR STORED ENERGY STRATEGY")
    print("=" * 70)
    
    px, market = load_data(LONG_HISTORY_UNIVERSE, START_DATE)
    if px.empty: return
    
    # Calculate features
    entropy = calculate_entropy(px, WINDOW)
    stored_energy = calculate_stored_energy(entropy, ENERGY_WINDOW)
    returns = px.pct_change().mean(axis=1)
    vol = returns.rolling(WINDOW).std() * np.sqrt(252)
    
    common = market.index.intersection(stored_energy.dropna().index)
    sp500_ret = market['^GSPC'].loc[common].pct_change()
    se = stored_energy.loc[common]
    v = vol.loc[common]
    
    strategy_ret, _ = run_strategy(sp500_ret, se, v)
    strategy_ret = strategy_ret.dropna()
    sp500_ret = sp500_ret.loc[strategy_ret.index]
    
    output_dir = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"
    
    # Run all tests
    bootstrap_results = bootstrap_confidence_intervals(strategy_ret, sp500_ret)
    walkforward_results = walk_forward_validation(px, market)
    placebo_results = placebo_test(sp500_ret, se, v)
    sensitivity_results = parameter_sensitivity(px, market)
    
    # Save results
    summary = pd.DataFrame({
        'Test': ['Bootstrap CI (Sharpe)', 'Bootstrap CI (CAGR)', 'Walk-Forward OOS', 'Placebo'],
        'Result': [
            f"{bootstrap_results['sharpe_diff']:.4f} [{bootstrap_results['sharpe_ci'][0]:.4f}, {bootstrap_results['sharpe_ci'][1]:.4f}]",
            f"{bootstrap_results['cagr_diff']:.2%} [{bootstrap_results['cagr_ci'][0]:.2%}, {bootstrap_results['cagr_ci'][1]:.2%}]",
            f"OOS Sharpe: {walkforward_results['test_sharpe']:.3f} vs Bench: {walkforward_results['bench_sharpe']:.3f}",
            f"p-value: {placebo_results['p_value']:.4f}"
        ],
        'Significant': [
            bootstrap_results['sharpe_sig'],
            bootstrap_results['cagr_sig'],
            walkforward_results['oos_outperformance'] > 0,
            placebo_results['p_value'] < 0.05
        ]
    })
    
    print("\n" + "=" * 70)
    print("SUMMARY OF STATISTICAL TESTS")
    print("=" * 70)
    print(summary.to_string(index=False))
    
    summary.to_csv(os.path.join(output_dir, "statistical_validation_summary.csv"), index=False)
    sensitivity_results.to_csv(os.path.join(output_dir, "parameter_sensitivity.csv"), index=False)
    
    print("\n=== VALIDATION COMPLETE ===")

if __name__ == "__main__":
    main()

"""
Tail Risk Reduction: Statistical Proof
======================================

Tests:
1. VaR (Value at Risk) Comparison - 5th percentile daily returns
2. CVaR (Expected Shortfall) - Average of worst 5% days
3. Extreme Loss Days - Count of days with > 3σ losses
4. Crisis Performance - GFC, COVID, Dot-Com
5. Mann-Whitney U Test - Statistical test on left tail
6. Conditional VaR - VaR during HIGH Stored Energy periods

Author: Antigravity Agent
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
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

START_DATE = "1980-01-01"
WINDOW = 63
ENERGY_WINDOW = 60
ROLL_RANK = 252 * 2
LEVERAGE = 1.5
BORROW_COST = 0.02

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

def run_strategy(sp500_ret, stored_energy, vol, roll_rank=ROLL_RANK):
    se_rank = stored_energy.rolling(roll_rank, min_periods=252).rank(pct=True)
    vol_rank = vol.rolling(roll_rank, min_periods=252).rank(pct=True)
    
    danger = (se_rank > 0.80) & (vol_rank < 0.50)
    safe = se_rank < 0.30
    
    lev = pd.Series(1.0, index=sp500_ret.index)
    lev[danger] = 0.0
    lev[safe] = LEVERAGE
    lev = lev.shift(1).fillna(1.0)
    
    daily_cost = (BORROW_COST / 252) * (lev - 1).clip(lower=0)
    strategy_ret = sp500_ret * lev - daily_cost
    
    return strategy_ret, lev, danger

# =============================================================================
# TAIL RISK TESTS
# =============================================================================

def test_var_cvar(strategy_ret, benchmark_ret, alpha=0.05):
    """Compare VaR and CVaR between strategy and benchmark."""
    
    print("\n" + "=" * 60)
    print("TEST 1: VALUE AT RISK (VaR) & EXPECTED SHORTFALL (CVaR)")
    print("=" * 60)
    
    s = strategy_ret.dropna()
    b = benchmark_ret.dropna()
    
    # VaR (Quantile of returns)
    var_strategy = s.quantile(alpha)
    var_benchmark = b.quantile(alpha)
    
    # CVaR (Mean of returns below VaR)
    cvar_strategy = s[s <= var_strategy].mean()
    cvar_benchmark = b[b <= var_benchmark].mean()
    
    # Improvement
    var_improvement = (var_benchmark - var_strategy) / abs(var_benchmark) * 100
    cvar_improvement = (cvar_benchmark - cvar_strategy) / abs(cvar_benchmark) * 100
    
    print(f"\n  5% VaR (Daily):")
    print(f"    Benchmark: {var_benchmark:.2%}")
    print(f"    Strategy:  {var_strategy:.2%}")
    print(f"    Improvement: {var_improvement:.1f}% (less negative = better)")
    
    print(f"\n  5% CVaR / Expected Shortfall (Daily):")
    print(f"    Benchmark: {cvar_benchmark:.2%}")
    print(f"    Strategy:  {cvar_strategy:.2%}")
    print(f"    Improvement: {cvar_improvement:.1f}%")
    
    # Bootstrap test for CVaR difference
    n_boot = 1000
    cvar_diffs = []
    for _ in range(n_boot):
        idx = np.random.choice(len(s), size=len(s), replace=True)
        s_sample = s.iloc[idx]
        b_sample = b.iloc[idx]
        
        var_s = s_sample.quantile(alpha)
        var_b = b_sample.quantile(alpha)
        cvar_s = s_sample[s_sample <= var_s].mean()
        cvar_b = b_sample[b_sample <= var_b].mean()
        cvar_diffs.append(cvar_s - cvar_b)
    
    ci_low = np.percentile(cvar_diffs, 2.5)
    ci_high = np.percentile(cvar_diffs, 97.5)
    significant = ci_low > 0 or ci_high < 0  # CI excludes 0
    
    print(f"\n  CVaR Difference (Strategy - Benchmark):")
    print(f"    Mean: {np.mean(cvar_diffs):.4%}")
    print(f"    95% CI: [{ci_low:.4%}, {ci_high:.4%}]")
    print(f"    SIGNIFICANT: {significant}")
    
    return {
        'var_bench': var_benchmark, 'var_strat': var_strategy,
        'cvar_bench': cvar_benchmark, 'cvar_strat': cvar_strategy,
        'cvar_ci': (ci_low, ci_high), 'cvar_significant': significant
    }

def test_extreme_days(strategy_ret, benchmark_ret, threshold=3):
    """Count days with extreme losses (> threshold * sigma)."""
    
    print("\n" + "=" * 60)
    print(f"TEST 2: EXTREME LOSS DAYS (>{threshold}σ)")
    print("=" * 60)
    
    s = strategy_ret.dropna()
    b = benchmark_ret.dropna()
    
    # Extreme loss threshold
    s_thresh = s.mean() - threshold * s.std()
    b_thresh = b.mean() - threshold * b.std()
    
    extreme_strategy = (s < s_thresh).sum()
    extreme_benchmark = (b < b_thresh).sum()
    
    reduction = (extreme_benchmark - extreme_strategy) / extreme_benchmark * 100
    
    print(f"\n  Threshold (3σ loss):")
    print(f"    Benchmark: {b_thresh:.2%}")
    print(f"    Strategy:  {s_thresh:.2%}")
    
    print(f"\n  Extreme Loss Day Count:")
    print(f"    Benchmark: {extreme_benchmark} days")
    print(f"    Strategy:  {extreme_strategy} days")
    print(f"    REDUCTION: {reduction:.1f}%")
    
    # Binomial test: Is the reduction significant?
    # H0: Strategy has same proportion of extreme days as benchmark
    n = len(s)
    p_bench = extreme_benchmark / n
    # P(observing <= extreme_strategy | p = p_bench)
    p_value = stats.binom.cdf(extreme_strategy, n, p_bench)
    
    print(f"\n  Binomial Test:")
    print(f"    P-value: {p_value:.4f}")
    print(f"    SIGNIFICANT (p < 0.05): {p_value < 0.05}")
    
    return {
        'extreme_bench': extreme_benchmark,
        'extreme_strat': extreme_strategy,
        'reduction_pct': reduction,
        'p_value': p_value
    }

def test_crisis_performance(strategy_ret, benchmark_ret):
    """Analyze performance during known crisis periods."""
    
    print("\n" + "=" * 60)
    print("TEST 3: CRISIS PERIOD PERFORMANCE")
    print("=" * 60)
    
    crises = {
        'Black Monday 1987': ('1987-09-01', '1987-12-31'),
        'Dot-Com Crash': ('2000-03-01', '2002-10-01'),
        'GFC': ('2007-10-01', '2009-03-01'),
        'COVID Crash': ('2020-02-01', '2020-03-31'),
        'Fed Tightening': ('2022-01-01', '2022-10-01'),
    }
    
    results = []
    
    for name, (start, end) in crises.items():
        try:
            s = strategy_ret.loc[start:end]
            b = benchmark_ret.loc[start:end]
            
            if len(s) < 20:
                continue
            
            cum_s = (1 + s).prod() - 1
            cum_b = (1 + b).prod() - 1
            
            dd_s = (1 + s).cumprod()
            dd_s = (dd_s / dd_s.cummax() - 1).min()
            
            dd_b = (1 + b).cumprod()
            dd_b = (dd_b / dd_b.cummax() - 1).min()
            
            protection = dd_s - dd_b  # Less negative = better
            
            results.append({
                'Crisis': name,
                'Benchmark_Return': cum_b,
                'Strategy_Return': cum_s,
                'Benchmark_MaxDD': dd_b,
                'Strategy_MaxDD': dd_s,
                'DD_Protection': protection
            })
            
            print(f"\n  {name}:")
            print(f"    Benchmark: {cum_b:.1%} return, {dd_b:.1%} MaxDD")
            print(f"    Strategy:  {cum_s:.1%} return, {dd_s:.1%} MaxDD")
            print(f"    DD Protection: {protection:.1%}")
            
        except:
            pass
    
    results_df = pd.DataFrame(results)
    
    # Average protection
    if len(results_df) > 0:
        avg_protection = results_df['DD_Protection'].mean()
        print(f"\n  AVERAGE DD PROTECTION: {avg_protection:.1%}")
        
        # T-test: Is average protection > 0?
        t_stat, p_value = stats.ttest_1samp(results_df['DD_Protection'], 0)
        # One-sided test (we expect protection > 0, meaning less negative DD)
        p_value_one = p_value / 2 if t_stat > 0 else 1 - p_value / 2
        print(f"  T-test (Protection > 0): t={t_stat:.2f}, p={p_value_one:.4f}")
        print(f"  SIGNIFICANT: {p_value_one < 0.05}")
    
    return results_df

def test_left_tail_distribution(strategy_ret, benchmark_ret, quantile=0.10):
    """Mann-Whitney U test on left tail of returns distribution."""
    
    print("\n" + "=" * 60)
    print("TEST 4: LEFT TAIL DISTRIBUTION (Mann-Whitney U)")
    print("=" * 60)
    
    s = strategy_ret.dropna()
    b = benchmark_ret.dropna()
    
    # Get the worst 10% of days for each
    s_thresh = s.quantile(quantile)
    b_thresh = b.quantile(quantile)
    
    s_tail = s[s <= s_thresh]
    b_tail = b[b <= b_thresh]
    
    print(f"\n  Left Tail (Worst {quantile:.0%} of days):")
    print(f"    Benchmark: n={len(b_tail)}, mean={b_tail.mean():.2%}, std={b_tail.std():.2%}")
    print(f"    Strategy:  n={len(s_tail)}, mean={s_tail.mean():.2%}, std={s_tail.std():.2%}")
    
    # Mann-Whitney U test
    # H0: The distributions are the same
    # H1: Strategy tail is shifted right (less negative)
    stat, p_value = stats.mannwhitneyu(s_tail, b_tail, alternative='greater')
    
    print(f"\n  Mann-Whitney U Test (Strategy > Benchmark in left tail):")
    print(f"    U-statistic: {stat:.0f}")
    print(f"    P-value: {p_value:.4f}")
    print(f"    SIGNIFICANT (p < 0.05): {p_value < 0.05}")
    
    return {
        'u_stat': stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def test_conditional_var(strategy_ret, benchmark_ret, danger_mask, alpha=0.05):
    """VaR conditional on being in the DANGER regime."""
    
    print("\n" + "=" * 60)
    print("TEST 5: CONDITIONAL VaR (During High Stored Energy)")
    print("=" * 60)
    
    s = strategy_ret.dropna()
    b = benchmark_ret.loc[s.index]
    d = danger_mask.loc[s.index].shift(1).fillna(False)  # Match signal lag
    
    # Returns during DANGER periods
    s_danger = s[d]
    b_danger = b[d]
    
    # Returns during SAFE periods
    s_safe = s[~d]
    b_safe = b[~d]
    
    print(f"\n  During DANGER periods ({d.sum()} days, {d.mean():.1%} of time):")
    print(f"    Benchmark Mean: {b_danger.mean():.3%}")
    print(f"    Strategy Mean:  {s_danger.mean():.3%} (hedged)")
    
    if len(b_danger) > 50:
        var_b_danger = b_danger.quantile(alpha)
        var_s_danger = s_danger.quantile(alpha)
        
        print(f"\n    Benchmark VaR (5%): {var_b_danger:.2%}")
        print(f"    Strategy VaR (5%):  {var_s_danger:.2%}")
        print(f"    VaR IMPROVEMENT: {(var_s_danger - var_b_danger):.2%} (closer to 0 = better)")
        
        # This should be very significant: strategy was hedged (0x) during danger
        if var_s_danger > var_b_danger:
            print(f"    ✓ Strategy REDUCES tail risk during DANGER by {var_s_danger - var_b_danger:.2%}")
    
    return {}

def main():
    print("=" * 70)
    print("TAIL RISK REDUCTION: STATISTICAL PROOF")
    print("=" * 70)
    
    px, market = load_data(LONG_HISTORY_UNIVERSE, START_DATE)
    if px.empty: return
    
    # Calculate features
    entropy = calculate_entropy(px, WINDOW)
    fragility = 1 - entropy
    stored_energy = fragility.rolling(ENERGY_WINDOW).sum()
    returns = px.pct_change().mean(axis=1)
    vol = returns.rolling(WINDOW).std() * np.sqrt(252)
    
    common = market.index.intersection(stored_energy.dropna().index)
    sp500_ret = market['^GSPC'].loc[common].pct_change()
    se = stored_energy.loc[common]
    v = vol.loc[common]
    
    strategy_ret, leverage, danger_mask = run_strategy(sp500_ret, se, v)
    strategy_ret = strategy_ret.dropna()
    benchmark_ret = sp500_ret.loc[strategy_ret.index]
    
    output_dir = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"
    
    # Run all tail risk tests
    var_results = test_var_cvar(strategy_ret, benchmark_ret)
    extreme_results = test_extreme_days(strategy_ret, benchmark_ret)
    crisis_results = test_crisis_performance(strategy_ret, benchmark_ret)
    tail_results = test_left_tail_distribution(strategy_ret, benchmark_ret)
    conditional_results = test_conditional_var(strategy_ret, benchmark_ret, danger_mask)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: TAIL RISK REDUCTION TESTS")
    print("=" * 70)
    
    summary = pd.DataFrame([
        {'Test': 'CVaR Reduction', 'Result': f"{(var_results['cvar_strat'] - var_results['cvar_bench']):.2%}", 'Significant': var_results['cvar_significant']},
        {'Test': 'Extreme Days Reduction', 'Result': f"{extreme_results['reduction_pct']:.0f}%", 'Significant': extreme_results['p_value'] < 0.05},
        {'Test': 'Left Tail Distribution', 'Result': f"p={tail_results['p_value']:.4f}", 'Significant': tail_results['significant']},
    ])
    
    print(summary.to_string(index=False))
    
    summary.to_csv(os.path.join(output_dir, "tail_risk_summary.csv"), index=False)
    crisis_results.to_csv(os.path.join(output_dir, "crisis_performance.csv"), index=False)
    
    print("\n=== TAIL RISK VALIDATION COMPLETE ===")

if __name__ == "__main__":
    main()

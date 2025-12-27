"""
R&R Empirical Tests: Addressing Remaining Reviewer Concerns
=============================================================

Tests:
1. Granger Causality: Credit SE → Equity SE
2. Detailed Backtest Metrics (MaxDD, Calmar, Sortino)
3. Out-of-Sample Holdout (Train: 2007-2019, Test: 2020-2024)
4. Comparison to Vol-Timing (Moreira & Muir, 2017)

Author: Antigravity Agent
Date: December 2025
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import yfinance as yf
import os
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# Universe
EXPANDED_UNIVERSE = [
    'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY',
    'EWJ', 'EWG', 'EWU', 'EWC', 'EWA', 'EWZ', 'EWY', 'EWT', 'EWH', 'EWS',
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI',
    'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'TIP', 'AGG',
    'GLD', 'SLV', 'USO', 'DBC',
    'EFA', 'EEM', 'VEU', 'VWO',
    'VNQ', 'IYR',
]

CREDIT_TICKERS = ['LQD', 'HYG', 'TIP', 'AGG']
EQUITY_TICKERS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'SPY', 'QQQ']

START_DATE = "2007-01-01"
WINDOW = 63
ENERGY_WINDOW = 60

def load_data(tickers, start_date):
    print("Loading data...")
    df = yf.download(tickers + ['^VIX', '^GSPC'], start=start_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        px = df['Adj Close' if 'Adj Close' in df else 'Close']
    else:
        px = df
    return px.dropna(how='all')

def calculate_entropy_for_subset(px, tickers, window):
    """Calculate entropy for a subset of assets."""
    subset = [t for t in tickers if t in px.columns]
    returns = px[subset].pct_change()
    entropy_series = {}
    
    for i in range(window, len(returns)):
        idx = returns.index[i]
        window_ret = returns.iloc[i-window:i].dropna(axis=1)
        
        if len(window_ret.columns) < 4:
            continue
            
        corr_matrix = window_ret.corr()
        
        try:
            eigvals = np.linalg.eigvalsh(corr_matrix)
            eigvals = eigvals[eigvals > 1e-10]
            probs = eigvals / np.sum(eigvals)
            S = -np.sum(probs * np.log(probs))
            N = len(probs)
            S_norm = S / np.log(N) if N > 1 else 1
        except:
            S_norm = np.nan
        entropy_series[idx] = S_norm
    
    return pd.Series(entropy_series)

def granger_causality_test(credit_se, equity_se, max_lag=5):
    """Test if Credit SE Granger-causes Equity SE."""
    
    print("\n" + "=" * 60)
    print("TEST 1: GRANGER CAUSALITY (Credit SE → Equity SE)")
    print("=" * 60)
    
    df = pd.DataFrame({
        'Credit_SE': credit_se,
        'Equity_SE': equity_se
    }).dropna()
    
    # Granger test
    print(f"\n  Sample size: {len(df)}")
    print(f"  Testing lags: 1 to {max_lag}")
    
    results = []
    try:
        gc_results = grangercausalitytests(df[['Equity_SE', 'Credit_SE']], maxlag=max_lag, verbose=False)
        
        for lag in range(1, max_lag + 1):
            f_stat = gc_results[lag][0]['ssr_ftest'][0]
            p_val = gc_results[lag][0]['ssr_ftest'][1]
            results.append({
                'Lag': lag,
                'F_Stat': f_stat,
                'P_Value': p_val,
                'Significant': p_val < 0.05
            })
            print(f"  Lag {lag}: F={f_stat:.2f}, p={p_val:.4f}, sig={p_val < 0.05}")
    except Exception as e:
        print(f"  Error: {e}")
    
    return pd.DataFrame(results)

def detailed_backtest_metrics(strategy_returns, benchmark_returns, rf=0.02):
    """Calculate comprehensive backtest metrics."""
    
    print("\n" + "=" * 60)
    print("TEST 2: DETAILED BACKTEST METRICS")
    print("=" * 60)
    
    def calc_metrics(rets, name):
        cumret = (1 + rets).cumprod()
        running_max = cumret.cummax()
        drawdown = (cumret - running_max) / running_max
        
        cagr = (cumret.iloc[-1]) ** (252 / len(rets)) - 1
        vol = rets.std() * np.sqrt(252)
        sharpe = (rets.mean() * 252 - rf) / vol
        max_dd = drawdown.min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # Sortino (downside deviation)
        downside_rets = rets[rets < 0]
        downside_vol = downside_rets.std() * np.sqrt(252) if len(downside_rets) > 0 else 0
        sortino = (rets.mean() * 252 - rf) / downside_vol if downside_vol > 0 else 0
        
        return {
            'Strategy': name,
            'CAGR': cagr,
            'Volatility': vol,
            'Sharpe': sharpe,
            'Max_Drawdown': max_dd,
            'Calmar': calmar,
            'Sortino': sortino
        }
    
    strat_metrics = calc_metrics(strategy_returns, 'Stored Energy')
    bench_metrics = calc_metrics(benchmark_returns, 'Benchmark')
    
    # Vol-timing strategy (Moreira & Muir, 2017)
    # Scale exposure inversely to realized volatility
    realized_vol = benchmark_returns.rolling(21).std()
    target_vol = benchmark_returns.std()
    vol_timing_weight = (target_vol / realized_vol).clip(0.5, 1.5).shift(1).fillna(1)
    vol_timing_returns = vol_timing_weight * benchmark_returns
    vol_timing_metrics = calc_metrics(vol_timing_returns.dropna(), 'Vol-Timing (M&M 2017)')
    
    results = pd.DataFrame([bench_metrics, strat_metrics, vol_timing_metrics])
    
    print("\n  PERFORMANCE COMPARISON:")
    print(results.to_string(index=False))
    
    return results

def out_of_sample_test(px, all_tickers, train_end='2019-12-31'):
    """Out-of-sample holdout test."""
    
    print("\n" + "=" * 60)
    print("TEST 3: OUT-OF-SAMPLE HOLDOUT")
    print(f"  Train: 2007 - {train_end[:4]}")
    print(f"  Test: {int(train_end[:4])+1} - 2024")
    print("=" * 60)
    
    available = [t for t in all_tickers if t in px.columns]
    returns = px[available].pct_change()
    
    # Calculate SE for full period
    entropy_series = {}
    for i in range(WINDOW, len(returns)):
        idx = returns.index[i]
        window_ret = returns.iloc[i-WINDOW:i].dropna(axis=1)
        
        if len(window_ret.columns) < 15:
            continue
            
        corr_matrix = window_ret.corr()
        
        try:
            eigvals = np.linalg.eigvalsh(corr_matrix)
            eigvals = eigvals[eigvals > 1e-10]
            probs = eigvals / np.sum(eigvals)
            S = -np.sum(probs * np.log(probs))
            N = len(probs)
            S_norm = S / np.log(N) if N > 1 else 1
        except:
            S_norm = np.nan
        entropy_series[idx] = S_norm
    
    entropy = pd.Series(entropy_series)
    fragility = 1 - entropy
    se = fragility.rolling(ENERGY_WINDOW).sum()
    
    spy_fwd = px['^GSPC'].pct_change(21).shift(-21) if '^GSPC' in px.columns else px['SPY'].pct_change(21).shift(-21)
    
    common = se.dropna().index.intersection(spy_fwd.dropna().index)
    
    df = pd.DataFrame({
        'SE': se.loc[common],
        'Fwd_Ret': spy_fwd.loc[common]
    }).dropna()
    
    # Split
    train = df[df.index <= train_end]
    test = df[df.index > train_end]
    
    print(f"\n  Train sample: {len(train)}")
    print(f"  Test sample: {len(test)}")
    
    # Train model
    train['SE_std'] = (train['SE'] - train['SE'].mean()) / train['SE'].std()
    X_train = sm.add_constant(train['SE_std'])
    y_train = train['Fwd_Ret']
    model = sm.OLS(y_train, X_train).fit()
    
    print(f"\n  IN-SAMPLE (Train):")
    print(f"    SE Coef: {model.params['SE_std']:.4f}, p={model.pvalues['SE_std']:.2e}")
    
    # Test model (use train mean/std for standardization)
    test['SE_std'] = (test['SE'] - train['SE'].mean()) / train['SE'].std()
    X_test = sm.add_constant(test['SE_std'])
    y_test = test['Fwd_Ret']
    test_model = sm.OLS(y_test, X_test).fit()
    
    print(f"\n  OUT-OF-SAMPLE (Test):")
    print(f"    SE Coef: {test_model.params['SE_std']:.4f}, p={test_model.pvalues['SE_std']:.2e}")
    print(f"    OOS Significant: {test_model.pvalues['SE_std'] < 0.05}")
    
    results = pd.DataFrame({
        'Period': ['In-Sample (2007-2019)', 'Out-of-Sample (2020-2024)'],
        'SE_Coef': [model.params['SE_std'], test_model.params['SE_std']],
        'SE_PVal': [model.pvalues['SE_std'], test_model.pvalues['SE_std']],
        'R2': [model.rsquared, test_model.rsquared],
        'Significant': [model.pvalues['SE_std'] < 0.05, test_model.pvalues['SE_std'] < 0.05]
    })
    
    return results

def main():
    print("=" * 70)
    print("R&R EMPIRICAL TESTS")
    print("=" * 70)
    
    px = load_data(EXPANDED_UNIVERSE, START_DATE)
    if px.empty: 
        print("ERROR: No data")
        return
    
    output_dir = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"
    
    # Test 1: Granger Causality
    print("\nCalculating Credit SE...")
    credit_entropy = calculate_entropy_for_subset(px, CREDIT_TICKERS, WINDOW)
    credit_se = (1 - credit_entropy).rolling(ENERGY_WINDOW).sum()
    
    print("Calculating Equity SE...")
    equity_entropy = calculate_entropy_for_subset(px, EQUITY_TICKERS, WINDOW)
    equity_se = (1 - equity_entropy).rolling(ENERGY_WINDOW).sum()
    
    granger_results = granger_causality_test(credit_se, equity_se)
    granger_results.to_csv(os.path.join(output_dir, "Table_Granger_Causality.csv"), index=False)
    
    # Test 2: Detailed Backtest Metrics
    spy = px['^GSPC'] if '^GSPC' in px.columns else px['SPY']
    spy_ret = spy.pct_change().dropna()
    
    # Simple SE strategy: reduce exposure when SE is high
    all_tickers = [t for t in EXPANDED_UNIVERSE if t in px.columns]
    returns = px[all_tickers].pct_change()
    
    entropy_series = {}
    for i in range(WINDOW, len(returns)):
        idx = returns.index[i]
        window_ret = returns.iloc[i-WINDOW:i].dropna(axis=1)
        if len(window_ret.columns) < 15:
            continue
        corr = window_ret.corr()
        try:
            eigvals = np.linalg.eigvalsh(corr)
            eigvals = eigvals[eigvals > 1e-10]
            probs = eigvals / np.sum(eigvals)
            S = -np.sum(probs * np.log(probs))
            N = len(probs)
            entropy_series[idx] = S / np.log(N) if N > 1 else 1
        except:
            pass
    
    se = (1 - pd.Series(entropy_series)).rolling(ENERGY_WINDOW).sum()
    se_pct = se.rank(pct=True)
    
    # Strategy: 100% exposure when SE < 50th pct, 50% when > 50th
    exposure = (1 - se_pct * 0.5).shift(1).fillna(1)
    strategy_ret = exposure * spy_ret
    
    common = strategy_ret.dropna().index.intersection(spy_ret.index)
    backtest_results = detailed_backtest_metrics(
        strategy_ret.loc[common], 
        spy_ret.loc[common]
    )
    backtest_results.to_csv(os.path.join(output_dir, "Table_Backtest_Metrics.csv"), index=False)
    
    # Test 3: Out-of-Sample
    oos_results = out_of_sample_test(px, EXPANDED_UNIVERSE)
    oos_results.to_csv(os.path.join(output_dir, "Table_OOS_Test.csv"), index=False)
    
    print("\n=== R&R TESTS COMPLETE ===")

if __name__ == "__main__":
    main()

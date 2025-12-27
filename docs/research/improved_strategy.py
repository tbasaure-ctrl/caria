"""
Improved SE Strategy: Exploiting SE × VIX Interaction
======================================================

Strategy Logic:
1. HIGH SE + LOW VIX = DANGER (reduce exposure to 25%)
2. HIGH SE + HIGH VIX = Crisis already priced (50%)
3. LOW SE + LOW VIX = Healthy expansion (100%)
4. LOW SE + HIGH VIX = Buying opportunity (110%)

This exploits the positive interaction term we found.

Author: Antigravity Agent
Date: December 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

EXPANDED_UNIVERSE = [
    'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY',
    'EWJ', 'EWG', 'EWU', 'EWC', 'EWA', 'EWZ', 'EWY', 'EWT', 'EWH', 'EWS',
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI',
    'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'TIP', 'AGG',
    'GLD', 'SLV', 'USO', 'DBC',
    'EFA', 'EEM', 'VEU', 'VWO',
    'VNQ', 'IYR',
]

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

def calculate_se(px, all_tickers, window, energy_window):
    available = [t for t in all_tickers if t in px.columns]
    returns = px[available].pct_change()
    
    entropy_series = {}
    for i in range(window, len(returns)):
        idx = returns.index[i]
        window_ret = returns.iloc[i-window:i].dropna(axis=1)
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
    
    se = (1 - pd.Series(entropy_series)).rolling(energy_window).sum()
    return se

def calc_metrics(rets, rf=0.02):
    cumret = (1 + rets).cumprod()
    running_max = cumret.cummax()
    drawdown = (cumret - running_max) / running_max
    
    cagr = (cumret.iloc[-1]) ** (252 / len(rets)) - 1
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252 - rf) / vol
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    downside_rets = rets[rets < 0]
    downside_vol = downside_rets.std() * np.sqrt(252) if len(downside_rets) > 0 else 0
    sortino = (rets.mean() * 252 - rf) / downside_vol if downside_vol > 0 else 0
    
    return {
        'CAGR': cagr,
        'Volatility': vol,
        'Sharpe': sharpe,
        'Max_Drawdown': max_dd,
        'Calmar': calmar,
        'Sortino': sortino
    }

def strategy_simple(se_pct, vix_pct):
    """Original simple strategy."""
    return (1 - se_pct * 0.5)

def strategy_interaction(se_pct, vix_pct):
    """
    Improved strategy using SE × VIX interaction.
    
    Quadrants:
    - HIGH SE + LOW VIX: DANGER (complacency) → 25%
    - HIGH SE + HIGH VIX: Crisis priced → 50%  
    - LOW SE + LOW VIX: Healthy → 100%
    - LOW SE + HIGH VIX: Opportunity → 110%
    """
    exposure = pd.Series(index=se_pct.index, dtype=float)
    
    for i in range(len(se_pct)):
        se = se_pct.iloc[i]
        vix = vix_pct.iloc[i]
        
        if pd.isna(se) or pd.isna(vix):
            exposure.iloc[i] = 1.0
        elif se > 0.7 and vix < 0.3:  # HIGH SE, LOW VIX = DANGER
            exposure.iloc[i] = 0.25
        elif se > 0.7 and vix >= 0.3:  # HIGH SE, HIGH VIX = Crisis priced
            exposure.iloc[i] = 0.50
        elif se <= 0.3 and vix > 0.7:  # LOW SE, HIGH VIX = Opportunity
            exposure.iloc[i] = 1.10
        else:  # Normal
            exposure.iloc[i] = 1.0 - se * 0.3
    
    return exposure

def strategy_aggressive(se_pct, vix_pct):
    """
    More aggressive de-risking with leverage on opportunities.
    """
    exposure = pd.Series(index=se_pct.index, dtype=float)
    
    for i in range(len(se_pct)):
        se = se_pct.iloc[i]
        vix = vix_pct.iloc[i]
        
        if pd.isna(se) or pd.isna(vix):
            exposure.iloc[i] = 1.0
        elif se > 0.8:  # Very high SE
            if vix < 0.3:  # Complacency
                exposure.iloc[i] = 0.0  # Full cash
            else:
                exposure.iloc[i] = 0.25
        elif se > 0.5:  # Elevated SE
            exposure.iloc[i] = 0.50
        elif se < 0.3 and vix > 0.7:  # Low SE, High VIX = Recovery
            exposure.iloc[i] = 1.25  # Leveraged
        else:
            exposure.iloc[i] = 1.0
    
    return exposure

def backtest_strategies(px, se, vix, spy_ret):
    """Compare all strategy variants."""
    
    # Align all data first
    common_idx = se.dropna().index.intersection(vix.dropna().index).intersection(spy_ret.dropna().index)
    
    se_aligned = se.loc[common_idx]
    vix_aligned = vix.loc[common_idx]
    spy_aligned = spy_ret.loc[common_idx]
    
    se_pct = se_aligned.rank(pct=True)
    vix_pct = vix_aligned.rank(pct=True)
    
    strategies = {
        'Benchmark': pd.Series(1.0, index=common_idx),
        'Simple (Old)': strategy_simple(se_pct, vix_pct).shift(1).fillna(1),
        'Interaction': strategy_interaction(se_pct, vix_pct).shift(1).fillna(1),
        'Aggressive': strategy_aggressive(se_pct, vix_pct).shift(1).fillna(1),
    }
    
    # Vol-timing (Moreira & Muir)
    realized_vol = spy_aligned.rolling(21).std()
    target_vol = spy_aligned.std()
    vol_timing_weight = (target_vol / realized_vol).clip(0.5, 1.5).shift(1).fillna(1)
    strategies['Vol-Timing'] = vol_timing_weight
    
    results = []
    equity_curves = {}
    
    for name, exposure in strategies.items():
        strat_ret = (exposure * spy_aligned).dropna()
        
        metrics = calc_metrics(strat_ret)
        metrics['Strategy'] = name
        results.append(metrics)
        
        equity_curves[name] = (1 + strat_ret).cumprod()
    
    return pd.DataFrame(results), equity_curves

def main():
    print("=" * 70)
    print("IMPROVED SE STRATEGY COMPARISON")
    print("=" * 70)
    
    px = load_data(EXPANDED_UNIVERSE, START_DATE)
    if px.empty: 
        print("ERROR: No data")
        return
    
    output_dir = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"
    
    # Calculate SE
    print("Calculating SE...")
    se = calculate_se(px, EXPANDED_UNIVERSE, WINDOW, ENERGY_WINDOW)
    
    # VIX
    vix = px['^VIX'] if '^VIX' in px.columns else None
    
    # SPY returns
    spy = px['^GSPC'] if '^GSPC' in px.columns else px['SPY']
    spy_ret = spy.pct_change()
    
    # Backtest
    results, equity_curves = backtest_strategies(px, se, vix, spy_ret)
    
    # Print results
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)
    
    cols = ['CAGR', 'Volatility', 'Sharpe', 'Max_Drawdown', 'Calmar', 'Sortino']
    print(results[cols].to_string())
    
    results.to_csv(os.path.join(output_dir, "Table_Strategy_Comparison.csv"))
    
    # Best strategy
    best = results['Sharpe'].idxmax()
    print(f"\n  BEST SHARPE: {best} ({results.loc[best, 'Sharpe']:.3f})")
    
    best_calmar = results['Calmar'].idxmax()
    print(f"  BEST CALMAR: {best_calmar} ({results.loc[best_calmar, 'Calmar']:.3f})")
    
    # Plot equity curves
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = {'Benchmark (Buy & Hold)': 'gray', 'Simple (Old)': 'blue', 
              'Interaction (New)': 'green', 'Aggressive': 'red', 'Vol-Timing (M&M)': 'purple'}
    
    for name, eq in equity_curves.items():
        ax.semilogy(eq.index, eq.values, label=name, color=colors.get(name, 'black'), 
                    linewidth=2 if 'New' in name or 'Aggr' in name else 1,
                    alpha=0.8 if 'New' in name or 'Aggr' in name else 0.5)
    
    ax.set_title('Strategy Comparison: Equity Curves (Log Scale)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (Log Scale)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Figure_Strategy_Comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n=== STRATEGY COMPARISON COMPLETE ===")

if __name__ == "__main__":
    main()

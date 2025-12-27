"""
Interaction Regression: SE × VIX
================================

Hypothesis: Stored Energy AMPLIFIES the effect of VIX on CVaR.
When SE is high (fragile regime), VIX matters more for tail risk.

Model: Forward_CVaR ~ SE + VIX + SE×VIX + ε

If β(SE×VIX) is significant and negative:
    "High SE + High VIX → Much worse CVaR"
    SE is a regime filter that conditions VIX's effect.

Author: Antigravity Agent
Date: December 2025
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
import os
import warnings

warnings.filterwarnings("ignore")

SYSTEM_UNIVERSE = [
    "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "XLV", "XLP",
    "LQD", "HYG", "EFA", "EEM", "TLT", "IEF", "GLD"
]

START_DATE = "2005-01-01"
WINDOW = 63
ENERGY_WINDOW = 60

def load_data(tickers, start_date):
    print("Loading data...")
    df = yf.download(tickers + ["^VIX"], start=start_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        px = df['Adj Close' if 'Adj Close' in df else 'Close']
    else:
        px = df
    return px.dropna(how='all')

def calculate_entropy(px, window):
    returns = px.pct_change()
    entropy_series = {}

    for i in range(window, len(returns)):
        idx = returns.index[i]
        window_ret = returns.iloc[i-window : i]
        valid_cols = window_ret.dropna(axis=1, thresh=int(window*0.8)).columns
        if len(valid_cols) < 8:
            continue
            
        corr_matrix = window_ret[valid_cols].corr()
        corr_matrix = corr_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
        
        if not corr_matrix.empty and len(corr_matrix) >= 8:
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

def run_interaction_regression():
    print("=== INTERACTION REGRESSION: SE × VIX ===")
    
    px = load_data(SYSTEM_UNIVERSE, START_DATE)
    if px.empty: return
    
    # Calculate SE
    system_tickers = [t for t in SYSTEM_UNIVERSE if t in px.columns]
    entropy = calculate_entropy(px[system_tickers], WINDOW)
    fragility = 1 - entropy
    stored_energy = fragility.rolling(ENERGY_WINDOW).sum()
    
    # VIX
    vix = px['^VIX'] if '^VIX' in px.columns else None
    
    # Forward returns (for CVaR proxy)
    spy_fwd_ret = px['SPY'].pct_change(21).shift(-21)
    
    # Build dataframe
    common = stored_energy.dropna().index
    df = pd.DataFrame({
        'SE': stored_energy.loc[common],
        'VIX': vix.reindex(common) if vix is not None else np.nan,
        'Forward_Ret': spy_fwd_ret.reindex(common)
    }).dropna()
    
    # Standardize SE and VIX for interpretability
    df['SE_std'] = (df['SE'] - df['SE'].mean()) / df['SE'].std()
    df['VIX_std'] = (df['VIX'] - df['VIX'].mean()) / df['VIX'].std()
    
    # Interaction term
    df['SE_x_VIX'] = df['SE_std'] * df['VIX_std']
    
    # Model 1: Without interaction
    print("\n" + "=" * 60)
    print("MODEL 1: Forward_Ret ~ SE + VIX (No Interaction)")
    print("=" * 60)
    
    X1 = df[['SE_std', 'VIX_std']]
    X1 = sm.add_constant(X1)
    y = df['Forward_Ret']
    
    model1 = sm.OLS(y, X1).fit()
    print(model1.summary())
    
    # Model 2: With interaction
    print("\n" + "=" * 60)
    print("MODEL 2: Forward_Ret ~ SE + VIX + SE×VIX (With Interaction)")
    print("=" * 60)
    
    X2 = df[['SE_std', 'VIX_std', 'SE_x_VIX']]
    X2 = sm.add_constant(X2)
    
    model2 = sm.OLS(y, X2).fit()
    print(model2.summary())
    
    # Key interpretation
    print("\n" + "=" * 60)
    print("KEY INTERPRETATION")
    print("=" * 60)
    
    interaction_coef = model2.params['SE_x_VIX']
    interaction_pval = model2.pvalues['SE_x_VIX']
    
    print(f"\n  Interaction Term (SE × VIX):")
    print(f"    Coefficient: {interaction_coef:.6f}")
    print(f"    P-value: {interaction_pval:.6f}")
    print(f"    Significant (p < 0.05): {interaction_pval < 0.05}")
    
    if interaction_coef < 0 and interaction_pval < 0.05:
        print(f"\n  ✓ SIGNIFICANT NEGATIVE INTERACTION")
        print(f"    When BOTH SE and VIX are high, forward returns are WORSE.")
        print(f"    SE amplifies the effect of VIX on tail risk.")
        print(f"    This supports SE as a 'regime filter'.")
    elif interaction_coef > 0 and interaction_pval < 0.05:
        print(f"\n  ✓ SIGNIFICANT POSITIVE INTERACTION")
        print(f"    High SE + High VIX → Less bad than expected.")
        print(f"    Possible: Crisis already priced in when both are elevated.")
    else:
        print(f"\n  ✗ Interaction NOT significant at 5% level.")
        print(f"    SE and VIX effects are additive, not multiplicative.")
    
    # R² improvement
    r2_improvement = model2.rsquared - model1.rsquared
    print(f"\n  R² Improvement from Interaction: {r2_improvement:.4f} ({r2_improvement/model1.rsquared*100:.1f}% relative)")
    
    # Save results
    output_dir = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"
    
    results = pd.DataFrame({
        'Model': ['Without Interaction', 'With Interaction'],
        'R2': [model1.rsquared, model2.rsquared],
        'AIC': [model1.aic, model2.aic],
        'SE_Coef': [model1.params['SE_std'], model2.params['SE_std']],
        'SE_PVal': [model1.pvalues['SE_std'], model2.pvalues['SE_std']],
        'VIX_Coef': [model1.params['VIX_std'], model2.params['VIX_std']],
        'VIX_PVal': [model1.pvalues['VIX_std'], model2.pvalues['VIX_std']],
        'Interaction_Coef': [np.nan, model2.params['SE_x_VIX']],
        'Interaction_PVal': [np.nan, model2.pvalues['SE_x_VIX']]
    })
    results.to_csv(os.path.join(output_dir, "Table4_Interaction_Regression.csv"), index=False)
    
    print(f"\n  Saved: Table4_Interaction_Regression.csv")
    print("\n=== REGRESSION COMPLETE ===")

if __name__ == "__main__":
    run_interaction_regression()

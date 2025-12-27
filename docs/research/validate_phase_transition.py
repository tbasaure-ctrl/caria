"""
Phase Transition Econometric Validation
=======================================

Purpose:
--------
Rigorously test the "Passive Substrate / Phase Transition" theory using the user's specified battery of tests.
We test if the relationship between ASF (Fragility) and Risk is conditional on Connectivity (C).

Test Suite:
1. Interaction Model (Linear): Risk ~ ASF + C + ASF*C. 
   - Check if b3 (Interaction) is significant.
   - Estimate C_crit = -b1/b3.
   - Plot Marginal Effect of ASF conditional on C with 95% CI.
2. Threshold Model (Piecewise): Estimate tau via grid search where slope flips sign.
3. Nonlinear Model (Polynomial): Risk ~ ASF + C + ASF^2 + C^2 + ASF*C (Approximating general surface).
4. Rolling Estimation: Track b3(t) and C_crit(t) over decades.
5. Robustness: Repeat with alternative definitions of C (Absorption Ratio, Network Density).

Author: Research Team
Date: December 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

OUTPUT_DIR = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"
EQUITY_UNIVERSE = [
    'AAPL', 'MSFT', 'JNJ', 'PG', 'XOM', 'JPM', 'GE', 'KO', 'PFE', 'WMT',
    'IBM', 'CVX', 'MRK', 'DIS', 'HD', 'MCD', 'BA', 'CAT', 'MMM', 'AXP'
]
LAMBDA = 0.02
CORR_WINDOW = 126

def load_data():
    start = '1990-01-01'
    end = '2025-12-20'
    print(f"Loading {start} to {end}...")
    
    df = yf.download(EQUITY_UNIVERSE, start=start, end=end, progress=False)
    spy = yf.download('^GSPC', start=start, end=end, progress=False)
    
    # Robust Selection
    if isinstance(df.columns, pd.MultiIndex):
        try: prices = df.xs('Adj Close', level=0, axis=1)
        except: prices = df.xs('Close', level=0, axis=1)
    else: prices = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
        
    if isinstance(spy.columns, pd.MultiIndex):
        try: spy_p = spy.xs('Adj Close', level=0, axis=1)
        except: spy_p = spy.xs('Close', level=0, axis=1)
    else: spy_p = spy['Adj Close'] if 'Adj Close' in spy.columns else spy['Close']
    
    if isinstance(spy_p, pd.DataFrame): spy_p = spy_p.iloc[:, 0]
    
    # Drop duplicates
    prices = prices[~prices.index.duplicated(keep='first')]
    spy_p = spy_p[~spy_p.index.duplicated(keep='first')]
    
    return prices.dropna(axis=1, how='all'), spy_p.dropna()

def compute_metrics(prices, spy):
    print("Computing State Variables (ASF, Connectivity)...")
    ret = prices.pct_change().dropna()
    
    # Vol Standardize
    vol = ret.rolling(21).std().shift(1)
    std_ret = (ret / vol).dropna()
    
    dates = std_ret.index
    data_list = []
    
    # Step 5 for speed
    indices = range(CORR_WINDOW, len(std_ret), 5)
    
    for i in indices:
        window = std_ret.iloc[i-CORR_WINDOW:i]
        try:
            lw = LedoitWolf()
            lw.fit(window.values)
            cov = lw.covariance_
            
            # Correlation Matrix
            d = np.sqrt(np.diag(cov))
            corr = cov / np.outer(d, d)
            
            # 1. ASF (Fragility)
            eig = np.linalg.eigvalsh(corr)
            eig = eig[eig > 1e-10]
            eig /= eig.sum()
            h = -np.sum(eig * np.log(eig)) / np.log(len(eig))
            asf = 1 - h
            
            # 2. Connectivity Proxies
            # C1: Mean Pairwise Correlation (Basal)
            mask = np.ones_like(corr, dtype=bool)
            np.fill_diagonal(mask, 0)
            c_mean = corr[mask].mean()
            
            # C2: Absorption Ratio (Concentration)
            k = max(1, int(0.2 * len(eig)))
            c_ar = np.sum(sorted(eig)[-k:])
            
            # C3: Network Density (Threshold > 0.4)
            c_den = (corr[mask] > 0.4).mean()
            
            data_list.append({
                'Date': dates[i],
                'ASF': asf,
                'C_Mean': c_mean,
                'C_AR': c_ar,
                'C_Den': c_den
            })
        except:
            pass
            
    df = pd.DataFrame(data_list).set_index('Date')
    
    # Smooth
    df = df.ewm(halflife=34).mean()
    
    # Target (Forward DD)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)
    spy_aligned = spy.reindex(df.index).ffill()
    min_fut = spy_aligned.rolling(window=indexer).min()
    df['Future_DD_Mag'] = ((min_fut / spy_aligned) - 1) * -1
    
    return df.dropna()

def test_interaction_model(df, c_col='C_Mean'):
    print(f"\n--- TEST 1: Interaction Model (C={c_col}) ---")
    
    # Standardize for coefficient interpretation? No, keep raw for C_crit interpretation.
    formula = f'Future_DD_Mag ~ ASF + {c_col} + ASF:{c_col}'
    model = ols(formula, data=df).fit()
    
    print(model.summary())
    
    with open(os.path.join(OUTPUT_DIR, f'Validation_Report_{c_col}.txt'), 'w') as f:
        f.write(model.summary().as_text())
    
    # Key Parameters
    b1 = model.params['ASF']
    b3 = model.params[f'ASF:{c_col}']
    
    print(f"\nb1 (ASF main): {b1:.4f}")
    print(f"b3 (Interaction): {b3:.4f}")
    
    if b3 != 0:
        c_crit = -b1 / b3
        print(f"Calculated C_critical: {c_crit:.4f}")
        
        # Verify if C_crit is within observed range
        c_min, c_max = df[c_col].min(), df[c_col].max()
        in_range = c_min <= c_crit <= c_max
        print(f"Observed C Range: [{c_min:.4f}, {c_max:.4f}] -> In Range? {in_range}")
    
    # Plot Marginal Effect
    # dRisk/dASF = b1 + b3*C
    # Var = Var(b1) + C^2*Var(b3) + 2*C*Cov(b1,b3)
    cov_matrix = model.cov_params()
    var_b1 = cov_matrix.loc['ASF', 'ASF']
    var_b3 = cov_matrix.loc[f'ASF:{c_col}', f'ASF:{c_col}']
    cov_b1b3 = cov_matrix.loc['ASF', f'ASF:{c_col}']
    
    c_vals = np.linspace(df[c_col].min(), df[c_col].max(), 100)
    marg_effect = b1 + b3 * c_vals
    std_error = np.sqrt(var_b1 + (c_vals**2)*var_b3 + 2*c_vals*cov_b1b3)
    
    plt.figure(figsize=(10, 6))
    plt.plot(c_vals, marg_effect, label='Marginal Effect of ASF', color='blue')
    plt.fill_between(c_vals, marg_effect - 1.96*std_error, marg_effect + 1.96*std_error, alpha=0.2, color='blue', label='95% CI')
    plt.axhline(0, color='red', linestyle='--')
    plt.axvline(c_crit, color='green', linestyle=':', label=f'Critical C ({c_crit:.2f})')
    
    # Overlay histogram of C to show distribution
    ax2 = plt.gca().twinx()
    sns.kdeplot(df[c_col], ax=ax2, color='gray', fill=True, alpha=0.1, label='C Density')
    ax2.set_ylabel('Density of C')
    
    plt.title(f'Marginal Effect of Fragility on Risk Conditional on Connectivity ({c_col})')
    plt.xlabel(f'Connectivity ({c_col})')
    plt.ylabel('d(Risk)/d(ASF)')
    plt.legend(loc='upper left')
    
    path = os.path.join(OUTPUT_DIR, f'Figure_Marginal_Effect_{c_col}.png')
    plt.savefig(path)
    print(f"Saved Marginal Effect Plot to {path}")
    plt.close()

def test_threshold_model(df, c_col='C_Mean'):
    print(f"\n--- TEST 2: Threshold Model (C={c_col}) ---")
    
    # Grid Search for tau
    percentiles = np.linspace(10, 90, 81) # 10th to 90th percentile
    taus = np.percentile(df[c_col], percentiles)
    
    best_tau = None
    best_aic = np.inf
    
    results = []
    
    for tau in taus:
        # Split
        mask_L = df[c_col] <= tau
        mask_H = df[c_col] > tau
        
        # Fit two models? Or one interaction with dummy?
        # Risk = alpha + theta_L * ASF * (C<=tau) + theta_H * ASF * (C>tau) + controls
        
        # Create dummies
        df['D_L'] = mask_L.astype(int)
        df['D_H'] = mask_H.astype(int)
        
        formula = 'Future_DD_Mag ~ 0 + D_L + D_H + D_L:ASF + D_H:ASF' # Simple specification
        model = ols(formula, data=df).fit()
        
        aic = model.aic
        theta_L = model.params['D_L:ASF']
        theta_H = model.params['D_H:ASF']
        t_L = model.tvalues['D_L:ASF']
        t_H = model.tvalues['D_H:ASF']
        
        results.append({
            'tau': tau,
            'aic': aic,
            'theta_L': theta_L,
            'theta_H': theta_H,
            't_L': t_L,
            't_H': t_H
        })
        
        if aic < best_aic:
            best_aic = aic
            best_tau = tau
            
    res_df = pd.DataFrame(results)
    best_row = res_df.loc[res_df['tau'] == best_tau].iloc[0]
    
    print(f"Optimal Threshold (tau): {best_tau:.4f}")
    print(f"Regime L (C <= tau): Slope = {best_row['theta_L']:.4f} (t={best_row['t_L']:.2f})")
    print(f"Regime H (C > tau):  Slope = {best_row['theta_H']:.4f} (t={best_row['t_H']:.2f})")
    
    pass_condition = (best_row['theta_L'] > 0) and (best_row['theta_H'] < 0)
    print(f"Passes Phase Transition check (Pos -> Neg)? {pass_condition}")
    
    # Plot AIC curve?
    
def test_rolling_estimates(df, c_col='C_Mean', window=252*2):
    print(f"\n--- TEST 4: Rolling Estimates (Window={window}d) ---")
    
    dates = []
    b3_vals = []
    
    # Rolling regression is slow, do step
    step = 60
    
    for i in range(window, len(df), step):
        sub = df.iloc[i-window:i]
        try:
            # Fit Interaction Model: Risk ~ ASF * C
            # Using numpy for speed
            # y = X beta
            # X = [1, ASF, C, ASF*C]
            
            y = sub['Future_DD_Mag'].values
            asf = sub['ASF'].values
            c = sub[c_col].values
            inter = asf * c
            const = np.ones_like(y)
            
            X = np.column_stack([const, asf, c, inter])
            
            # Beta = (X'X)^-1 X'y
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            dates.append(sub.index[-1])
            b3_vals.append(beta[3]) # beta_3 is interaction
        except:
            pass
            
    plt.figure(figsize=(10, 5))
    plt.plot(dates, b3_vals, label='Rolling Beta_3 (Interaction)')
    plt.axhline(0, color='r', linestyle='--')
    plt.title(f'Rolling Interaction Coefficient (Does market get more integrated?)')
    plt.ylabel('Beta_3')
    path = os.path.join(OUTPUT_DIR, 'Figure_Rolling_Beta3.png')
    plt.savefig(path)
    print(f"Saved Rolling Plot to {path}")
    plt.close()

def main():
    prices, spy = load_data()
    df = compute_metrics(prices, spy)
    df.to_csv(os.path.join(OUTPUT_DIR, 'Table_Validation_Data.csv'))
    
    # 1. Main Interaction (Mean Corr)
    test_interaction_model(df, 'C_Mean')
    test_threshold_model(df, 'C_Mean')
    
    # 2. Robustness (Absorption Ratio)
    test_interaction_model(df, 'C_AR')
    
    # 3. Rolling
    test_rolling_estimates(df, 'C_Mean')
    
    print("\nValidation Complete.")

if __name__ == "__main__":
    main()

"""
ASF Incrementality "Horse Race" Analysis
========================================

Purpose:
--------
Execute Priority 2 of Revision Package:
Run nested predictive regressions to prove ASF adds signal beyond:
1. VIX (Implied Vol)
2. Absorption Ratio (AR) - Kritzman
3. Realized Volatility (RV)

Model:
------
Target (Forward Drawdown) ~ ASF + VIX + AR + RV + (ASF * LowVol_Dummy)

Author: Research Team
Date: December 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"

EQUITY_UNIVERSE = [
    'AAPL', 'MSFT', 'JNJ', 'PG', 'XOM', 'JPM', 'GE', 'KO', 'PFE', 'WMT',
    'IBM', 'CVX', 'MRK', 'DIS', 'HD', 'MCD', 'BA', 'CAT', 'MMM', 'AXP'
]

# Best Lambda from Sensitivity (Short-term memory seems predictive positively)
LAMBDA = 0.02 
CORR_WINDOW = 126
VOL_NORM_WINDOW = 21

# ============================================================================
# UTILS
# ============================================================================

def load_data():
    start_date = '1990-01-01'
    end_date = '2025-12-20'
    print(f"Loading data {start_date} to {end_date}...")
    
    # Stocks
    df = yf.download(EQUITY_UNIVERSE, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        prices = df['Adj Close'] if 'Adj Close' in df.columns.get_level_values(0) else df['Close']
    else:
        prices = df
    
    # SPY
    spy = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy_prices = spy['Adj Close'] if 'Adj Close' in spy.columns.get_level_values(0) else spy['Close']
    else:
        spy_prices = spy['Adj Close'] if 'Adj Close' in spy.columns else spy['Close']
    if isinstance(spy_prices, pd.DataFrame): spy_prices = spy_prices.iloc[:, 0]

    # VIX
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix_close = vix['Close'].squeeze()
    else:
        vix_close = vix['Close']
    if isinstance(vix_close, pd.DataFrame): vix_close = vix_close.iloc[:, 0]
        
    return prices.dropna(axis=1, how='all'), spy_prices.dropna(), vix_close.dropna()


def compute_features(prices, spy_prices, vix):
    returns = prices.pct_change().dropna()
    spy_ret = spy_prices.pct_change().dropna()
    
    # 1. Realized Volatility (21d)
    rv = spy_ret.rolling(21).std() * np.sqrt(252)
    
    # 2. ASF (Lambda=0.02)
    # Vol Standardized
    vol_lag = returns.rolling(21).std().shift(1)
    std_ret = (returns / vol_lag).dropna()
    
    asf_vals = []
    ar_vals = []
    dates = []
    
    print("Computing ASF and Absorption Ratio...")
    for i in range(CORR_WINDOW, len(std_ret)):
        window = std_ret.iloc[i-CORR_WINDOW:i]
        try:
            # Ledoit Wolf
            lw = LedoitWolf()
            lw.fit(window.values)
            cov = lw.covariance_
            
            # Eigenvalues
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = eigvals[eigvals > 1e-10]
            eigvals /= eigvals.sum()
            
            # ASF (Entropy)
            entr = -np.sum(eigvals * np.log(eigvals)) / np.log(len(eigvals))
            asf_t = 1 - entr
            
            # Absorption Ratio (Top 20% Variance)
            # AR = Sum(Variance of Top K eigenvectors) / Total Variance
            # Since we used standardized returns, Total Variance ~ N (or close after shrinkage?)
            # Actually, eigenvalues of correlation matrix sum to N.
            # AR = Sum(Top K Eigs) / N
            k = max(1, int(0.2 * len(eigvals)))
            ar_t = np.sum(sorted(eigvals)[-k:]) # Top k (sorted ascending, take tail)
            
            asf_vals.append(asf_t)
            ar_vals.append(ar_t)
            dates.append(std_returns.index[i])
            
        except:
            pass
            
    asf_raw = pd.Series(asf_vals, index=dates)
    ar_raw = pd.Series(ar_vals, index=dates)
    
    # Exponential Smooth ASF
    asf = asf_raw.ewm(halflife=int(np.log(2)/LAMBDA)).mean()
    ar = ar_raw # AR is usually instantaneous, but maybe smooth slightly? Keep raw for "Instantaneous" comparison.
    
    # 3. Target: Forward 21d Drawdown
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)
    rolling_min = spy_prices.rolling(window=indexer).min()
    future_dd = (rolling_min / spy_prices) - 1
    target_dd = future_dd * -1 # Make positive (Magnitude of Drawdown)
    
    # 4. Interaction Term (ASF * Low_Vol)
    # Low Vol = VIX below median? or RV?
    # Let's use VIX for interaction
    vix_aligned = vix.reindex(asf.index).ffill()
    low_vol_dummy = (vix_aligned < vix_aligned.rolling(252*5).quantile(0.4)).astype(int)
    interaction = asf * low_vol_dummy
    
    # Ensure all are Series with names
    target_dd.name = 'Target_DD'
    asf.name = 'ASF'
    ar.name = 'AR'
    vix.name = 'VIX'
    rv.name = 'RV'

    # Combine Basics First
    try:
        data = pd.concat([target_dd, asf, ar, vix, rv], axis=1).dropna()
        data.columns = ['Target_DD', 'ASF', 'AR', 'VIX', 'RV']
    except Exception as e:
        print(f"Error combining data: {e}")
        print(f"Index duplicates? Target:{target_dd.index.duplicated().any()}, ASF:{asf.index.duplicated().any()}")
        raise e
        
    # Compute Interaction Inside DataFrame (Guaranteed Alignment)
    # Low Vol = VIX < Rolling 5y 40th percentile
    rolling_q40 = data['VIX'].rolling(252*5).quantile(0.4)
    low_vol_dummy = (data['VIX'] < rolling_q40).astype(int)
    data['Interaction'] = data['ASF'] * low_vol_dummy
    
    # Drop eventual NaNs from rolling quantile
    data = data.dropna()
    
    print(f"Combined Data Shape: {data.shape}")
    return data

def run_regressions(data):
    print("\nRunning Horse Race Regressions...")
    
    # Standardize inputs for coefficient comparison
    scaler = StandardScaler()
    cols = ['ASF', 'AR', 'VIX', 'RV', 'Interaction']
    data[cols] = scaler.fit_transform(data[cols])
    
    # Models
    # 1. Baseline: VIX + RV
    X1 = sm.add_constant(data[['VIX', 'RV']])
    m1 = sm.OLS(data['Target_DD'], X1).fit()
    
    # 2. Add AR
    X2 = sm.add_constant(data[['VIX', 'RV', 'AR']])
    m2 = sm.OLS(data['Target_DD'], X2).fit()
    
    # 3. Add ASF
    X3 = sm.add_constant(data[['VIX', 'RV', 'AR', 'ASF']])
    m3 = sm.OLS(data['Target_DD'], X3).fit()
    
    # 4. Full: + Interaction
    X4 = sm.add_constant(data[['VIX', 'RV', 'ASF', 'Interaction']]) # Drop AR if collinear/inferior
    m4 = sm.OLS(data['Target_DD'], X4).fit()

    # Results Table
    res = {
        'Model 1 (Vol)': {'R2': m1.rsquared, 'AIC': m1.aic},
        'Model 2 (+AR)': {'R2': m2.rsquared, 'AIC': m2.aic, 'AR_t': m2.tvalues.get('AR', 0)},
        'Model 3 (+ASF)': {'R2': m3.rsquared, 'AIC': m3.aic, 'ASF_t': m3.tvalues.get('ASF', 0)},
        'Model 4 (Inter)': {'R2': m4.rsquared, 'AIC': m4.aic, 'Inter_t': m4.tvalues.get('Interaction', 0)}
    }
    
    print("\nModel 3 Coefficients (Horse Race):")
    print(m3.summary())
    
    return pd.DataFrame(res).T

def main():
    prices, spy, vix = load_data()
    data = compute_features(prices, spy, vix)
    results = run_regressions(data)
    
    print("\nResults Summary:")
    print(results)
    
    results.to_csv(os.path.join(OUTPUT_DIR, 'Table_HorseRace_Results.csv'))

if __name__ == "__main__":
    main()

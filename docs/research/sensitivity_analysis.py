"""
ASF Sensitivity Analysis & Robustness Check
===========================================

Purpose:
--------
Execute Priority 1 of Revision Package:
1. Test sensitivity to lambda (decay parameter) across [0.001, 0.003, 0.005, 0.01, 0.02].
2. Compare covariance estimators (Ledoit-Wolf vs Standard vs OAS).
3. Evaluate stability of crisis prediction (AUC for forward drawdowns).

Metric:
-------
AUC-ROC for predicting "Major Crisis" (Next month drawdown < -10%)

Author: Research Team
Date: December 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf, OAS, EmpiricalCovariance
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"

# Universe (Long history stocks to capture 2000, 2008, 2020)
EQUITY_UNIVERSE = [
    'AAPL', 'MSFT', 'JNJ', 'PG', 'XOM', 'JPM', 'GE', 'KO', 'PFE', 'WMT',
    'IBM', 'CVX', 'MRK', 'DIS', 'HD', 'MCD', 'BA', 'CAT', 'MMM', 'AXP'
]

# Parameters to Test
LAMBDAS = [0.001, 0.003, 0.005, 0.01, 0.02]  # Half-lives: ~3y, 1y, 6m, 3m, 1.5m
ESTIMATORS = {
    'Ledoit-Wolf': LedoitWolf,
    'OAS': OAS,
    'Standard': EmpiricalCovariance
}

CORR_WINDOW = 126
VOL_NORM_WINDOW = 21

# Crisis Definition
CRISIS_THRESHOLD = -0.10  # Forward 21-day drawdown < -10%


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_data(tickers, start_date='1990-01-01', end_date='2025-12-20'):
    print(f"Loading data {start_date} to {end_date}...")
    df = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # Handle MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        prices = df['Adj Close'] if 'Adj Close' in df.columns.get_level_values(0) else df['Close']
    else:
        prices = df
    
    # SPY Benchmark
    spy = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy_prices = spy['Adj Close'] if 'Adj Close' in spy.columns.get_level_values(0) else spy['Close']
    else:
        spy_prices = spy['Adj Close'] if 'Adj Close' in spy.columns else spy['Close']
    
    if isinstance(spy_prices, pd.DataFrame):
        spy_prices = spy_prices.iloc[:, 0]
        
    return prices.dropna(axis=1, how='all'), spy_prices.dropna()


def volatility_standardize(returns, window=21):
    """Standardize with Shift(1) to avoid lookahead."""
    vol = returns.rolling(window).std().shift(1)
    return (returns / vol).dropna()


def calculate_entropy(corr_matrix):
    try:
        eigvals = np.linalg.eigvalsh(corr_matrix)
        eigvals = eigvals[eigvals > 1e-10]
        probs = eigvals / np.sum(eigvals)
        S = -np.sum(probs * np.log(probs))
        N = len(probs)
        return S / np.log(N) if N > 1 else 1.0
    except:
        return np.nan


def compute_asf(returns, decay_lambda, estimator_class):
    """Compute ASF for a specific lambda and estimator."""
    std_returns = volatility_standardize(returns, VOL_NORM_WINDOW)
    
    fragility = {}
    
    # Optimization: Use rolling windows more efficiently if possible
    # For now, standard loop for clarity and different estimators
    
    dates = std_returns.index
    n_days = len(dates)
    
    # Pre-loop initialization
    asf_val = 0.5  # Neutral start
    asf_series = []
    valid_dates = []
    
    print(f"  > Computing ASF (lambda={decay_lambda}, est={estimator_class.__name__})...")
    
    for i in range(CORR_WINDOW, n_days):
        window = std_returns.iloc[i-CORR_WINDOW:i]
        
        # Skip if too much missing data
        if window.shape[1] < 5: 
            continue
            
        try:
            # Estimate Covariance
            est = estimator_class()
            est.fit(window.values)
            cov = est.covariance_
            
            # Convert to Correlation
            d = np.sqrt(np.diag(cov))
            corr = cov / np.outer(d, d)
            
            # Entropy
            H_t = 1 - calculate_entropy(corr)
            
            # Accumulate
            asf_val = (1 - decay_lambda) * asf_val + decay_lambda * H_t
            
            asf_series.append(asf_val)
            valid_dates.append(dates[i])
            
        except Exception as e:
            continue
            
    return pd.Series(asf_series, index=valid_dates)


def create_target_variable(spy_prices, lookahead=21, threshold=CRISIS_THRESHOLD):
    """Binary target: 1 if Max Drawdown in next 21 days < threshold."""
    # Forward max drawdown
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=lookahead)
    rolling_min = spy_prices.rolling(window=indexer).min()
    
    # Drawdown from current price to future min
    future_dd = (rolling_min / spy_prices) - 1
    
    target = (future_dd < threshold).astype(int)
    return target


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("ASF SENSITIVITY ANALYSIS (ROBUSTNESS CHECK)")
    print("="*60)
    
    # 1. Load Data
    prices, spy_prices = load_data(EQUITY_UNIVERSE)
    returns = prices.pct_change().dropna()
    print(f"Data Loaded: {len(prices)} days, {len(prices.columns)} assets.")
    
    # 2. Create Target (Crisis Prediction)
    target = create_target_variable(spy_prices)
    print(f"Crisis Target Created: {target.sum()} crisis days identified ({(target.mean()*100):.1f}%).")
    
    results = []
    
    # 3. Main Loop
    for est_name, est_class in ESTIMATORS.items():
        for lam in LAMBDAS:
            asf = compute_asf(returns, lam, est_class)
            
            # Align
            common_idx = asf.index.intersection(target.index)
            if len(common_idx) < 100:
                print("    ! Not enough overlap data.")
                continue
                
            X = asf.loc[common_idx]
            y = target.loc[common_idx]
            
            # Shift ASF (it's a state variable at T, predicting T+1..T+21)
            # Already implied: ASF computed at T uses T-126..T data. Target is forward T..T+21.
            # So direct comparison is valid predictively.
            
            if len(y.unique()) > 1:
                auc = roc_auc_score(y, X)
                print(f"    -> AUC: {auc:.3f}")
                results.append({
                    'Estimator': est_name,
                    'Lambda': lam,
                    'HalfLife_Days': int(np.log(2)/lam),
                    'AUC': auc
                })
            else:
                print("    ! Target has no variance in overlap period.")
    
    # 4. Compile Results
    df_res = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(df_res.sort_values('AUC', ascending=False).to_string(index=False))
    
    # Save CSV
    df_res.to_csv(os.path.join(OUTPUT_DIR, 'Table_Sensitivity_Results.csv'), index=False)
    
    # 5. Plot Heatmap
    plt.figure(figsize=(10, 6))
    pivot = df_res.pivot(index='Estimator', columns='HalfLife_Days', values='AUC')
    sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
    plt.title('ASF Crisis Prediction Robustness (AUC Scores)')
    plt.xlabel('Half-Life (Days)')
    plt.ylabel('Covariance Estimator')
    
    fig_path = os.path.join(OUTPUT_DIR, 'Figure_Sensitivity_Heatmap.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved Heatmap to: {fig_path}")

if __name__ == "__main__":
    main()


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
DATA_PATH = "c:/key/wise_adviser_cursor_context/Caria_repo/caria/docs/research/media/Global_Macro_Prices_FMP.csv"
OUTPUT_DIR = "c:/key/wise_adviser_cursor_context/Caria_repo/caria/docs/research/outputs"
WINDOW = 63  # 3 months
LAMBDA = 0.995 # Proxy for ~139d half-life (simple EWMA for now or previous logic)
HORIZON = 21

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ---------------- UTILS ----------------
def compute_entropy(cov_matrix):
    """Computes Normalized Spectral Entropy."""
    eigvals = np.linalg.eigvalsh(cov_matrix)
    # Filter negatives (numerical noise)
    eigvals = eigvals[eigvals > 0]
    # Normalize to probability dist
    if len(eigvals) < 2: return 0
    weights = eigvals / np.sum(eigvals)
    # Entropy
    entropy = -np.sum(weights * np.log(weights))
    # Normalize by log(N)
    return entropy / np.log(len(eigvals))

def ledoit_wolf_estimation(returns):
    """Robust covariance estimation."""
    try:
        lw = LedoitWolf(assume_centered=False)
        cov = lw.fit(returns).covariance_
        return cov
    except:
        return returns.cov().values

# ---------------- MAIN ----------------
def main():
    print("Loading Global Macro Data...")
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    
    # Calculate Returns
    returns = df.pct_change().dropna()
    
    # Rolling Analysis
    results = []
    
    # We need a rolling loop. To be fast, step=5 days
    indices = range(WINDOW, len(returns)-HORIZON, 5)
    
    print(f"Processing {len(indices)} rolling windows...")
    
    for i in indices:
        window_ret = returns.iloc[i-WINDOW:i]
        
        # Require 80% data presence
        valid_cols = window_ret.columns[window_ret.notna().sum() > (WINDOW * 0.8)]
        if len(valid_cols) < 5: continue # Need minimum assets
        
        sub_ret = window_ret[valid_cols].fillna(0) # Fill remaining gaps with 0
        
        # Covariance & Correlation
        cov = ledoit_wolf_estimation(sub_ret)
        # Convert to Corr for Connectivity
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
        
        # Metrics
        entropy = compute_entropy(cov) # Use Cov eigenvalues for PCA consistency or Corr? 
        # Paper says Spectral Entropy of CORRELATION matrix
        entropy_corr = compute_entropy(corr)
        
        connectivity = np.mean(corr[np.triu_indices_from(corr, k=1)])
        
        # Future Risk: Equal Weight Portfolio Drawdown
        future_ret = returns[valid_cols].iloc[i:i+HORIZON] # Future returns of SAME assets
        portfolio_curve = (1 + future_ret.mean(axis=1)).cumprod()
        drawdown = (portfolio_curve.min() - 1) # Max loss in horizon
        
        results.append({
            'Date': returns.index[i],
            'Entropy': entropy_corr,
            'Fragility': 1 - entropy_corr,
            'Connectivity': connectivity,
            'Future_Drawdown': drawdown * -1 # Make positive risk metric
        })
        
    res_df = pd.DataFrame(results).set_index('Date')
    
    # Accumulation (ASF)
    # Simple EWMA of Fragility
    res_df['ASF'] = res_df['Fragility'].ewm(halflife=139).mean() # Match paper half-life
    
    # Remove NaNs
    res_df.dropna(inplace=True)
    
    print(f"Analzying {len(res_df)} data points...")
    
    # ---------------- THRESHOLD REGRESSION ----------------
    # Search for Threshold
    thresholds = np.percentile(res_df['Connectivity'], np.arange(10, 90, 5))
    best_tau = None
    best_ssr = np.inf
    best_model = None
    
    for tau in thresholds:
        # Split Data
        mask_low = res_df['Connectivity'] <= tau
        mask_high = res_df['Connectivity'] > tau
        
        # Fit Piecewise Model: Risk ~ ASF (Low) + ASF (High)
        # We also control for C itself
        
        # Construct Design Matrix
        # Intercepts
        X = pd.DataFrame(index=res_df.index)
        X['Low_Const'] = 1 * mask_low
        X['High_Const'] = 1 * mask_high
        
        # Slopes
        X['Low_Slope'] = res_df['ASF'] * mask_low
        X['High_Slope'] = res_df['ASF'] * mask_high
        
        # Controls
        X['Connectivity'] = res_df['Connectivity']
        
        y = res_df['Future_Drawdown']
        
        try:
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
            ssr = model.ssr
            if ssr < best_ssr:
                best_ssr = ssr
                best_tau = tau
                best_model = model
        except:
            continue
            
    print("\n" + "="*50)
    print("GLOBAL MULTI-ASSET PHASE TRANSITION RESULTS")
    print("="*50)
    print(f"Optimal Connectivity Threshold (Global): {best_tau:.4f}")
    
    print("\nModel Summary:")
    print(best_model.summary())
    
    # Save Results
    with open(os.path.join(OUTPUT_DIR, 'Global_Phase_Transition.txt'), 'w') as f:
        f.write("GLOBAL MACRO PHASE TRANSITION TEST\n")
        f.write(f"Universe: {len(df.columns)} Global Assets (Equities, Rates, Credit, Commodities)\n")
        f.write(f"Optimal Threshold: {best_tau:.4f}\n\n")
        f.write(best_model.summary().as_text())
        
    # Check Hypothesis
    # We expect Low_Slope > 0 (Contagion) and High_Slope < 0 (Disintegration)
    params = best_model.params
    print(f"\nRESULTS CHECK:")
    print(f"Contagion Regime Slope (Theta_L): {params['Low_Slope']:.4f}")
    print(f"Disintegration Regime Slope (Theta_H): {params['High_Slope']:.4f}")
    
    if params['Low_Slope'] > 0 and params['High_Slope'] < 0:
        print(">> HYPOTHESIS VALIDATED: Sign Flip Observed.")
    else:
        print(">> HYPOTHESIS MIXED/FAILED.")

if __name__ == "__main__":
    main()

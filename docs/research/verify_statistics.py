
import pandas as pd
import numpy as np
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data():
    """Load the pre-calculated theory data."""
    path = r'c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs\Table_Theory_Data.csv'
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # Rename Basal_Corr to Connectivity for consistency with code
        if 'Basal_Corr' in df.columns:
            df = df.rename(columns={'Basal_Corr': 'Connectivity'})
        return df.dropna()
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def bayesian_bootstrap_threshold(df, n_iter=1000):
    """
    Simulate a Bayesian Posterior for Tau using Weighted Likelihood Bootstrap.
    """
    print(f"Running Bayesian Bootstrap for Threshold (N={n_iter})...")
    
    taus = []
    # Grid of potential thresholds
    grid = np.linspace(df['Connectivity'].quantile(0.1), df['Connectivity'].quantile(0.9), 50)
    
    X = df[['ASF', 'Connectivity']].values
    y = df['Future_DD_Mag'].values
    n = len(y)
    
    for i in range(n_iter):
        if i % 100 == 0: print(f"Iter {i}...")
        
        # Dirichlet Weights (Bayesian Bootstrap)
        weights = np.random.dirichlet(np.ones(n), 1)[0] * n
        
        best_tau = None
        best_ssr = np.inf
        
        # Fast Grid Search
        # Pre-calculate weighted y to save time? No, weights change every iter.
        w_sqrt = np.sqrt(weights)
        y_w = y * w_sqrt
        
        for tau in grid:
            mask_L = X[:, 1] <= tau
            mask_H = X[:, 1] > tau
            
            asf_L = X[:, 0] * mask_L
            asf_H = X[:, 0] * mask_H
            const = np.ones(n)
            
            X_design = np.column_stack([const, asf_L, asf_H])
            X_w = X_design * w_sqrt[:, np.newaxis]
            
            try:
                beta = np.linalg.lstsq(X_w, y_w, rcond=None)[0]
                residuals = y_w - X_w @ beta
                ssr = np.sum(residuals**2)
                
                if ssr < best_ssr:
                    best_ssr = ssr
                    best_tau = tau
            except:
                continue
                
        if best_tau is not None:
            taus.append(best_tau)
            
    return np.array(taus)

def hysteresis_permutation_test(df, n_iter=1000):
    """
    Test if the 'Loop Area' is statistically significant.
    """
    print(f"Running Hysteresis Permutation Test (N={n_iter})...")
    
    # 1. Calculate Real Area on smoothed path
    df['ASF_Smooth'] = df['ASF'].rolling(26, center=True).mean()
    df['Risk_Smooth'] = df['Future_DD_Mag'].rolling(26, center=True).mean()
    path = df.dropna(subset=['ASF_Smooth', 'Risk_Smooth'])
    
    def calc_area(x, y):
        # simple polygon area (shoelace)
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    real_area = calc_area(path['ASF_Smooth'].values, path['Risk_Smooth'].values)
    print(f"Real Loop Area: {real_area:.6f}")
    
    # 2. Permutation
    null_areas = []
    real_risk = path['Risk_Smooth'].values
    
    for i in range(n_iter):
        # Cyclic shift to preserve time structure but destroy ASF-Risk alignment
        shift = np.random.randint(1, len(real_risk))
        shuffled_risk = np.roll(real_risk, shift)
        
        null_area = calc_area(path['ASF_Smooth'].values, shuffled_risk)
        null_areas.append(null_area)
        
    null_areas = np.array(null_areas)
    p_value = (null_areas >= real_area).mean()
    
    return real_area, null_areas, p_value

def main():
    df = load_data()
    if df is None: return
    
    # --- 1. Bayesian Threshold ---
    posterior_taus = bayesian_bootstrap_threshold(df, n_iter=200) # Reduced for speed in demo
    
    plt.figure(figsize=(10, 6))
    sns.histplot(posterior_taus, kde=True, color='purple', label='Posterior Density')
    
    mean_tau = np.mean(posterior_taus)
    ci_lower = np.percentile(posterior_taus, 2.5)
    ci_upper = np.percentile(posterior_taus, 97.5)
    
    plt.axvline(mean_tau, color='red', linestyle='--', label=f'Mean: {mean_tau:.3f}')
    plt.axvline(ci_lower, color='black', linestyle=':', label='95% CI')
    plt.axvline(ci_upper, color='black', linestyle=':')
    
    plt.title('Bayesian Posterior Distribution of Critical Threshold ($\\tau$)', fontsize=14)
    plt.xlabel('Connectivity Threshold', fontsize=12)
    plt.legend()
    # Save absolute path
    out_dir = r'c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, 'Figure_Bayesian_Threshold.png'))
    print(f"Bayesian Threshold: {mean_tau:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # --- 2. Hysteresis Test ---
    real_area, null_areas, p_val = hysteresis_permutation_test(df, n_iter=1000)
    
    print(f"Hysteresis Area: {real_area:.5f}")
    print(f"Null Area Mean: {null_areas.mean():.5f}")
    print(f"P-Value: {p_val:.5f}")
    
    if p_val < 0.05:
        print("RESULT: Hysteresis is Statistically Significant!")
    else:
        print("RESULT: Hysteresis is not significant.")

if __name__ == "__main__":
    main()

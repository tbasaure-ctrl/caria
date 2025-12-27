import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import sys
import traceback

# Configuration
DATA_DIR = "coodination_data"
SECTORS = {
    'FixedIncome': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'TIP', 'BND', 'AGG', 'MUB', 'MBB'],
    'Commodities': ['GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBB', 'DBC', 'PALL', 'PPLT']
}
LOOKBACK_WINDOW = 252 # 1 Year for Correlation
THETA = 0.995 # Persistence parameter

def load_sector_data(sector_name, symbols):
    print(f"Loading data for {sector_name}...")
    dfs = []
    for sym in symbols:
        path = os.path.join(DATA_DIR, f"{sym}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Standardize date column
            d_col = 'date' if 'date' in df.columns else 'Date'
            df[d_col] = pd.to_datetime(df[d_col])
            df = df.set_index(d_col).sort_index()
            # Standardize price column
            # Standardize price column
            target_col = None
            if sym in df.columns:
                target_col = sym
            elif 'adjClose' in df.columns:
                target_col = 'adjClose'
            elif 'close' in df.columns:
                target_col = 'close'
            elif len(df.columns) >= 2:
                # Fallback: assume 2nd column is price if not date
                target_col = df.columns[1] if df.columns[1] != d_col else df.columns[0]
            
            if target_col:
                series = df[target_col].rename(sym)
                dfs.append(series)
            else:
                print(f"Error: Could not identify price column for {sym}")
        else:
            print(f"Warning: Missing {sym}")
            
    if not dfs: return None, None
    
    # Merge and Align
    full_df = pd.concat(dfs, axis=1)
    # Forward fill to handle varying holidays/intervals, drop leading NaNs
    full_df = full_df.ffill().dropna()
    
    print(f"Loaded {len(full_df)} rows, {full_df.shape[1]} assets.")
    
    # Calculate Returns
    returns = full_df.pct_change()
    return full_df, returns

def compute_spectral_entropy(corr_matrix):
    # Eigenvalues
    evals, _ = np.linalg.eigh(corr_matrix)
    # Normalize (abs to handle slight precision negatives)
    evals = np.abs(evals)
    evals = evals / evals.sum()
    
    # Entropy
    # H = - (1/logN) * sum(p * log p)
    N = len(evals)
    # Avoid log(0)
    evals = evals[evals > 1e-10]
    H = -np.sum(evals * np.log(evals)) / np.log(N)
    return H

def build_sector_asf(returns):
    print("Computing Rolling Correlation and ASF (this may take a moment)...")
    dates = returns.index
    n_rows = len(returns)
    asf_series = []
    h_series = []
    
    # Initialize ASF
    current_asf = 0.5 
    
    # Need rolling window. 
    # Efficient: Rolling correlation is heavy.
    # We can iterate, or use pandas rolling?
    # Pandas rolling cov then corr is feasible.
    
    # Optimization: Only recompute every X days? 
    # Manuscript says "daily"? Let's do daily but simple loop for clarity.
    # Actually, pandas rolling().corr() returns a MultiIndex.
    
    # Let's use a stride if it's too slow, but 5000 rows is fine.
    # Actually, let's just do it step-by-step for the loop to apply recursive ASF.
    
    rolling_corr = returns.rolling(window=LOOKBACK_WINDOW).corr()
    
    # Loop through days
    # rolling_corr index is (Date, Asset). 
    # We need to grab the matrix for each date.
    
    unique_dates = returns.index[LOOKBACK_WINDOW:]
    
    # Pre-calculate Entropies?
    # Iterating the multi-index is slow.
    # Better: Loop distinct dates.
    
    for date in unique_dates:
        # Get matrix
        try:
            # Slicing the MultiIndex
            # This is the slow part. 
            # Alternative: don't precompute rolling. Compute on the fly.
            window_rets = returns.loc[:date].iloc[-LOOKBACK_WINDOW:]
            if len(window_rets) < LOOKBACK_WINDOW: 
                h_series.append(np.nan)
                continue
                
            corr = window_rets.corr()
            h = compute_spectral_entropy(corr.values)
            h_series.append(h)
            
            # Recursive ASF
            # ASF_t = theta * ASF_t-1 + (1-theta)*(1 - H_t)
            # Note: 1 - H_t is "Compression" or "Fragility"
            fragility_flow = 1 - h
            next_asf = THETA * current_asf + (1 - THETA) * fragility_flow
            asf_series.append(next_asf)
            current_asf = next_asf
            
        except Exception as e:
            h_series.append(np.nan)
            asf_series.append(np.nan)
            
    asf_df = pd.DataFrame({'Entropy': h_series, 'ASF': asf_series}, index=unique_dates)
    return asf_df

def run_regression(asf_df, sector_returns):
    # Align
    # Create Benchmark Return (Equal Weight of sector?) or use a major proxy (TLT/GLD)
    # Let's use Equal Weight of the sector constituents as the "Sector Index"
    
    sector_index = sector_returns.mean(axis=1).rename("Sector_Ret")
    
    data = asf_df.join(sector_index).dropna()
    
    # Calculate MaxDD Forward (Risk)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)
    rolling_min = data['Sector_Ret'].rolling(window=indexer).min() # This isn't quite maxdd
    # Drawdown of price path:
    price = (1 + data['Sector_Ret']).cumprod()
    # Forward Max DD
    # We need a loop or efficient implementation. 
    # Let's use the same func as before.
    
    fwd_maxdd = []
    for i in range(len(price) - 21):
        window = price.iloc[i:i+21]
        peak = window.cummax()
        dd = (window - peak) / peak
        fwd_maxdd.append(dd.min())
    fwd_maxdd = pd.Series(fwd_maxdd, index=price.index[:-21])
    
    data['Fwd_MaxDD'] = fwd_maxdd
    data = data.dropna()
    data['Fwd_MaxDD'] = -data['Fwd_MaxDD'] # Positive Magnitude
    
    # Regression
    X = sm.add_constant(data['ASF'])
    y = data['Fwd_MaxDD']
    
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    
    print(f"  Beta: {model.params['ASF']:.4f}")
    print(f"  T-Stat: {model.tvalues['ASF']:.4f}")
    print(f"  R2: {model.rsquared:.4f}")
    
    return model.tvalues['ASF']

def run_analysis():
    results = []
    
    for sector, symbols in SECTORS.items():
        print(f"\nAnalyzing Sector: {sector}")
        price_df, ret_df = load_sector_data(sector, symbols)
        if price_df is None: continue
        
        asf_df = build_sector_asf(ret_df)
        
        print(f"Regressing ASF vs {sector} Crash Risk...")
        t_stat = run_regression(asf_df, ret_df)
        
        results.append({'Sector': sector, 'ASF_T_Stat': t_stat})
        
    print("\n--- Final Universality Results ---")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    try:
        with open('sector_results.txt', 'w') as f:
            sys.stdout = f
            run_analysis()
            sys.stdout = sys.__stdout__
            
        print("Analysis complete. Saved to sector_results.txt")
        with open('sector_results.txt', 'r') as f:
            print(f.read())
    except Exception:
        sys.stdout = sys.__stdout__
        traceback.print_exc()

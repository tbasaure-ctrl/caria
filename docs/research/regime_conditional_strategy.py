"""
Regime-Conditional Risk Control (Hypothesis Falsification Test)
===============================================================

Hypothesis: 
Conditioning on Structural Regime (Connectivity) materially improves 
risk management (CVaR/MaxDD) relative to stationary rules (VIX Control).

Design:
1. Regimes: Defined by Median Connectivity (C). 
   - Low C (Contagion World)
   - High C (Integration World)
   
2. Signals (Regime-Dependent):
   - In Low C: Danger if ASF (Coupling) increases > 80th percentile.
   - In High C: Danger if ASF (Coupling) decreases < 20th percentile (Entropy Spike).
   
3. Rules:
   - Baseline: 100% SPY
   - Danger: 50% SPY (De-risk)
   - No Shorting, No Leverage.
   
4. Benchmark: Naive VIX Control
   - Danger if VIX > 80th percentile.
   
Author: Research Team
Date: December 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf
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
    
    df = yf.download(EQUITY_UNIVERSE, start=start, end=end, progress=False, auto_adjust=False)
    spy = yf.download('^GSPC', start=start, end=end, progress=False, auto_adjust=False)
    vix = yf.download('^VIX', start=start, end=end, progress=False, auto_adjust=False)
    
    with open(os.path.join(OUTPUT_DIR, 'Debug_Regime.txt'), 'w') as f:
        f.write(f"DF Columns: {df.columns}\n")
        f.write(f"DF Shape: {df.shape}\n")
    
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
    
    if isinstance(vix.columns, pd.MultiIndex):
        try: vix_c = vix.xs('Close', level=0, axis=1)
        except: vix_c = vix['Close']
    else: vix_c = vix['Close']
    if isinstance(vix_c, pd.DataFrame): vix_c = vix_c.iloc[:, 0]
    
    # Drop duplicates
    prices = prices[~prices.index.duplicated(keep='first')]
    spy_p = spy_p[~spy_p.index.duplicated(keep='first')]
    vix_c = vix_c[~vix_c.index.duplicated(keep='first')]
    
    # Strip Timezones
    if prices.index.tz is not None: prices.index = prices.index.tz_localize(None)
    if spy_p.index.tz is not None: spy_p.index = spy_p.index.tz_localize(None)
    if vix_c.index.tz is not None: vix_c.index = vix_c.index.tz_localize(None)
    
    return prices.dropna(axis=1, how='all'), spy_p.dropna(), vix_c.dropna()

def compute_signals(prices, spy, vix):
    print("Computing Regime State & Signals...")
    ret = prices.pct_change() # Keep NaNs for now
    
    # Vol Standardize
    vol = ret.rolling(21).std().shift(1)
    std_ret = (ret / vol).replace([np.inf, -np.inf], np.nan) # Handle Infs
    
    dates = std_ret.index
    data_list = []
    
    indices = range(CORR_WINDOW, len(std_ret), 21) # Monthly step to prevent timeout
    
    with open(os.path.join(OUTPUT_DIR, 'Debug_Loop.txt'), 'w') as f:
        f.write(f"Std Ret Shape: {std_ret.shape}\n")
        f.write(f"Indices: {list(indices)[:5]}...{list(indices)[-5:]}\n")
        
        count = 0
        for i in indices:
            window = std_ret.iloc[i-CORR_WINDOW:i]
            window_clean = window.dropna(axis=1, how='any')
            
            if count < 5:
                f.write(f"Idx {i}: Window Clean Shape {window_clean.shape}\n")
            
            if window_clean.shape[1] < 10:
                continue
                
            try:
                lw = LedoitWolf()
                lw.fit(window_clean.values)
                cov = lw.covariance_
                d = np.sqrt(np.diag(cov))
                corr = cov / np.outer(d, d)
                
                # ASF
                eig = np.linalg.eigvalsh(corr)
                eig = eig[eig > 1e-10]
                eig /= eig.sum()
                h = -np.sum(eig * np.log(eig)) / np.log(len(eig))
                asf = 1 - h
                
                # C (Mean Pairwise)
                mask = np.ones_like(corr, dtype=bool)
                np.fill_diagonal(mask, 0)
                c_mean = corr[mask].mean()
                
                data_list.append({'Date': dates[i], 'ASF': asf, 'C': c_mean})
                count += 1
            except Exception as e:
                f.write(f"Error at {i}: {e}\n")
                pass
            
    print(f"Data List Length: {len(data_list)}")
    if len(data_list) > 0:
        print(f"Sample Entry: {data_list[0]}")
            
    df = pd.DataFrame(data_list).set_index('Date')
    df = df.ewm(halflife=34).mean()
    
    # Align VIX and Returns
    df['VIX'] = vix.reindex(df.index).ffill()
    
    # Debug
    print(f"DF Head:\n{df.head()}")
    
    df['SPY_Ret'] = spy.pct_change().reindex(df.index).shift(-1) # Forward return 5d (since step=5)
    # Actually, let's use daily resolution for backtest logic, so reindex daily
    
    # Upsample to daily for simulation
    df_daily = df.reindex(spy.index).ffill().dropna()
    df_daily['SPY_Ret'] = spy.pct_change()
    
    # Calculate Percentiles (Rolling)
    window_roll = 252 * 5 # 5 year history for percentiles
    
    df_daily['ASF_80'] = df_daily['ASF'].rolling(window_roll).quantile(0.8)
    df_daily['ASF_20'] = df_daily['ASF'].rolling(window_roll).quantile(0.2)
    df_daily['VIX_80'] = df_daily['VIX'].rolling(window_roll).quantile(0.8)
    df_daily['C_Median'] = df_daily['C'].rolling(window_roll).median()
    
    return df_daily.dropna()

def run_strategy(df):
    print("Running Hypothesis Test Strategies...")
    
    # 1. Regime Identification
    # C_Median defines the regime
    # Low C: C < Median
    # High C: C >= Median
    
    is_low_c = df['C'] < df['C_Median']
    is_high_c = df['C'] >= df['C_Median']
    
    # 2. Signals
    # Low C (Contagion): Danger if ASF is HIGH (Spike in coupling)
    signal_low_c = (df['ASF'] > df['ASF_80']) & is_low_c
    
    # High C (Integration): Danger if ASF is LOW (Entropy spike / Coupling fracture)
    signal_high_c = (df['ASF'] < df['ASF_20']) & is_high_c
    
    # Combined Signal (Regime-Conditional)
    asf_danger = signal_low_c | signal_high_c
    
    # VIX Signal (Naive Benchmark)
    vix_danger = df['VIX'] > df['VIX_80']
    
    print(f"Low C Signals: {signal_low_c.sum()}")
    print(f"High C Signals: {signal_high_c.sum()}")
    print(f"Total Regime Signals: {asf_danger.sum()}")
    print(f"VIX Signals: {vix_danger.sum()}")
    
    with open(os.path.join(OUTPUT_DIR, 'Signal_Counts.txt'), 'w') as f:
        f.write(f"Low C Signals: {signal_low_c.sum()}\n")
        f.write(f"High C Signals: {signal_high_c.sum()}\n")
        f.write(f"Total Regime Signals: {asf_danger.sum()}\n")
        f.write(f"VIX Signals: {vix_danger.sum()}\n")
    
    # 3. Portfolios
    # Strategy 1: Buy & Hold
    # Strategy 2: VIX Control (50% risk off)
    # Strategy 3: Regime-Conditional (50% risk off)
    
    alloc_bh = pd.Series(1.0, index=df.index)
    
    alloc_vix = pd.Series(1.0, index=df.index)
    alloc_vix[vix_danger] = 0.5
    
    alloc_regime = pd.Series(1.0, index=df.index)
    alloc_regime[asf_danger] = 0.5
    
    # Returns
    # Note: SPY_Ret is t+1 return already? No, standard pct_change is (P_t / P_t-1) - 1.
    # We allocate at Close_t for Return_t+1. So shift allocation.
    
    strat_bh = (alloc_bh.shift(1) * df['SPY_Ret']).dropna()
    strat_vix = (alloc_vix.shift(1) * df['SPY_Ret']).dropna()
    strat_regime = (alloc_regime.shift(1) * df['SPY_Ret']).dropna()
    
    return strat_bh, strat_vix, strat_regime

def compute_stats(name, returns):
    # Cumulative
    cum = (1 + returns).cumprod()
    cagr = (cum.iloc[-1]**(252/len(cum))) - 1
    
    # Drawdown
    dd = (cum / cum.cummax()) - 1
    max_dd = dd.min()
    
    # CVaR (5%)
    cvar = returns[returns < returns.quantile(0.05)].mean()
    
    # Time Underwater (Fraction of days in DD < -5%)
    # Let's say underwater if DD < -0.01 for simplicity of metric
    time_under = (dd < -0.05).mean()
    
    return {
        'Strategy': name,
        'CAGR': cagr,
        'MaxDD': max_dd,
        'CVaR_5%': cvar,
        'Time_Under_5%': time_under
    }

def main():
    try:
        prices, spy, vix = load_data()
        df = compute_signals(prices, spy, vix)
        s_bh, s_vix, s_regime = run_strategy(df)
        
        # Stats
        stats = []
        stats.append(compute_stats("Buy & Hold", s_bh))
        stats.append(compute_stats("Naive VIX Control", s_vix))
        stats.append(compute_stats("Regime-Conditional (ASF)", s_regime))
        
        res_df = pd.DataFrame(stats)
        print("\n=== HYPOTHESIS TEST RESULTS ===")
        print(res_df.to_string(index=False))
        
        # Save
        res_df.to_csv(os.path.join(OUTPUT_DIR, 'Table_Strategy_Falsification.csv'), index=False)
        
        # Plot
        cum_df = pd.DataFrame({
            'Buy&Hold': (1+s_bh).cumprod(),
            'VIX_Control': (1+s_vix).cumprod(),
            'Regime_ASF': (1+s_regime).cumprod()
        })
        
        plt.figure(figsize=(10, 6))
        plt.plot(cum_df, linewidth=1.5)
        plt.legend(cum_df.columns)
        plt.title("Falsification Test: Regime-Conditional Risk Control vs Benchmarks")
        plt.yscale('log')
        plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_Strategy_Test.png'))
        
        plt.figure(figsize=(10, 4))
        dd_df = cum_df / cum_df.cummax() - 1
        plt.plot(dd_df, linewidth=1)
        plt.legend(dd_df.columns)
        plt.title("Drawdowns")
        plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_Strategy_DD.png'))
        
        print(f"\nSaved plots to {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

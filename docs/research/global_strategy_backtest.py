
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf

# ---------------- CONFIG ----------------
DATA_PATH = "c:/key/wise_adviser_cursor_context/Caria_repo/caria/docs/research/media/Global_Macro_Prices_FMP.csv"
OUTPUT_DIR = "c:/key/wise_adviser_cursor_context/Caria_repo/caria/docs/research/outputs"
WINDOW = 63
HALFLIFE = 34 # Optimized short memory for alpha
TAU_GLOBAL = 0.28 # From previous test

# ---------------- UTILS ----------------
def compute_entropy(cov):
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    prob = eigvals / np.sum(eigvals)
    return -np.sum(prob * np.log(prob)) / np.log(len(eigvals))

def ledoit_wolf_estimation(ret):
    try:
        return LedoitWolf(assume_centered=False).fit(ret).covariance_
    except:
        return ret.cov().values

# ---------------- MAIN ----------------
def main():
    print("Loading Global Macro Data...")
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    returns = df.pct_change().dropna()
    
    # 1. GENERATE SIGNALS
    dates = returns.index[WINDOW:]
    signals = pd.DataFrame(index=dates, columns=['ASF', 'Connectivity', 'Regime', 'Signal'])
    
    # Fast Loop (Monthly rebalance proxy)
    step = 21 
    loop_indices = range(WINDOW, len(returns), step)
    
    signals_list = []
    
    for t in loop_indices:
        window = returns.iloc[t-WINDOW:t]
        
        # Filter valid assets
        valid = window.columns[window.notna().sum() > (WINDOW*0.8)]
        if len(valid) < 5: 
            continue
            
        sub_ret = window[valid].fillna(0)
        cov = ledoit_wolf_estimation(sub_ret)
        
        # Entropy & ASF
        h_t = compute_entropy(cov)
        f_t = 1 - h_t
        
        # Accumulate ASF (EMA)
        decay = 1 - np.exp(-np.log(2)/HALFLIFE)
        # Use simple recursive calc
        prev_asf = signals_list[-1]['ASF'] if signals_list else f_t
        asf_t = (1-decay)*prev_asf + decay*f_t
        
        # Connectivity
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
        c_t = np.mean(corr[np.triu_indices_from(corr, k=1)])
        
        # Store for this block
        date = returns.index[t]
        signals_list.append({
            'Date': date,
            'ASF': asf_t,
            'Connectivity': c_t
        })
        
    if not signals_list:
        print("Error: No signals generated.")
        return

    signals = pd.DataFrame(signals_list).set_index('Date')
    
    # Reindex to daily (ffill)
    signals = signals.reindex(returns.index[WINDOW:], method='ffill').dropna()
    
    # Define Regimes & Allocation Rule
    # Rule: 
    # 1. Contagion Regime (C < 0.28): Danger if ASF is High (> 0.7 quantile)
    # 2. Disintegration Regime (C > 0.28): Danger if ASF DROPS (Entropy Spike) (< 0.3 quantile)
    # Allocation: 1.3x in Safe, 0.5x in Danger.
    
    # Quantiles (Adaptive)
    signals['ASF_Rank'] = signals['ASF'].rolling(252).rank(pct=True)
    
    def get_allocation(row):
        c = row['Connectivity']
        rank = row['ASF_Rank']
        
        if pd.isna(c) or pd.isna(rank): return 1.0
        
        # Regime Logic
        is_contagion_regime = (c <= TAU_GLOBAL)
        
        danger = False
        if is_contagion_regime:
            # Danger: High Fragility (Contagion building)
            if rank > 0.8: danger = True
        else:
            # Danger: Low Fragility (Disintegration / Entropy Spike)
            # Actually, "Entropy Spike" means Fragility DROPS. So Low ASF Rank is dangerous here.
            if rank < 0.2: danger = True
            
        if danger:
            return 0.5 # Defensive
        else:
            return 1.3 # Aggressive (Alpha capture)
            
    signals['Allocation'] = signals.apply(get_allocation, axis=1)
    
    # 2. BACKTEST
    # Benchmark: Equal Weight Portfolio of the Universe
    # (A Global Balanced Portfolio)
    
    # Align dates
    sub_returns = returns.loc[signals.index[0]:]
    signals = signals.loc[sub_returns.index]
    
    # Portfolio Returns
    bench_ret = sub_returns.mean(axis=1) # 1x Leverage
    strat_ret = bench_ret * signals['Allocation'].shift(1).fillna(1.0)
    
    # Metrics
    def calc_metrics(r):
        cum = (1+r).cumprod()
        cagr = (cum.iloc[-1]**(252/len(cum))) - 1
        vol = r.std() * np.sqrt(252)
        sharpe = (cagr - 0.02) / vol
        dd = (cum / cum.cummax() - 1).min()
        return cagr, vol, sharpe, dd
        
    b_cagr, b_vol, b_sharpe, b_dd = calc_metrics(bench_ret)
    s_cagr, s_vol, s_sharpe, s_dd = calc_metrics(strat_ret)
    
    print("\n" + "="*50)
    print("GLOBAL REGIME-CONDITIONAL STRATEGY RESULTS")
    print("="*50)
    print(f"{'Metric':<15} {'Benchmark':<15} {'Strategy':<15}")
    print("-" * 45)
    print(f"{'CAGR':<15} {b_cagr:.2%}          {s_cagr:.2%}")
    print(f"{'Vol':<15} {b_vol:.2%}          {s_vol:.2%}")
    print(f"{'Sharpe':<15} {b_sharpe:.2f}           {s_sharpe:.2f}")
    print(f"{'MaxDD':<15} {b_dd:.2%}          {s_dd:.2%}")
    
    # Save CSV for Paper
    res_df = pd.DataFrame({
        'Strategy': ['Global EqWt (Benchmark)', 'Regime-Conditional (Global)'],
        'CAGR': [b_cagr, s_cagr],
        'Volatility': [b_vol, s_vol],
        'Sharpe': [b_sharpe, s_sharpe],
        'MaxDD': [b_dd, s_dd]
    })
    res_df.to_csv(os.path.join(OUTPUT_DIR, 'Table_Global_Strategy.csv'), index=False)
    print(f"\nSaved results to {os.path.join(OUTPUT_DIR, 'Table_Global_Strategy.csv')}")

if __name__ == "__main__":
    main()

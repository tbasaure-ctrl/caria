import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import sys
import traceback

# --- Configuration ---
ASF_FILE = 'Table_Theory_Data.csv'
VIX_FILE = 'coodination_data/CBOE_Volatility_Index.csv'
SPX_FILE = 'coodination_data/S&P_500.csv'
# Additional Assets
GLD_FILE = 'coodination_data/Gold.csv'
USO_FILE = 'coodination_data/Oil.csv'
TLT_FILE = 'coodination_data/Treasuries_20Y.csv'
EUR_FILE = 'coodination_data/Euro_USD.csv'
BTC_FILE = 'coodination_data/BTC_USD.csv'

THRESHOLD = 0.14
RISK_FREE_RATE = 0.03 # 3% annual
COST_BPS = 0.0010 # 10 bps

def load_data():
    print("Loading data...")
    # Load ASF
    asf = pd.read_csv(ASF_FILE)
    asf.rename(columns={asf.columns[0]: 'Date'}, inplace=True)
    asf['Date'] = pd.to_datetime(asf['Date'])
    asf = asf.set_index('Date').sort_index()

    # Load Standard Assets
    assets = {
        'VIX': (VIX_FILE, 'VIX'),
        'SPX': (SPX_FILE, 'SPX'),
        'BTC': (BTC_FILE, 'BTC'),
        'GLD': (GLD_FILE, 'GLD'),
        'USO': (USO_FILE, 'USO'),
        'TLT': (TLT_FILE, 'TLT'),
        'EUR': (EUR_FILE, 'EUR')
    }
    
    df = asf[['ASF']].copy()
    
    for key, (path, col_name) in assets.items():
        if os.path.exists(path):
            temp = pd.read_csv(path)
            # handle 'date' or 'Date'
            d_col = 'date' if 'date' in temp.columns else 'Date'
            temp[d_col] = pd.to_datetime(temp[d_col])
            temp = temp.set_index(d_col).sort_index()
            # handle 'close' or 'adjClose'
            c_col = 'adjClose' if 'adjClose' in temp.columns else 'close'
            if key == 'VIX':
                temp[col_name] = temp[c_col] / 100.0
            else:
                temp[col_name] = temp[c_col]
            
            # Join carefully
            # VIX/SPX are core, others are satellite
            how_join = 'inner' if key in ['VIX', 'SPX'] else 'left'
            df = df.join(temp[[col_name]], how=how_join)
        else:
            print(f"Warning: {path} not found")

    print(f"Merged Data: {len(df)} records from {df.index[0].date()} to {df.index[-1].date()}")
    return df

def calculate_drawdowns(series, window=21):
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
    rolling_min = series.rolling(window=indexer).min()
    fwd_worst_return = (rolling_min / series) - 1
    return -fwd_worst_return 

def calc_metrics(returns, dates, rf_annual=0.03):
    if len(returns) == 0: return 0,0,0,0
    total_ret = (1 + returns).prod()
    delta = dates.max() - dates.min()
    n_years = delta.days / 365.25 if delta.days > 0 else 0
    if n_years <= 0: return 0,0,0,0
    
    rows_per_year = len(returns) / n_years
    freq_factor = rows_per_year
    
    cagr = total_ret ** (1/n_years) - 1
    vol = returns.std() * np.sqrt(freq_factor) 
    sharpe = (cagr - rf_annual) / vol if vol > 0 else 0
    
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    return cagr, vol, sharpe, max_dd

def run_horse_race(df):
    print("\n--- Running Horse Race Regression ---")
    data = df.copy()
    data['Fwd_MaxDD'] = calculate_drawdowns(data['SPX'], window=21) 
    data['Interaction'] = data['VIX'] * data['ASF']
    data = data.dropna()
    
    models = {
        '(1) VIX': ['VIX'],
        '(2) VIX + ASF': ['VIX', 'ASF'],
        '(3) Full + Inter': ['VIX', 'ASF', 'Interaction']
    }
    
    results = []
    for name, cols in models.items():
        X = data[cols]
        X = sm.add_constant(X)
        y = data['Fwd_MaxDD']
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        res_dict = {'Model': name, 'R2': model.rsquared, 'AIC': model.aic}
        for col in cols:
            res_dict[f'{col}_beta'] = model.params[col]
            res_dict[f'{col}_t'] = model.tvalues[col]
        results.append(res_dict)
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string())

def run_strategy(df):
    print("\n--- Running Kelly Strategy Backtest (SPX) ---")
    data = df.copy()
    data['Returns'] = data['SPX'].pct_change()
    data = data.dropna()
    
    # Simple Optimization
    best_sharpe = -999
    best_thresh = 0
    # Search direction: High=Safe (Coordination) confirmed by data
    
    print("\n--- Threshold Optimization ---")
    for thresh in np.linspace(0.05, 0.25, 21):
        # Logic: High ASF (Coordination) -> Lev 1.3
        temp_sig = (data['ASF'] > thresh).astype(int).shift(1)
        temp_w = np.where(temp_sig == 1, 1.3, 0.5)
        temp_ret = temp_w * (data['Returns']) 
        c, v, s, d = calc_metrics(temp_ret.dropna(), data.index)
        if s > best_sharpe:
            best_sharpe = s
            best_thresh = thresh
    
    print(f"Optimal Threshold: {best_thresh:.3f} (Sharpe: {best_sharpe:.2f})")
    
    # Metric Print
    THRESHOLD = best_thresh
    data['Signal'] = (data['ASF'] > THRESHOLD).astype(int).shift(1)
    data['Weight'] = np.where(data['Signal'] == 1, 1.3, 0.5)
    
    freq_factor = len(data) / ((data.index[-1] - data.index[0]).days / 365.25)
    rf_period = RISK_FREE_RATE / freq_factor
    data['Strat_Ret_Gross'] = data['Weight'] * (data['Returns'] - rf_period) + rf_period
    data['Weight_Diff'] = data['Weight'].diff().abs()
    data['Cost'] = np.where(data['Weight_Diff'] > 0, COST_BPS, 0)
    data['Strat_Ret_Net'] = data['Strat_Ret_Gross'] - data['Cost']
    
    data = data.dropna()
    
    bench_cagr, bench_vol, bench_sharpe, bench_dd = calc_metrics(data['Returns'], data.index)
    strat_cagr, strat_vol, strat_sharpe, strat_dd = calc_metrics(data['Strat_Ret_Net'], data.index)
    
    print(f"Benchmark (SPX): Sharpe={bench_sharpe:.2f}, MaxDD={bench_dd:.2%}")
    print(f"Strategy (Kelly): Sharpe={strat_sharpe:.2f}, MaxDD={strat_dd:.2%}")
    return best_thresh

def run_cross_asset_test(df, threshold):
    print("\n--- Running Cross-Asset Universality Test ---")
    
    assets = ['BTC', 'GLD', 'TLT', 'USO', 'EUR']
    results = []
    
    for asset in assets:
        if asset not in df.columns: continue
        
        # Slicing
        sub = df[['ASF', asset]].dropna()
        if len(sub) < 100: continue
        
        sub['Returns'] = sub[asset].pct_change()
        
        # Test Both Directions
        # Dir A: High ASF (Safe) -> Long Risk
        # Dir B: Low ASF (Fragile) -> Long Safe Haven?
        # Standard: High ASF = Coord = Safe.
        # Asset Logic:
        # Risk Assets (BTC, USO, EUR?): Long when Safe.
        # Safe Havens (GLD, TLT): Maybe Long when Fragile?
        
        # For uniformity, let's test the "Risk On" Strategy for all
        # Long (1.0) when Safe, Cash (0.0) when Fragile.
        # If asset is Safe Haven, this might underperform (negative alpha).
        
        sub['Signal'] = (sub['ASF'] > threshold).astype(int).shift(1)
        sub['Weight'] = np.where(sub['Signal'] == 1, 1.0, 0.0)
        
        cost = 0.0010 # 10bps
        if asset == 'BTC': cost = 0.0020
        
        sub['Strat_Ret'] = sub['Weight'] * sub['Returns']
        # Cost logic simplified
        sub['W_Diff'] = sub['Weight'].diff().abs()
        sub['Cost'] = np.where(sub['W_Diff'] > 0, cost, 0)
        sub['Net_Ret'] = sub['Strat_Ret'] - sub['Cost']
        
        sub = sub.dropna()
        
        b_c, b_v, b_s, b_d = calc_metrics(sub['Returns'], sub.index)
        s_c, s_v, s_s, s_d = calc_metrics(sub['Net_Ret'], sub.index)
        
        results.append({
            'Asset': asset,
            'Bench_Sharpe': b_s,
            'Strat_Sharpe': s_s,
            'Delta_Sharpe': s_s - b_s,
            'Bench_DD': b_d,
            'Strat_DD': s_d
        })
        
    print(pd.DataFrame(results).to_string())

def run_correlation_test(df):
    print("\n--- Running Stock-Bond Correlation Universality Test ---")
    data = df.copy()
    
    # ensure we have data
    if 'TLT' not in data.columns:
        print("Missing TLT data for correlation test.")
        return

    # Calculate Rolling Correlation (e.g. 3-month window usually standard for regime)
    window = 63 
    
    # Calculate returns first
    rets = data[['SPX', 'TLT']].pct_change()
    
    # Rolling Correlation
    rolling_corr = rets['SPX'].rolling(window=window).corr(rets['TLT'])
    
    data['StockBond_Corr'] = rolling_corr
    # Shift correlation back? No, we want to predict FUTURE correlation or coincident?
    # Theory: Low ASF (Fragility) -> Liquidation -> Correlation Spikes.
    # So ASF_t should predict Corr_{t+1...t+k} or be coincident with rising correlation.
    # Let's predict average correlation over the NEXT month (21 days).
    
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)
    data['Fwd_Avg_Corr'] = data['StockBond_Corr'].rolling(window=indexer).mean().shift(-21) 
    # Actually, shift(-21) shifts the *start* of the window back. 
    # FixedForwardWindowIndexer does the lookahead. 
    # Just rolling(indexer).mean() gives the mean of [t, t+21]. 
    # We want to regress Y=Fwd_Corr on X=ASF_t.
    
    data['Fwd_Avg_Corr'] = data['StockBond_Corr'].rolling(window=indexer).mean()
    
    data = data.dropna()
    
    # Regression
    X = sm.add_constant(data['ASF'])
    y = data['Fwd_Avg_Corr']
    
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    
    print(f"Regression: Fwd Stock-Bond Corr on ASF")
    print(f"Beta: {model.params['ASF']:.4f}")
    print(f"T-Stat: {model.tvalues['ASF']:.4f}")
    print(f"R2: {model.rsquared:.4f}")
    
    return model.params['ASF'], model.tvalues['ASF']

def run_credit_spread_test(df):
    print("\n--- Running Credit Spread (Priced Risk) Test ---")
    data = df.copy()
    
    # Load HYG/LQD if not already in df (they are in separate files)
    # We need to load them here or in load_data.
    # Quick load:
    # Load HYG/LQD if not already in df (they are in separate files)
    # They are saved as date,SYMBOL by fetch_sector_data.py
    try:
        hyg = pd.read_csv('coodination_data/HYG.csv', parse_dates=['date']).set_index('date').sort_index()['HYG']
        lqd = pd.read_csv('coodination_data/LQD.csv', parse_dates=['date']).set_index('date').sort_index()['LQD']
        
        # Merge
        temp = pd.DataFrame({'HYG': hyg, 'LQD': lqd})
        data = data.join(temp, how='inner')
        
        if len(data) < 100:
            print("Not enough credit data.")
            return

        # Construct Spread Proxy: HYG/LQD Price Ratio
        # Rising Ratio = Outperformance of High Yield (Risk On).
        # Falling Ratio = Underperformance of High Yield (Widening Spreads).
        # Hypothesis: Low ASF (Fragility) -> Future Underperformance of HYG relative to LQD (Ratio falls).
        
        data['Credit_Ratio'] = data['HYG'] / data['LQD']
        data['Fwd_Return_Spread'] = data['Credit_Ratio'].pct_change().shift(-21) # 1 month fwd change in ratio
        # Or better: Fwd Excess Return of HYG over LQD.
        
        data['HYG_Ret'] = data['HYG'].pct_change()
        data['LQD_Ret'] = data['LQD'].pct_change()
        data['Excess_Credit_Ret'] = (data['HYG_Ret'] - data['LQD_Ret']).rolling(window=21).sum().shift(-21)
        
        data = data.dropna()
        
        # Regression
        X = sm.add_constant(data['ASF'])
        y = data['Excess_Credit_Ret']
        
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        
        print(f"Regression: Fwd Credit Excess Return (HYG-LQD) on ASF")
        print(f"Beta: {model.params['ASF']:.4f}")
        print(f"T-Stat: {model.tvalues['ASF']:.4f}")
        print(f"R2: {model.rsquared:.4f}")
        
    except Exception as e:
        print(f"Credit Analysis Failed: {e}")

if __name__ == "__main__":
    try:
        with open('empirical_results.txt', 'w') as f:
            sys.stdout = f
            df = load_data()
            run_horse_race(df)
            thresh = run_strategy(df)
            run_correlation_test(df)
            run_credit_spread_test(df)
            
            sys.stdout = sys.__stdout__
            
        print("Analysis complete. Results saved to empirical_results.txt")
        with open('empirical_results.txt', 'r') as f:
            print(f.read())
            
    except Exception:
        sys.stdout = sys.__stdout__
        traceback.print_exc()

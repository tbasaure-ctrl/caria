"""
Simple Horse Race Analysis
==========================
Robust version using pd.merge to avoid alignment errors.

Model: Target ~ ASF + VIX + AR + RV + Inter
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from functools import reduce
import os

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
    vix = yf.download('^VIX', start=start, end=end, progress=False)
    
    # Drop duplicates
    if isinstance(df.index, pd.DatetimeIndex): df = df[~df.index.duplicated(keep='first')]
    if isinstance(spy.index, pd.DatetimeIndex): spy = spy[~spy.index.duplicated(keep='first')]
    if isinstance(vix.index, pd.DatetimeIndex): vix = vix[~vix.index.duplicated(keep='first')]
    
    # Robust column selection
    if isinstance(df.columns, pd.MultiIndex):
        try:
            prices = df.xs('Adj Close', level=0, axis=1)
        except KeyError:
            prices = df.xs('Close', level=0, axis=1)
    else:
        prices = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
    
    # SPY
    if isinstance(spy.columns, pd.MultiIndex):
        spy_p = spy.xs('Adj Close', level=0, axis=1) if 'Adj Close' in spy.columns.get_level_values(0) else spy.xs('Close', level=0, axis=1)
    else:
        spy_p = spy['Adj Close'] if 'Adj Close' in spy.columns else spy['Close']
        
    # VIX
    if isinstance(vix.columns, pd.MultiIndex):
        vix_c = vix.xs('Close', level=0, axis=1)
    else:
        vix_c = vix['Close']
    
    # Force single level series and drop timezones if any
    if isinstance(spy_p, pd.DataFrame): spy_p = spy_p.iloc[:, 0]
    if isinstance(vix_c, pd.DataFrame): vix_c = vix_c.iloc[:, 0]
    
    spy_p.index = spy_p.index.tz_localize(None)
    vix_c.index = vix_c.index.tz_localize(None)
    prices.index = prices.index.tz_localize(None)
    
    return prices.dropna(axis=1, how='all'), spy_p.dropna(), vix_c.dropna()

def compute_metrics(prices, spy_p, vix_c):
    ret = prices.pct_change().dropna()
    spy_ret = spy_p.pct_change().dropna()
    
    # 1. RV
    rv = (spy_ret.rolling(21).std() * np.sqrt(252)).dropna()
    rv.name = 'RV'
    
    # 2. ASF & AR
    vol_lag = ret.rolling(21).std().shift(1)
    std_ret = (ret / vol_lag).dropna()
    
    asf_dict = {}
    ar_dict = {}
    
    print("Computing metrics...")
    indices = range(CORR_WINDOW, len(std_ret))
    dates = std_ret.index
    
    for i in indices:
        window = std_ret.iloc[i-CORR_WINDOW:i]
        try:
            lw = LedoitWolf()
            lw.fit(window.values)
            cov = lw.covariance_
            
            eig = np.linalg.eigvalsh(cov)
            eig = eig[eig > 1e-10]
            eig /= eig.sum()
            
            # ASF (Entropy)
            h = -np.sum(eig * np.log(eig)) / np.log(len(eig))
            asf_dict[dates[i]] = 1 - h
            
            # AR (Top 20%)
            k = max(1, int(0.2 * len(eig)))
            ar_dict[dates[i]] = np.sum(sorted(eig)[-k:])
        except:
            pass
            
    asf = pd.Series(asf_dict).ewm(halflife=int(np.log(2)/LAMBDA)).mean()
    asf.name = 'ASF'
    
    ar = pd.Series(ar_dict)
    ar.name = 'AR'
    
    # 3. Target (Forward DD)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)
    # Ensure SPY matches timeframe
    spy_aligned = spy_p.reindex(asf.index).ffill()
    min_fut = spy_aligned.rolling(window=indexer).min()
    dd = (min_fut / spy_aligned) - 1
    target = dd * -1 # Magnitude
    target.name = 'Target_DD'
    
    # 4. VIX
    vix_s = vix_c.reindex(asf.index).ffill()
    vix_s.name = 'VIX'
    
    # Merge
    dfs = [target, asf, ar, vix_s, rv]
    data = reduce(lambda left,right: pd.merge(left,right,left_index=True,right_index=True, how='inner'), dfs)
    
    # Interaction
    q40 = data['VIX'].rolling(252*5).quantile(0.4)
    data['Interaction'] = data['ASF'] * (data['VIX'] < q40).astype(int)
    
    return data.dropna()

def main():
    try:
        prices, spy, vix = load_data()
        data = compute_metrics(prices, spy, vix)
        
        print(f"Data Shape: {data.shape}")
        
        # Standardize
        scaler = StandardScaler()
        cols = ['ASF', 'AR', 'VIX', 'RV', 'Interaction']
        data[cols] = scaler.fit_transform(data[cols])
        
        # Regressions
        res = {}
        for name, feats in [
            ('Model 1 (Vol)', ['VIX', 'RV']),
            ('Model 2 (+AR)', ['VIX', 'RV', 'AR']),
            ('Model 3 (+ASF)', ['VIX', 'RV', 'AR', 'ASF']),
            ('Model 4 (Inter)', ['VIX', 'RV', 'ASF', 'Interaction'])
        ]:
            X = sm.add_constant(data[feats])
            m = sm.OLS(data['Target_DD'], X).fit()
            res[name] = {'R2': m.rsquared, 'AIC': m.aic, 'Params': m.params.to_dict(), 'T-stats': m.tvalues.to_dict()}
            
            if name == 'Model 3 (+ASF)':
                print("\nModel 3 Summary:")
                print(m.summary())
            if name == 'Model 4 (Inter)':
                print("\nModel 4 Summary:")
                print(m.summary())

        df_res = pd.DataFrame(res).T
        print("\nOverview:")
        print(df_res[['R2', 'AIC']])
        
        df_res.to_csv(os.path.join(OUTPUT_DIR, 'Table_SimpleHorseRace.csv'))
        
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    main()

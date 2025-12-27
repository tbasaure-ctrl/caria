"""
Regime Coherence Test
=====================
Check AUC performance by decade to see if 1990-2000 noise is dragging down results.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

EQUITY_UNIVERSE = [
    'AAPL', 'MSFT', 'JNJ', 'PG', 'XOM', 'JPM', 'GE', 'KO', 'PFE', 'WMT',
    'IBM', 'CVX', 'MRK', 'DIS', 'HD', 'MCD', 'BA', 'CAT', 'MMM', 'AXP'
]
LAMBDA = 0.02
CRISIS_THRESHOLD = -0.10

def load_data():
    start = '1990-01-01'
    end = '2025-12-20'
    
    df = yf.download(EQUITY_UNIVERSE, start=start, end=end, progress=False)
    # Robust column selection
    if isinstance(df.columns, pd.MultiIndex):
        try:
            prices = df.xs('Adj Close', level=0, axis=1)
        except KeyError:
            prices = df.xs('Close', level=0, axis=1)
    else:
        prices = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
        
    spy = yf.download('^GSPC', start=start, end=end, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        try:
            spy_p = spy.xs('Adj Close', level=0, axis=1)
        except KeyError:
            spy_p = spy.xs('Close', level=0, axis=1)
    else:
        spy_p = spy['Adj Close'] if 'Adj Close' in spy.columns else spy['Close']
        
    if isinstance(spy_p, pd.DataFrame): spy_p = spy_p.iloc[:, 0]
    
    return prices.dropna(axis=1, how='all'), spy_p.dropna()

def main():
    print("Loading Data...")
    prices, spy = load_data()
    
    # Compute ASF
    ret = prices.pct_change().dropna()
    vol = ret.rolling(21).std().shift(1)
    std_ret = (ret / vol).dropna()
    
    dates = std_ret.index
    asf_vals = []
    valid_dates = []
    
    print("Computing ASF...")
    for i in range(126, len(std_ret), 5): # Step 5 for speed
        window = std_ret.iloc[i-126:i]
        try:
            lw = LedoitWolf()
            lw.fit(window.values)
            cov = lw.covariance_
            eig = np.linalg.eigvalsh(cov)
            eig = eig[eig > 1e-10]
            eig /= eig.sum()
            h = -np.sum(eig * np.log(eig)) / np.log(len(eig))
            asf_vals.append(1-h)
            valid_dates.append(dates[i])
        except:
            pass
            
    asf = pd.Series(asf_vals, index=valid_dates).ewm(halflife=int(np.log(2)/LAMBDA)).mean()
    
    # Target
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)
    spy_aligned = spy.reindex(asf.index).ffill()
    fut_min = spy_aligned.rolling(window=indexer).min()
    target_dd = (fut_min / spy_aligned) - 1
    target = (target_dd < CRISIS_THRESHOLD).astype(int)
    
    df = pd.concat([asf, target], axis=1).dropna()
    df.columns = ['ASF', 'Crisis']
    
    print("\nAUC by Era:")
    eras = [
        ('1990-1999', '1990-01-01', '1999-12-31'),
        ('2000-2009', '2000-01-01', '2009-12-31'),
        ('2010-2019', '2010-01-01', '2019-12-31'),
        ('2020-2025', '2020-01-01', '2025-12-31'),
        ('Post-2000', '2000-01-01', '2025-12-31')
    ]
    
    for name, start, end in eras:
        sub = df.loc[start:end]
        if len(sub) > 100 and sub['Crisis'].nunique() > 1:
            auc = roc_auc_score(sub['Crisis'], sub['ASF'])
            print(f"{name}: AUC = {auc:.3f} (N={len(sub)}, CrisisDays={sub['Crisis'].sum()})")
        else:
            print(f"{name}: Insufficient data or variance")

if __name__ == "__main__":
    main()

"""
Proving the Regime Inversion Theory
===================================

Hypothesis: The relationship between Structural Fragility (Coupling) and Risk 
is regime-dependent, governed by the "Basal Connectivity" of the market (Passive/ETF dominance).

Theory Steps:
1. Calculate ASF (Coupling State) daily.
2. Calculate "Basal Correlation" (Median pairwise correlation) as proxy for ETF dominance.
3. Test 3 Models:
   - Model A: Regime Split (Pre vs Post 2000).
   - Model B: Interaction (Risk ~ ASF + ASF * Basal_Corr).
   - Model C: Non-Linear "Goldilocks" (Risk ~ ASF + ASF^2).

Goal: Find the "Measurable General Rule" that explains why AUC flips from 0.70 to 0.37.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf
import statsmodels.api as sm
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
CRISIS_THRESHOLD = -0.05 # Sensitive threshold to get enough signal points

def load_data():
    start = '1990-01-01'
    end = '2025-12-20'
    print(f"Loading {start} to {end}...")
    
    df = yf.download(EQUITY_UNIVERSE, start=start, end=end, progress=False)
    
    # Robust Selection
    if isinstance(df.columns, pd.MultiIndex):
        try:
            prices = df.xs('Adj Close', level=0, axis=1)
        except KeyError:
            prices = df.xs('Close', level=0, axis=1)
    else:
        prices = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
        
    spy = yf.download('^GSPC', start=start, end=end, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy_p = spy.xs('Adj Close', level=0, axis=1) if 'Adj Close' in spy.columns.get_level_values(0) else spy.xs('Close', level=0, axis=1)
    else:
        spy_p = spy['Adj Close'] if 'Adj Close' in spy.columns else spy['Close']
    if isinstance(spy_p, pd.DataFrame): spy_p = spy_p.iloc[:, 0]
    
    return prices.dropna(axis=1, how='all'), spy_p.dropna()

def compute_regime_metrics(prices, spy):
    print("Computing metrics...")
    ret = prices.pct_change().dropna()
    spy_ret = spy.pct_change().dropna()
    
    # Vol Standardize
    vol = ret.rolling(21).std().shift(1)
    std_ret = (ret / vol).dropna()
    
    dates = std_ret.index
    asf_vals = []
    med_corr_vals = [] # Proxy for "Basal Connectivity"
    valid_dates = []
    
    # Step 5 for speed
    indices = range(126, len(std_ret), 5)
    
    for i in indices:
        window = std_ret.iloc[i-126:i]
        try:
            # 1. Ledoit Wolf (State)
            lw = LedoitWolf()
            lw.fit(window.values)
            cov = lw.covariance_
            
            # Corr
            d = np.sqrt(np.diag(cov))
            corr = cov / np.outer(d, d)
            
            # Measures
            # A. ASF (Coupling/Low Entropy) - Our State Variable
            eig = np.linalg.eigvalsh(corr)
            eig = eig[eig > 1e-10]
            eig /= eig.sum()
            h = -np.sum(eig * np.log(eig)) / np.log(len(eig))
            asf_vals.append(1-h) 
            
            # B. Median Correlation - Our "Regime" Variable
            # Get off-diagonal elements
            off_diag = corr[np.triu_indices_from(corr, k=1)]
            med_corr_vals.append(np.median(off_diag))
            
            valid_dates.append(dates[i])
        except:
            pass
            
    # Create DF
    data = pd.DataFrame({
        'ASF': asf_vals,
        'Basal_Corr': med_corr_vals
    }, index=valid_dates)
    
    # Smooth
    data['ASF'] = data['ASF'].ewm(halflife=34).mean()
    data['Basal_Corr'] = data['Basal_Corr'].rolling(126).mean() # Smoother regime proxy
    
    # Target (Forward DD)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)
    spy_aligned = spy.reindex(data.index).ffill()
    min_fut = spy_aligned.rolling(window=indexer).min()
    data['Future_DD_Mag'] = ((min_fut / spy_aligned) - 1) * -1
    
    return data.dropna()

def analyze_theory(data):
    print("\nRunning Theoretical Tests...")
    
    # 1. Visual Proof (Scatter with Interactions)
    plt.figure(figsize=(10, 6))
    
    # Define Eras
    data['Era'] = np.where(data.index.year < 2000, '1990-1999 (Pre-ETF)', '2000-2025 (ETF Era)')
    
    sns.lmplot(data=data, x='ASF', y='Future_DD_Mag', hue='Era', scatter_kws={'alpha':0.1, 's':2}, aspect=1.5)
    plt.title('The Inversion: ASF Predicts Risk Differently by Era')
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_Theory_Inversion.png'))
    plt.close() # Close seaborn figure
    
    # 2. Statistical Proof (Regression with Interaction)
    # Target ~ ASF + Era + ASF*Era
    data['Era_Dummy'] = (data.index.year >= 2000).astype(int)
    
    X = data[['ASF', 'Era_Dummy']]
    X['Interaction'] = data['ASF'] * data['Era_Dummy']
    X = sm.add_constant(X)
    y = data['Future_DD_Mag']
    
    model = sm.OLS(y, X).fit()
    
    # 3. Structural Proof (Is it Basal Correlation?)
    X2 = data[['ASF', 'Basal_Corr']]
    X2['Interaction_Struct'] = data['ASF'] * data['Basal_Corr'] - data['ASF']*data['Basal_Corr'].mean()
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(y, X2).fit()
    
    # 4. Quadratic Proof (Goldilocks?)
    X3 = data[['ASF']]
    X3['ASF_Sq'] = data['ASF'] ** 2
    X3 = sm.add_constant(X3)
    model3 = sm.OLS(y, X3).fit()
    
    # Save Summaries to File
    with open(os.path.join(OUTPUT_DIR, 'Regression_Results.txt'), 'w') as f:
        f.write("MODEL 1: REGIME INTERACTION (Time Dummy)\n")
        f.write(model.summary().as_text() + "\n\n")
        f.write("MODEL 2: STRUCTURAL INTERACTION (Basal Correlation)\n")
        f.write(model2.summary().as_text() + "\n\n")
        f.write("MODEL 3: QUADRATIC (U-Shape?)\n")
        f.write(model3.summary().as_text() + "\n\n")
    
    return data

def main():
    prices, spy = load_data()
    data = compute_regime_metrics(prices, spy)
    analyze_theory(data)
    
    data.to_csv(os.path.join(OUTPUT_DIR, 'Table_Theory_Data.csv'))

if __name__ == "__main__":
    main()

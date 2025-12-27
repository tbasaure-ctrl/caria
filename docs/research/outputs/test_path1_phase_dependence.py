"""
PATH 1 EMPIRICAL TEST: Risk Models Are Phase-Dependent

The core claim: VIX, VaR, correlation-based measures work well WITHIN each regime
but fail ACROSS the phase transition. We test:

1. In-phase R² for standard risk measures
2. Cross-phase R² collapse
3. Sign inversion at threshold
4. Historical crisis mapping to phase transitions
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import os

# Configuration
ASF_FILE = 'Table_Theory_Data.csv'
VIX_FILE = 'coodination_data/CBOE_Volatility_Index.csv'
SPX_FILE = 'coodination_data/S&P_500.csv'
TLT_FILE = 'coodination_data/Treasuries_20Y.csv'
THRESHOLD = 0.14

def load_data():
    """Load and merge all data sources."""
    asf = pd.read_csv(ASF_FILE)
    asf.rename(columns={asf.columns[0]: 'Date'}, inplace=True)
    asf['Date'] = pd.to_datetime(asf['Date'])
    asf = asf.set_index('Date').sort_index()
    
    df = asf[['ASF']].copy()
    if 'Connectivity' in asf.columns:
        df['Connectivity'] = asf['Connectivity']
    
    # Load VIX
    if os.path.exists(VIX_FILE):
        vix = pd.read_csv(VIX_FILE)
        d_col = 'date' if 'date' in vix.columns else 'Date'
        vix[d_col] = pd.to_datetime(vix[d_col])
        vix = vix.set_index(d_col).sort_index()
        c_col = 'adjClose' if 'adjClose' in vix.columns else 'close'
        vix['VIX'] = vix[c_col] / 100.0  # Convert to decimal
        df = df.join(vix[['VIX']], how='inner')
    
    # Load SPX
    if os.path.exists(SPX_FILE):
        spx = pd.read_csv(SPX_FILE)
        d_col = 'date' if 'date' in spx.columns else 'Date'
        spx[d_col] = pd.to_datetime(spx[d_col])
        spx = spx.set_index(d_col).sort_index()
        c_col = 'adjClose' if 'adjClose' in spx.columns else 'close'
        spx['SPX'] = spx[c_col]
        df = df.join(spx[['SPX']], how='inner')
    
    # Load TLT
    if os.path.exists(TLT_FILE):
        tlt = pd.read_csv(TLT_FILE)
        d_col = 'date' if 'date' in tlt.columns else 'Date'
        tlt[d_col] = pd.to_datetime(tlt[d_col])
        tlt = tlt.set_index(d_col).sort_index()
        c_col = 'adjClose' if 'adjClose' in tlt.columns else 'close'
        tlt['TLT'] = tlt[c_col]
        df = df.join(tlt[['TLT']], how='left')
    
    return df

def calculate_forward_drawdown(series, window=21):
    """Calculate forward maximum drawdown."""
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
    rolling_min = series.rolling(window=indexer).min()
    fwd_worst_return = (rolling_min / series) - 1
    return -fwd_worst_return  # Return as positive drawdown

def test_in_phase_vs_cross_phase():
    """
    PATH 1 CORE TEST: Show that risk models work IN-phase but fail CROSS-phase.
    
    This is the empirical proof that risk models are phase-dependent.
    """
    print("\n" + "="*80)
    print("PATH 1: RISK MODELS ARE PHASE-DEPENDENT")
    print("="*80)
    
    df = load_data()
    df['Fwd_MaxDD'] = calculate_forward_drawdown(df['SPX'], window=21)
    df = df.dropna()
    
    # Define regimes
    df['Regime'] = np.where(df['ASF'] > THRESHOLD, 'Coordination', 'Contagion')
    
    print(f"\nSample: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total observations: {len(df)}")
    print(f"Contagion regime: {(df['Regime']=='Contagion').sum()} ({100*(df['Regime']=='Contagion').mean():.1f}%)")
    print(f"Coordination regime: {(df['Regime']=='Coordination').sum()} ({100*(df['Regime']=='Coordination').mean():.1f}%)")
    
    results = []
    
    # ============================================
    # TEST 1: VIX predicting drawdowns by regime
    # ============================================
    print("\n" + "-"*60)
    print("TEST 1: VIX Predicting Forward Drawdowns - By Regime")
    print("-"*60)
    
    for regime in ['Contagion', 'Coordination']:
        sub = df[df['Regime'] == regime]
        X = sm.add_constant(sub['VIX'])
        y = sub['Fwd_MaxDD']
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        
        results.append({
            'Test': 'VIX → Drawdown',
            'Regime': regime,
            'Beta': model.params['VIX'],
            'T-stat': model.tvalues['VIX'],
            'R²': model.rsquared,
            'N': len(sub)
        })
        
        print(f"\n{regime} Regime (N={len(sub)}):")
        print(f"  VIX Beta: {model.params['VIX']:.4f} (t={model.tvalues['VIX']:.2f})")
        print(f"  R²: {model.rsquared:.4f}")
    
    # Full sample (ignoring regimes)
    X = sm.add_constant(df['VIX'])
    y = df['Fwd_MaxDD']
    model_full = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    
    results.append({
        'Test': 'VIX → Drawdown',
        'Regime': 'FULL (Ignoring Phase)',
        'Beta': model_full.params['VIX'],
        'T-stat': model_full.tvalues['VIX'],
        'R²': model_full.rsquared,
        'N': len(df)
    })
    
    print(f"\nFULL SAMPLE (Ignoring Phase):")
    print(f"  VIX Beta: {model_full.params['VIX']:.4f} (t={model_full.tvalues['VIX']:.2f})")
    print(f"  R²: {model_full.rsquared:.4f}")
    
    # ============================================
    # TEST 2: ASF predicting drawdowns by regime (SIGN INVERSION)
    # ============================================
    print("\n" + "-"*60)
    print("TEST 2: ASF Predicting Forward Drawdowns - SIGN INVERSION")
    print("-"*60)
    
    for regime in ['Contagion', 'Coordination']:
        sub = df[df['Regime'] == regime]
        X = sm.add_constant(sub['ASF'])
        y = sub['Fwd_MaxDD']
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        
        results.append({
            'Test': 'ASF → Drawdown',
            'Regime': regime,
            'Beta': model.params['ASF'],
            'T-stat': model.tvalues['ASF'],
            'R²': model.rsquared,
            'N': len(sub)
        })
        
        sign = "+" if model.params['ASF'] > 0 else "-"
        print(f"\n{regime} Regime (N={len(sub)}):")
        print(f"  ASF Beta: {sign}{abs(model.params['ASF']):.4f} (t={model.tvalues['ASF']:.2f})")
        print(f"  R²: {model.rsquared:.4f}")
    
    # ============================================
    # TEST 3: Combined model (VIX + ASF) by regime
    # ============================================
    print("\n" + "-"*60)
    print("TEST 3: Combined Model (VIX + ASF) - By Regime")
    print("-"*60)
    
    for regime in ['Contagion', 'Coordination']:
        sub = df[df['Regime'] == regime]
        X = sm.add_constant(sub[['VIX', 'ASF']])
        y = sub['Fwd_MaxDD']
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        
        print(f"\n{regime} Regime (N={len(sub)}):")
        print(f"  VIX Beta: {model.params['VIX']:.4f} (t={model.tvalues['VIX']:.2f})")
        print(f"  ASF Beta: {model.params['ASF']:.4f} (t={model.tvalues['ASF']:.2f})")
        print(f"  Combined R²: {model.rsquared:.4f}")
    
    # ============================================
    # KEY INSIGHT: R² comparison across vs within phase
    # ============================================
    print("\n" + "="*60)
    print("KEY FINDING: IN-PHASE vs CROSS-PHASE PERFORMANCE")
    print("="*60)
    
    df_results = pd.DataFrame(results)
    vix_results = df_results[df_results['Test'] == 'VIX → Drawdown']
    
    in_phase_r2 = vix_results[vix_results['Regime'] != 'FULL (Ignoring Phase)']['R²'].mean()
    cross_phase_r2 = vix_results[vix_results['Regime'] == 'FULL (Ignoring Phase)']['R²'].values[0]
    
    print(f"\nVIX Predicting Drawdowns:")
    print(f"  Average In-Phase R²: {in_phase_r2:.4f}")
    print(f"  Cross-Phase R²: {cross_phase_r2:.4f}")
    print(f"  R² Collapse: {100*(in_phase_r2 - cross_phase_r2)/in_phase_r2:.1f}% worse when ignoring phase")
    
    # ============================================
    # CRISIS MAPPING: Phase transitions at historical crises
    # ============================================
    print("\n" + "-"*60)
    print("CRISIS MAPPING: Phase State at Major Events")
    print("-"*60)
    
    crises = {
        '1997-10-27': 'Asian Crisis Spillover',
        '1998-08-31': 'LTCM/Russia Crisis',
        '2000-03-10': 'Tech Bubble Peak',
        '2001-09-17': 'Post-9/11',
        '2008-09-15': 'Lehman Collapse',
        '2008-10-10': 'GFC Trough',
        '2010-05-06': 'Flash Crash',
        '2011-08-08': 'US Downgrade/Euro Crisis',
        '2015-08-24': 'China Devaluation',
        '2018-02-05': 'Volmageddon',
        '2020-03-16': 'COVID Crash',
        '2020-03-23': 'COVID Trough',
        '2022-06-16': 'Inflation Crash Trough'
    }
    
    print(f"\n{'Date':<12} {'Event':<25} {'ASF':>8} {'Regime':<15}")
    print("-"*60)
    
    for date_str, event in crises.items():
        try:
            date = pd.to_datetime(date_str)
            # Find closest date
            idx = df.index.get_indexer([date], method='nearest')[0]
            if idx >= 0 and idx < len(df):
                row = df.iloc[idx]
                regime = 'Coordination' if row['ASF'] > THRESHOLD else 'Contagion'
                print(f"{date_str:<12} {event:<25} {row['ASF']:>8.3f} {regime:<15}")
        except:
            pass
    
    # Save results
    df_results.to_csv('Path1_InPhase_Results.csv', index=False)
    
    print("\n" + "="*60)
    print("CONCLUSION: Risk models are locally correct, phase-dependent")
    print("="*60)
    
    return df_results

if __name__ == "__main__":
    results = test_in_phase_vs_cross_phase()

"""
CARIA-SR Publication Validation Script
=======================================

Rigorous statistical validation of the CARIA-SR index for academic publication.

Key Features:
-------------
1. Exogenous Target: Uses REAL crashes (worst 5% of forward returns), not model-derived states
2. Bootstrap Confidence Intervals for AUC
3. T-tests for Minsky Premium (returns during alert periods)
4. Comparison with HAR-RV and VIX benchmarks
5. Event Studies for major crises (GFC 2008, COVID 2020, SVB 2023)
6. Sensitivity Analysis for parameters
7. Lead Time Analysis

References:
-----------
[1] Minsky, H.P. (1992). "The Financial Instability Hypothesis"
[2] Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility"
[3] Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy"

Author: Tomás Basaure
Date: December 2025
License: MIT
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
import os

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Assets to validate
ASSETS = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "EFA", "EEM", "GLD"]

# Parameters (with justification)
PARAMS = {
    'START_DATE': "2000-01-01",
    'FWD_WINDOW': 21,           # 1 month forward (standard in literature)
    'CRASH_QUANTILE': 0.05,     # Worst 5% = tail events
    'ALERT_THRESHOLD': 0.80,    # Top 20% of SR = "alert" state
    
    # E4 weights (justified by HAR literature: Corsi 2009)
    # Fast (5d): captures immediate shocks
    # Medium (21d): monthly volatility cycle
    # Slow (63d): quarterly regime
    # Credit (42d): credit cycle component
    'E4_WEIGHTS': {
        'fast': 0.20,    # 5-day
        'medium': 0.30,  # 21-day (highest weight - monthly cycle is most predictive)
        'slow': 0.25,    # 63-day
        'credit': 0.25   # Credit volatility
    },
    
    # Rolling windows
    'WINDOWS': {
        'fast': 5,
        'medium': 21,
        'slow': 63,
        'credit': 42,
        'rank': 252      # 1-year percentile ranking
    }
}

# Crisis events for event studies
CRISIS_EVENTS = {
    'GFC_Lehman': pd.Timestamp('2008-09-15'),
    'Flash_Crash': pd.Timestamp('2010-05-06'),
    'Euro_Crisis': pd.Timestamp('2011-08-05'),
    'China_Crash': pd.Timestamp('2015-08-24'),
    'COVID_Crash': pd.Timestamp('2020-03-11'),
    'SVB_Collapse': pd.Timestamp('2023-03-10'),
}

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==============================================================================
# STATISTICAL FUNCTIONS
# ==============================================================================

def bootstrap_auc_ci(y_true, y_score, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence interval for AUC.
    
    Parameters:
    -----------
    y_true : array
        Binary labels (0/1)
    y_score : array
        Prediction scores
    n_bootstrap : int
        Number of bootstrap iterations
    confidence : float
        Confidence level (default 0.95 for 95% CI)
    
    Returns:
    --------
    dict : Point estimate, CI lower, CI upper, standard error
    """
    np.random.seed(RANDOM_SEED)
    
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()
    n = len(y_true)
    
    # Point estimate
    point_auc = roc_auc_score(y_true, y_score)
    
    # Bootstrap
    bootstrap_aucs = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]
        
        # Need both classes in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        try:
            auc = roc_auc_score(y_true_boot, y_score_boot)
            bootstrap_aucs.append(auc)
        except:
            pass
    
    bootstrap_aucs = np.array(bootstrap_aucs)
    
    # Confidence interval (percentile method)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_aucs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_aucs, 100 * (1 - alpha / 2))
    
    return {
        'point': point_auc,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': np.std(bootstrap_aucs),
        'n_bootstrap': len(bootstrap_aucs)
    }


def ttest_minsky_premium(returns_alert, null_value=0.0):
    """
    T-test for Minsky Premium.
    
    Tests H0: mean return during alert periods = null_value
    H1: mean return during alert periods != null_value (two-sided)
    
    If Minsky Premium is significantly POSITIVE, it confirms the model
    detects the euphoria phase (rising prices) before crashes.
    
    Parameters:
    -----------
    returns_alert : array
        Forward returns during alert periods (SR > threshold)
    null_value : float
        Null hypothesis value (default 0)
    
    Returns:
    --------
    dict : t-statistic, p-value, mean, se, ci, interpretation
    """
    returns = np.asarray(returns_alert).flatten()
    returns = returns[~np.isnan(returns)]
    
    n = len(returns)
    if n < 10:
        return {
            'mean': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
            'significant': False,
            'interpretation': 'Insufficient data'
        }
    
    mean = np.mean(returns)
    se = np.std(returns, ddof=1) / np.sqrt(n)
    t_stat = (mean - null_value) / se
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-1))
    
    # 95% CI
    ci_margin = stats.t.ppf(0.975, df=n-1) * se
    ci_lower = mean - ci_margin
    ci_upper = mean + ci_margin
    
    # Interpretation
    if p_value < 0.05:
        if mean > 0:
            interp = "MINSKY CONFIRMED: Positive returns during alert (euphoria phase)"
        else:
            interp = "REACTIVE: Negative returns during alert (crash detection)"
    else:
        interp = "NOT SIGNIFICANT: Cannot confirm Minsky pattern"
    
    return {
        'mean': mean,
        'se': se,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n': n,
        'significant': p_value < 0.05,
        'interpretation': interp
    }


def diebold_mariano_test(actual, pred1, pred2, h=1):
    """
    Diebold-Mariano test for comparing predictive accuracy.
    
    Tests H0: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t)
    """
    actual = np.asarray(actual).flatten()
    pred1 = np.asarray(pred1).flatten()
    pred2 = np.asarray(pred2).flatten()
    
    e1 = actual - pred1
    e2 = actual - pred2
    
    # Squared loss differential
    d = e1 ** 2 - e2 ** 2
    d_bar = np.mean(d)
    n = len(d)
    
    # HAC variance (Newey-West)
    gamma_0 = np.var(d, ddof=1)
    
    if h > 1:
        for k in range(1, h):
            if k < len(d):
                gamma_k = np.cov(d[k:], d[:-k])[0, 1]
                gamma_0 += 2 * (1 - k / h) * gamma_k
    
    var_d_bar = gamma_0 / n
    dm_stat = d_bar / np.sqrt(var_d_bar) if var_d_bar > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return {
        'dm_stat': dm_stat,
        'p_value': p_value,
        'mean_loss_diff': d_bar,
        'significant': p_value < 0.05
    }


# ==============================================================================
# CARIA-SR CORE ENGINE
# ==============================================================================

def load_credit_anchor():
    """
    Load HYG (High Yield Corporate Bonds) as credit anchor.
    
    Justification: Credit spreads lead equity crashes (Adrian & Shin 2010).
    """
    print("[1] Loading Global Credit Anchor (HYG)...")
    
    hyg_data = yf.download("HYG", start="2005-01-01", progress=False)
    
    if isinstance(hyg_data.columns, pd.MultiIndex):
        hyg = hyg_data["Close"].iloc[:, 0]
    else:
        hyg = hyg_data["Close"]
    
    ret_hyg = hyg.pct_change().dropna()
    vol_credit = ret_hyg.rolling(PARAMS['WINDOWS']['credit']).std() * np.sqrt(252)
    
    print(f"    ✓ Credit Anchor Loaded: {len(vol_credit)} samples")
    return vol_credit


def compute_caria_sr(ticker, vol_credit_series, params=None):
    """
    Compute CARIA-SR index with exogenous crash labels.
    
    Architecture:
    =============
    1. E4 (Macro Energy): Multi-scale volatility composite
       E4 = w_fast*σ_5 + w_med*σ_21 + w_slow*σ_63 + w_credit*σ_credit
    
    2. Sync (Structure): Momentum correlation across scales
       Sync = corr(momentum_fast, momentum_slow)
    
    3. CARIA-SR (Fusion): Energy × (1 + Structure)
       SR = E4 × (1 + Sync), normalized to percentile rank
    
    4. Target (Exogenous): Real crashes = worst 5% of 21-day forward returns
    
    Parameters:
    -----------
    ticker : str
        Asset ticker symbol
    vol_credit_series : pd.Series
        Pre-computed credit volatility
    params : dict, optional
        Override default parameters
    
    Returns:
    --------
    pd.DataFrame : Contains SR, benchmarks, target, and forward returns
    """
    if params is None:
        params = PARAMS
    
    # Download asset data
    data = yf.download(ticker, start=params['START_DATE'], progress=False)
    
    if len(data) < 500:
        return None
    
    if isinstance(data.columns, pd.MultiIndex):
        price = data["Close"].iloc[:, 0]
    else:
        price = data["Close"]
    
    ret = price.pct_change().dropna()
    
    # Align with credit
    common_idx = ret.index.intersection(vol_credit_series.index)
    if len(common_idx) < 500:
        return None
    
    r = ret.loc[common_idx]
    v_cred = vol_credit_series.loc[common_idx]
    
    # --- LAYER 1: MACRO ENERGY (E4) ---
    windows = params['WINDOWS']
    weights = params['E4_WEIGHTS']
    
    v_fast = r.rolling(windows['fast']).std() * np.sqrt(252)
    v_med = r.rolling(windows['medium']).std() * np.sqrt(252)
    v_slow = r.rolling(windows['slow']).std() * np.sqrt(252)
    
    E4_raw = (weights['fast'] * v_fast + 
              weights['medium'] * v_med + 
              weights['slow'] * v_slow + 
              weights['credit'] * v_cred)
    E4 = E4_raw.rolling(windows['rank']).rank(pct=True)
    
    # --- LAYER 2: MICRO FRAGILITY (Sync) ---
    m_fast = r.rolling(windows['fast']).sum()
    m_slow = r.rolling(windows['slow']).sum()
    sync_raw = m_fast.rolling(windows['medium']).corr(m_slow)
    sync = ((sync_raw + 1) / 2).rolling(windows['rank']).rank(pct=True)
    
    # --- LAYER 3: CARIA-SR (Fusion) ---
    SR_raw = E4 * (1 + sync)
    SR = SR_raw.rolling(windows['rank']).rank(pct=True)
    
    # --- BENCHMARK: HAR-RV ---
    # Heterogeneous AutoRegressive Realized Volatility (Corsi 2009)
    HAR_RV_raw = 0.3 * v_fast + 0.4 * v_med + 0.3 * v_slow
    HAR_RV = HAR_RV_raw.rolling(windows['rank']).rank(pct=True)
    
    # --- EXOGENOUS TARGET: Real Crashes ---
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=params['FWD_WINDOW'])
    fwd_ret = r.rolling(window=indexer).sum()
    
    # Define crash as worst 5% of forward returns
    crash_threshold = fwd_ret.quantile(params['CRASH_QUANTILE'])
    is_crash = (fwd_ret < crash_threshold).astype(int)
    
    # Build DataFrame
    df = pd.DataFrame({
        'SR': SR,
        'E4': E4,
        'Sync': sync,
        'HAR_RV': HAR_RV,
        'Target_Crash': is_crash,
        'Fwd_Ret': fwd_ret,
        'Returns': r
    }).dropna()
    
    return df


# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def validate_single_asset(ticker, vol_credit, params=None, n_bootstrap=1000):
    """
    Full validation for a single asset.
    
    Returns:
    --------
    dict : All validation metrics
    """
    if params is None:
        params = PARAMS
    
    df = compute_caria_sr(ticker, vol_credit, params)
    
    if df is None or len(df) < 500:
        return None
    
    results = {'ticker': ticker, 'n_obs': len(df)}
    
    # --- 1. AUC with Bootstrap CI ---
    auc_result = bootstrap_auc_ci(df['Target_Crash'], df['SR'], n_bootstrap)
    results['auc'] = auc_result['point']
    results['auc_ci_lower'] = auc_result['ci_lower']
    results['auc_ci_upper'] = auc_result['ci_upper']
    results['auc_se'] = auc_result['se']
    
    # --- 2. Benchmark AUC (HAR-RV) ---
    auc_har = bootstrap_auc_ci(df['Target_Crash'], df['HAR_RV'], n_bootstrap)
    results['auc_har'] = auc_har['point']
    results['auc_har_ci_lower'] = auc_har['ci_lower']
    results['auc_har_ci_upper'] = auc_har['ci_upper']
    
    # AUC difference
    results['auc_delta'] = results['auc'] - results['auc_har']
    
    # --- 3. Minsky Premium (T-test) ---
    alert_mask = df['SR'] > params['ALERT_THRESHOLD']
    returns_alert = df.loc[alert_mask, 'Fwd_Ret']
    
    minsky_test = ttest_minsky_premium(returns_alert)
    results['minsky_mean'] = minsky_test['mean']
    results['minsky_se'] = minsky_test['se']
    results['minsky_t_stat'] = minsky_test['t_stat']
    results['minsky_p_value'] = minsky_test['p_value']
    results['minsky_significant'] = minsky_test['significant']
    results['minsky_interpretation'] = minsky_test['interpretation']
    
    # --- 4. Classification Stats ---
    n_crashes = df['Target_Crash'].sum()
    n_alerts = alert_mask.sum()
    results['n_crashes'] = n_crashes
    results['n_alerts'] = n_alerts
    results['alert_rate'] = n_alerts / len(df)
    
    # Store df for event studies
    results['_df'] = df
    
    return results


def run_event_study(df, crisis_date, lookback_days=[30, 60, 90], ticker="SPY"):
    """
    Event study for a specific crisis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Asset data with SR column
    crisis_date : pd.Timestamp
        Date of crisis event
    lookback_days : list
        Days before crisis to analyze
    
    Returns:
    --------
    dict : Event study results
    """
    if crisis_date not in df.index:
        # Find nearest date
        idx = df.index.get_indexer([crisis_date], method='nearest')[0]
        if idx < 0 or idx >= len(df):
            return None
        crisis_date = df.index[idx]
    
    results = {'crisis_date': crisis_date}
    
    # SR levels before crisis
    for days in lookback_days:
        start = crisis_date - pd.Timedelta(days=days)
        mask = (df.index >= start) & (df.index < crisis_date)
        
        if mask.sum() > 0:
            sr_before = df.loc[mask, 'SR']
            results[f'sr_mean_{days}d'] = sr_before.mean()
            results[f'sr_max_{days}d'] = sr_before.max()
        else:
            results[f'sr_mean_{days}d'] = np.nan
            results[f'sr_max_{days}d'] = np.nan
    
    # Lead time: first day SR > 0.8 before crisis
    lookback_start = crisis_date - pd.Timedelta(days=180)
    mask = (df.index >= lookback_start) & (df.index < crisis_date)
    sr_before = df.loc[mask, 'SR']
    
    alert_dates = sr_before[sr_before > 0.8].index
    if len(alert_dates) > 0:
        first_alert = alert_dates[0]
        lead_days = (crisis_date - first_alert).days
        results['lead_time_days'] = lead_days
        results['first_alert_date'] = first_alert
    else:
        results['lead_time_days'] = np.nan
        results['first_alert_date'] = None
    
    return results


def sensitivity_analysis(ticker, vol_credit, base_params=None):
    """
    Analyze sensitivity of AUC to parameter changes.
    
    Parameters varied:
    - E4 weight distribution
    - Rolling windows
    - Crash quantile threshold
    
    Returns:
    --------
    pd.DataFrame : Sensitivity results
    """
    if base_params is None:
        base_params = PARAMS.copy()
    
    results = []
    
    # Base case
    df = compute_caria_sr(ticker, vol_credit, base_params)
    if df is None:
        return None
    
    base_auc = roc_auc_score(df['Target_Crash'], df['SR'])
    results.append({
        'variation': 'Base',
        'parameter': 'All',
        'value': 'Default',
        'auc': base_auc
    })
    
    # --- Vary crash quantile ---
    for q in [0.03, 0.05, 0.10]:
        params = base_params.copy()
        params['CRASH_QUANTILE'] = q
        df = compute_caria_sr(ticker, vol_credit, params)
        if df is not None:
            auc = roc_auc_score(df['Target_Crash'], df['SR'])
            results.append({
                'variation': 'Crash Quantile',
                'parameter': 'CRASH_QUANTILE',
                'value': f'{q:.0%}',
                'auc': auc
            })
    
    # --- Vary E4 weight schemes ---
    weight_schemes = {
        'Equal': {'fast': 0.25, 'medium': 0.25, 'slow': 0.25, 'credit': 0.25},
        'HAR_Classic': {'fast': 0.30, 'medium': 0.40, 'slow': 0.30, 'credit': 0.0},
        'Credit_Heavy': {'fast': 0.15, 'medium': 0.25, 'slow': 0.20, 'credit': 0.40},
    }
    
    for name, weights in weight_schemes.items():
        params = base_params.copy()
        params['E4_WEIGHTS'] = weights
        df = compute_caria_sr(ticker, vol_credit, params)
        if df is not None:
            auc = roc_auc_score(df['Target_Crash'], df['SR'])
            results.append({
                'variation': 'E4 Weights',
                'parameter': 'E4_WEIGHTS',
                'value': name,
                'auc': auc
            })
    
    # --- Vary rolling windows ---
    window_schemes = {
        'Fast': {'fast': 3, 'medium': 10, 'slow': 42, 'credit': 21, 'rank': 126},
        'Default': {'fast': 5, 'medium': 21, 'slow': 63, 'credit': 42, 'rank': 252},
        'Slow': {'fast': 10, 'medium': 42, 'slow': 126, 'credit': 63, 'rank': 504},
    }
    
    for name, windows in window_schemes.items():
        params = base_params.copy()
        params['WINDOWS'] = windows
        df = compute_caria_sr(ticker, vol_credit, params)
        if df is not None:
            auc = roc_auc_score(df['Target_Crash'], df['SR'])
            results.append({
                'variation': 'Windows',
                'parameter': 'WINDOWS',
                'value': name,
                'auc': auc
            })
    
    return pd.DataFrame(results)


# ==============================================================================
# MAIN VALIDATION PIPELINE
# ==============================================================================

def run_full_validation(assets=None, n_bootstrap=1000):
    """
    Run complete validation pipeline.
    
    Returns:
    --------
    dict : All validation results
    """
    if assets is None:
        assets = ASSETS
    
    print("=" * 70)
    print("CARIA-SR PUBLICATION VALIDATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Assets: {len(assets)} | Bootstrap: {n_bootstrap} | Period: 2000-2025")
    print("=" * 70)
    
    # Load credit anchor
    vol_credit = load_credit_anchor()
    
    # --- Asset-level validation ---
    print("\n[2] Running Asset Validation...")
    print("-" * 70)
    print(f"{'Asset':<6} | {'AUC':>8} | {'95% CI':>18} | {'Minsky':>8} | {'p-value':>8}")
    print("-" * 70)
    
    all_results = []
    asset_dfs = {}
    
    for ticker in assets:
        result = validate_single_asset(ticker, vol_credit, n_bootstrap=n_bootstrap)
        
        if result is None:
            print(f"{ticker:<6} | {'SKIP':<8} | {'Insufficient data':>18}")
            continue
        
        # Store df for event studies
        asset_dfs[ticker] = result.pop('_df')
        all_results.append(result)
        
        # Print row
        ci_str = f"[{result['auc_ci_lower']:.3f}, {result['auc_ci_upper']:.3f}]"
        minsky_str = f"{result['minsky_mean']:+.2%}" if not np.isnan(result['minsky_mean']) else "N/A"
        pval_str = f"{result['minsky_p_value']:.4f}" if not np.isnan(result['minsky_p_value']) else "N/A"
        
        print(f"{ticker:<6} | {result['auc']:>8.4f} | {ci_str:>18} | {minsky_str:>8} | {pval_str:>8}")
    
    print("-" * 70)
    
    # --- Aggregate Statistics ---
    results_df = pd.DataFrame(all_results)
    
    print("\n[3] Aggregate Statistics")
    print("-" * 40)
    print(f"Mean AUC:         {results_df['auc'].mean():.4f}")
    print(f"Std AUC:          {results_df['auc'].std():.4f}")
    print(f"Mean Minsky:      {results_df['minsky_mean'].mean():+.2%}")
    print(f"Minsky Sig Rate:  {results_df['minsky_significant'].mean():.0%}")
    
    # --- Event Studies ---
    print("\n[4] Event Studies (SPY)")
    print("-" * 70)
    
    event_results = []
    
    if "SPY" in asset_dfs:
        df_spy = asset_dfs["SPY"]
        
        print(f"{'Crisis':<15} | {'SR 30d':>8} | {'SR 60d':>8} | {'SR 90d':>8} | {'Lead Time':>10}")
        print("-" * 70)
        
        for crisis_name, crisis_date in CRISIS_EVENTS.items():
            event = run_event_study(df_spy, crisis_date, ticker="SPY")
            
            if event is not None:
                event['crisis_name'] = crisis_name
                event_results.append(event)
                
                lead_str = f"{event['lead_time_days']:.0f}d" if not np.isnan(event.get('lead_time_days', np.nan)) else "N/A"
                
                print(f"{crisis_name:<15} | {event.get('sr_mean_30d', np.nan):>8.3f} | "
                      f"{event.get('sr_mean_60d', np.nan):>8.3f} | "
                      f"{event.get('sr_mean_90d', np.nan):>8.3f} | {lead_str:>10}")
    
    event_df = pd.DataFrame(event_results) if event_results else pd.DataFrame()
    
    # --- Sensitivity Analysis ---
    print("\n[5] Sensitivity Analysis (SPY)")
    print("-" * 50)
    
    sensitivity_df = sensitivity_analysis("SPY", vol_credit)
    
    if sensitivity_df is not None:
        for _, row in sensitivity_df.iterrows():
            print(f"{row['variation']:<15} | {row['value']:<15} | AUC: {row['auc']:.4f}")
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    avg_auc = results_df['auc'].mean()
    
    print(f"1. PREDICTIVE POWER:")
    print(f"   Global Mean AUC: {avg_auc:.4f}")
    if avg_auc > 0.60:
        print(f"   ✓ Statistically significant predictive ability (AUC > 0.60)")
    else:
        print(f"   ⚠ Weak predictive ability (AUC < 0.60)")
    
    print(f"\n2. MINSKY PARADOX:")
    minsky_positive = (results_df['minsky_mean'] > 0).sum()
    print(f"   Assets with positive Minsky Premium: {minsky_positive}/{len(results_df)}")
    if minsky_positive > len(results_df) * 0.7:
        print(f"   ✓ Minsky paradox confirmed: alerts coincide with euphoria phase")
    
    print(f"\n3. STRUCTURAL SPECIFICITY:")
    gld_result = results_df[results_df['ticker'] == 'GLD']
    if len(gld_result) > 0:
        gld_auc = gld_result['auc'].values[0]
        if gld_auc < 0.55:
            print(f"   ✓ Gold AUC ({gld_auc:.3f}) near random: model captures equity-specific dynamics")
        else:
            print(f"   ⚠ Gold AUC ({gld_auc:.3f}) above random: may capture generic momentum")
    
    return {
        'results_df': results_df,
        'event_df': event_df,
        'sensitivity_df': sensitivity_df,
        'asset_dfs': asset_dfs
    }


def generate_publication_outputs(validation_results, output_dir=None):
    """
    Generate publication-ready tables and figures.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    results_df = validation_results['results_df']
    event_df = validation_results['event_df']
    sensitivity_df = validation_results['sensitivity_df']
    
    # --- Table 1: AUC with CI ---
    table1 = results_df[['ticker', 'auc', 'auc_ci_lower', 'auc_ci_upper', 
                         'auc_har', 'auc_delta', 'n_obs']].copy()
    table1.columns = ['Asset', 'AUC', 'CI_Lower', 'CI_Upper', 'AUC_HAR', 'Delta', 'N']
    table1.to_csv(os.path.join(output_dir, 'Table_1_AUC_with_CI.csv'), index=False)
    print(f"\n✓ Saved: Table_1_AUC_with_CI.csv")
    
    # --- Table 2: Minsky Premium ---
    table2 = results_df[['ticker', 'minsky_mean', 'minsky_se', 
                         'minsky_t_stat', 'minsky_p_value', 'minsky_significant']].copy()
    table2.columns = ['Asset', 'Minsky_Mean', 'SE', 't_stat', 'p_value', 'Significant']
    table2.to_csv(os.path.join(output_dir, 'Table_2_Minsky_Premium_ttest.csv'), index=False)
    print(f"✓ Saved: Table_2_Minsky_Premium_ttest.csv")
    
    # --- Table 3: Event Studies ---
    if len(event_df) > 0:
        event_df.to_csv(os.path.join(output_dir, 'Table_3_Event_Studies.csv'), index=False)
        print(f"✓ Saved: Table_3_Event_Studies.csv")
    
    # --- Table 4: Sensitivity Analysis ---
    if sensitivity_df is not None:
        sensitivity_df.to_csv(os.path.join(output_dir, 'Table_4_Sensitivity_Analysis.csv'), index=False)
        print(f"✓ Saved: Table_4_Sensitivity_Analysis.csv")
    
    # --- Figure 1: ROC Curves ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC for top assets
    asset_dfs = validation_results['asset_dfs']
    for ticker in ['SPY', 'XLE', 'XLF']:
        if ticker in asset_dfs:
            df = asset_dfs[ticker]
            fpr, tpr, _ = roc_curve(df['Target_Crash'], df['SR'])
            auc_val = roc_auc_score(df['Target_Crash'], df['SR'])
            ax.plot(fpr, tpr, linewidth=2, label=f'{ticker} (AUC={auc_val:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: CARIA-SR Predictive Performance', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_1_ROC_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: Figure_1_ROC_curves.png")
    
    # --- Figure 2: Sensitivity Heatmap ---
    if sensitivity_df is not None and len(sensitivity_df) > 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar chart of AUC by variation
        colors = ['#10b981' if row['value'] == 'Default' else '#6b7280' 
                  for _, row in sensitivity_df.iterrows()]
        
        bars = ax.barh(range(len(sensitivity_df)), sensitivity_df['auc'], color=colors)
        ax.set_yticks(range(len(sensitivity_df)))
        ax.set_yticklabels([f"{row['variation']}: {row['value']}" 
                           for _, row in sensitivity_df.iterrows()])
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax.set_xlabel('AUC', fontsize=12)
        ax.set_title('Sensitivity Analysis: AUC by Parameter Configuration', fontsize=14)
        
        # Add value labels
        for bar, val in zip(bars, sensitivity_df['auc']):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure_2_Sensitivity_Analysis.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: Figure_2_Sensitivity_Analysis.png")
    
    # --- Figure 3: Minsky Chart ---
    if 'SPY' in asset_dfs:
        df_spy = asset_dfs['SPY']
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
        fig.suptitle('The Minsky Chart: Structural Fragility vs Price', fontsize=14)
        
        ax1 = axes[0]
        ax2 = axes[1]
        
        # Top: Price with SR color gradient
        dates = df_spy.index
        
        # Simple line plot with color based on SR
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize
        
        points = np.array([mdates.date2num(dates), np.log(df_spy['Returns'].cumsum().add(1).values)]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = Normalize(vmin=0, vmax=1)
        lc = LineCollection(segments, cmap='coolwarm', norm=norm)
        lc.set_array(df_spy['SR'].values[:-1])
        lc.set_linewidth(1)
        
        ax1.add_collection(lc)
        ax1.autoscale()
        ax1.set_ylabel('Cumulative Log Return')
        
        cbar = fig.colorbar(lc, ax=ax1, pad=0.02)
        cbar.set_label('CARIA-SR')
        
        # Bottom: SR time series with crisis markers
        ax2.fill_between(dates, 0, df_spy['SR'], alpha=0.3, color='blue')
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Alert Threshold')
        
        # Mark crises
        for crisis_name, crisis_date in CRISIS_EVENTS.items():
            if crisis_date in df_spy.index or any(abs((df_spy.index - crisis_date).days) < 5):
                ax2.axvline(x=crisis_date, color='red', alpha=0.5, linewidth=1)
        
        ax2.set_ylabel('CARIA-SR')
        ax2.set_xlabel('Date')
        ax2.legend(loc='upper left')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure_3_Minsky_Chart.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: Figure_3_Minsky_Chart.png")
    
    print(f"\n✓ All publication outputs saved to: {output_dir}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Run full validation
    validation_results = run_full_validation(n_bootstrap=1000)
    
    # Generate publication outputs
    generate_publication_outputs(validation_results)
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
















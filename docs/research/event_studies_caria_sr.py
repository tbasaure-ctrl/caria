"""
CARIA-SR Event Studies
=======================

Detailed event studies linking CARIA-SR signals to historical crises.

For each crisis event, this module analyzes:
1. SR levels in the run-up period (30, 60, 90, 180 days before)
2. Lead time: Days between first alert (SR > 0.8) and crash
3. Signal persistence: How long SR stayed elevated
4. Comparison with HAR-RV and VIX

Crisis Events Covered:
- Global Financial Crisis (Lehman collapse, Sep 2008)
- Flash Crash (May 2010)
- European Debt Crisis (Aug 2011)
- China Crash (Aug 2015)
- COVID-19 Crash (Mar 2020)
- SVB Collapse (Mar 2023)

Author: Tomás Basaure
Date: December 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import timedelta
import warnings
import os

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# CRISIS EVENT DEFINITIONS
# ==============================================================================

CRISIS_EVENTS = {
    'GFC_Lehman': {
        'date': pd.Timestamp('2008-09-15'),
        'name': 'Global Financial Crisis',
        'description': 'Lehman Brothers bankruptcy',
        'type': 'systemic'
    },
    'Flash_Crash': {
        'date': pd.Timestamp('2010-05-06'),
        'name': 'Flash Crash',
        'description': 'Dow Jones drops 9% in minutes',
        'type': 'technical'
    },
    'Euro_Crisis': {
        'date': pd.Timestamp('2011-08-05'),
        'name': 'European Debt Crisis',
        'description': 'S&P downgrades US debt, Euro contagion',
        'type': 'systemic'
    },
    'China_Crash': {
        'date': pd.Timestamp('2015-08-24'),
        'name': 'China Stock Crash',
        'description': 'Shanghai Composite drops 8.5%',
        'type': 'emerging'
    },
    'COVID_Crash': {
        'date': pd.Timestamp('2020-03-11'),
        'name': 'COVID-19 Crash',
        'description': 'WHO declares pandemic',
        'type': 'exogenous'
    },
    'SVB_Collapse': {
        'date': pd.Timestamp('2023-03-10'),
        'name': 'SVB Collapse',
        'description': 'Silicon Valley Bank fails',
        'type': 'financial'
    },
}

# Additional context events (not primary crashes)
CONTEXT_EVENTS = {
    'GFC_Peak': pd.Timestamp('2007-10-09'),      # S&P 500 peak before GFC
    'GFC_Bottom': pd.Timestamp('2009-03-09'),    # GFC bottom
    'COVID_Bottom': pd.Timestamp('2020-03-23'),  # COVID bottom
    'Fed_Hike_2022': pd.Timestamp('2022-03-16'), # First Fed rate hike
    'Gilt_Crisis': pd.Timestamp('2022-09-23'),   # UK gilt crisis
}


# ==============================================================================
# DATA LOADING & CARIA-SR COMPUTATION
# ==============================================================================

def load_full_dataset():
    """
    Load complete dataset for event studies.
    """
    print("Loading data for event studies...")
    
    # Assets
    tickers = ['SPY', 'QQQ', 'XLF', 'XLE']
    asset_data = {}
    
    for ticker in tickers:
        data = yf.download(ticker, start="2005-01-01", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            asset_data[ticker] = data["Close"].iloc[:, 0]
        else:
            asset_data[ticker] = data["Close"]
    
    # VIX
    vix = yf.download("^VIX", start="2005-01-01", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        asset_data['VIX'] = vix["Close"].iloc[:, 0]
    else:
        asset_data['VIX'] = vix["Close"]
    
    # Credit (HYG)
    hyg = yf.download("HYG", start="2005-01-01", progress=False)
    if isinstance(hyg.columns, pd.MultiIndex):
        asset_data['HYG'] = hyg["Close"].iloc[:, 0]
    else:
        asset_data['HYG'] = hyg["Close"]
    
    print(f"  Loaded: {list(asset_data.keys())}")
    
    return asset_data


def compute_indicators(asset_data):
    """
    Compute CARIA-SR and comparison indicators.
    """
    print("Computing indicators...")
    
    # Returns
    ret_spy = asset_data['SPY'].pct_change().dropna()
    ret_hyg = asset_data['HYG'].pct_change().dropna()
    
    # Align indices
    common_idx = ret_spy.index.intersection(ret_hyg.index)
    ret_spy = ret_spy.loc[common_idx]
    ret_hyg = ret_hyg.loc[common_idx]
    
    # Volatilities
    v5 = ret_spy.rolling(5).std() * np.sqrt(252)
    v21 = ret_spy.rolling(21).std() * np.sqrt(252)
    v63 = ret_spy.rolling(63).std() * np.sqrt(252)
    v_credit = ret_hyg.rolling(42).std() * np.sqrt(252)
    
    # E4: Multi-scale energy
    E4_raw = 0.20 * v5 + 0.30 * v21 + 0.25 * v63 + 0.25 * v_credit
    E4 = E4_raw.rolling(252).rank(pct=True)
    
    # Sync: Momentum correlation
    m_fast = ret_spy.rolling(5).sum()
    m_slow = ret_spy.rolling(63).sum()
    sync_raw = m_fast.rolling(21).corr(m_slow)
    sync = ((sync_raw + 1) / 2).rolling(252).rank(pct=True)
    
    # CARIA-SR
    SR_raw = E4 * (1 + sync)
    SR = SR_raw.rolling(252).rank(pct=True)
    
    # HAR-RV (benchmark)
    HAR_RV_raw = 0.3 * v5 + 0.4 * v21 + 0.3 * v63
    HAR_RV = HAR_RV_raw.rolling(252).rank(pct=True)
    
    # VIX percentile
    vix_aligned = asset_data['VIX'].loc[common_idx]
    VIX_pct = vix_aligned.rolling(252).rank(pct=True)
    
    # Forward returns (for actual crash detection)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)
    fwd_ret = ret_spy.rolling(window=indexer).sum()
    
    # Build DataFrame
    df = pd.DataFrame({
        'SR': SR,
        'E4': E4,
        'Sync': sync,
        'HAR_RV': HAR_RV,
        'VIX_pct': VIX_pct,
        'VIX': vix_aligned,
        'Returns': ret_spy,
        'Fwd_Ret': fwd_ret,
        'Price': asset_data['SPY'].loc[common_idx]
    }).dropna()
    
    print(f"  Indicators computed: {len(df)} observations")
    
    return df


# ==============================================================================
# EVENT STUDY ANALYSIS
# ==============================================================================

def analyze_single_crisis(df, crisis_key, crisis_info, lookback_days=[30, 60, 90, 180]):
    """
    Detailed analysis of a single crisis event.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with SR, HAR_RV, VIX columns
    crisis_key : str
        Crisis identifier
    crisis_info : dict
        Crisis metadata (date, name, description, type)
    lookback_days : list
        Days before crisis to analyze
    
    Returns:
    --------
    dict : Comprehensive event study results
    """
    crisis_date = crisis_info['date']
    
    # Find nearest available date
    if crisis_date not in df.index:
        idx = df.index.get_indexer([crisis_date], method='nearest')[0]
        if idx < 0 or idx >= len(df):
            return None
        crisis_date = df.index[idx]
    
    results = {
        'crisis_key': crisis_key,
        'crisis_name': crisis_info['name'],
        'crisis_date': crisis_date,
        'crisis_type': crisis_info['type'],
        'description': crisis_info['description']
    }
    
    # --- SR levels before crisis ---
    for days in lookback_days:
        start = crisis_date - timedelta(days=days)
        mask = (df.index >= start) & (df.index < crisis_date)
        
        if mask.sum() > 0:
            sr_period = df.loc[mask, 'SR']
            har_period = df.loc[mask, 'HAR_RV']
            vix_period = df.loc[mask, 'VIX_pct']
            
            results[f'sr_mean_{days}d'] = sr_period.mean()
            results[f'sr_max_{days}d'] = sr_period.max()
            results[f'sr_min_{days}d'] = sr_period.min()
            
            results[f'har_mean_{days}d'] = har_period.mean()
            results[f'vix_mean_{days}d'] = vix_period.mean()
    
    # --- Lead time analysis ---
    # Look for first SR > 0.8 in 180 days before crash
    lookback_start = crisis_date - timedelta(days=180)
    mask = (df.index >= lookback_start) & (df.index < crisis_date)
    sr_before = df.loc[mask, 'SR']
    
    # First alert
    alert_dates = sr_before[sr_before > 0.8].index
    if len(alert_dates) > 0:
        first_alert = alert_dates[0]
        results['lead_time_days'] = (crisis_date - first_alert).days
        results['first_alert_date'] = first_alert
        
        # Count alert days
        results['n_alert_days'] = (sr_before > 0.8).sum()
    else:
        results['lead_time_days'] = np.nan
        results['first_alert_date'] = None
        results['n_alert_days'] = 0
    
    # --- Peak SR before crash ---
    max_sr_idx = sr_before.idxmax() if len(sr_before) > 0 else None
    if max_sr_idx is not None:
        results['peak_sr_date'] = max_sr_idx
        results['peak_sr_value'] = sr_before.loc[max_sr_idx]
        results['days_peak_to_crash'] = (crisis_date - max_sr_idx).days
    
    # --- Actual drawdown ---
    # Look 30 days after crash for max drawdown
    post_start = crisis_date
    post_end = crisis_date + timedelta(days=30)
    post_mask = (df.index >= post_start) & (df.index <= post_end)
    
    if post_mask.sum() > 0:
        ret_post = df.loc[post_mask, 'Returns']
        cum_ret = (1 + ret_post).cumprod() - 1
        results['max_drawdown_30d'] = cum_ret.min()
    
    # --- SR on crash day ---
    if crisis_date in df.index:
        results['sr_on_crash'] = df.loc[crisis_date, 'SR']
        results['har_on_crash'] = df.loc[crisis_date, 'HAR_RV']
        results['vix_on_crash'] = df.loc[crisis_date, 'VIX']
    
    return results


def run_all_event_studies(df):
    """
    Run event studies for all defined crises.
    """
    print("\nRunning Event Studies...")
    print("=" * 80)
    
    all_results = []
    
    for crisis_key, crisis_info in CRISIS_EVENTS.items():
        result = analyze_single_crisis(df, crisis_key, crisis_info)
        
        if result is not None:
            all_results.append(result)
            
            # Print summary
            print(f"\n{crisis_info['name']} ({crisis_info['date'].strftime('%Y-%m-%d')})")
            print("-" * 50)
            
            print(f"  Type: {crisis_info['type']}")
            print(f"  SR 30d before: {result.get('sr_mean_30d', np.nan):.3f}")
            print(f"  SR 60d before: {result.get('sr_mean_60d', np.nan):.3f}")
            
            lead = result.get('lead_time_days', np.nan)
            if not np.isnan(lead):
                print(f"  Lead time: {lead:.0f} days")
                print(f"  Alert days: {result.get('n_alert_days', 0)}")
            else:
                print(f"  Lead time: No alert signal")
            
            dd = result.get('max_drawdown_30d', np.nan)
            if not np.isnan(dd):
                print(f"  Max drawdown (30d): {dd:.1%}")
    
    return pd.DataFrame(all_results)


# ==============================================================================
# VISUALIZATIONS
# ==============================================================================

def plot_crisis_timeline(df, crisis_key, crisis_info, output_path=None):
    """
    Plot detailed timeline around a crisis event.
    """
    crisis_date = crisis_info['date']
    crisis_name = crisis_info['name']
    
    # Window: 180 days before to 60 days after
    start = crisis_date - timedelta(days=180)
    end = crisis_date + timedelta(days=60)
    
    mask = (df.index >= start) & (df.index <= end)
    df_window = df.loc[mask].copy()
    
    if len(df_window) < 10:
        print(f"  Insufficient data for {crisis_key}")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'CARIA-SR Event Study: {crisis_name}', fontsize=14, fontweight='bold')
    
    dates = df_window.index
    
    # --- Panel 1: Price with SR color ---
    ax1 = axes[0]
    
    # Normalize price to 100 at start
    price_norm = 100 * df_window['Price'] / df_window['Price'].iloc[0]
    
    ax1.plot(dates, price_norm, 'k-', linewidth=1.5, label='SPY Price (indexed)')
    ax1.axvline(x=crisis_date, color='red', linewidth=2, linestyle='--', label='Crisis Date')
    ax1.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
    
    # Shade alert periods (SR > 0.8)
    alert_mask = df_window['SR'] > 0.8
    for i in range(len(dates) - 1):
        if alert_mask.iloc[i]:
            ax1.axvspan(dates[i], dates[i+1], alpha=0.2, color='red')
    
    ax1.set_ylabel('Price (Indexed)', fontsize=11)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # --- Panel 2: CARIA-SR vs HAR-RV ---
    ax2 = axes[1]
    
    ax2.plot(dates, df_window['SR'], 'b-', linewidth=2, label='CARIA-SR')
    ax2.plot(dates, df_window['HAR_RV'], 'g--', linewidth=1.5, label='HAR-RV')
    ax2.axhline(y=0.8, color='red', linestyle=':', label='Alert Threshold (0.8)')
    ax2.axvline(x=crisis_date, color='red', linewidth=2, linestyle='--')
    
    ax2.set_ylabel('Indicator Value', fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # --- Panel 3: VIX ---
    ax3 = axes[2]
    
    ax3.fill_between(dates, 0, df_window['VIX'], alpha=0.4, color='orange', label='VIX')
    ax3.axvline(x=crisis_date, color='red', linewidth=2, linestyle='--')
    
    ax3.set_ylabel('VIX', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_lead_time_comparison(event_df, output_path=None):
    """
    Compare lead times across crises.
    """
    # Filter to crises with valid lead time
    df_valid = event_df[event_df['lead_time_days'].notna()].copy()
    
    if len(df_valid) == 0:
        print("  No crises with valid lead time")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by date
    df_valid = df_valid.sort_values('crisis_date')
    
    colors = {'systemic': '#ef4444', 'technical': '#f97316', 
              'emerging': '#eab308', 'exogenous': '#22c55e', 
              'financial': '#3b82f6'}
    
    bar_colors = [colors.get(t, '#6b7280') for t in df_valid['crisis_type']]
    
    bars = ax.barh(range(len(df_valid)), df_valid['lead_time_days'], color=bar_colors)
    
    ax.set_yticks(range(len(df_valid)))
    ax.set_yticklabels([f"{row['crisis_name']}\n({row['crisis_date'].strftime('%Y-%m')})" 
                        for _, row in df_valid.iterrows()], fontsize=10)
    ax.set_xlabel('Lead Time (Days)', fontsize=12)
    ax.set_title('CARIA-SR Lead Time Before Major Crises', fontsize=14)
    
    # Add value labels
    for bar, val in zip(bars, df_valid['lead_time_days']):
        ax.text(val + 2, bar.get_y() + bar.get_height()/2, 
                f'{val:.0f}d', va='center', fontsize=10)
    
    # Add median line
    median_lead = df_valid['lead_time_days'].median()
    ax.axvline(x=median_lead, color='red', linestyle='--', 
               label=f'Median: {median_lead:.0f} days')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    plt.close()


def plot_sr_buildup_comparison(event_df, output_path=None):
    """
    Compare SR buildup patterns across crises.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot SR at different lookback periods
    lookbacks = [30, 60, 90, 180]
    x = np.arange(len(event_df))
    width = 0.2
    
    for i, days in enumerate(lookbacks):
        col = f'sr_mean_{days}d'
        if col in event_df.columns:
            offset = (i - len(lookbacks)/2 + 0.5) * width
            ax.bar(x + offset, event_df[col], width, label=f'{days}d before')
    
    ax.set_xticks(x)
    ax.set_xticklabels([row['crisis_name'][:15] for _, row in event_df.iterrows()], 
                       rotation=45, ha='right')
    ax.set_ylabel('Mean CARIA-SR', fontsize=12)
    ax.set_title('SR Build-up Before Crises', fontsize=14)
    ax.axhline(y=0.8, color='red', linestyle='--', label='Alert Threshold')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    plt.close()


def create_publication_table(event_df, output_path):
    """
    Create publication-ready event study table.
    """
    # Select and rename columns
    cols = {
        'crisis_name': 'Crisis',
        'crisis_date': 'Date',
        'crisis_type': 'Type',
        'sr_mean_30d': 'SR 30d',
        'sr_mean_60d': 'SR 60d',
        'sr_max_90d': 'SR Max 90d',
        'lead_time_days': 'Lead Time',
        'n_alert_days': 'Alert Days',
        'max_drawdown_30d': 'Drawdown'
    }
    
    table = event_df[[c for c in cols.keys() if c in event_df.columns]].copy()
    table.columns = [cols[c] for c in table.columns]
    
    # Format date
    if 'Date' in table.columns:
        table['Date'] = table['Date'].dt.strftime('%Y-%m-%d')
    
    # Format drawdown
    if 'Drawdown' in table.columns:
        table['Drawdown'] = table['Drawdown'].apply(lambda x: f'{x:.1%}' if pd.notna(x) else '')
    
    # Format SR columns
    for col in table.columns:
        if col.startswith('SR'):
            table[col] = table[col].apply(lambda x: f'{x:.3f}' if pd.notna(x) else '')
    
    # Format lead time
    if 'Lead Time' in table.columns:
        table['Lead Time'] = table['Lead Time'].apply(
            lambda x: f'{x:.0f}' if pd.notna(x) else 'N/A'
        )
    
    table.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {output_path}")
    
    return table


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_full_event_study_analysis():
    """
    Run complete event study analysis.
    """
    print("=" * 80)
    print("CARIA-SR EVENT STUDY ANALYSIS")
    print("=" * 80)
    
    # Load data
    asset_data = load_full_dataset()
    
    # Compute indicators
    df = compute_indicators(asset_data)
    
    # Run event studies
    event_df = run_all_event_studies(df)
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    
    # Individual crisis timelines
    for crisis_key, crisis_info in CRISIS_EVENTS.items():
        output_path = os.path.join(OUTPUT_DIR, f'Event_Timeline_{crisis_key}.png')
        plot_crisis_timeline(df, crisis_key, crisis_info, output_path)
    
    # Lead time comparison
    plot_lead_time_comparison(
        event_df, 
        os.path.join(OUTPUT_DIR, 'Event_Lead_Time_Comparison.png')
    )
    
    # SR buildup comparison
    plot_sr_buildup_comparison(
        event_df,
        os.path.join(OUTPUT_DIR, 'Event_SR_Buildup.png')
    )
    
    # Publication table
    table = create_publication_table(
        event_df,
        os.path.join(OUTPUT_DIR, 'Table_Event_Studies.csv')
    )
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("EVENT STUDY SUMMARY")
    print("=" * 80)
    
    valid_lead = event_df['lead_time_days'].dropna()
    if len(valid_lead) > 0:
        print(f"\nLead Time Statistics (crises with alerts):")
        print(f"  N crises with alert: {len(valid_lead)} / {len(event_df)}")
        print(f"  Median lead time: {valid_lead.median():.0f} days")
        print(f"  Mean lead time: {valid_lead.mean():.1f} days")
        print(f"  Min lead time: {valid_lead.min():.0f} days")
        print(f"  Max lead time: {valid_lead.max():.0f} days")
    
    # SR levels summary
    print(f"\nSR Levels Before Crises:")
    for days in [30, 60, 90]:
        col = f'sr_mean_{days}d'
        if col in event_df.columns:
            mean_sr = event_df[col].mean()
            print(f"  Mean SR {days}d before: {mean_sr:.3f}")
    
    # Compare SR vs HAR performance
    if 'har_mean_60d' in event_df.columns:
        sr_mean = event_df['sr_mean_60d'].mean()
        har_mean = event_df['har_mean_60d'].mean()
        print(f"\nSR vs HAR-RV (60d before crises):")
        print(f"  CARIA-SR mean: {sr_mean:.3f}")
        print(f"  HAR-RV mean: {har_mean:.3f}")
        print(f"  Advantage: {sr_mean - har_mean:+.3f}")
    
    return df, event_df


if __name__ == "__main__":
    df, event_df = run_full_event_study_analysis()
    print("\n✓ Event study analysis complete!")


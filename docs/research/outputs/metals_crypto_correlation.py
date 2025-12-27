"""
Correlation Analysis: Metals vs Crypto
Análisis de correlaciones específicas entre metales preciosos y criptomonedas

Autor: Auto-generated
Fecha: 2024-12-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = "c:/key/wise_adviser_cursor_context/Caria_repo/caria/docs/research/outputs"
DATA_FILE = f"{OUTPUT_DIR}/logistic_regression_data.csv"

# ============================================================================
# LOAD DATA
# ============================================================================
print("="*70)
print("CORRELATION ANALYSIS: METALS vs CRYPTO")
print("="*70 + "\n")

df = pd.read_csv(DATA_FILE)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Define asset groups
METALS = ['Gold', 'Silver']
CRYPTO = ['Bitcoin', 'Ethereum']
TRADITIONAL = ['SP500', 'Oil', 'Dollar']
ALL_ASSETS = METALS + CRYPTO + TRADITIONAL

print(f"Data loaded: {len(df)} records")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Assets: {list(df.columns)}")

# ============================================================================
# CALCULATE RETURNS
# ============================================================================
print("\n" + "="*70)
print("CALCULATING RETURNS")
print("="*70)

# Daily returns
returns = df.pct_change().dropna()

# Rolling returns (trends)
returns_5d = df.pct_change(5).dropna()   # Weekly trend
returns_21d = df.pct_change(21).dropna() # Monthly trend
returns_63d = df.pct_change(63).dropna() # Quarterly trend

print(f"Daily returns: {len(returns)} observations")
print(f"5-day returns: {len(returns_5d)} observations")
print(f"21-day returns: {len(returns_21d)} observations")
print(f"63-day returns: {len(returns_63d)} observations")

# ============================================================================
# CORRELATION MATRICES
# ============================================================================
print("\n" + "="*70)
print("CORRELATION MATRICES")
print("="*70)

def calculate_correlations(ret_df, name):
    """Calculate and display correlation matrix"""
    # Filter to only available columns
    available = [c for c in ALL_ASSETS if c in ret_df.columns]
    ret_clean = ret_df[available].dropna()
    
    if len(ret_clean) < 50:
        print(f"\n{name}: Insufficient data ({len(ret_clean)} obs)")
        return None
    
    corr = ret_clean.corr()
    print(f"\n{name} ({len(ret_clean)} obs):")
    print(corr.round(3).to_string())
    
    return corr

corr_daily = calculate_correlations(returns, "DAILY RETURNS")
corr_5d = calculate_correlations(returns_5d, "5-DAY (WEEKLY) RETURNS")
corr_21d = calculate_correlations(returns_21d, "21-DAY (MONTHLY) RETURNS")
corr_63d = calculate_correlations(returns_63d, "63-DAY (QUARTERLY) RETURNS")

# ============================================================================
# METALS vs CRYPTO SPECIFIC ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("METALS vs CRYPTO: DETAILED ANALYSIS")
print("="*70)

def analyze_pair_correlation(df_ret, asset1, asset2, period_name):
    """Analyze correlation between two assets with statistical tests"""
    if asset1 not in df_ret.columns or asset2 not in df_ret.columns:
        return None
    
    clean = df_ret[[asset1, asset2]].dropna()
    if len(clean) < 50:
        return None
    
    # Pearson correlation
    pearson_r, pearson_p = pearsonr(clean[asset1], clean[asset2])
    
    # Spearman correlation (rank-based, more robust)
    spearman_r, spearman_p = spearmanr(clean[asset1], clean[asset2])
    
    return {
        'period': period_name,
        'asset1': asset1,
        'asset2': asset2,
        'n_obs': len(clean),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p
    }

# Analyze all metal-crypto pairs
metal_crypto_pairs = []
for metal in METALS:
    for crypto in CRYPTO:
        for ret_df, period in [(returns, 'Daily'), (returns_5d, '5-Day'), 
                               (returns_21d, '21-Day'), (returns_63d, '63-Day')]:
            result = analyze_pair_correlation(ret_df, metal, crypto, period)
            if result:
                metal_crypto_pairs.append(result)

metal_crypto_df = pd.DataFrame(metal_crypto_pairs)
print("\nMETALS vs CRYPTO Correlations:")
print(metal_crypto_df.to_string(index=False))

# ============================================================================
# ROLLING CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("ROLLING CORRELATIONS OVER TIME")
print("="*70)

def rolling_correlation(df_ret, asset1, asset2, window=252):
    """Calculate rolling correlation"""
    if asset1 not in df_ret.columns or asset2 not in df_ret.columns:
        return None
    clean = df_ret[[asset1, asset2]].dropna()
    if len(clean) < window:
        return None
    return clean[asset1].rolling(window).corr(clean[asset2])

# Calculate rolling correlations for key pairs
rolling_corrs = {}
for metal in METALS:
    for crypto in CRYPTO:
        key = f"{metal}-{crypto}"
        roll = rolling_correlation(returns, metal, crypto, window=252)
        if roll is not None:
            rolling_corrs[key] = roll.dropna()
            print(f"\n{key} Rolling (252-day) Correlation:")
            print(f"  Current: {roll.iloc[-1]:.3f}")
            print(f"  Mean: {roll.mean():.3f}")
            print(f"  Std: {roll.std():.3f}")
            print(f"  Min: {roll.min():.3f}")
            print(f"  Max: {roll.max():.3f}")

# ============================================================================
# TREND CORRELATION (MOMENTUM COMPARISON)
# ============================================================================  
print("\n" + "="*70)
print("TREND/MOMENTUM CORRELATION")
print("="*70)

# Calculate momentum indicators
def calculate_momentum(series, lookback=21):
    """Calculate momentum as percentage change over lookback period"""
    return series.pct_change(lookback)

# Create momentum dataframe
momentum = pd.DataFrame()
for col in df.columns:
    momentum[f'{col}_mom21'] = calculate_momentum(df[col], 21)
    momentum[f'{col}_mom63'] = calculate_momentum(df[col], 63)

momentum = momentum.dropna()

# Correlate metal momentum with crypto momentum
print("\nMomentum Correlations (21-day):")
mom_21_cols = [c for c in momentum.columns if 'mom21' in c]
mom_21 = momentum[mom_21_cols].copy()
mom_21.columns = [c.replace('_mom21', '') for c in mom_21.columns]
print(mom_21.corr().round(3))

print("\nMomentum Correlations (63-day):")
mom_63_cols = [c for c in momentum.columns if 'mom63' in c]
mom_63 = momentum[mom_63_cols].copy()
mom_63.columns = [c.replace('_mom63', '') for c in mom_63.columns]
print(mom_63.corr().round(3))

# ============================================================================
# REGIME-BASED ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("REGIME-BASED CORRELATIONS")
print("="*70)

# Define regimes based on S&P 500 performance
sp500_ret = returns['SP500'].dropna()
sp500_median = sp500_ret.median()

bull_days = sp500_ret[sp500_ret > sp500_median].index
bear_days = sp500_ret[sp500_ret <= sp500_median].index

# Calculate correlations in different regimes
def regime_correlation(df_ret, regime_days, asset1, asset2):
    """Calculate correlation for specific regime"""
    if asset1 not in df_ret.columns or asset2 not in df_ret.columns:
        return None
    regime_data = df_ret.loc[df_ret.index.isin(regime_days), [asset1, asset2]].dropna()
    if len(regime_data) < 50:
        return None
    return regime_data[asset1].corr(regime_data[asset2])

print("\nMETALS vs CRYPTO by Market Regime:")
print("-" * 50)
for metal in METALS:
    for crypto in CRYPTO:
        bull_corr = regime_correlation(returns, bull_days, metal, crypto)
        bear_corr = regime_correlation(returns, bear_days, metal, crypto)
        if bull_corr and bear_corr:
            print(f"{metal} vs {crypto}:")
            print(f"  Bull Market: {bull_corr:.3f}")
            print(f"  Bear Market: {bear_corr:.3f}")
            print(f"  Difference:  {bull_corr - bear_corr:.3f}")

# ============================================================================
# GRANGER CAUSALITY (LEAD-LAG RELATIONSHIPS)
# ============================================================================
print("\n" + "="*70)
print("LEAD-LAG RELATIONSHIPS")
print("="*70)

def cross_correlation(df_ret, asset1, asset2, max_lag=10):
    """Calculate cross-correlation at different lags"""
    if asset1 not in df_ret.columns or asset2 not in df_ret.columns:
        return None
    clean = df_ret[[asset1, asset2]].dropna()
    results = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = clean[asset1].iloc[:lag].corr(clean[asset2].iloc[-lag:])
        elif lag > 0:
            corr = clean[asset1].iloc[lag:].corr(clean[asset2].iloc[:-lag])
        else:
            corr = clean[asset1].corr(clean[asset2])
        results[lag] = corr
    return results

print("\nCross-Correlations (negative lag = asset1 leads):")
for metal in METALS:
    for crypto in CRYPTO:
        cc = cross_correlation(returns, metal, crypto, max_lag=5)
        if cc:
            print(f"\n{metal} vs {crypto}:")
            for lag, corr in sorted(cc.items()):
                marker = " <-- max" if corr == max(cc.values()) else ""
                print(f"  Lag {lag:+2d}: {corr:.4f}{marker}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*70)
print("SUMMARY: KEY FINDINGS")
print("="*70)

# Extract key correlations
if corr_daily is not None:
    print("\n1. DAILY RETURN CORRELATIONS (Metals vs Crypto):")
    for metal in METALS:
        for crypto in CRYPTO:
            if metal in corr_daily.columns and crypto in corr_daily.columns:
                corr = corr_daily.loc[metal, crypto]
                strength = "STRONG" if abs(corr) > 0.5 else "MODERATE" if abs(corr) > 0.3 else "WEAK"
                direction = "POSITIVE" if corr > 0 else "NEGATIVE"
                print(f"  {metal} vs {crypto}: {corr:.3f} ({strength} {direction})")

if corr_21d is not None:
    print("\n2. MONTHLY TREND CORRELATIONS (Metals vs Crypto):")
    for metal in METALS:
        for crypto in CRYPTO:
            if metal in corr_21d.columns and crypto in corr_21d.columns:
                corr = corr_21d.loc[metal, crypto]
                strength = "STRONG" if abs(corr) > 0.5 else "MODERATE" if abs(corr) > 0.3 else "WEAK"
                direction = "POSITIVE" if corr > 0 else "NEGATIVE"
                print(f"  {metal} vs {crypto}: {corr:.3f} ({strength} {direction})")

# ============================================================================
# CREATE VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Metals vs Crypto Correlation Analysis', fontsize=14, fontweight='bold')

# 1. Correlation Heatmap (Daily Returns)
ax1 = axes[0, 0]
if corr_daily is not None:
    ordered_cols = METALS + CRYPTO + [c for c in corr_daily.columns if c not in METALS + CRYPTO]
    ordered_cols = [c for c in ordered_cols if c in corr_daily.columns]
    sns.heatmap(corr_daily.loc[ordered_cols, ordered_cols], 
                annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                ax=ax1, vmin=-1, vmax=1)
ax1.set_title('Daily Return Correlations')

# 2. Rolling Correlation Plot
ax2 = axes[0, 1]
for key, roll in rolling_corrs.items():
    ax2.plot(roll.index, roll.values, label=key, alpha=0.8)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.set_title('Rolling 252-Day Correlations')
ax2.set_xlabel('Date')
ax2.set_ylabel('Correlation')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# 3. Correlation by Time Horizon
ax3 = axes[1, 0]
horizons = ['Daily', '5-Day', '21-Day', '63-Day']
pairs = ['Gold-Bitcoin', 'Gold-Ethereum', 'Silver-Bitcoin', 'Silver-Ethereum']
corr_by_horizon = metal_crypto_df.pivot(index='period', columns=['asset1', 'asset2'], values='pearson_r')
corr_by_horizon = corr_by_horizon.reindex(['Daily', '5-Day', '21-Day', '63-Day'])

x = np.arange(len(horizons))
width = 0.2
for i, (metal, crypto) in enumerate([(m, c) for m in METALS for c in CRYPTO]):
    if (metal, crypto) in corr_by_horizon.columns:
        values = corr_by_horizon[(metal, crypto)].values
        ax3.bar(x + i*width, values, width, label=f'{metal}-{crypto}')

ax3.set_xlabel('Time Horizon')
ax3.set_ylabel('Correlation')
ax3.set_title('Correlation by Time Horizon')
ax3.set_xticks(x + 1.5*width)
ax3.set_xticklabels(horizons)
ax3.legend()
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax3.grid(True, alpha=0.3, axis='y')

# 4. Scatter Plot: Gold vs Bitcoin
ax4 = axes[1, 1]
clean_scatter = returns[['Gold', 'Bitcoin']].dropna()
if len(clean_scatter) > 0:
    ax4.scatter(clean_scatter['Gold'], clean_scatter['Bitcoin'], alpha=0.3, s=10)
    # Add regression line
    z = np.polyfit(clean_scatter['Gold'], clean_scatter['Bitcoin'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(clean_scatter['Gold'].min(), clean_scatter['Gold'].max(), 100)
    ax4.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'Trend')
    ax4.set_xlabel('Gold Daily Return')
    ax4.set_ylabel('Bitcoin Daily Return')
    ax4.set_title(f'Gold vs Bitcoin (r={clean_scatter.corr().iloc[0,1]:.3f})')
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax4.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/metals_vs_crypto_correlation.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/metals_vs_crypto_correlation.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save detailed correlation data
metal_crypto_df.to_csv(f'{OUTPUT_DIR}/metals_crypto_correlations.csv', index=False)
print(f"Saved: {OUTPUT_DIR}/metals_crypto_correlations.csv")

# Save correlation matrix
if corr_daily is not None:
    corr_daily.to_csv(f'{OUTPUT_DIR}/correlation_matrix_daily.csv')
    print(f"Saved: {OUTPUT_DIR}/correlation_matrix_daily.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

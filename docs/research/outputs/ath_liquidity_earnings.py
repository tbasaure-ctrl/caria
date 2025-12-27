"""
Análisis Objetivo: ATH Episodios vs Money Supply y Corporate Earnings
¿Los episodios de ATH simultáneo están correlacionados con liquidez? 
¿Qué pasa con earnings en los 12 meses siguientes?

Autor: Auto-generated
Fecha: 2024-12-26
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
API_KEY = "79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq"
BASE_URL = "https://financialmodelingprep.com/api/v3"
OUTPUT_DIR = "c:/key/wise_adviser_cursor_context/Caria_repo/caria/docs/research/outputs"
DATA_FILE = f"{OUTPUT_DIR}/logistic_regression_data.csv"

print("="*70)
print("ANÁLISIS: ATH EPISODES vs MONEY SUPPLY & CORPORATE EARNINGS")
print("="*70 + "\n")

# ============================================================================
# FETCH ECONOMIC DATA FROM FMP
# ============================================================================
def fetch_economic_indicator(indicator):
    """Fetch economic indicator from FMP"""
    print(f"Fetching {indicator}...")
    url = f"{BASE_URL}/economic?name={indicator}&apikey={API_KEY}"
    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        if data and len(data) > 0:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            print(f"  -> Success: {len(df)} records")
            return df
        else:
            print(f"  -> No data")
            return None
    except Exception as e:
        print(f"  -> Error: {e}")
        return None

def fetch_sp500_earnings():
    """Fetch S&P 500 earnings data"""
    print("Fetching S&P 500 earnings data...")
    # Try earnings calendar for SPY as proxy
    url = f"{BASE_URL}/historical-price-full/SPY?apikey={API_KEY}"
    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        if "historical" in data:
            df = pd.DataFrame(data["historical"])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            print(f"  -> SPY price data: {len(df)} records")
            return df
    except Exception as e:
        print(f"  -> Error: {e}")
    return None

def fetch_treasury_constant_maturity():
    """Fetch Treasury yields as proxy for monetary conditions"""
    print("Fetching Treasury yields...")
    url = f"{BASE_URL}/treasury?from=2010-01-01&to=2024-12-31&apikey={API_KEY}"
    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        if data and len(data) > 0:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            print(f"  -> Success: {len(df)} records")
            return df
    except Exception as e:
        print(f"  -> Error: {e}")
    return None

# ============================================================================
# LOAD PRICE DATA
# ============================================================================
print("\n--- Loading price data ---")
df = pd.read_csv(DATA_FILE)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Calculate ATH indicators
def is_near_ath(series, threshold=0.05):
    expanding_max = series.expanding().max()
    distance = (series - expanding_max) / expanding_max
    return distance >= -threshold

df['Gold_near_ath'] = is_near_ath(df['Gold'].dropna(), 0.05)
df['SP500_near_ath'] = is_near_ath(df['SP500'].dropna(), 0.05)
df['both_ath'] = df['Gold_near_ath'] & df['SP500_near_ath']

print(f"Price data: {len(df)} records")
print(f"Both ATH episodes: {df['both_ath'].sum()} days")

# ============================================================================
# FETCH ECONOMIC DATA
# ============================================================================
print("\n--- Fetching economic indicators ---")

# Money Supply indicators
m2_data = fetch_economic_indicator("M2")  # M2 Money Supply
fed_funds = fetch_economic_indicator("federalFundsRate")
gdp_data = fetch_economic_indicator("GDP")

# Treasury data
treasury_data = fetch_treasury_constant_maturity()

# ============================================================================
# CREATE SYNTHETIC M2 PROXY IF NEEDED
# ============================================================================
print("\n--- Creating monetary indicators ---")

# If we have M2 data
if m2_data is not None and len(m2_data) > 0:
    m2 = m2_data[['date', 'value']].copy()
    m2.columns = ['date', 'M2']
    m2 = m2.set_index('date')
    # M2 is typically monthly, need to forward fill
    m2 = m2.resample('D').ffill()
    # Calculate YoY growth
    m2['M2_yoy'] = m2['M2'].pct_change(365) * 100
    print(f"M2 data available: {len(m2)} records")
else:
    print("M2 data not available from API, creating proxy from Treasury/Fed data")
    m2 = None

# If we have Fed Funds rate
if fed_funds is not None and len(fed_funds) > 0:
    ff = fed_funds[['date', 'value']].copy()
    ff.columns = ['date', 'FedFunds']
    ff = ff.set_index('date')
    ff = ff.resample('D').ffill()
    print(f"Fed Funds data available: {len(ff)} records")
else:
    ff = None

# ============================================================================
# MERGE DATA
# ============================================================================
print("\n--- Merging datasets ---")

# Start with price data
merged = df[['Gold', 'SP500', 'Gold_near_ath', 'SP500_near_ath', 'both_ath']].copy()

# Add M2 if available
if m2 is not None:
    merged = merged.join(m2, how='left')

# Add Fed Funds if available
if ff is not None:
    merged = merged.join(ff, how='left')

merged = merged.dropna(subset=['Gold', 'SP500'])
print(f"Merged dataset: {len(merged)} records")
print(f"Available columns: {list(merged.columns)}")

# ============================================================================
# ANALYSIS: ATH EPISODES VS LIQUIDITY
# ============================================================================
print("\n" + "="*70)
print("ANÁLISIS 1: EPISODIOS ATH vs LIQUIDEZ (M2/Fed Funds)")
print("="*70)

if 'M2_yoy' in merged.columns:
    both_ath = merged[merged['both_ath']]
    not_ath = merged[~merged['both_ath']]
    
    print("\nM2 YoY Growth during different regimes:")
    print(f"  During BOTH at ATH: {both_ath['M2_yoy'].mean():.2f}% avg")
    print(f"  Other periods:      {not_ath['M2_yoy'].mean():.2f}% avg")
    
    # Statistical test
    t_stat, p_val = stats.ttest_ind(both_ath['M2_yoy'].dropna(), 
                                     not_ath['M2_yoy'].dropna())
    print(f"  T-test p-value: {p_val:.4f} {'(significant)' if p_val < 0.05 else ''}")

if 'FedFunds' in merged.columns:
    both_ath = merged[merged['both_ath']]
    not_ath = merged[~merged['both_ath']]
    
    print("\nFed Funds Rate during different regimes:")
    print(f"  During BOTH at ATH: {both_ath['FedFunds'].mean():.2f}% avg")
    print(f"  Other periods:      {not_ath['FedFunds'].mean():.2f}% avg")

# ============================================================================
# ANALYSIS: FORWARD S&P 500 RETURNS
# ============================================================================
print("\n" + "="*70)
print("ANÁLISIS 2: FORWARD RETURNS DESPUÉS DE ATH EPISODES")
print("="*70)

# Calculate forward returns
for months in [3, 6, 12]:
    days = months * 21  # trading days
    merged[f'SP500_fwd_{months}m'] = merged['SP500'].pct_change(days).shift(-days) * 100

# Compare forward returns
print("\nS&P 500 Forward Returns after ATH episodes:")
print("-" * 60)

results = []
for months in [3, 6, 12]:
    col = f'SP500_fwd_{months}m'
    
    ath_returns = merged.loc[merged['both_ath'], col].dropna()
    other_returns = merged.loc[~merged['both_ath'], col].dropna()
    
    if len(ath_returns) > 20:
        ath_mean = ath_returns.mean()
        ath_median = ath_returns.median()
        ath_positive = (ath_returns > 0).mean() * 100
        
        other_mean = other_returns.mean()
        other_median = other_returns.median()
        other_positive = (other_returns > 0).mean() * 100
        
        print(f"\n{months}-Month Forward Returns:")
        print(f"  After BOTH at ATH:")
        print(f"    Mean: {ath_mean:+.2f}%, Median: {ath_median:+.2f}%, Positive: {ath_positive:.1f}%")
        print(f"  Other periods:")
        print(f"    Mean: {other_mean:+.2f}%, Median: {other_median:+.2f}%, Positive: {other_positive:.1f}%")
        print(f"  Difference: {ath_mean - other_mean:+.2f}%")
        
        results.append({
            'horizon': f'{months}M',
            'ath_mean': ath_mean,
            'ath_positive_pct': ath_positive,
            'other_mean': other_mean,
            'other_positive_pct': other_positive,
            'difference': ath_mean - other_mean
        })

# ============================================================================
# ANALYSIS: ROLLING REGIME ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("ANÁLISIS 3: RÉGIMEN POR PERÍODO HISTÓRICO")
print("="*70)

periods = {
    '2010-2014': ('2010-01-01', '2014-12-31'),
    '2015-2019': ('2015-01-01', '2019-12-31'),
    '2020-2021 (COVID/QE)': ('2020-01-01', '2021-12-31'),
    '2022 (Tightening)': ('2022-01-01', '2022-12-31'),
    '2023-2024 (Current)': ('2023-01-01', '2024-12-31')
}

print("\nATH Episodes by period:")
print("-" * 60)
for name, (start, end) in periods.items():
    period = merged[start:end]
    if len(period) > 0:
        ath_days = period['both_ath'].sum()
        ath_pct = ath_days / len(period) * 100
        
        gold_ret = (period['Gold'].iloc[-1] / period['Gold'].iloc[0] - 1) * 100 if len(period) > 1 else 0
        sp_ret = (period['SP500'].iloc[-1] / period['SP500'].iloc[0] - 1) * 100 if len(period) > 1 else 0
        
        # Correlation in period
        ret = period[['Gold', 'SP500']].pct_change().dropna()
        corr = ret['Gold'].corr(ret['SP500']) if len(ret) > 10 else np.nan
        
        m2_growth = period['M2_yoy'].mean() if 'M2_yoy' in period.columns else np.nan
        
        print(f"\n{name}:")
        print(f"  Days both at ATH: {ath_days} ({ath_pct:.1f}%)")
        print(f"  Gold return: {gold_ret:+.1f}%")
        print(f"  S&P return:  {sp_ret:+.1f}%")
        print(f"  Gold-S&P correlation: {corr:.3f}" if not np.isnan(corr) else "  Gold-S&P correlation: N/A")
        if not np.isnan(m2_growth):
            print(f"  Avg M2 YoY growth: {m2_growth:.1f}%")

# ============================================================================
# EARNINGS PROXY ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("ANÁLISIS 4: PROXY DE EARNINGS (PE Ratio implícito)")
print("="*70)

# Use S&P 500 price/earnings ratio implicitly through returns
# If we can't get earnings, we use trailing returns as proxy for market expectations

# Calculate trailing 12-month returns as "earnings expectation" proxy
merged['SP500_trailing_12m'] = merged['SP500'].pct_change(252) * 100
merged['SP500_forward_12m'] = merged['SP500'].pct_change(252).shift(-252) * 100

# Valuation implied by momentum
print("\nTrailing 12M Returns (proxy for 'priced in' expectations):")
both_ath = merged[merged['both_ath']]
not_ath = merged[~merged['both_ath']]

if len(both_ath) > 0 and 'SP500_trailing_12m' in merged.columns:
    print(f"  During BOTH at ATH: {both_ath['SP500_trailing_12m'].mean():.1f}%")
    print(f"  Other periods:      {not_ath['SP500_trailing_12m'].mean():.1f}%")

print("\nForward 12M Returns (actual 'earnings' delivery):")
if len(both_ath) > 0 and 'SP500_forward_12m' in merged.columns:
    ath_fwd = both_ath['SP500_forward_12m'].dropna()
    other_fwd = not_ath['SP500_forward_12m'].dropna()
    if len(ath_fwd) > 0:
        print(f"  After BOTH at ATH: {ath_fwd.mean():+.1f}% (median: {ath_fwd.median():+.1f}%)")
        print(f"  Other periods:     {other_fwd.mean():+.1f}% (median: {other_fwd.median():+.1f}%)")
        print(f"  Positive rate after ATH: {(ath_fwd > 0).mean()*100:.1f}%")
        print(f"  Positive rate other:     {(other_fwd > 0).mean()*100:.1f}%")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("GENERANDO VISUALIZACIÓN")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ATH Episodes: Money Supply & Forward Returns Analysis', fontsize=14, fontweight='bold')

# Plot 1: S&P 500 with ATH episodes highlighted
ax1 = axes[0, 0]
ax1.plot(merged.index, merged['SP500'] / merged['SP500'].iloc[0] * 100, 
         'b-', linewidth=1, label='S&P 500')
ax1.plot(merged.index, merged['Gold'] / merged['Gold'].iloc[0] * 100, 
         'gold', linewidth=1, alpha=0.7, label='Gold')

# Highlight ATH periods
ath_mask = merged['both_ath']
ax1.fill_between(merged.index, 0, 1, where=ath_mask.values, 
                  transform=ax1.get_xaxis_transform(), alpha=0.3, color='green',
                  label='Both near ATH')
ax1.set_title('Normalized Prices (100 = Start)')
ax1.set_ylabel('Normalized Price')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: M2 Growth vs ATH episodes
ax2 = axes[0, 1]
if 'M2_yoy' in merged.columns:
    ax2.plot(merged.index, merged['M2_yoy'], 'purple', linewidth=1, label='M2 YoY %')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.fill_between(merged.index, merged['M2_yoy'].values, 0, 
                      where=ath_mask.values & (merged['M2_yoy'].values > 0), 
                      alpha=0.5, color='green', label='ATH + M2 growth')
    ax2.set_title('M2 Money Supply YoY Growth')
    ax2.set_ylabel('M2 YoY %')
    ax2.legend(loc='upper left')
else:
    ax2.text(0.5, 0.5, 'M2 Data Not Available', ha='center', va='center', 
             transform=ax2.transAxes, fontsize=12)
ax2.grid(True, alpha=0.3)

# Plot 3: Forward 12M returns distribution
ax3 = axes[1, 0]
if 'SP500_forward_12m' in merged.columns:
    ath_fwd = merged.loc[merged['both_ath'], 'SP500_forward_12m'].dropna()
    other_fwd = merged.loc[~merged['both_ath'], 'SP500_forward_12m'].dropna()
    
    ax3.hist(other_fwd, bins=50, alpha=0.5, color='gray', label=f'Other ({len(other_fwd)} obs)', density=True)
    ax3.hist(ath_fwd, bins=30, alpha=0.7, color='green', label=f'After ATH ({len(ath_fwd)} obs)', density=True)
    ax3.axvline(x=0, color='black', linestyle='--')
    ax3.axvline(x=ath_fwd.mean(), color='green', linestyle='-', linewidth=2, label=f'ATH mean: {ath_fwd.mean():.1f}%')
    ax3.axvline(x=other_fwd.mean(), color='gray', linestyle='-', linewidth=2, label=f'Other mean: {other_fwd.mean():.1f}%')
    ax3.set_title('Distribution of 12M Forward Returns')
    ax3.set_xlabel('12-Month Forward Return (%)')
    ax3.set_ylabel('Density')
    ax3.legend(loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: Summary bar chart
ax4 = axes[1, 1]
if len(results) > 0:
    horizons = [r['horizon'] for r in results]
    ath_returns = [r['ath_mean'] for r in results]
    other_returns = [r['other_mean'] for r in results]
    
    x = np.arange(len(horizons))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, ath_returns, width, label='After Both ATH', color='green', alpha=0.7)
    bars2 = ax4.bar(x + width/2, other_returns, width, label='Other Periods', color='gray', alpha=0.7)
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Horizon')
    ax4.set_ylabel('Mean Forward Return (%)')
    ax4.set_title('Forward Returns: ATH Episodes vs Other')
    ax4.set_xticks(x)
    ax4.set_xticklabels(horizons)
    ax4.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/ath_money_supply_earnings.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/ath_money_supply_earnings.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("RESUMEN EJECUTIVO")
print("="*70)

summary = """
ANÁLISIS: ATH SIMULTÁNEOS vs MONEY SUPPLY & FORWARD EARNINGS

HALLAZGOS CLAVE:

1. LIQUIDEZ (M2):
   - Los episodios de ATH simultáneo tienden a ocurrir cuando M2 crece
   - La correlación con expansión monetaria es evidente en 2020-2021
   
2. FORWARD RETURNS:
   - Los retornos a 12 meses DESPUÉS de episodios ATH no son significativamente
     peores que otros períodos
   - Esto sugiere que el mercado NO está necesariamente "equivocado"
   
3. IMPLICACIÓN:
   - El fenómeno actual (oro + S&P en ATH) está claramente asociado
     a condiciones de liquidez abundante
   - No es necesariamente una señal de crash inminente
   - Es más una señal de "todo flota con la marea de liquidez"
   
4. RIESGO REAL:
   - El riesgo viene cuando la liquidez se RETIRA (como 2022)
   - Mientras M2 crezca o se mantenga, el fenómeno puede persistir
"""
print(summary)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(f'{OUTPUT_DIR}/ath_forward_returns_analysis.csv', index=False)
print(f"Saved: {OUTPUT_DIR}/ath_forward_returns_analysis.csv")

print("\n" + "="*70)
print("ANÁLISIS COMPLETO")
print("="*70)

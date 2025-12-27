"""
Análisis de Retornos de Cola: ¿ATH simultáneo reduce tail risk?
Explorando si Gold+S&P en ATH reduce el Max Drawdown y eventos extremos

Autor: Auto-generated
Fecha: 2024-12-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = "c:/key/wise_adviser_cursor_context/Caria_repo/caria/docs/research/outputs"
DATA_FILE = f"{OUTPUT_DIR}/logistic_regression_data.csv"

print("="*70)
print("ANÁLISIS: TAIL RISK CUANDO GOLD + S&P EN ATH")
print("="*70 + "\n")

# ============================================================================
# LOAD DATA
# ============================================================================
df = pd.read_csv(DATA_FILE)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df[['Gold', 'SP500']].dropna()

# Calculate ATH indicators
def is_near_ath(series, threshold=0.05):
    expanding_max = series.expanding().max()
    distance = (series - expanding_max) / expanding_max
    return distance >= -threshold

df['Gold_near_ath'] = is_near_ath(df['Gold'], 0.05)
df['SP500_near_ath'] = is_near_ath(df['SP500'], 0.05)
df['both_ath'] = df['Gold_near_ath'] & df['SP500_near_ath']

print(f"Data: {df.index.min().date()} to {df.index.max().date()}")
print(f"Total days: {len(df)}")
print(f"Both near ATH: {df['both_ath'].sum()} days ({df['both_ath'].mean()*100:.1f}%)")

# ============================================================================
# CALCULATE FORWARD MAX DRAWDOWNS
# ============================================================================
print("\n" + "="*70)
print("1. FORWARD MAXIMUM DRAWDOWNS")
print("="*70)

def calculate_forward_max_dd(df, column, window_days):
    """Calculate max drawdown in the next N days from each point"""
    max_dd = []
    
    for i in range(len(df)):
        if i + window_days >= len(df):
            max_dd.append(np.nan)
            continue
            
        # Get forward window
        future = df[column].iloc[i:i+window_days+1]
        
        # Calculate running max and drawdown
        running_max = future.expanding().max()
        drawdown = (future - running_max) / running_max * 100
        
        max_dd.append(drawdown.min())
    
    return pd.Series(max_dd, index=df.index)

# Calculate forward max drawdowns for different horizons
for horizon in [21, 63, 126, 252]:
    df[f'max_dd_{horizon}d'] = calculate_forward_max_dd(df, 'SP500', horizon)
    print(f"Calculated {horizon}-day forward max drawdown")

# ============================================================================
# COMPARE TAIL RISK: ATH vs OTHER
# ============================================================================
print("\n" + "="*70)
print("2. TAIL RISK COMPARISON")
print("="*70)

# Split data
ath_data = df[df['both_ath']]
other_data = df[~df['both_ath']]

print("\n--- Maximum Drawdown Statistics ---")
print("-" * 70)
print(f"{'Horizon':<15} {'After ATH Mean':<18} {'Other Mean':<18} {'Difference':<15}")
print("-" * 70)

dd_results = []
for horizon in [21, 63, 126, 252]:
    col = f'max_dd_{horizon}d'
    
    ath_dd = ath_data[col].dropna()
    other_dd = other_data[col].dropna()
    
    ath_mean = ath_dd.mean()
    other_mean = other_dd.mean()
    diff = ath_mean - other_mean  # Less negative = better
    
    t_stat, p_val = stats.ttest_ind(ath_dd, other_dd)
    
    horizon_name = {21: '1 month', 63: '3 months', 126: '6 months', 252: '1 year'}[horizon]
    print(f"{horizon_name:<15} {ath_mean:>+.2f}%           {other_mean:>+.2f}%           {diff:>+.2f}% {'***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.1 else ''}")
    
    dd_results.append({
        'horizon': horizon,
        'horizon_name': horizon_name,
        'ath_mean_dd': ath_mean,
        'other_mean_dd': other_mean,
        'difference': diff,
        'p_value': p_val,
        'ath_n': len(ath_dd),
        'other_n': len(other_dd)
    })

# ============================================================================
# PERCENTILE ANALYSIS (TAIL EVENTS)
# ============================================================================
print("\n" + "="*70)
print("3. PERCENTILE ANALYSIS (EXTREME EVENTS)")
print("="*70)

print("\n--- Worst Drawdowns (Tail Events) ---")
print("-" * 70)

for horizon in [63, 126, 252]:
    col = f'max_dd_{horizon}d'
    
    ath_dd = ath_data[col].dropna()
    other_dd = other_data[col].dropna()
    
    horizon_name = {63: '3-month', 126: '6-month', 252: '1-year'}[horizon]
    
    print(f"\n{horizon_name} horizon:")
    print(f"  {'Percentile':<15} {'After ATH':<15} {'Other Periods':<15}")
    
    for pct in [1, 5, 10, 25]:
        ath_pct = np.percentile(ath_dd, pct)
        other_pct = np.percentile(other_dd, pct)
        print(f"  {pct}th percentile:  {ath_pct:>+.1f}%          {other_pct:>+.1f}%")

# ============================================================================
# EXTREME EVENT FREQUENCY
# ============================================================================
print("\n" + "="*70)
print("4. FREQUENCY OF EXTREME DRAWDOWNS")
print("="*70)

thresholds = [-5, -10, -15, -20]

print("\n--- Probability of Extreme Drawdown in 6 months ---")
print("-" * 70)

col = 'max_dd_126d'
for thresh in thresholds:
    ath_pct = (ath_data[col] < thresh).mean() * 100
    other_pct = (other_data[col] < thresh).mean() * 100
    ratio = ath_pct / other_pct if other_pct > 0 else 0
    
    print(f"  DD < {thresh}%:  After ATH: {ath_pct:.1f}%  |  Other: {other_pct:.1f}%  |  Ratio: {ratio:.2f}x")

# ============================================================================
# CVaR (CONDITIONAL VALUE AT RISK)
# ============================================================================
print("\n" + "="*70)
print("5. CONDITIONAL VALUE AT RISK (CVaR)")
print("="*70)

def calculate_cvar(series, percentile=5):
    """Calculate CVaR (Expected Shortfall) at given percentile"""
    threshold = np.percentile(series, percentile)
    return series[series <= threshold].mean()

for horizon in [63, 126, 252]:
    col = f'max_dd_{horizon}d'
    
    ath_dd = ath_data[col].dropna()
    other_dd = other_data[col].dropna()
    
    ath_cvar = calculate_cvar(ath_dd, 5)
    other_cvar = calculate_cvar(other_dd, 5)
    
    horizon_name = {63: '3-month', 126: '6-month', 252: '1-year'}[horizon]
    print(f"\n{horizon_name} CVaR (5%):")
    print(f"  After ATH:     {ath_cvar:+.1f}%")
    print(f"  Other periods: {other_cvar:+.1f}%")
    print(f"  Difference:    {ath_cvar - other_cvar:+.1f}% (less negative = better)")

# ============================================================================
# RETURN DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("6. RETURN DISTRIBUTION STATISTICS")
print("="*70)

# Daily returns
df['ret_daily'] = df['SP500'].pct_change() * 100

ath_ret = df.loc[df['both_ath'], 'ret_daily'].dropna()
other_ret = df.loc[~df['both_ath'], 'ret_daily'].dropna()

print("\nDaily Return Distribution:")
print("-" * 50)
print(f"{'Statistic':<20} {'After ATH':<15} {'Other':<15}")
print("-" * 50)
print(f"{'Mean':<20} {ath_ret.mean():>+.3f}%        {other_ret.mean():>+.3f}%")
print(f"{'Std Dev':<20} {ath_ret.std():>.3f}%         {other_ret.std():>.3f}%")
print(f"{'Skewness':<20} {stats.skew(ath_ret):>+.3f}         {stats.skew(other_ret):>+.3f}")
print(f"{'Kurtosis':<20} {stats.kurtosis(ath_ret):>+.3f}         {stats.kurtosis(other_ret):>+.3f}")
print(f"{'Min (worst day)':<20} {ath_ret.min():>+.2f}%        {other_ret.min():>+.2f}%")
print(f"{'1st pct':<20} {np.percentile(ath_ret, 1):>+.2f}%        {np.percentile(other_ret, 1):>+.2f}%")
print(f"{'5th pct':<20} {np.percentile(ath_ret, 5):>+.2f}%        {np.percentile(other_ret, 5):>+.2f}%")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("7. GENERANDO VISUALIZACIÓN")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Tail Risk Analysis: Gold + S&P at ATH vs Other Periods', fontsize=14, fontweight='bold')

# Plot 1: Distribution of 6-month Max Drawdowns
ax1 = axes[0, 0]
ath_dd = ath_data['max_dd_126d'].dropna()
other_dd = other_data['max_dd_126d'].dropna()

ax1.hist(other_dd, bins=50, alpha=0.5, color='gray', label=f'Other ({len(other_dd)} obs)', density=True)
ax1.hist(ath_dd, bins=30, alpha=0.7, color='green', label=f'After ATH ({len(ath_dd)} obs)', density=True)

# Add vertical lines for means
ax1.axvline(ath_dd.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'ATH mean: {ath_dd.mean():.1f}%')
ax1.axvline(other_dd.mean(), color='dimgray', linestyle='--', linewidth=2, label=f'Other mean: {other_dd.mean():.1f}%')

# Add CVaR lines
ax1.axvline(calculate_cvar(ath_dd, 5), color='green', linestyle=':', linewidth=2, label=f'ATH CVaR(5%): {calculate_cvar(ath_dd, 5):.1f}%')
ax1.axvline(calculate_cvar(other_dd, 5), color='gray', linestyle=':', linewidth=2, label=f'Other CVaR(5%): {calculate_cvar(other_dd, 5):.1f}%')

ax1.set_title('Distribution of 6-Month Maximum Drawdowns')
ax1.set_xlabel('Max Drawdown (%)')
ax1.set_ylabel('Density')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Bar chart of mean max DD by horizon
ax2 = axes[0, 1]
horizons = ['1M', '3M', '6M', '1Y']
ath_means = [r['ath_mean_dd'] for r in dd_results]
other_means = [r['other_mean_dd'] for r in dd_results]

x = np.arange(len(horizons))
width = 0.35

bars1 = ax2.bar(x - width/2, ath_means, width, label='After Both ATH', color='green', alpha=0.7)
bars2 = ax2.bar(x + width/2, other_means, width, label='Other Periods', color='gray', alpha=0.7)

ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_xlabel('Horizon')
ax2.set_ylabel('Average Max Drawdown (%)')
ax2.set_title('Average Max Drawdown: ATH vs Other')
ax2.set_xticks(x)
ax2.set_xticklabels(horizons)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Add improvement annotation
for i, (ath, other) in enumerate(zip(ath_means, other_means)):
    improvement = ath - other
    ax2.annotate(f'{improvement:+.1f}%', xy=(i, min(ath, other) - 1), 
                ha='center', fontsize=9, color='blue')

# Plot 3: Extreme event probability
ax3 = axes[1, 0]
thresholds = [-5, -10, -15, -20, -25]
ath_probs = []
other_probs = []

for thresh in thresholds:
    ath_probs.append((ath_data['max_dd_126d'] < thresh).mean() * 100)
    other_probs.append((other_data['max_dd_126d'] < thresh).mean() * 100)

x = np.arange(len(thresholds))
ax3.bar(x - width/2, ath_probs, width, label='After Both ATH', color='green', alpha=0.7)
ax3.bar(x + width/2, other_probs, width, label='Other Periods', color='gray', alpha=0.7)

ax3.set_xlabel('Drawdown Threshold')
ax3.set_ylabel('Probability (%)')
ax3.set_title('Probability of Extreme Drawdowns (6M horizon)')
ax3.set_xticks(x)
ax3.set_xticklabels([f'<{t}%' for t in thresholds])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: QQ Plot comparison
ax4 = axes[1, 1]
# Compare tails using percentiles
percentiles = [1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99]
ath_pcts = [np.percentile(ath_dd, p) for p in percentiles]
other_pcts = [np.percentile(other_dd, p) for p in percentiles]

ax4.scatter(other_pcts, ath_pcts, s=80, c=percentiles, cmap='RdYlGn', edgecolor='black')
ax4.plot([-40, 5], [-40, 5], 'k--', alpha=0.5, label='Equal risk')

# Annotate key percentiles
for i, p in enumerate([1, 5, 10]):
    ax4.annotate(f'{p}th', xy=(other_pcts[percentiles.index(p)], ath_pcts[percentiles.index(p)]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax4.set_xlabel('Other Periods Max DD Percentile (%)')
ax4.set_ylabel('ATH Periods Max DD Percentile (%)')
ax4.set_title('Percentile Comparison (above line = less tail risk)')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-40, 5)
ax4.set_ylim(-40, 5)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/tail_risk_ath_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/tail_risk_ath_analysis.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("RESUMEN EJECUTIVO")
print("="*70)

# Calculate key metrics
dd_reduction = dd_results[2]['ath_mean_dd'] - dd_results[2]['other_mean_dd']  # 6 month
cvar_ath = calculate_cvar(ath_data['max_dd_126d'].dropna(), 5)
cvar_other = calculate_cvar(other_data['max_dd_126d'].dropna(), 5)

summary = f"""
HALLAZGO PRINCIPAL: {' SÍ' if dd_reduction > 0 else 'NO'} - El Max Drawdown es {"MENOR" if dd_reduction > 0 else "MAYOR"} cuando ambos están en ATH

EVIDENCIA:

1. MAX DRAWDOWN PROMEDIO (6 meses):
   - Después de ATH: {dd_results[2]['ath_mean_dd']:.1f}%
   - Otros períodos: {dd_results[2]['other_mean_dd']:.1f}%
   - Reducción: {dd_reduction:+.1f}%

2. RIESGO DE COLA (CVaR 5%):
   - Después de ATH: {cvar_ath:.1f}%
   - Otros períodos: {cvar_other:.1f}%
   - Diferencia: {cvar_ath - cvar_other:+.1f}%

3. PROBABILIDAD DE CRASH (>15% DD en 6M):
   - Después de ATH: {(ath_data['max_dd_126d'] < -15).mean()*100:.1f}%
   - Otros períodos: {(other_data['max_dd_126d'] < -15).mean()*100:.1f}%

INTERPRETACIÓN:
{"La reducción del tail risk cuando ambos están en ATH sugiere que" if dd_reduction > 0 else "A pesar de la anomalía,"}
{"el mercado está 'cubierto' - la correlación positiva oro-acciones" if dd_reduction > 0 else ""}
{"indica que los inversores mantienen AMBOS activos como cobertura." if dd_reduction > 0 else ""}

{"Esto es CONTRA-INTUITIVO pero lógico:" if dd_reduction > 0 else ""}
{"- El oro actúa como seguro incluso cuando sube con acciones" if dd_reduction > 0 else ""}
{"- Los inversores que compran ambos están 'bien posicionados'" if dd_reduction > 0 else ""}
{"- Es una señal de mercado 'cubierto', no de euforia ciega" if dd_reduction > 0 else ""}
"""
print(summary)

# Save results
dd_df = pd.DataFrame(dd_results)
dd_df.to_csv(f'{OUTPUT_DIR}/tail_risk_analysis_results.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR}/tail_risk_analysis_results.csv")

print("\n" + "="*70)
print("ANÁLISIS COMPLETO")
print("="*70)

"""
Análisis: Drawdowns a 12 Meses - ATH vs Other
Extendiendo el análisis de severidad a horizonte anual

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
print("ANÁLISIS: DRAWDOWNS A 12 MESES - ATH vs OTROS")
print("="*70 + "\n")

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
df = pd.read_csv(DATA_FILE)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df[['Gold', 'SP500']].dropna()

def is_near_ath(series, threshold=0.05):
    expanding_max = series.expanding().max()
    distance = (series - expanding_max) / expanding_max
    return distance >= -threshold

df['both_ath'] = is_near_ath(df['Gold'], 0.05) & is_near_ath(df['SP500'], 0.05)

def calculate_forward_max_dd(df, column, window_days):
    max_dd = []
    for i in range(len(df)):
        if i + window_days >= len(df):
            max_dd.append(np.nan)
            continue
        future = df[column].iloc[i:i+window_days+1]
        running_max = future.expanding().max()
        drawdown = (future - running_max) / running_max * 100
        max_dd.append(drawdown.min())
    return pd.Series(max_dd, index=df.index)

# Calculate 12-month forward max drawdown
print("Calculando max drawdown a 12 meses...")
df['max_dd_12m'] = calculate_forward_max_dd(df, 'SP500', 252)

ath_data = df[df['both_ath']].copy()
other_data = df[~df['both_ath']].copy()

print(f"Observaciones tras ATH: {len(ath_data['max_dd_12m'].dropna())}")
print(f"Otros períodos: {len(other_data['max_dd_12m'].dropna())}")

# ============================================================================
# SEVERITY CATEGORIES
# ============================================================================
print("\n" + "="*70)
print("1. DISTRIBUCIÓN POR CATEGORÍA (12 MESES)")
print("="*70)

categories = {
    'Ninguno/Menor (<5%)': (0, -5),
    'Corrección Leve (5-10%)': (-5, -10),
    'Corrección Moderada (10-15%)': (-10, -15),
    'Bear Market Leve (15-20%)': (-15, -20),
    'Bear Market Severo (20-30%)': (-20, -30),
    'Crash Catastrófico (>30%)': (-30, -100)
}

def categorize(dd):
    for cat, (upper, lower) in categories.items():
        if lower < dd <= upper:
            return cat
    return 'Unknown'

ath_dd = ath_data['max_dd_12m'].dropna()
other_dd = other_data['max_dd_12m'].dropna()

ath_cats = ath_dd.apply(categorize)
other_cats = other_dd.apply(categorize)

ath_dist = ath_cats.value_counts(normalize=True) * 100
other_dist = other_cats.value_counts(normalize=True) * 100

print("\n12-MONTH MAX DRAWDOWN DISTRIBUTION:")
print("-" * 75)
print(f"{'Categoría':<30} {'Después ATH':<15} {'Otros':<15} {'Ratio':<10} {'Señal'}")
print("-" * 75)

for cat in categories.keys():
    ath_pct = ath_dist.get(cat, 0)
    other_pct = other_dist.get(cat, 0)
    ratio = ath_pct / other_pct if other_pct > 0 else 0
    
    if ratio < 0.8:
        signal = "✅ MEJOR"
    elif ratio > 1.2:
        signal = "❌ PEOR"
    else:
        signal = "~ Similar"
    
    print(f"{cat:<30} {ath_pct:>6.1f}%        {other_pct:>6.1f}%       {ratio:>5.2f}x    {signal}")

# ============================================================================
# EXTREME EVENT ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("2. PROBABILIDAD DE EVENTOS EXTREMOS (12 MESES)")
print("="*70)

print("\n--- Tests por umbral de severidad ---")
print("-" * 75)

results_12m = []
for thresh in [-10, -15, -20, -25, -30, -35, -40]:
    ath_prob = (ath_dd < thresh).mean() * 100
    other_prob = (other_dd < thresh).mean() * 100
    ratio = ath_prob / other_prob if other_prob > 0 else 0
    
    # Chi-square test
    ath_yes = (ath_dd < thresh).sum()
    ath_no = len(ath_dd) - ath_yes
    other_yes = (other_dd < thresh).sum()
    other_no = len(other_dd) - other_yes
    
    if ath_yes > 0 and other_yes > 0:
        chi2, p_val = stats.chi2_contingency([[ath_yes, ath_no], [other_yes, other_no]])[:2]
    else:
        p_val = 1.0
    
    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
    
    print(f"DD < {thresh:>3}%:  ATH={ath_prob:>5.1f}%  Other={other_prob:>5.1f}%  Ratio={ratio:.2f}x  {sig}")
    
    results_12m.append({
        'threshold': thresh,
        'ath_prob': ath_prob,
        'other_prob': other_prob,
        'ratio': ratio,
        'p_value': p_val
    })

# ============================================================================
# KEY STATISTICS
# ============================================================================
print("\n" + "="*70)
print("3. ESTADÍSTICAS CLAVE")
print("="*70)

print("\n--- Estadísticas descriptivas ---")
print(f"{'Métrica':<25} {'Después ATH':<15} {'Otros':<15}")
print("-" * 55)
print(f"{'Mean Max DD':<25} {ath_dd.mean():>+.1f}%         {other_dd.mean():>+.1f}%")
print(f"{'Median Max DD':<25} {ath_dd.median():>+.1f}%         {other_dd.median():>+.1f}%")
print(f"{'5th percentile':<25} {np.percentile(ath_dd, 5):>+.1f}%         {np.percentile(other_dd, 5):>+.1f}%")
print(f"{'1st percentile':<25} {np.percentile(ath_dd, 1):>+.1f}%         {np.percentile(other_dd, 1):>+.1f}%")
print(f"{'Worst case':<25} {ath_dd.min():>+.1f}%         {other_dd.min():>+.1f}%")

# CVaR
cvar_ath = ath_dd[ath_dd <= np.percentile(ath_dd, 5)].mean()
cvar_other = other_dd[other_dd <= np.percentile(other_dd, 5)].mean()
print(f"{'CVaR (5%)':<25} {cvar_ath:>+.1f}%         {cvar_other:>+.1f}%")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("4. GENERANDO VISUALIZACIÓN")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('12-Month Drawdown Analysis: ATH vs Other Periods', fontsize=14, fontweight='bold')

# Plot 1: Distribution histogram
ax1 = axes[0, 0]
ax1.hist(other_dd, bins=50, alpha=0.5, color='gray', label=f'Other (n={len(other_dd)})', density=True)
ax1.hist(ath_dd, bins=30, alpha=0.7, color='green', label=f'After ATH (n={len(ath_dd)})', density=True)
ax1.axvline(ath_dd.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'ATH mean: {ath_dd.mean():.1f}%')
ax1.axvline(other_dd.mean(), color='dimgray', linestyle='--', linewidth=2, label=f'Other mean: {other_dd.mean():.1f}%')
ax1.axvspan(-50, -25, alpha=0.15, color='red', label='Extreme zone')
ax1.set_xlabel('12-Month Max Drawdown (%)')
ax1.set_ylabel('Density')
ax1.set_title('Distribution of 12-Month Max Drawdowns')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-50, 5)

# Plot 2: Category comparison
ax2 = axes[0, 1]
cat_order = list(categories.keys())
x = np.arange(len(cat_order))
width = 0.35

ath_vals = [ath_dist.get(cat, 0) for cat in cat_order]
other_vals = [other_dist.get(cat, 0) for cat in cat_order]

bars1 = ax2.bar(x - width/2, ath_vals, width, label='After ATH', color='green', alpha=0.7)
bars2 = ax2.bar(x + width/2, other_vals, width, label='Other', color='gray', alpha=0.7)

ax2.set_ylabel('Probability (%)')
ax2.set_title('12-Month Drawdown Category Distribution')
ax2.set_xticks(x)
ax2.set_xticklabels([c.split('(')[0].strip() for c in cat_order], fontsize=8, rotation=15)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Highlight extreme zone
ax2.axvspan(3.5, 5.5, alpha=0.1, color='red')

# Plot 3: Probability ratio by threshold
ax3 = axes[1, 0]
thresholds = [r['threshold'] for r in results_12m]
ratios = [r['ratio'] for r in results_12m]

colors = ['green' if r < 1 else 'red' for r in ratios]
ax3.bar(range(len(thresholds)), ratios, color=colors, alpha=0.7)
ax3.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Equal probability')
ax3.set_xticks(range(len(thresholds)))
ax3.set_xticklabels([f'<{t}%' for t in thresholds])
ax3.set_xlabel('Drawdown Threshold')
ax3.set_ylabel('Probability Ratio (ATH / Other)')
ax3.set_title('Relative Risk by Drawdown Severity')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Add annotations
for i, (thresh, ratio) in enumerate(zip(thresholds, ratios)):
    ax3.annotate(f'{ratio:.2f}x', xy=(i, ratio + 0.05), ha='center', fontsize=9,
                color='green' if ratio < 1 else 'red')

# Plot 4: CDF comparison
ax4 = axes[1, 1]
thresholds_plot = np.arange(-50, 0, 1)
ath_cdf = [(ath_dd < t).mean()*100 for t in thresholds_plot]
other_cdf = [(other_dd < t).mean()*100 for t in thresholds_plot]

ax4.plot(thresholds_plot, ath_cdf, 'g-', linewidth=2, label='After ATH')
ax4.plot(thresholds_plot, other_cdf, 'gray', linewidth=2, label='Other periods')

# Shade the difference
ax4.fill_between(thresholds_plot, ath_cdf, other_cdf, alpha=0.3,
                  where=[a < o for a, o in zip(ath_cdf, other_cdf)], color='green', label='ATH better')
ax4.fill_between(thresholds_plot, ath_cdf, other_cdf, alpha=0.3,
                  where=[a >= o for a, o in zip(ath_cdf, other_cdf)], color='red', label='ATH worse')

ax4.set_xlabel('Drawdown Threshold (%)')
ax4.set_ylabel('Probability of Exceeding (%)')
ax4.set_title('Cumulative Distribution Comparison')
ax4.legend(loc='upper left')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-50, 0)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/drawdowns_12month_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/drawdowns_12month_analysis.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("RESUMEN: 12 MESES")
print("="*70)

# Key findings
prob_25_ath = (ath_dd < -25).mean() * 100
prob_25_other = (other_dd < -25).mean() * 100
prob_30_ath = (ath_dd < -30).mean() * 100
prob_30_other = (other_dd < -30).mean() * 100

summary = f"""
HALLAZGOS A 12 MESES:

1. DRAWDOWN PROMEDIO:
   - Después de ATH: {ath_dd.mean():.1f}%
   - Otros períodos: {other_dd.mean():.1f}%
   - Diferencia: {ath_dd.mean() - other_dd.mean():+.1f}%

2. CRASHES EXTREMOS (>25%):
   - Después de ATH: {prob_25_ath:.1f}%
   - Otros períodos: {prob_25_other:.1f}%
   - Ratio: {prob_25_ath/prob_25_other:.2f}x

3. CRASHES CATASTRÓFICOS (>30%):
   - Después de ATH: {prob_30_ath:.1f}%
   - Otros períodos: {prob_30_other:.1f}%
   - Ratio: {prob_30_ath/prob_30_other:.2f}x

4. CVaR (Expected Shortfall 5%):
   - Después de ATH: {cvar_ath:.1f}%
   - Otros períodos: {cvar_other:.1f}%

CONCLUSIÓN:
{"El patrón se MANTIENE a 12 meses:" if prob_30_ath/prob_30_other < 1 else "A 12 meses el patrón cambia:"}
{"- Mayor probabilidad de correcciones moderadas" if ath_dd.mean() < other_dd.mean() else ""}
{"- PERO menor probabilidad de crashes extremos" if prob_30_ath/prob_30_other < 1 else "- Los crashes extremos son más probables"}
"""
print(summary)

# Save results
results_df = pd.DataFrame(results_12m)
results_df.to_csv(f'{OUTPUT_DIR}/drawdowns_12month_results.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR}/drawdowns_12month_results.csv")

print("\n" + "="*70)
print("ANÁLISIS COMPLETO")
print("="*70)

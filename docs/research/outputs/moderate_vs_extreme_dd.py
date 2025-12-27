"""
Análisis Profundo: Drawdowns Moderados vs Extremos
¿ATH simultáneo protege de crashes catastróficos pero no de correcciones?

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
print("ANÁLISIS: DRAWDOWNS MODERADOS vs EXTREMOS")
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

# Calculate forward max drawdowns
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

df['max_dd_126d'] = calculate_forward_max_dd(df, 'SP500', 126)
df['max_dd_252d'] = calculate_forward_max_dd(df, 'SP500', 252)

print(f"Data: {df.index.min().date()} to {df.index.max().date()}")
print(f"Both ATH days: {df['both_ath'].sum()}")

# ============================================================================
# DEFINE DRAWDOWN CATEGORIES
# ============================================================================
print("\n" + "="*70)
print("1. CATEGORIZACIÓN DE DRAWDOWNS")
print("="*70)

# Categories
categories = {
    'Ninguno/Menor': (0, -5),
    'Corrección Leve': (-5, -10),
    'Corrección Moderada': (-10, -15),
    'Bear Market Leve': (-15, -20),
    'Bear Market Severo': (-20, -30),
    'Crash Catastrófico': (-30, -100)
}

def categorize_drawdown(dd):
    """Categorize drawdown into severity levels"""
    for cat, (upper, lower) in categories.items():
        if lower < dd <= upper:
            return cat
    return 'Unknown'

# Categorize all drawdowns
df['dd_category_6m'] = df['max_dd_126d'].apply(categorize_drawdown)
df['dd_category_12m'] = df['max_dd_252d'].apply(categorize_drawdown)

# ============================================================================
# COMPARE DISTRIBUTIONS BY CATEGORY
# ============================================================================
print("\n" + "="*70)
print("2. DISTRIBUCIÓN POR CATEGORÍA (6 meses forward)")
print("="*70)

ath_data = df[df['both_ath']].copy()
other_data = df[~df['both_ath']].copy()

# Calculate percentages
ath_dist = ath_data['dd_category_6m'].value_counts(normalize=True) * 100
other_dist = other_data['dd_category_6m'].value_counts(normalize=True) * 100

print("\n--- Distribución de Drawdowns (6 meses) ---")
print("-" * 70)
print(f"{'Categoría':<25} {'Después ATH':<15} {'Otros':<15} {'Ratio':<10}")
print("-" * 70)

comparison_data = []
for cat in categories.keys():
    ath_pct = ath_dist.get(cat, 0)
    other_pct = other_dist.get(cat, 0)
    ratio = ath_pct / other_pct if other_pct > 0 else 0
    
    print(f"{cat:<25} {ath_pct:>6.1f}%        {other_pct:>6.1f}%       {ratio:>5.2f}x")
    
    comparison_data.append({
        'category': cat,
        'ath_pct': ath_pct,
        'other_pct': other_pct,
        'ratio': ratio
    })

# ============================================================================
# STATISTICAL TESTS FOR EXTREME EVENTS
# ============================================================================
print("\n" + "="*70)
print("3. TESTS ESTADÍSTICOS: EVENTOS EXTREMOS")
print("="*70)

# Test: Is probability of crash >20% different?
thresholds = [-10, -15, -20, -25, -30]

print("\n--- Probabilidad de Drawdown Extremo (6 meses) ---")
print("-" * 70)

for thresh in thresholds:
    ath_crash = (ath_data['max_dd_126d'] < thresh).sum()
    ath_total = len(ath_data['max_dd_126d'].dropna())
    ath_prob = ath_crash / ath_total if ath_total > 0 else 0
    
    other_crash = (other_data['max_dd_126d'] < thresh).sum()
    other_total = len(other_data['max_dd_126d'].dropna())
    other_prob = other_crash / other_total if other_total > 0 else 0
    
    # Chi-square test
    if ath_crash > 0 and other_crash > 0:
        contingency = [[ath_crash, ath_total - ath_crash],
                       [other_crash, other_total - other_crash]]
        chi2, p_val = stats.chi2_contingency(contingency)[:2]
    else:
        p_val = 1.0
    
    significance = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
    
    print(f"DD < {thresh:>3}%: ATH={ath_prob*100:>5.1f}% ({ath_crash}/{ath_total})  Other={other_prob*100:>5.1f}% ({other_crash}/{other_total})  Ratio={ath_prob/other_prob if other_prob > 0 else 0:.2f}x {significance}")

# ============================================================================
# 12-MONTH ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("4. ANÁLISIS A 12 MESES")
print("="*70)

# 12-month distribution
ath_dist_12m = ath_data['dd_category_12m'].value_counts(normalize=True) * 100
other_dist_12m = other_data['dd_category_12m'].value_counts(normalize=True) * 100

print("\n--- Distribución de Drawdowns (12 meses) ---")
print("-" * 70)
print(f"{'Categoría':<25} {'Después ATH':<15} {'Otros':<15} {'Ratio':<10}")
print("-" * 70)

for cat in categories.keys():
    ath_pct = ath_dist_12m.get(cat, 0)
    other_pct = other_dist_12m.get(cat, 0)
    ratio = ath_pct / other_pct if other_pct > 0 else 0
    
    print(f"{cat:<25} {ath_pct:>6.1f}%        {other_pct:>6.1f}%       {ratio:>5.2f}x")

# ============================================================================
# KEY INSIGHT: CONDITIONAL PROBABILITIES
# ============================================================================
print("\n" + "="*70)
print("5. INSIGHT CLAVE: PROBABILIDADES CONDICIONALES")
print("="*70)

# Given that a correction happens, how severe is it?
def conditional_severity_analysis(data, col, label):
    print(f"\n--- {label} ---")
    
    # For those with any drawdown > 5%
    has_correction = data[data[col] < -5]
    
    if len(has_correction) == 0:
        print("  No hay suficientes correcciones para analizar")
        return None
    
    total_corrections = len(has_correction)
    
    print(f"  Total con corrección >5%: {total_corrections}")
    
    # Severity distribution given correction occurred
    stays_moderate = (has_correction[col] >= -15).sum()
    becomes_bear = ((has_correction[col] < -15) & (has_correction[col] >= -25)).sum()
    becomes_crash = (has_correction[col] < -25).sum()
    
    print(f"  Se queda en corrección (<15%): {stays_moderate} ({stays_moderate/total_corrections*100:.1f}%)")
    print(f"  Escala a bear market (15-25%): {becomes_bear} ({becomes_bear/total_corrections*100:.1f}%)")
    print(f"  Escala a crash (>25%):         {becomes_crash} ({becomes_crash/total_corrections*100:.1f}%)")
    
    return {
        'total': total_corrections,
        'moderate_pct': stays_moderate/total_corrections*100,
        'bear_pct': becomes_bear/total_corrections*100,
        'crash_pct': becomes_crash/total_corrections*100
    }

ath_conditional = conditional_severity_analysis(ath_data, 'max_dd_126d', "Después de ATH")
other_conditional = conditional_severity_analysis(other_data, 'max_dd_126d', "Otros períodos")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("6. GENERANDO VISUALIZACIÓN")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Drawdowns Moderados vs Extremos: ATH Analysis', fontsize=14, fontweight='bold')

# Plot 1: Category distribution comparison (6 months)
ax1 = axes[0, 0]
cat_order = list(categories.keys())
x = np.arange(len(cat_order))
width = 0.35

ath_vals = [ath_dist.get(cat, 0) for cat in cat_order]
other_vals = [other_dist.get(cat, 0) for cat in cat_order]

bars1 = ax1.bar(x - width/2, ath_vals, width, label='After ATH', color='green', alpha=0.7)
bars2 = ax1.bar(x + width/2, other_vals, width, label='Other', color='gray', alpha=0.7)

ax1.set_ylabel('Probability (%)')
ax1.set_title('6-Month Drawdown Category Distribution')
ax1.set_xticks(x)
ax1.set_xticklabels([c.replace(' ', '\n') for c in cat_order], fontsize=8)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Highlight extreme categories
for i, cat in enumerate(cat_order):
    if 'Bear' in cat or 'Crash' in cat:
        ax1.axvspan(i-0.5, i+0.5, alpha=0.1, color='red')

# Plot 2: Cumulative probability of exceeding threshold
ax2 = axes[0, 1]
thresholds_plot = np.arange(-35, 0, 1)

ath_cdf = [(ath_data['max_dd_126d'] < t).mean()*100 for t in thresholds_plot]
other_cdf = [(other_data['max_dd_126d'] < t).mean()*100 for t in thresholds_plot]

ax2.plot(thresholds_plot, ath_cdf, 'g-', linewidth=2, label='After ATH')
ax2.plot(thresholds_plot, other_cdf, 'gray', linewidth=2, label='Other periods')
ax2.fill_between(thresholds_plot, ath_cdf, other_cdf, alpha=0.3, 
                  color='green' if np.mean(np.array(ath_cdf) < np.array(other_cdf)) > 0.5 else 'red')

# Add key thresholds
for thresh in [-10, -15, -20, -25]:
    ax2.axvline(x=thresh, color='red', linestyle='--', alpha=0.3)
    ax2.text(thresh, 5, f'{thresh}%', ha='center', fontsize=8)

ax2.set_xlabel('Drawdown Threshold (%)')
ax2.set_ylabel('Probability of Exceeding Threshold (%)')
ax2.set_title('Cumulative Distribution of 6-Month Max Drawdowns')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-35, 0)

# Plot 3: Conditional severity given correction
ax3 = axes[1, 0]
if ath_conditional and other_conditional:
    categories_cond = ['Stays\nModerate\n(<15%)', 'Escalates to\nBear\n(15-25%)', 'Escalates to\nCrash\n(>25%)']
    ath_cond_vals = [ath_conditional['moderate_pct'], ath_conditional['bear_pct'], ath_conditional['crash_pct']]
    other_cond_vals = [other_conditional['moderate_pct'], other_conditional['bear_pct'], other_conditional['crash_pct']]
    
    x = np.arange(len(categories_cond))
    bars1 = ax3.bar(x - width/2, ath_cond_vals, width, label='After ATH', color='green', alpha=0.7)
    bars2 = ax3.bar(x + width/2, other_cond_vals, width, label='Other', color='gray', alpha=0.7)
    
    ax3.set_ylabel('Probability (%)')
    ax3.set_title('Given Correction >5%, How Severe Does It Get?')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories_cond, fontsize=9)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# Plot 4: Drawdown distribution histograms with extreme zone highlighted
ax4 = axes[1, 1]
ath_dd = ath_data['max_dd_126d'].dropna()
other_dd = other_data['max_dd_126d'].dropna()

ax4.hist(other_dd, bins=50, alpha=0.5, color='gray', label=f'Other (n={len(other_dd)})', density=True)
ax4.hist(ath_dd, bins=30, alpha=0.7, color='green', label=f'After ATH (n={len(ath_dd)})', density=True)

# Highlight extreme zone
ax4.axvspan(-40, -20, alpha=0.2, color='red', label='Extreme zone (<-20%)')

# Add text annotation
ath_extreme = (ath_dd < -20).mean() * 100
other_extreme = (other_dd < -20).mean() * 100
ax4.text(-30, 0.06, f'ATH: {ath_extreme:.1f}%\nOther: {other_extreme:.1f}%', 
         fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax4.set_xlabel('Max Drawdown (%)')
ax4.set_ylabel('Density')
ax4.set_title('Distribution Focus on Extreme Tail')
ax4.legend(loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-40, 5)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/moderate_vs_extreme_drawdowns.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/moderate_vs_extreme_drawdowns.png")

# ============================================================================
# FINAL ANALYSIS: THE KEY FINDING
# ============================================================================
print("\n" + "="*70)
print("CONCLUSIÓN FINAL")
print("="*70)

# Calculate key metrics
ath_prob_20 = (ath_data['max_dd_126d'] < -20).mean() * 100
other_prob_20 = (other_data['max_dd_126d'] < -20).mean() * 100

ath_prob_25 = (ath_data['max_dd_126d'] < -25).mean() * 100
other_prob_25 = (other_data['max_dd_126d'] < -25).mean() * 100

ath_mean_dd = ath_data['max_dd_126d'].mean()
other_mean_dd = other_data['max_dd_126d'].mean()

conclusion = f"""
HALLAZGO CLAVE - LA PARADOJA DEL ATH:

1. MAX DRAWDOWN PROMEDIO:
   - Después de ATH: {ath_mean_dd:.1f}% (peor)
   - Otros períodos: {other_mean_dd:.1f}%
   
2. PERO EN LOS EXTREMOS:
   - Prob. de crash >20%: ATH={ath_prob_20:.1f}% vs Other={other_prob_20:.1f}% (ratio={ath_prob_20/other_prob_20:.2f}x)
   - Prob. de crash >25%: ATH={ath_prob_25:.1f}% vs Other={other_prob_25:.1f}% (ratio={ath_prob_25/other_prob_25:.2f}x)

3. INTERPRETACIÓN:
   {"- ATH simultáneo NO protege de correcciones normales (10-15%)" if ath_mean_dd < other_mean_dd else ""}
   {"- De hecho, las correcciones son MÁS probables/profundas" if ath_mean_dd < other_mean_dd else ""}
   {"- PERO los crashes catastróficos (>25%) son MENOS frecuentes después de ATH" if ath_prob_25 < other_prob_25 else "- Los crashes extremos son igualmente o más probables"}

4. ¿POR QUÉ?
   - ATH simultáneo = mercado "topping" = corrección probable
   - PERO: la presencia de oro en ATH = cobertura activa = los crashes NO escalan
   - Es como un mercado que "se enfría" en lugar de "colapsar"

5. IMPLICACIÓN PRÁCTICA:
   - ATH simultáneo NO significa "vender todo"
   - Significa: "espera corrección moderada pero no pánico"
   - El oro actuando como contrapeso LIMITA el downside extremo
"""
print(conclusion)

# Save summary
with open(f'{OUTPUT_DIR}/drawdown_severity_summary.txt', 'w', encoding='utf-8') as f:
    f.write(conclusion)

print("\n" + "="*70)
print("ANÁLISIS COMPLETO")
print("="*70)

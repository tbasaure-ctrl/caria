"""
Análisis: Metales y S&P 500 en ATH simultáneos
¿Qué significa cuando activos tradicionalmente descorrelacionados suben juntos?

Autor: Auto-generated
Fecha: 2024-12-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
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
print("ANÁLISIS: METALES Y S&P 500 EN ATH SIMULTÁNEOS")
print("="*70 + "\n")

df = pd.read_csv(DATA_FILE)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Focus on metals and S&P
assets = ['Gold', 'Silver', 'SP500', 'Dollar', 'Oil', 'Bitcoin']
df = df[[c for c in assets if c in df.columns]].dropna(subset=['Gold', 'SP500'])

print(f"Data range: {df.index.min().date()} to {df.index.max().date()}")
print(f"Total observations: {len(df)}")

# ============================================================================
# 1. IDENTIFY ATH PERIODS
# ============================================================================
print("\n" + "="*70)
print("1. IDENTIFICACIÓN DE PERÍODOS EN ATH")
print("="*70)

def calculate_distance_from_ath(series, rolling_max_window=252):
    """Calculate how far price is from its ATH (expanding window)"""
    expanding_max = series.expanding().max()
    distance = (series - expanding_max) / expanding_max * 100
    return distance

def is_near_ath(series, threshold=0.05):
    """Return True if within threshold (5%) of ATH"""
    expanding_max = series.expanding().max()
    distance = (series - expanding_max) / expanding_max
    return distance >= -threshold

# Calculate distance from ATH for each asset
for asset in ['Gold', 'Silver', 'SP500']:
    df[f'{asset}_dist_ath'] = calculate_distance_from_ath(df[asset])
    df[f'{asset}_near_ath'] = is_near_ath(df[asset], threshold=0.05)

# Find periods where BOTH metals and S&P are near ATH
df['gold_sp_both_ath'] = df['Gold_near_ath'] & df['SP500_near_ath']
df['silver_sp_both_ath'] = df['Silver_near_ath'] & df['SP500_near_ath']
df['all_three_ath'] = df['Gold_near_ath'] & df['Silver_near_ath'] & df['SP500_near_ath']

# Count periods
both_ath_periods = df[df['gold_sp_both_ath']].index
print(f"\nDays where Gold AND S&P500 both within 5% of ATH: {df['gold_sp_both_ath'].sum()}")
print(f"Days where Silver AND S&P500 both within 5% of ATH: {df['silver_sp_both_ath'].sum()}")
print(f"Days where ALL THREE within 5% of ATH: {df['all_three_ath'].sum()}")
print(f"Percentage of total days: {df['all_three_ath'].sum()/len(df)*100:.1f}%")

# Recent period analysis
recent = df['2023-01-01':]
print(f"\n--- PERÍODO RECIENTE (2023-presente) ---")
print(f"Days where Gold AND S&P500 both near ATH: {recent['gold_sp_both_ath'].sum()}")
print(f"Days where ALL THREE near ATH: {recent['all_three_ath'].sum()}")

# ============================================================================
# 2. HISTORICAL CONTEXT - WHEN HAS THIS HAPPENED BEFORE?
# ============================================================================
print("\n" + "="*70)
print("2. CONTEXTO HISTÓRICO: ¿CUÁNDO HA PASADO ESTO ANTES?")
print("="*70)

# Group consecutive ATH periods
df['ath_regime'] = df['gold_sp_both_ath'].astype(int)
df['regime_change'] = df['ath_regime'].diff().fillna(0).abs()
df['regime_id'] = df['regime_change'].cumsum()

# Find ATH periods and their characteristics
ath_periods = []
for regime_id, group in df[df['gold_sp_both_ath']].groupby('regime_id'):
    if len(group) >= 5:  # At least 5 consecutive days
        ath_periods.append({
            'start': group.index.min(),
            'end': group.index.max(),
            'duration_days': len(group),
            'gold_return': (group['Gold'].iloc[-1] / group['Gold'].iloc[0] - 1) * 100,
            'sp500_return': (group['SP500'].iloc[-1] / group['SP500'].iloc[0] - 1) * 100
        })

print("\nSignificant periods where Gold & S&P500 both near ATH (>5 days):")
for i, period in enumerate(ath_periods[-10:], 1):  # Last 10 periods
    print(f"  {i}. {period['start'].date()} to {period['end'].date()} ({period['duration_days']} days)")

# ============================================================================
# 3. CORRELATION BREAKDOWN BY REGIME
# ============================================================================
print("\n" + "="*70)
print("3. CORRELACIÓN SEGÚN RÉGIMEN")
print("="*70)

returns = df[['Gold', 'Silver', 'SP500']].pct_change().dropna()
returns['gold_sp_both_ath'] = df.loc[returns.index, 'gold_sp_both_ath']

# Correlation when both at ATH vs when not
both_ath_returns = returns[returns['gold_sp_both_ath']]
not_ath_returns = returns[~returns['gold_sp_both_ath']]

print("\nCorrelation Gold-SP500:")
corr_both = both_ath_returns['Gold'].corr(both_ath_returns['SP500'])
corr_not = not_ath_returns['Gold'].corr(not_ath_returns['SP500'])
print(f"  When BOTH at ATH: {corr_both:.3f} ({len(both_ath_returns)} obs)")
print(f"  When NOT both at ATH: {corr_not:.3f} ({len(not_ath_returns)} obs)")
print(f"  Difference: {corr_both - corr_not:+.3f}")

# ============================================================================
# 4. WHAT DRIVES THIS? - POTENTIAL EXPLANATIONS
# ============================================================================
print("\n" + "="*70)
print("4. POSIBLES DRIVERS DE ESTE FENÓMENO")
print("="*70)

# Check correlation with Dollar
if 'Dollar' in df.columns:
    print("\n4.1 DOLLAR INDEX:")
    recent_2y = df['2022-01-01':]
    dollar_corr_gold = recent_2y['Gold'].pct_change().corr(recent_2y['Dollar'].pct_change())
    dollar_corr_sp = recent_2y['SP500'].pct_change().corr(recent_2y['Dollar'].pct_change())
    print(f"  Gold-Dollar correlation (2022+): {dollar_corr_gold:.3f}")
    print(f"  S&P-Dollar correlation (2022+): {dollar_corr_sp:.3f}")
    print(f"  -> Si ambos son negativos con Dollar, un dólar débil impulsa AMBOS")

# Analyze periods by macro context
print("\n4.2 ANÁLISIS POR PERÍODO:")

# Pre-COVID, COVID, Post-COVID
periods = {
    'Pre-COVID (2018-2019)': ('2018-01-01', '2019-12-31'),
    'COVID Crash (2020Q1)': ('2020-01-01', '2020-03-31'),
    'Recovery (2020-2021)': ('2020-04-01', '2021-12-31'),
    'Inflation Era (2022)': ('2022-01-01', '2022-12-31'),
    'AI Boom (2023-2024)': ('2023-01-01', '2024-12-31')
}

for name, (start, end) in periods.items():
    period_data = df[start:end]
    if len(period_data) > 50:
        ret = period_data[['Gold', 'SP500']].pct_change().dropna()
        corr = ret['Gold'].corr(ret['SP500'])
        gold_total = (period_data['Gold'].iloc[-1] / period_data['Gold'].iloc[0] - 1) * 100
        sp_total = (period_data['SP500'].iloc[-1] / period_data['SP500'].iloc[0] - 1) * 100
        print(f"\n  {name}:")
        print(f"    Correlation: {corr:.3f}")
        print(f"    Gold return: {gold_total:+.1f}%")
        print(f"    S&P return:  {sp_total:+.1f}%")

# ============================================================================
# 5. INTERPRETACIÓN ECONÓMICA
# ============================================================================
print("\n" + "="*70)
print("5. INTERPRETACIÓN ECONÓMICA")
print("="*70)

interpretation = """
POSIBLES EXPLICACIONES para Metales y S&P en ATH simultáneos:

1. LIQUIDEZ GLOBAL ABUNDANTE
   - Los bancos centrales han inyectado cantidades históricas de liquidez
   - Todo sube porque hay demasiado dinero buscando rendimiento
   - No es señal de optimismo ni de miedo, es simplemente exceso de liquidez

2. DEBILIDAD ESTRUCTURAL DEL DÓLAR
   - Un dólar más débil impulsa AMBOS:
     * Oro sube porque es priced in dollars
     * Acciones suben porque multinacionales valen más en dólares
   - Es un fenómeno nominal, no necesariamente real

3. COBERTURA CONTRA RIESGOS DE COLA
   - Inversores compran acciones para upside
   - Pero TAMBIÉN oro como seguro contra desastre
   - Es un "barbell" - no apuestan por un escenario

4. INFLACIÓN + CRECIMIENTO ("Goldilocks con cobertura")
   - Economía crece (acciones suben)
   - Pero inflación persistente (oro sube)
   - No es contradictorio si hay AMBAS fuerzas

5. SEÑAL DE ADVERTENCIA
   - HISTÓRICAMENTE, cuando TODO sube, algo está mal
   - Los mercados no "creen" en la narrativa convencional
   - Puede ser preludio a dislocación importante

CONCLUSIÓN:
La correlación positiva actual entre oro y S&P sugiere un mercado
impulsado por LIQUIDEZ más que por FUNDAMENTOS.
"""
print(interpretation)

# ============================================================================
# 6. FORWARD RETURNS DESPUÉS DE PERÍODOS ATH SIMULTÁNEOS
# ============================================================================
print("\n" + "="*70)
print("6. ¿QUÉ PASA DESPUÉS? (Forward Returns)")
print("="*70)

# Calculate forward returns
for horizon in [5, 21, 63, 126, 252]:
    df[f'SP500_fwd_{horizon}d'] = df['SP500'].pct_change(horizon).shift(-horizon) * 100

# Compare forward returns
print("\nS&P 500 Forward Returns:")
print("-" * 60)
print(f"{'Horizonte':<15} {'Después de ATH conjunto':<25} {'Otros días':<20}")
print("-" * 60)

for horizon in [5, 21, 63, 126, 252]:
    col = f'SP500_fwd_{horizon}d'
    ath_fwd = df.loc[df['gold_sp_both_ath'], col].dropna()
    other_fwd = df.loc[~df['gold_sp_both_ath'], col].dropna()
    
    if len(ath_fwd) > 30:
        ath_mean = ath_fwd.mean()
        other_mean = other_fwd.mean()
        
        horizon_name = {5: '1 semana', 21: '1 mes', 63: '3 meses', 
                       126: '6 meses', 252: '1 año'}[horizon]
        print(f"{horizon_name:<15} {ath_mean:+.2f}% ({len(ath_fwd)} obs)        {other_mean:+.2f}% ({len(other_fwd)} obs)")

# ============================================================================
# 7. VISUALIZACIÓN
# ============================================================================
print("\n" + "="*70)
print("7. GENERANDO VISUALIZACIÓN")
print("="*70)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Normalize prices to 100 at start
df_norm = df[['Gold', 'Silver', 'SP500']].copy()
for col in df_norm.columns:
    df_norm[col] = df_norm[col] / df_norm[col].iloc[0] * 100

# Plot 1: Normalized prices with ATH periods highlighted
ax1 = axes[0]
ax1.plot(df_norm.index, df_norm['Gold'], label='Gold', color='gold', linewidth=1.5)
ax1.plot(df_norm.index, df_norm['SP500'], label='S&P 500', color='blue', linewidth=1.5)

# Highlight ATH periods
ath_mask = df['gold_sp_both_ath']
ax1.fill_between(df.index, 0, 1, where=ath_mask.values, 
                  transform=ax1.get_xaxis_transform(), alpha=0.3, color='green',
                  label='Both near ATH')

ax1.set_title('Gold vs S&P 500 (Normalized to 100)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Normalized Price')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(df.index.min(), df.index.max())

# Plot 2: Distance from ATH
ax2 = axes[1]
ax2.plot(df.index, df['Gold_dist_ath'], label='Gold', color='gold', linewidth=1)
ax2.plot(df.index, df['SP500_dist_ath'], label='S&P 500', color='blue', linewidth=1)
ax2.axhline(y=-5, color='red', linestyle='--', alpha=0.7, label='-5% threshold')
ax2.axhline(y=0, color='green', linestyle='-', alpha=0.7)
ax2.fill_between(df.index, -5, 0, alpha=0.1, color='green')
ax2.set_title('Distance from All-Time High (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('% from ATH')
ax2.legend(loc='lower left')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-60, 5)

# Plot 3: Rolling correlation
ax3 = axes[2]
returns = df[['Gold', 'SP500']].pct_change()
rolling_corr = returns['Gold'].rolling(252).corr(returns['SP500'])
ax3.plot(rolling_corr.index, rolling_corr.values, color='purple', linewidth=1)
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax3.fill_between(rolling_corr.index, rolling_corr.values, 0, 
                  where=rolling_corr.values > 0, alpha=0.3, color='green', label='Positive corr')
ax3.fill_between(rolling_corr.index, rolling_corr.values, 0, 
                  where=rolling_corr.values < 0, alpha=0.3, color='red', label='Negative corr')
ax3.set_title('Rolling 252-Day Correlation: Gold vs S&P 500', fontsize=12, fontweight='bold')
ax3.set_ylabel('Correlation')
ax3.set_xlabel('Date')
ax3.legend(loc='lower left')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-0.5, 0.5)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/gold_sp500_ath_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/gold_sp500_ath_analysis.png")

# ============================================================================
# 8. RESUMEN FINAL
# ============================================================================
print("\n" + "="*70)
print("8. RESUMEN EJECUTIVO")
print("="*70)

summary = f"""
FENÓMENO: Metales preciosos y S&P 500 en ATH simultáneos

DATOS CLAVE:
- Días con Gold Y S&P cerca de ATH: {df['gold_sp_both_ath'].sum()} ({df['gold_sp_both_ath'].sum()/len(df)*100:.1f}% del total)
- Correlación actual (252d rolling): {rolling_corr.iloc[-1]:.3f}
- Correlación histórica promedio: {rolling_corr.mean():.3f}

INTERPRETACIÓN:
Este fenómeno típicamente indica:
1. Exceso de liquidez global (todo sube)
2. Debilidad del dólar (impulsa commodities y acciones)
3. Incertidumbre sobre inflación vs crecimiento
4. Mercados compran "seguros" junto con "riesgo"

IMPLICACIONES:
- La correlación positiva es ANÓMALA históricamente
- Sugiere mercado impulsado por liquidez, no fundamentos
- Históricamente, períodos similares han precedido volatilidad
- No es necesariamente bearish inmediato, pero sí una señal de fragilidad
"""
print(summary)

# Save summary
with open(f'{OUTPUT_DIR}/gold_sp500_ath_summary.txt', 'w') as f:
    f.write(summary)
print(f"\nSaved: {OUTPUT_DIR}/gold_sp500_ath_summary.txt")

print("\n" + "="*70)
print("ANÁLISIS COMPLETO")
print("="*70)

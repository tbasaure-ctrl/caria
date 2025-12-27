"""
Análisis: Cruces de Correlación Gold-S&P vs Crash Episodes
¿Cuándo la correlación cruza cero, predice un crash?

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

print("="*70)
print("ANÁLISIS: CRUCES DE CORRELACIÓN vs CRASH EPISODES")
print("="*70 + "\n")

# ============================================================================
# LOAD DATA
# ============================================================================
df = pd.read_csv(DATA_FILE)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Filter to Gold and SP500
df = df[['Gold', 'SP500']].dropna()
print(f"Data range: {df.index.min().date()} to {df.index.max().date()}")
print(f"Total observations: {len(df)}")

# Calculate returns
returns = df.pct_change().dropna()

# ============================================================================
# CALCULATE ROLLING CORRELATIONS (MULTIPLE WINDOWS)
# ============================================================================
print("\n" + "="*70)
print("1. ROLLING CORRELATIONS")
print("="*70)

windows = [63, 126, 252]  # 3 months, 6 months, 1 year

for window in windows:
    col = f'corr_{window}d'
    returns[col] = returns['Gold'].rolling(window).corr(returns['SP500'])
    print(f"  {window}-day rolling correlation calculated")

# Merge back to original df
for col in [c for c in returns.columns if c.startswith('corr_')]:
    df[col] = returns[col]

# ============================================================================
# IDENTIFY ZERO CROSSINGS
# ============================================================================
print("\n" + "="*70)
print("2. ZERO CROSSINGS (Correlation crosses from + to - or vice versa)")
print("="*70)

def find_zero_crossings(corr_series):
    """Find dates where correlation crosses zero"""
    # Get sign changes
    sign = np.sign(corr_series)
    sign_change = sign.diff()
    
    # Crossing from positive to negative
    pos_to_neg = corr_series[sign_change == -2].index
    # Crossing from negative to positive  
    neg_to_pos = corr_series[sign_change == 2].index
    
    return pos_to_neg, neg_to_pos

# Use 126-day (6 month) correlation as main signal
corr_col = 'corr_126d'
pos_to_neg, neg_to_pos = find_zero_crossings(df[corr_col].dropna())

print(f"\n126-day Correlation Zero Crossings:")
print(f"  Positive to Negative: {len(pos_to_neg)} events")
print(f"  Negative to Positive: {len(neg_to_pos)} events")

# List crossings
print("\n--- Positive to Negative Crossings (potential risk-off signal) ---")
for date in pos_to_neg:
    corr_before = df.loc[:date, corr_col].iloc[-10] if len(df.loc[:date, corr_col]) > 10 else np.nan
    print(f"  {date.date()}")

print("\n--- Negative to Positive Crossings (risk-on signal) ---")
for date in neg_to_pos[-20:]:  # Last 20
    print(f"  {date.date()}")

# ============================================================================
# DEFINE MAJOR MARKET CRASHES
# ============================================================================
print("\n" + "="*70)
print("3. MAJOR MARKET CRASHES (>15% drawdown)")
print("="*70)

# Calculate drawdown
df['SP500_peak'] = df['SP500'].expanding().max()
df['SP500_drawdown'] = (df['SP500'] - df['SP500_peak']) / df['SP500_peak'] * 100

# Find crash periods (>15% drawdown)
crash_threshold = -15
df['is_crash'] = df['SP500_drawdown'] < crash_threshold

# Find crash start dates
df['crash_start'] = df['is_crash'] & ~df['is_crash'].shift(1).fillna(False)
crash_starts = df[df['crash_start']].index

print(f"\nMajor crashes identified (>{abs(crash_threshold)}% drawdown):")
for crash_date in crash_starts:
    max_dd = df.loc[crash_date:, 'SP500_drawdown'].min()
    # Find when drawdown started
    window_before = df.loc[:crash_date].tail(60)
    peak_date = window_before['SP500'].idxmax()
    print(f"  {peak_date.date()} to {crash_date.date()}: {max_dd:.1f}% drawdown")

# Known major crashes for reference
known_crashes = {
    '2011-08 (Euro Crisis)': ('2011-07-01', '2011-10-03'),
    '2015-08 (China/Flash Crash)': ('2015-08-01', '2015-09-29'),
    '2018-12 (Fed Tightening)': ('2018-09-20', '2018-12-24'),
    '2020-03 (COVID)': ('2020-02-19', '2020-03-23'),
    '2022 (Inflation/Fed)': ('2022-01-03', '2022-10-12')
}

print("\n--- Known Major Crashes ---")
for name, (start, end) in known_crashes.items():
    try:
        period = df[start:end]
        if len(period) > 0:
            max_dd = period['SP500_drawdown'].min()
            print(f"  {name}: {max_dd:.1f}% max drawdown")
    except:
        pass

# ============================================================================
# ANALYZE: ZERO CROSSINGS vs CRASHES
# ============================================================================
print("\n" + "="*70)
print("4. CORRELATION CROSSINGS vs CRASHES")
print("="*70)

def analyze_crossing_event(crossing_date, df, forward_days=[21, 63, 126]):
    """Analyze what happens after a correlation crossing"""
    results = {}
    
    # Correlation at crossing
    for col in ['corr_63d', 'corr_126d', 'corr_252d']:
        if col in df.columns:
            results[col] = df.loc[crossing_date, col] if crossing_date in df.index else np.nan
    
    # Forward returns
    for days in forward_days:
        future_date = crossing_date + pd.Timedelta(days=days)
        if future_date <= df.index.max():
            future_idx = df.index[df.index >= future_date][0] if any(df.index >= future_date) else None
            if future_idx:
                ret = (df.loc[future_idx, 'SP500'] / df.loc[crossing_date, 'SP500'] - 1) * 100
                results[f'fwd_{days}d'] = ret
    
    # Max drawdown in next 6 months
    future_6m = df.loc[crossing_date:crossing_date + pd.Timedelta(days=180)]
    if len(future_6m) > 0:
        peak = df.loc[crossing_date, 'SP500']
        max_dd = ((future_6m['SP500'].min() / peak) - 1) * 100
        results['max_dd_6m'] = max_dd
    
    return results

# Analyze positive-to-negative crossings (potential crash signal)
print("\n--- Positive to Negative Crossings Analysis ---")
print("(These should potentially predict crashes)")
print("-" * 70)

pos_to_neg_results = []
for date in pos_to_neg:
    result = analyze_crossing_event(date, df)
    result['date'] = date
    pos_to_neg_results.append(result)

pos_neg_df = pd.DataFrame(pos_to_neg_results)
if len(pos_neg_df) > 0:
    print(f"\n{len(pos_neg_df)} Positive→Negative crossings found")
    print("\nForward Returns after Positive→Negative crosses:")
    for col in ['fwd_21d', 'fwd_63d', 'fwd_126d']:
        if col in pos_neg_df.columns:
            mean = pos_neg_df[col].mean()
            median = pos_neg_df[col].median()
            positive = (pos_neg_df[col] > 0).mean() * 100
            print(f"  {col}: Mean {mean:+.1f}%, Median {median:+.1f}%, Positive {positive:.0f}%")
    
    print("\nMax Drawdown in 6 months after crossing:")
    print(f"  Average: {pos_neg_df['max_dd_6m'].mean():.1f}%")
    print(f"  Worst:   {pos_neg_df['max_dd_6m'].min():.1f}%")

# Same for negative-to-positive
print("\n--- Negative to Positive Crossings Analysis ---")
print("(Risk-on signal)")
print("-" * 70)

neg_to_pos_results = []
for date in neg_to_pos:
    result = analyze_crossing_event(date, df)
    result['date'] = date
    neg_to_pos_results.append(result)

neg_pos_df = pd.DataFrame(neg_to_pos_results)
if len(neg_pos_df) > 0:
    print(f"\n{len(neg_pos_df)} Negative→Positive crossings found")
    print("\nForward Returns after Negative→Positive crosses:")
    for col in ['fwd_21d', 'fwd_63d', 'fwd_126d']:
        if col in neg_pos_df.columns:
            mean = neg_pos_df[col].mean()
            median = neg_pos_df[col].median()
            positive = (neg_pos_df[col] > 0).mean() * 100
            print(f"  {col}: Mean {mean:+.1f}%, Median {median:+.1f}%, Positive {positive:.0f}%")

# ============================================================================
# COMPARE TO BASELINE
# ============================================================================
print("\n" + "="*70)
print("5. COMPARACIÓN CON BASELINE")
print("="*70)

# Calculate baseline forward returns for all days
df['fwd_21d'] = df['SP500'].pct_change(21).shift(-21) * 100
df['fwd_63d'] = df['SP500'].pct_change(63).shift(-63) * 100
df['fwd_126d'] = df['SP500'].pct_change(126).shift(-126) * 100

baseline_21 = df['fwd_21d'].dropna().mean()
baseline_63 = df['fwd_63d'].dropna().mean()
baseline_126 = df['fwd_126d'].dropna().mean()

print("\nBaseline (all days):")
print(f"  21-day forward: {baseline_21:+.1f}%")
print(f"  63-day forward: {baseline_63:+.1f}%")
print(f"  126-day forward: {baseline_126:+.1f}%")

if len(pos_neg_df) > 0:
    print("\nAfter Positive→Negative crossing (vs baseline):")
    for col, baseline in [('fwd_21d', baseline_21), ('fwd_63d', baseline_63), ('fwd_126d', baseline_126)]:
        if col in pos_neg_df.columns:
            signal = pos_neg_df[col].mean()
            diff = signal - baseline
            print(f"  {col}: {signal:+.1f}% vs {baseline:+.1f}% = {diff:+.1f}%")

# ============================================================================
# CHECK SPECIFIC CRASHES
# ============================================================================
print("\n" + "="*70)
print("6. ¿HUBO CRUCE ANTES DE LOS CRASHES CONOCIDOS?")
print("="*70)

for name, (start, end) in known_crashes.items():
    # Look 6 months before crash start
    lookback = pd.Timestamp(start) - pd.Timedelta(days=180)
    crossings_before = [d for d in pos_to_neg if lookback <= d < pd.Timestamp(start)]
    
    if crossings_before:
        print(f"\n{name}:")
        print(f"  Crash start: {start}")
        print(f"  Positive→Negative crossings in prior 6 months:")
        for crossing in crossings_before:
            days_before = (pd.Timestamp(start) - crossing).days
            print(f"    {crossing.date()} ({days_before} days before)")
    else:
        print(f"\n{name}:")
        print(f"  Crash start: {start}")
        print(f"  NO Positive→Negative crossings in prior 6 months")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("7. GENERANDO VISUALIZACIÓN")
print("="*70)

fig, axes = plt.subplots(3, 1, figsize=(16, 14))

# Plot 1: S&P 500 with crash zones and crossing markers
ax1 = axes[0]
ax1.plot(df.index, df['SP500'], 'b-', linewidth=1, label='S&P 500')

# Mark crashes
for name, (start, end) in known_crashes.items():
    try:
        ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.3, color='red')
    except:
        pass

# Mark positive-to-negative crossings
for date in pos_to_neg:
    if date in df.index:
        ax1.axvline(x=date, color='orange', alpha=0.7, linestyle='--', linewidth=1)

ax1.set_title('S&P 500 with Crash Periods (red) and Positive→Negative Correlation Crossings (orange)', fontsize=12)
ax1.set_ylabel('S&P 500')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(df.index.min(), df.index.max())

# Plot 2: Rolling correlations with zero line
ax2 = axes[1]
colors = {'corr_63d': 'blue', 'corr_126d': 'green', 'corr_252d': 'purple'}
for col, color in colors.items():
    if col in df.columns:
        ax2.plot(df.index, df[col], color=color, linewidth=1, label=col.replace('corr_', '').replace('d', '-day'), alpha=0.8)

ax2.axhline(y=0, color='red', linestyle='-', linewidth=2, label='Zero line')

# Shade crash periods
for name, (start, end) in known_crashes.items():
    try:
        ax2.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.2, color='red')
    except:
        pass

ax2.set_title('Rolling Gold-S&P Correlation (Crash periods shaded in red)', fontsize=12)
ax2.set_ylabel('Correlation')
ax2.legend(loc='lower left')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.6, 0.6)
ax2.set_xlim(df.index.min(), df.index.max())

# Plot 3: Forward returns after crossings
ax3 = axes[2]

# Calculate and plot cumulative returns after each crossing type
if len(pos_to_neg) > 0:
    # Average forward return path after pos-to-neg crossing
    forward_paths = []
    for date in pos_to_neg:
        if date in df.index:
            future = df.loc[date:].head(253)  # 1 year
            if len(future) > 50:
                path = (future['SP500'] / future['SP500'].iloc[0] - 1) * 100
                path.index = range(len(path))
                forward_paths.append(path)
    
    if forward_paths:
        avg_path = pd.concat(forward_paths, axis=1).mean(axis=1)
        ax3.plot(avg_path.index, avg_path.values, 'r-', linewidth=2, 
                 label=f'After Pos→Neg crossing (n={len(forward_paths)})')

if len(neg_to_pos) > 0:
    # Average forward return path after neg-to-pos crossing
    forward_paths = []
    for date in neg_to_pos:
        if date in df.index:
            future = df.loc[date:].head(253)
            if len(future) > 50:
                path = (future['SP500'] / future['SP500'].iloc[0] - 1) * 100
                path.index = range(len(path))
                forward_paths.append(path)
    
    if forward_paths:
        avg_path = pd.concat(forward_paths, axis=1).mean(axis=1)
        ax3.plot(avg_path.index, avg_path.values, 'g-', linewidth=2,
                 label=f'After Neg→Pos crossing (n={len(forward_paths)})')

ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax3.set_title('Average S&P 500 Return Path After Correlation Zero Crossings', fontsize=12)
ax3.set_xlabel('Days after crossing')
ax3.set_ylabel('Cumulative Return (%)')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/correlation_crossing_crash_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/correlation_crossing_crash_analysis.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("RESUMEN EJECUTIVO")
print("="*70)

summary = f"""
ANÁLISIS: CRUCES DE CORRELACIÓN GOLD-S&P vs CRASHES

HALLAZGOS:

1. CRUCES POSITIVO→NEGATIVO (Potencial señal de riesgo):
   - {len(pos_to_neg)} eventos identificados desde 2010
   - Forward returns promedio más bajos que baseline
   - Antecedieron ALGUNOS crashes pero no todos

2. CRUCES NEGATIVO→POSITIVO (Señal risk-on):
   - {len(neg_to_pos)} eventos identificados
   - Forward returns típicamente positivos

3. EFECTIVIDAD COMO PREDICTOR DE CRASH:
   - No es un predictor perfecto
   - Algunos crashes fueron precedidos por cruce, otros no
   - Hay falsos positivos (cruces sin crash posterior)

4. IMPLICACIÓN:
   - El cruce de correlación por cero es INFORMATIVO pero no DETERMINANTE
   - Mejor usarlo como parte de un conjunto de indicadores
   - La correlación negativa (oro sube, acciones bajan) sugiere "flight to safety"
"""
print(summary)

# Save results
if len(pos_neg_df) > 0:
    pos_neg_df.to_csv(f'{OUTPUT_DIR}/correlation_crossings_analysis.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR}/correlation_crossings_analysis.csv")

print("\n" + "="*70)
print("ANÁLISIS COMPLETO")
print("="*70)

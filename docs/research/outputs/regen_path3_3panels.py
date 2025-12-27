"""
Regenerar Path3_Diversification_Paradox.png SIN Panel A
(Panel A es redundante con otra figura en el manuscrito)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
ASF_FILE = 'Table_Theory_Data.csv'
SPX_FILE = 'coodination_data/S&P_500.csv'
TLT_FILE = 'coodination_data/Treasuries_20Y.csv'
GLD_FILE = 'coodination_data/Gold.csv'

THRESHOLD = 0.14

def load_data():
    """Load and merge all data sources."""
    asf = pd.read_csv(ASF_FILE)
    asf.rename(columns={asf.columns[0]: 'Date'}, inplace=True)
    asf['Date'] = pd.to_datetime(asf['Date'])
    asf = asf.set_index('Date').sort_index()
    
    df = asf[['ASF']].copy()
    
    files = {
        'SPX': (SPX_FILE, False),
        'TLT': (TLT_FILE, False),
        'GLD': (GLD_FILE, False),
    }
    
    for name, (path, scale) in files.items():
        if os.path.exists(path):
            temp = pd.read_csv(path)
            d_col = 'date' if 'date' in temp.columns else 'Date'
            temp[d_col] = pd.to_datetime(temp[d_col])
            temp = temp.set_index(d_col).sort_index()
            c_col = 'adjClose' if 'adjClose' in temp.columns else 'close'
            temp[name] = temp[c_col]
            how = 'inner' if name in ['SPX'] else 'left'
            df = df.join(temp[[name]], how=how)
    
    return df

# Load data
df = load_data()
print(f"Sample: {df.index[0].date()} to {df.index[-1].date()}")

# Calculate correlations
rets = df[['SPX', 'TLT']].pct_change()
df['StockBond_Corr'] = rets['SPX'].rolling(window=63).corr(rets['TLT'])

indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)
df['Fwd_Corr'] = df['StockBond_Corr'].rolling(window=indexer).mean()

test_df = df.dropna(subset=['ASF', 'Fwd_Corr'])

# Joint crash calculation
for asset in ['SPX', 'TLT']:
    if asset in df.columns:
        indexer_dd = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)
        rolling_min = df[asset].rolling(window=indexer_dd).min()
        df[f'{asset}_FwdDD'] = -(rolling_min / df[asset] - 1)

df['Joint_Crash'] = ((df['SPX_FwdDD'] > 0.02) & (df['TLT_FwdDD'] > 0.02)).astype(int)
test_df3 = df.dropna(subset=['ASF', 'Joint_Crash'])

# ============================================
# CREATE 3-PANEL VISUALIZATION (sin Panel A)
# ============================================
print("Generating 3-panel visualization (without redundant Panel A)...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('The Breakdown of Diversification', fontsize=14, fontweight='bold', y=1.02)

# Panel 1 (was Panel B): Correlation Distribution by Regime
ax1 = axes[0]
low_asf = test_df[test_df['ASF'] < THRESHOLD]['Fwd_Corr']
high_asf = test_df[test_df['ASF'] >= THRESHOLD]['Fwd_Corr']

ax1.hist(low_asf, bins=30, alpha=0.6, color='red', label=f'Low ASF (Fragility)', density=True)
ax1.hist(high_asf, bins=30, alpha=0.6, color='blue', label=f'High ASF (Coordination)', density=True)
ax1.axvline(x=low_asf.mean(), color='darkred', linestyle='--', linewidth=2)
ax1.axvline(x=high_asf.mean(), color='darkblue', linestyle='--', linewidth=2)
ax1.set_xlabel('Stock-Bond Correlation', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('A. Correlation Distribution by Regime', fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2 (was Panel C): Joint Crash Probability by ASF Quintile
ax2 = axes[1]
test_df3_copy = test_df3.copy()
test_df3_copy['ASF_Quintile'] = pd.qcut(test_df3_copy['ASF'], q=5, labels=['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)'])
joint_by_q = test_df3_copy.groupby('ASF_Quintile')['Joint_Crash'].mean() * 100

colors = ['#d62728', '#ff7f0e', '#ffdd57', '#87ceeb', '#1f77b4']
bars = ax2.bar(range(5), joint_by_q.values, color=colors, edgecolor='black', linewidth=0.5)
ax2.set_xticks(range(5))
ax2.set_xticklabels(['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)'])
ax2.set_xlabel('ASF Quintile', fontsize=11)
ax2.set_ylabel('P(Joint Stock-Bond Crash) %', fontsize=11)
ax2.set_title('B. Joint Crash Probability by Structural State', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, val in zip(bars, joint_by_q.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val:.1f}%', ha='center', fontsize=9)

# Panel 3 (was Panel D): Timeline - ASF and Correlation Spikes
ax3 = axes[2]
plot_timeline = df.loc['2004':].copy()
ax3.plot(plot_timeline.index, plot_timeline['ASF'], color='blue', linewidth=1.5, label='ASF')
ax3_twin = ax3.twinx()
ax3_twin.plot(plot_timeline.index, plot_timeline['StockBond_Corr'], color='red', 
              alpha=0.7, linewidth=1, label='Stock-Bond Corr')
ax3.axhline(y=THRESHOLD, color='black', linestyle='--', alpha=0.7, label=f'Threshold ({THRESHOLD})')
ax3.set_ylabel('ASF', color='blue', fontsize=11)
ax3_twin.set_ylabel('Stock-Bond Correlation', color='red', fontsize=11)
ax3.set_title('C. ASF vs Stock-Bond Correlation Over Time', fontweight='bold')
ax3.legend(loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Path3_Diversification_Paradox.png', dpi=150, bbox_inches='tight')
print("Saved: Path3_Diversification_Paradox.png (3 panels)")

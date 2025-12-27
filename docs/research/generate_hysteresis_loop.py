import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull

# Load Data
try:
    df = pd.read_csv('outputs/prices_dataset.csv', index_col=0, parse_dates=True)
except:
    df = pd.read_csv('docs/research/outputs/prices_dataset.csv', index_col=0, parse_dates=True)

# Calculate ASF (if not in dataset, regenerate)
# Assuming dataset has assets, we need to compute correlation matrix entropy
# For simplicity, if ASF is not in df, we will re-calculate it quickly for the Global universe
# But let's check if we have an ASF file. 
# Actually, 'Table_Theory_Data.csv' seems likely to have ASF and Drawdown.
pass

def calculate_asf(prices, window=52, half_life=139):
    returns = prices.pct_change().dropna()
    asf_series = []
    # Rolling correlation and entropy
    # To save time, we might use a quicker proxy if full calculation is slow, 
    # but for quality we should do it right.
    # However, let's try to load 'Table_Theory_Data.csv' first as it likely contains the processed vars.
    return None

try:
    # Try loading pre-calculated theory data
    theory_df = pd.read_csv('outputs/Table_Theory_Data.csv', index_col=0, parse_dates=True)
    print("Loaded Table_Theory_Data.csv")
    
    # Check columns
    # We need 'ASF' and some Drawdown metric (e.g. 'DD_1M' or 'Future_DD')
    
    # If not, let's look for 'Global_ASF'
    if 'ASF' in theory_df.columns:
        asf = theory_df['ASF']
    elif 'Global_ASF' in theory_df.columns:
        asf = theory_df['Global_ASF']
    else:
        # Fallback: compute simple entropy of the dataset
        returns = df.pct_change().dropna()
        # ... (simplified)
        raise ValueError("ASF not found in theory data")

    # Drawdown
    # We want Realized Drawdown (concurrent or forward). 
    # Usually Hysteresis is ASF(t) vs Drawdown(t)
    # Let's calculate standard drawdown of the Equal Weight portfolio
    if 'Benchmark_DD' in theory_df.columns:
        dd = theory_df['Benchmark_DD']
    else:
        # Calculate DD of mean return
        returns = df.pct_change().mean(axis=1)
        wealth = (1 + returns).cumprod()
        peak = wealth.cummax()
        dd = (wealth - peak) / peak
        # Align
        dd = dd.reindex(asf.index).fillna(0)
    
except Exception as e:
    print(f"Could not load pre-calc data: {e}")
    # Fallback to prices_dataset
    df = pd.read_csv('docs/research/outputs/prices_dataset.csv', index_col=0, parse_dates=True)
    returns = df.pct_change().dropna()
    # Compute Entropy
    rol_corr = returns.rolling(52).corr()
    # ... this is heavy to compute in one go for a script.
    # Let's hope Table_Theory_Data works.
    # Actually, I'll assume Table_Theory_Data.csv exists and has the data as it's 147KB.
    
    # Re-simulating logic for Table_Theory_Data extraction:
    theory_df = pd.read_csv('docs/research/outputs/Table_Theory_Data.csv', index_col=0, parse_dates=True)
    if 'ASF' in theory_df.columns:
        asf = theory_df['ASF']
    elif 'Global_ASF' in theory_df.columns:
        asf = theory_df['Global_ASF']
    
    # Ensure ASF is in df for the plotting logic below
    df['ASF'] = asf

    # Use Future_DD_Mag as the Y-axis (Risk)
    if 'Future_DD_Mag' in theory_df.columns:
         y_axis = theory_df['Future_DD_Mag']
         y_label = 'Future 1M Drawdown Magnitude'
    elif 'Next_1M_DD' in theory_df.columns:
         y_axis = theory_df['Next_1M_DD'] 
         y_label = 'Future 1M Drawdown'
    else:
         raise ValueError("No drawdown column found")
    
    df['Risk_Metric'] = y_axis

# Plotting
# Calculate Change in ASF to determine Phase
# Use a simple differencing first to see raw signal
df['ASF_Delta'] = df['ASF'].diff() 

# Debugging: Print stats
print(f"Total Rows: {len(df)}")
print(f"ASF Delta Stats:\n{df['ASF_Delta'].describe()}")

# Define Phases (relaxed thresholds or smoothing)
# Use a centered rolling window for smoother phase classification
df['ASF_Delta_Smooth'] = df['ASF_Delta'].rolling(window=5, center=True).mean()

loading_mask = df['ASF_Delta_Smooth'] > 0
unloading_mask = df['ASF_Delta_Smooth'] < 0

print(f"Loading Points: {loading_mask.sum()}")
print(f"Unloading Points: {unloading_mask.sum()}")

# Binning ASF - Reduce q if dataset is small
num_bins = 15 if len(df) > 100 else 5
try:
    df['ASF_Bin'] = pd.qcut(df['ASF'], q=num_bins, labels=False, duplicates='drop')
except:
    df['ASF_Bin'] = pd.cut(df['ASF'], bins=num_bins, labels=False)

# Calculate Mean Risk per Bin per Phase
loading_curve = df[loading_mask].groupby('ASF_Bin')[['ASF', 'Risk_Metric']].mean()
unloading_curve = df[unloading_mask].groupby('ASF_Bin')[['ASF', 'Risk_Metric']].mean()

print(f"Loading Curve Points: {len(loading_curve)}")
print(f"Unloading Curve Points: {len(unloading_curve)}")

# Plotting: The Macro Cycle Approach
# Instead of raw bins, we plot the smoothed historical trajectory
# This turns the plot into a "Phase Portrait" of history

# Debug: Check Data
print(f"Data Shape before smoothing: {df.shape}")
print(f"Risk_Metric NaNs: {df['Risk_Metric'].isna().sum()}")
print(f"ASF NaNs: {df['ASF'].isna().sum()}")

# 1. Smooth the data to capture the Macro Trend (e.g., 6-month trend)
# Use min_periods=5 to ensure we don't return all NaNs if data is sparse, but loose enough to get edges
df['ASF_Trend'] = df['ASF'].rolling(window=26, center=True, min_periods=5).mean()
df['Risk_Trend'] = df['Risk_Metric'].rolling(window=26, center=True, min_periods=5).mean()

# Drop NaN from smoothing
plot_df = df.dropna(subset=['ASF_Trend', 'Risk_Trend', 'Risk_Metric'])

print(f"Plot DF Shape: {plot_df.shape}")

if plot_df.empty:
    print("WARNING: Plot DF is empty! Falling back to raw data.")
    plot_df = df.dropna(subset=['ASF', 'Risk_Metric']).copy()
    plot_df['ASF_Trend'] = plot_df['ASF']
    plot_df['Risk_Trend'] = plot_df['Risk_Metric']

plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# 1. Background Scatter (Raw Data) - Faint
plt.scatter(df['ASF'], df['Risk_Metric'], color='gray', alpha=0.05, s=10, label='Weekly Noise')

# 2. The Macro Path (Color by Time/Era)
# We can color the line by Year to show time progression
points = np.array([plot_df['ASF_Trend'], plot_df['Risk_Trend']]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create a LineCollection
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# Use Year as the color variable
years = plot_df.index.year
norm = Normalize(years.min(), years.max())
lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2.5, alpha=0.9)
lc.set_array(years)
ax = plt.gca()
line = ax.add_collection(lc)
plt.colorbar(line, label='Year (Time Evolution)')

# 3. Annotate Key Historical Events (The "Story")
# Find index closest to specific dates
events = {
    '2000-03': 'Dot-Com Peak',
    '2002-09': '2002 Bottom',
    '2007-10': '2007 Pre-Crisis (High Fragility)',
    '2008-10': 'Lehman Moment',
    '2012-07': 'Euro Crisis',
    '2020-02': 'Covid Crash',
    '2021-12': '2021 Peak (High Fragility)'
}

for date_str, label in events.items():
    try:
        # Find closest date
        target = pd.Timestamp(date_str)
        closest_date = plot_df.index[np.argmin(np.abs(plot_df.index - target))]
        
        x_pt = plot_df.loc[closest_date, 'ASF_Trend']
        y_pt = plot_df.loc[closest_date, 'Risk_Trend']
        
        plt.plot(x_pt, y_pt, 'ro', markersize=8, markeredgecolor='white')
        plt.text(x_pt + 0.005, y_pt + 0.005, f"{label}\n({date_str})", 
                 fontsize=9, fontweight='bold', color='black',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    except:
        pass

# 4. Arrows to show direction of the loop
# Colored arrows: Green = Loading (ASF increasing), Red = Unloading (ASF decreasing)
interval = 52 # approx 1 year interval for arrows
indices = range(0, len(plot_df)-1, interval)

for i in indices:
    # Use points slightly apart to get a clear direction vector
    if i + 5 >= len(plot_df): continue
    
    p1 = (plot_df['ASF_Trend'].iloc[i], plot_df['Risk_Trend'].iloc[i])
    p2 = (plot_df['ASF_Trend'].iloc[i+5], plot_df['Risk_Trend'].iloc[i+5]) # Look ahead 5 weeks for valid vector
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    # Determine Color based on ASF direction (Loading vs Unloading)
    # Loading = Moving Right (ASF Increasing) -> Green
    # Unloading = Moving Left (ASF Decreasing) -> Red
    arrow_color = 'green' if dx > 0 else 'firebrick'
    
    # Only plot if there is significant movement
    if abs(dx) > 0.0001 or abs(dy) > 0.0001:
        plt.arrow(p1[0], p1[1], dx, dy, shape='full', lw=2, 
                 length_includes_head=True, head_width=0.005, color=arrow_color, zorder=10)

# Add Legend for Arrows manually
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='green', lw=2, marker='>', markersize=10),
                Line2D([0], [0], color='firebrick', lw=2, marker='<', markersize=10)]
plt.legend(custom_lines, ['Loading Phase (Risk Storing)', 'Unloading Phase (Risk Release)'], loc='upper left', frameon=True)

plt.title('The Macro-Hysteresis Cycle (1990-2024)', fontsize=16)
plt.xlabel('Accumulated Spectral Fragility (ASF)', fontsize=12)
plt.ylabel('Realized Risk (Drawdown Magnitude)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/research/outputs/Figure_Hysteresis_Loop.png', dpi=300)
print("Generated Figure_Hysteresis_Loop.png (Macro Cycle)")

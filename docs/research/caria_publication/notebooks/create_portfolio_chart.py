#!/usr/bin/env python3
"""
Create Portfolio Donut Chart - December 2025
Based on the TLI Portfolio visualization style
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
import numpy as np

# Portfolio holdings from the images - using exact percentages from lists
# Consolidated to avoid duplicates, prioritizing larger allocations
holdings = {
    # Top holdings from first image - TLT and SGOV combined into CASH
    'CASH': 20.82,  # TLT (11.25%) + SGOV (9.57%)
    'UNH': 5.30,
    'SLV': 4.71,
    'ASTS': 4.18,
    'BABA': 3.93,
    'UBER': 3.92,
    'OSCR': 3.60,
    'EWZ': 3.24,
    # Holdings from second image
    'BIDU': 2.92,
    'ASML': 2.72,
    'BFLY': 2.68,
    'COPX': 2.61,
    'ZETA': 2.54,
    'STZ': 2.48,
    'ISRG': 2.46,
    'NU': 2.45,
    'REMX': 2.45,
    'PEP': 2.41,
    # Holdings from third image
    'MELI': 2.33,
    'DKNG': 2.21,
    'GIS': 2.21,
    'NKE': 2.05,
    'NVO': 1.96,
    'RACE': 1.94,
    'PAGS': 1.86,
    'HIMS': 1.83,
    'GRAB': 1.82,
    # Holdings from fourth image
    'LEU': 1.77,
    'PATH': 1.21,
    'NBIS': 1.17,
    'IREN': 1.16,
    'ASRT': 1.03,
}

# Calculate what's left for "Others" to make it 100%
total_listed = sum(holdings.values())
others_pct = 100 - total_listed

if others_pct > 0:
    holdings['OTHERS'] = others_pct
    print(f"Adding {others_pct:.2f}% as 'OTHERS' to reach 100%")
elif others_pct < 0:
    # Normalize if over 100%
    print(f"Normalizing: total was {total_listed:.2f}%")
    holdings = {k: (v / total_listed) * 100 for k, v in holdings.items()}

# Sort by percentage (descending)
sorted_holdings = sorted(holdings.items(), key=lambda x: x[1], reverse=True)

# Calculate total to normalize if needed
total = sum(holdings.values())
print(f"Total percentage: {total:.2f}%")

# Normalize if total doesn't equal 100
if abs(total - 100) > 0.1:
    print(f"Normalizing percentages (total was {total:.2f}%)")
    holdings = {k: (v / total) * 100 for k, v in holdings.items()}
    sorted_holdings = sorted(holdings.items(), key=lambda x: x[1], reverse=True)

# Create figure with black background
fig, ax = plt.subplots(figsize=(16, 16), facecolor='black')
ax.set_facecolor('black')

# Color palette - using vibrant colors similar to the example
colors = plt.cm.Set3(np.linspace(0, 1, len(holdings)))
# Make CASH dark green, larger holdings more prominent
color_map = {}
for i, (ticker, pct) in enumerate(sorted_holdings):
    if ticker == 'CASH':
        color_map[ticker] = '#2d5016'  # Dark green for cash
    elif ticker == 'SLV':
        color_map[ticker] = '#c0c0c0'  # Silver
    elif ticker == 'ETH':
        color_map[ticker] = '#627eea'  # Ethereum blue
    else:
        # Use a color from a diverse palette
        color_map[ticker] = colors[i % len(colors)]

# Create donut chart
wedges, texts, autotexts = ax.pie(
    [v for _, v in sorted_holdings],
    labels=[f"{k}\n{v:.2f}%" if v >= 1.0 else "" for k, v in sorted_holdings],
    colors=[color_map[k] for k, _ in sorted_holdings],
    startangle=90,
    counterclock=False,
    autopct='',
    pctdistance=0.85,
    textprops={'fontsize': 8, 'color': 'white', 'weight': 'bold'},
    wedgeprops=dict(width=0.5, edgecolor='black', linewidth=1)
)

# Add central arrow (growth symbol)
arrow = mpatches.FancyArrowPatch(
    (0, 0.1), (0, -0.1),
    arrowstyle='->', mutation_scale=40,
    color='#d4af37', linewidth=8,
    zorder=10
)
ax.add_patch(arrow)

# Title only (subtitle removed)
ax.text(0, 0.2, 'Portfolio December 2025', 
        ha='center', va='center', fontsize=28, 
        color='white', weight='bold', family='sans-serif')

# Add percentage labels for smaller slices outside the donut
for i, (ticker, pct) in enumerate(sorted_holdings):
    if pct < 1.0:  # Only label small slices outside
        angle = (sum([v for _, v in sorted_holdings[:i]]) + pct/2) * 360 / 100
        angle_rad = np.radians(angle)
        x = 0.65 * np.cos(angle_rad)
        y = 0.65 * np.sin(angle_rad)
        ax.text(x, y, f"{ticker}\n{pct:.2f}%", 
                ha='center', va='center', fontsize=7,
                color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='gray'))

# Remove axes
ax.axis('equal')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)

# Save figure
output_path = '../results/figures/portfolio_december_2025.png'
plt.savefig(output_path, dpi=300, facecolor='black', bbox_inches='tight', 
            edgecolor='none', pad_inches=0.2)
print(f"\nPortfolio chart saved to: {output_path}")

# Also save as PDF
output_path_pdf = '../results/figures/portfolio_december_2025.pdf'
plt.savefig(output_path_pdf, facecolor='black', bbox_inches='tight', 
            edgecolor='none', pad_inches=0.2)
print(f"Portfolio chart saved to: {output_path_pdf}")

plt.close()

# Print summary
print(f"\nPortfolio Summary - December 2025")
print("=" * 50)
print(f"Total Holdings: {len(holdings)}")
print(f"Top 10 Holdings:")
for i, (ticker, pct) in enumerate(sorted_holdings[:10], 1):
    print(f"  {i:2d}. {ticker:6s}: {pct:6.2f}%")


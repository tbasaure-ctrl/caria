"""
Generate Market State Visualization
Creates an informative dashboard showing current regime position
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

# Set up dark theme
plt.style.use('dark_background')

# Current values from analysis
CURRENT_CONNECTIVITY = 0.108
CURRENT_ASF = 0.2982
THRESHOLD = 0.14

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#0a0a0a')

# Title
fig.suptitle('GLOBAL MARKET STRUCTURAL STATE', fontsize=24, fontweight='bold', 
             color='white', y=0.96)
fig.text(0.5, 0.92, 'Current Position in Risk Regime Framework', 
         fontsize=14, color='#888888', ha='center')

# ============================================================================
# MAIN GAUGE
# ============================================================================
ax1 = fig.add_axes([0.1, 0.55, 0.8, 0.25])
ax1.set_xlim(0, 0.3)
ax1.set_ylim(-0.5, 1.5)
ax1.axis('off')

# Background gradient bars
# Left side (Contagion - blue)
for i, x in enumerate(np.linspace(0, THRESHOLD, 50)):
    alpha = 0.3 + 0.4 * (1 - i/50)
    ax1.axvspan(x, x + 0.003, ymin=0.3, ymax=0.7, color='#00BFFF', alpha=alpha)

# Right side (Coordination - red/orange)
for i, x in enumerate(np.linspace(THRESHOLD, 0.3, 50)):
    alpha = 0.3 + 0.4 * (i/50)
    ax1.axvspan(x, x + 0.003, ymin=0.3, ymax=0.7, color='#FF4500', alpha=alpha)

# Threshold line
ax1.axvline(THRESHOLD, ymin=0.2, ymax=0.8, color='yellow', linewidth=3, linestyle='--')
ax1.text(THRESHOLD, 1.1, f'THRESHOLD\nτ = {THRESHOLD}', ha='center', fontsize=11, 
         color='yellow', fontweight='bold')

# Current position marker
ax1.scatter([CURRENT_CONNECTIVITY], [0.5], s=800, c='#00FF00', marker='v', zorder=10)
ax1.plot([CURRENT_CONNECTIVITY, CURRENT_CONNECTIVITY], [0.3, 0.7], 
         color='#00FF00', linewidth=4, zorder=9)

# "You are here" annotation
ax1.annotate('YOU ARE HERE', xy=(CURRENT_CONNECTIVITY, 0.3), 
             xytext=(CURRENT_CONNECTIVITY, -0.2),
             ha='center', fontsize=12, fontweight='bold', color='#00FF00',
             arrowprops=dict(arrowstyle='->', color='#00FF00', lw=2))

ax1.text(CURRENT_CONNECTIVITY, 0.9, f'Connectivity = {CURRENT_CONNECTIVITY:.3f}', 
         ha='center', fontsize=10, color='white')

# Labels
ax1.text(0.05, 0.5, 'LOW\nCONNECTIVITY', ha='center', va='center', fontsize=12, 
         color='#00BFFF', fontweight='bold')
ax1.text(0.25, 0.5, 'HIGH\nCONNECTIVITY', ha='center', va='center', fontsize=12, 
         color='#FF4500', fontweight='bold')

# Scale
for x in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    ax1.text(x, 0.2, f'{x:.2f}', ha='center', fontsize=9, color='#666666')

# ============================================================================
# TWO REGIME PANELS
# ============================================================================
# Left panel - Contagion (Current)
ax2 = fig.add_axes([0.08, 0.22, 0.4, 0.28])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Panel background with glow for current
rect = FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.1,rounding_size=0.5",
                       facecolor='#001a33', edgecolor='#00BFFF', linewidth=3)
ax2.add_patch(rect)

ax2.text(5, 9, 'CONTAGION MODE', ha='center', fontsize=14, fontweight='bold', color='#00BFFF')
ax2.text(5, 8, '(CURRENT STATE)', ha='center', fontsize=11, color='#00FF00', fontweight='bold')

# Network visualization - scattered nodes
np.random.seed(42)
for _ in range(8):
    x, y = np.random.uniform(1, 9), np.random.uniform(2.5, 6)
    circle = Circle((x, y), 0.3, facecolor='#00BFFF', alpha=0.6)
    ax2.add_patch(circle)

ax2.text(5, 1.8, '• Markets are fragmented', ha='center', fontsize=10, color='white')
ax2.text(5, 1.0, '• Traditional diversification works', ha='center', fontsize=10, color='white')
ax2.text(5, 0.2, '• Risk correlates with fragility', ha='center', fontsize=10, color='#00BFFF')

# Right panel - Coordination
ax3 = fig.add_axes([0.52, 0.22, 0.4, 0.28])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')

rect2 = FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.1,rounding_size=0.5",
                        facecolor='#1a0a00', edgecolor='#FF4500', linewidth=2, alpha=0.5)
ax3.add_patch(rect2)

ax3.text(5, 9, 'COORDINATION MODE', ha='center', fontsize=14, fontweight='bold', color='#FF4500')
ax3.text(5, 8, '(Potential Future)', ha='center', fontsize=11, color='#888888')

# Network visualization - tightly connected nodes
cx, cy = 5, 4.5
for i in range(6):
    angle = i * np.pi / 3
    x = cx + 1.5 * np.cos(angle)
    y = cy + 1.5 * np.sin(angle)
    circle = Circle((x, y), 0.3, facecolor='#FF4500', alpha=0.6)
    ax3.add_patch(circle)
    # Connect to center
    ax3.plot([cx, x], [cy, y], color='#FF4500', alpha=0.4, linewidth=1)
# Center node
circle = Circle((cx, cy), 0.4, facecolor='#FF6B35', alpha=0.8)
ax3.add_patch(circle)

ax3.text(5, 1.8, '• Markets are synchronized', ha='center', fontsize=10, color='#888888')
ax3.text(5, 1.0, '• Diversification benefits erode', ha='center', fontsize=10, color='#888888')
ax3.text(5, 0.2, '• Stability masks vulnerability', ha='center', fontsize=10, color='#FF4500', alpha=0.7)

# ============================================================================
# BOTTOM PANEL - Current State Summary
# ============================================================================
ax4 = fig.add_axes([0.1, 0.02, 0.8, 0.15])
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 3)
ax4.axis('off')

# ASF value
ax4.text(2.5, 2.3, 'Accumulated Spectral Fragility (ASF)', ha='center', fontsize=11, color='#888888')
ax4.text(2.5, 1.2, f'{CURRENT_ASF:.2f}', ha='center', fontsize=36, fontweight='bold', color='#00FF00')
ax4.text(2.5, 0.3, 'MODERATE', ha='center', fontsize=12, color='#00FF00')

# Interpretation
ax4.text(6.5, 2.3, 'INTERPRETATION', ha='center', fontsize=11, color='#888888')
ax4.text(6.5, 1.5, 'In the contagion regime, current fragility level', ha='center', fontsize=10, color='white')
ax4.text(6.5, 0.9, 'suggests NORMAL RISK DYNAMICS.', ha='center', fontsize=11, fontweight='bold', color='#00BFFF')
ax4.text(6.5, 0.3, 'Standard risk models apply.', ha='center', fontsize=10, color='#666666')

# Divider
ax4.axvline(4.8, ymin=0.1, ymax=0.9, color='#333333', linewidth=1)

# Save
plt.savefig('market_state_dashboard.png', dpi=150, facecolor='#0a0a0a', 
            edgecolor='none', bbox_inches='tight')
plt.close()

print("Dashboard saved to market_state_dashboard.png")
print("\nKey insights shown:")
print(f"  - Current connectivity: {CURRENT_CONNECTIVITY:.3f} (below threshold {THRESHOLD})")
print(f"  - Current regime: CONTAGION (normal risk dynamics)")
print(f"  - Current ASF: {CURRENT_ASF:.2f} (moderate)")
print("  - Implications: Traditional diversification works, standard models apply")

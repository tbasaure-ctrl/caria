
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

# Create Output Directory if needed
OUTPUT_DIR = "c:/key/wise_adviser_cursor_context/Caria_repo/caria/docs/research/media"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
# ---------------- SYNTHETIC MODEL FOR VISUALIZATION ----------------
# We recreate the interaction surface based on the estimated coefficients
# Risk = Alpha + B1*ASF + B2*C + B3*(ASF*C)

# Coefficients from our rigor testing (approximate, tuned to match Empirical Tau = 0.14)
ALPHA = 0.05
B_ASF = 4.30  # Contagion Slope (Positive)
B_C = -0.50
B_INTER = -30.7 # Strong negative interaction to flip the slope at C ~ 0.14 (-4.3 / -30.7 = 0.14)

# Grid
asf_range = np.linspace(0, 1, 30)
c_range = np.linspace(0, 1, 30)
X, Y = np.meshgrid(asf_range, c_range)

# Surface Equation (Risk Magnitude)
# Z represents "Future Distress" (Drawdown Magnitude)
Z = ALPHA + B_ASF * X + B_C * Y + B_INTER * (X * Y)

# ---------------- PLOT 2D CONTOUR (HEATMAP) ----------------
# A 2D "Phase Diagram" is often clearer than 3D
fig, ax = plt.subplots(figsize=(10, 8))

# Contour Filled Plot (Heatmap)
# Levels of Risk: Blue (Low) to Red (High)
# We define levels manually to control coloring
levels = np.linspace(Z.min(), Z.max(), 30)
contour = ax.contourf(X, Y, Z, levels=levels, cmap='RdYlBu_r', alpha=0.9, extend='both')

# Add White Contour Lines for precision
ax.contour(X, Y, Z, levels=10, colors='white', alpha=0.3, linewidths=0.5)

# Add Threshold Line (Where slope flips)
# dRisk/dASF = B_ASF + B_INTER * C = 0
# C_critical = -B_ASF / B_INTER
c_crit = -B_ASF / B_INTER
# Let's plot this line if it's in range
if 0 <= c_crit <= 1:
    ax.axhline(c_crit, color='white', linestyle='--', linewidth=2, label=f'Critical Connectivity ($\\tau \\approx {c_crit:.2f}$)')

# Annotations (Regimes)
# Contagion Regime (Low C)
ax.text(0.5, 0.05, "CONTAGION REGIME\n(Fragility $\\to$ Risk)", color='black', 
        fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# Disintegration Regime (High C)
ax.text(0.5, 0.8, "DISINTEGRATION REGIME\n(Fragility = Safety)", color='white', 
        fontweight='bold', ha='center', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

# Labels
ax.set_xlabel('Structural Fragility (ASF)', fontsize=12)
ax.set_ylabel('Market Connectivity (C)', fontsize=12)
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('Predicted Tail Risk Magnitude', fontsize=12)

# Title
plt.title("Phase Transition Diagram: The Inversion of Risk", fontsize=14, fontweight='bold')

# Legend
ax.legend(loc='lower right')

# Save
output_filename = 'Figure_Phase_Transition_Contour.png'
# Try to save in correct dir
save_dirs = ['docs/research/outputs', 'c:/key/wise_adviser_cursor_context/Caria_repo/caria/docs/research/outputs']
saved = False
for d in save_dirs:
    if os.path.exists(d):
        plt.savefig(os.path.join(d, output_filename), dpi=300, bbox_inches='tight')
        print(f"Saved {output_filename} to {d}")
        saved = True
        break
    
if not saved:
    # Fallback to current
    plt.savefig(output_filename, dpi=300)
    print(f"Saved {output_filename} locally")

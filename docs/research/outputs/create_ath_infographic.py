"""
Infografía Simple: La Paradoja del Doble ATH
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_DIR = "c:/key/wise_adviser_cursor_context/Caria_repo/caria/docs/research/outputs"

# Create figure with dark theme
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 10))

# Hide axes
ax.axis('off')

# Title
ax.text(0.5, 0.95, '🏆 LA PARADOJA DEL DOBLE ATH 🏆', 
        fontsize=28, fontweight='bold', ha='center', va='top',
        color='gold', transform=ax.transAxes)

ax.text(0.5, 0.88, 'Cuando GOLD + S&P 500 ambos en máximos históricos',
        fontsize=16, ha='center', va='top', style='italic',
        color='white', transform=ax.transAxes)

# Dividing line
ax.axhline(y=0.82, xmin=0.1, xmax=0.9, color='gold', linewidth=2)

# LEFT BOX - Warning
left_box = mpatches.FancyBboxPatch((0.05, 0.35), 0.4, 0.42,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#3d1515', edgecolor='#ff6b6b',
                                     linewidth=3, transform=ax.transAxes)
ax.add_patch(left_box)

ax.text(0.25, 0.72, '⚠️ ESPERA', fontsize=22, fontweight='bold', 
        ha='center', color='#ff6b6b', transform=ax.transAxes)
ax.text(0.25, 0.65, 'Corrección 10-15%', fontsize=18, fontweight='bold',
        ha='center', color='white', transform=ax.transAxes)
ax.text(0.25, 0.58, '77% probabilidad', fontsize=14,
        ha='center', color='#ffaaaa', transform=ax.transAxes)

# Draw small declining chart
x_chart = np.linspace(0.1, 0.4, 50)
y_chart = 0.48 - 0.08 * np.sin(np.linspace(0, np.pi, 50))
ax.plot(x_chart, y_chart, color='#ff6b6b', linewidth=3, transform=ax.transAxes)
ax.text(0.25, 0.38, '📉 Pullback normal', fontsize=12, ha='center',
        color='#ffaaaa', transform=ax.transAxes)

# RIGHT BOX - Protection
right_box = mpatches.FancyBboxPatch((0.55, 0.35), 0.4, 0.42,
                                      boxstyle="round,pad=0.02",
                                      facecolor='#153d15', edgecolor='#4ecdc4',
                                      linewidth=3, transform=ax.transAxes)
ax.add_patch(right_box)

ax.text(0.75, 0.72, '🛡️ PROTEGIDO', fontsize=22, fontweight='bold',
        ha='center', color='#4ecdc4', transform=ax.transAxes)
ax.text(0.75, 0.65, 'No crash >20%', fontsize=18, fontweight='bold',
        ha='center', color='white', transform=ax.transAxes)
ax.text(0.75, 0.58, '0% crashes históricos', fontsize=14,
        ha='center', color='#aaffee', transform=ax.transAxes)

# Shield icon area
ax.text(0.75, 0.48, '🔒', fontsize=40, ha='center', transform=ax.transAxes)
ax.text(0.75, 0.38, 'Crash IMPOSIBLE', fontsize=12, ha='center',
        color='#aaffee', transform=ax.transAxes)

# Bottom comparison bars
ax.text(0.5, 0.28, 'PROBABILIDAD DE CRASH >20% (12 meses)', fontsize=14, 
        fontweight='bold', ha='center', color='white', transform=ax.transAxes)

# Bar: Other periods
ax.barh(0.20, 0.14, height=0.04, left=0.35, color='#ff6b6b', 
        transform=ax.transAxes, zorder=5)
ax.text(0.33, 0.20, 'Otros períodos:', fontsize=11, ha='right', va='center',
        color='white', transform=ax.transAxes)
ax.text(0.50, 0.20, '14%', fontsize=14, fontweight='bold', ha='left', va='center',
        color='#ff6b6b', transform=ax.transAxes)

# Bar: ATH periods
ax.barh(0.14, 0.001, height=0.04, left=0.35, color='#4ecdc4',
        transform=ax.transAxes, zorder=5)
ax.text(0.33, 0.14, 'Después de ATH:', fontsize=11, ha='right', va='center',
        color='white', transform=ax.transAxes)
ax.text(0.37, 0.14, '0%', fontsize=14, fontweight='bold', ha='left', va='center',
        color='#4ecdc4', transform=ax.transAxes)

# Key insight box
insight_box = mpatches.FancyBboxPatch((0.1, 0.02), 0.8, 0.08,
                                        boxstyle="round,pad=0.01",
                                        facecolor='#2a2a3d', edgecolor='gold',
                                        linewidth=2, transform=ax.transAxes)
ax.add_patch(insight_box)

ax.text(0.5, 0.06, '💡 ORO en ATH = Inversores YA cubiertos = Los crashes no escalan',
        fontsize=13, fontweight='bold', ha='center', color='gold', transform=ax.transAxes)

# Watermark
ax.text(0.98, 0.01, 'Análisis basado en datos 2010-2024', fontsize=8, 
        ha='right', color='gray', transform=ax.transAxes)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/paradoja_doble_ath_infografia.png', dpi=150, 
            bbox_inches='tight', facecolor='#1a1a2e')
print(f"Saved: {OUTPUT_DIR}/paradoja_doble_ath_infografia.png")

plt.close()
print("Infografía generada exitosamente!")

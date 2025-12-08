import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
output_dir = "validation_results"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------------------
# 1. Descarga de datos
# -------------------------------------------------------------------
print("Descargando datos de SPY y HYG ...")
spy_df = yf.download("SPY", start="2007-01-01", progress=False)
hyg_df = yf.download("HYG", start="2007-01-01", progress=False)

# Extraemos Close y aseguramos que sean Series
spy_close = spy_df["Close"]
hyg_close = hyg_df["Close"]
# Si son DataFrames, extraemos la primera columna; si son Series, las usamos directamente
if isinstance(spy_close, pd.DataFrame):
    spy = spy_close.iloc[:, 0].dropna()
else:
    spy = spy_close.dropna()
if isinstance(hyg_close, pd.DataFrame):
    hyg = hyg_close.iloc[:, 0].dropna()
else:
    hyg = hyg_close.dropna()

# Alineamos índices
common = spy.index.intersection(hyg.index)
spy = spy.reindex(common)
hyg = hyg.reindex(common)

# Retornos logarítmicos
ret = np.log(spy).diff().dropna()
spy = spy.loc[ret.index]        # alineamos precio con retornos
hyg = hyg.loc[ret.index]

# -------------------------------------------------------------------
# 2. Volatilidades y energía E4
# -------------------------------------------------------------------
def ann_vol(x, w):
    return x.rolling(w).std() * np.sqrt(252)

vol_5   = ann_vol(ret, 5)
vol_21  = ann_vol(ret, 21)
vol_63  = ann_vol(ret, 63)
vol_cred = hyg.pct_change().rolling(42).std() * np.sqrt(252)

# Energía 4-escala
E4_raw = 0.20*vol_5 + 0.30*vol_21 + 0.25*vol_63 + 0.25*vol_cred

# Normalización por percentil en ventana 252
E4 = E4_raw.rolling(252).rank(pct=True)
# Aseguramos que E4 sea una Series
if isinstance(E4, pd.DataFrame):
    E4 = E4.iloc[:, 0] if E4.shape[1] == 1 else E4.squeeze()
E4 = pd.Series(E4)

# -------------------------------------------------------------------
# 3. Sincronía (proxy simple) y CARIA-SR
# -------------------------------------------------------------------
sync = (vol_21.rolling(21).corr(vol_63) + 1) / 2  # [0,1]
# Aseguramos que sync sea una Series
if isinstance(sync, pd.DataFrame):
    sync = sync.iloc[:, 0] if sync.shape[1] == 1 else sync.squeeze()
sync = pd.Series(sync)

SR_raw = E4 * (1 + sync)
SR = SR_raw.rolling(252).rank(pct=True)
# Aseguramos que SR sea una Series
if isinstance(SR, pd.DataFrame):
    SR = SR.iloc[:, 0] if SR.shape[1] == 1 else SR.squeeze()
SR = pd.Series(SR)

# -------------------------------------------------------------------
# 4. Estado estructural y pérdidas futuras
# -------------------------------------------------------------------
# Definición de estado frágil (cuantil 80 de E4 y Sync)
# Primero eliminamos NaN para calcular cuantiles
sync_clean = sync.dropna()
E4_clean = E4.dropna()

sync_q80 = sync_clean.quantile(0.8)
E4_q80   = E4_clean.quantile(0.8)

state = (((sync > sync_q80) & (E4 > E4_q80)).fillna(0).astype(int))

# Pérdida futura a 10 días
future_loss_10d = ret.rolling(10).sum().shift(-10)

# -------------------------------------------------------------------
# 5. Ensamblar DataFrame limpio
# -------------------------------------------------------------------
# Encontramos el índice común de todas las Series
common_idx = ret.index.intersection(E4.index).intersection(sync.index).intersection(SR.index).intersection(state.index)

# Verificamos que tenemos datos
if len(common_idx) == 0:
    raise ValueError("No hay índices comunes entre las Series. Verifica los cálculos.")

# Creamos el DataFrame directamente extrayendo valores
# Convertimos todo a Series simples primero
def to_series(s, idx):
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0] if s.shape[1] == 1 else s.squeeze()
    if not isinstance(s, pd.Series):
        return pd.Series(s, index=idx)
    # Extraemos valores y creamos nueva Series para evitar problemas de índice
    vals = s.reindex(idx).values
    if vals.ndim > 1:
        vals = vals.flatten()[:len(idx)]
    return pd.Series(vals, index=idx)

df = pd.DataFrame(index=common_idx)
df["Close"] = to_series(spy, common_idx)
df["ret"] = to_series(ret, common_idx)
df["E4"] = to_series(E4, common_idx)
df["sync"] = to_series(sync, common_idx)
df["SR"] = to_series(SR, common_idx)
df["state"] = to_series(state, common_idx)
df["future_loss_10d"] = to_series(future_loss_10d, common_idx)

# Eliminamos solo filas donde las columnas críticas son NaN
# (no todas las columnas, ya que future_loss_10d tiene NaN al final por el shift)
print(f"Antes de dropna: {len(df)} filas")
print(f"NaN en E4: {df['E4'].isna().sum()}")
print(f"NaN en sync: {df['sync'].isna().sum()}")
print(f"NaN en SR: {df['SR'].isna().sum()}")
print(f"NaN en state: {df['state'].isna().sum()}")

df = df.dropna(subset=["E4", "sync", "SR", "state"])

print(f"Datos listos. N = {len(df)} filas válidas.")

# -------------------------------------------------------------------
# 6. Métricas numéricas (AUC + pérdidas por régimen)
# -------------------------------------------------------------------
auc_state = roc_auc_score(df["state"], df["SR"])
print(f"\nAUC(SR vs state) = {auc_state:.3f}")

m0 = df.loc[df["state"] == 0, "future_loss_10d"].mean()
m1 = df.loc[df["state"] == 1, "future_loss_10d"].mean()
print(f"Future 10d loss (state=0, normal):  {m0:+.4f}")
print(f"Future 10d loss (state=1, frágil):  {m1:+.4f}")

# -------------------------------------------------------------------
# 7. FIGURA 1: Curva ROC
# -------------------------------------------------------------------
fpr, tpr, _ = roc_curve(df["state"], df["SR"])

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"CARIA-SR (AUC = {auc_state:.3f})")
plt.plot([0,1],[0,1], linestyle="--", color="grey", label="Azar")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC: CARIA-SR como clasificador de fragilidad estructural")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve_caria_sr.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"\nGráfico 1 guardado: {output_dir}/roc_curve_caria_sr.png")

# -------------------------------------------------------------------
# 8. FIGURA 2: Scatter E4 vs SR coloreado por estado
# -------------------------------------------------------------------
plt.figure(figsize=(7,5))
scatter = plt.scatter(
    df["E4"], df["SR"],
    c=df["state"], cmap="coolwarm", alpha=0.7, s=10
)
cbar = plt.colorbar(scatter)
cbar.set_label("Fragile (state)")
plt.xlabel("Energía sistémica normalizada (E4)")
plt.ylabel("Índice CARIA-SR")
plt.title("Relación entre energía y CARIA-SR\ncoloreado por estado estructural (0=normal, 1=frágil)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "scatter_e4_vs_sr.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"Gráfico 2 guardado: {output_dir}/scatter_e4_vs_sr.png")

print("\n¡Análisis completado! Gráficos guardados en validation_results/")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. DOWNLOAD GLOBAL CREDIT VOL FOR ALL ASSETS (HYG)
# ============================================================
print("Descargando HYG (crédito) ...")

hyg_data = yf.download("HYG", start="2007-01-01", auto_adjust=True, progress=False)
hyg = hyg_data["Close"].squeeze().dropna() if isinstance(hyg_data, pd.DataFrame) else hyg_data.dropna()
ret_hyg = hyg.pct_change()

# 42d credit volatility - asegurar que es Series
vol_credit = (ret_hyg.rolling(42).std() * np.sqrt(252))
if isinstance(vol_credit, pd.DataFrame):
    vol_credit = vol_credit.squeeze()
vol_credit.name = "vol_credit"

print(f"Crédito cargado: {len(vol_credit)} muestras")

# ============================================================
# 2. FUNCTION: Compute CARIA-SR for any asset
# ============================================================
def compute_sr(ticker, vol_credit):
    """
    Devuelve:
        df  -> DataFrame con columnas: price, ret, sync, E4, SR, state
        auc -> AUC para detección de estado estructural
        m0  -> media de pérdidas a 10 días en estado normal
        m1  -> media de pérdidas a 10 días en estado frágil
    """
    print(f"\nProcesando {ticker} ...")

    px_data = yf.download(ticker, start="2007-01-01", auto_adjust=True, progress=False)
    # Extraer Close como Series
    if isinstance(px_data, pd.DataFrame):
        px = px_data["Close"].squeeze().dropna()
    else:
        px = px_data.dropna()
    
    ret = px.pct_change()

    # Alinear crédito con este asset - crear DataFrame explícitamente
    # Alinear índices primero y asegurar que son Series 1D
    common_idx = px.index.intersection(ret.index).intersection(vol_credit.index)
    if len(common_idx) < 300:
        print(f"{ticker}: muy pocos datos comunes → skip")
        return None, None, None, None
    
    # Extraer valores como arrays 1D
    px_vals = px.loc[common_idx].values
    ret_vals = ret.loc[common_idx].values
    vol_vals = vol_credit.loc[common_idx].values
    
    # Si alguno es 2D, hacer squeeze
    if px_vals.ndim > 1:
        px_vals = px_vals.squeeze()
    if ret_vals.ndim > 1:
        ret_vals = ret_vals.squeeze()
    if vol_vals.ndim > 1:
        vol_vals = vol_vals.squeeze()
    
    df = pd.DataFrame({
        "price": px_vals,
        "ret": ret_vals,
        "vol_credit": vol_vals
    }, index=common_idx).dropna()
    
    if len(df) < 300:
        print(f"{ticker}: muy pocos datos después de dropna → skip")
        return None, None, None, None

    # ----------------------------
    # Synchronization proxy
    # ----------------------------
    roc = df["ret"].rolling(21).sum()
    mom_norm = (roc - roc.rolling(252).mean()) / roc.rolling(252).std()
    sync = ((mom_norm.rolling(21).corr(df["vol_credit"]) + 1) / 2)
    sync.name = "sync"

    # ----------------------------
    # E4 Energy (4-scale)
    # ----------------------------
    vol5  = df["ret"].rolling(5).std() * np.sqrt(252)
    vol21 = df["ret"].rolling(21).std() * np.sqrt(252)
    vol63 = df["ret"].rolling(63).std() * np.sqrt(252)
    E4 = (0.20*vol5 + 0.30*vol21 + 0.25*vol63 + 0.25*df["vol_credit"])
    E4.name = "E4"

    # ----------------------------
    # CARIA-SR SIGNAL
    # ----------------------------
    SR = ((E4.rank(pct=True) * (1 + sync))).rank(pct=True)
    SR.name = "SR"

    df = pd.concat([df, sync, E4, SR], axis=1).dropna()

    # ----------------------------
    # STRUCTURAL STATE (Regime)
    # ----------------------------
    q_sync = df["sync"].quantile(0.80)
    q_E4   = df["E4"].quantile(0.80)
    df["state"] = ((df["sync"] > q_sync) & (df["E4"] > q_E4)).astype(int)

    if df["state"].sum() < 5:
        print(f"{ticker}: muy pocos eventos estructurales → skip")
        return None, None, None, None

    # ----------------------------
    # AUC: ¿SR detecta state=1?
    # ----------------------------
    auc = roc_auc_score(df["state"], df["SR"])

    # ----------------------------
    # 10-day future loss
    # ----------------------------
    future = df["ret"].rolling(10).sum().shift(-10)
    future.name = "future10"
    df = pd.concat([df, future], axis=1).dropna()

    m0 = df[df["state"] == 0]["future10"].mean()
    m1 = df[df["state"] == 1]["future10"].mean()

    return df, auc, m0, m1


# ============================================================
# 3. MULTI-ASSET EVALUATION
# ============================================================

assets = ["SPY","QQQ","IWM","DIA","XLF","XLE","XLK","EFA","EEM","GLD"]

results = []

for t in assets:
    df, auc, m0, m1 = compute_sr(t, vol_credit)
    if df is None:
        continue
    results.append((t, auc, m0, m1))

# ============================================================
# 4. SHOW RESULTS
# ============================================================
print("\n\n================= RESULTADOS =================")
print("Ticker |   AUC   | FutureLoss Normal | FutureLoss Fragile")
print("---------------------------------------------------------")

for t, auc, m0, m1 in results:
    print(f"{t:<5} | {auc:6.3f} | {m0:>+8.4f} | {m1:>+8.4f}")

print("\nTotal evaluado:", len(results))

# ============================================================
# 5. OPTIONAL: VISUALIZAR UN ASSET (SPY)
# ============================================================
asset = "SPY"
df, auc, m0, m1 = compute_sr(asset, vol_credit)

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df.index, df["price"], label="Precio")
ax.set_title(f"{asset} Price + Structural Fragility (SR)\nAUC={auc:.3f}")
ax2 = ax.twinx()
ax2.plot(df.index, df["SR"], color="red", alpha=0.3, label="SR")
plt.savefig(f"{asset}_caria_sr.png", dpi=150, bbox_inches='tight')
print(f"\nGràfico guardado en: {asset}_caria_sr.png")

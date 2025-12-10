import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import roc_auc_score
from scipy import stats

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
assets = ["SPY","QQQ","IWM","DIA","XLF","XLE","XLK","EFA","EEM","GLD"]
start_date = "1990-01-01"
credit_start = "2007-01-01"

# ---------------------------------------------------------
# DOWNLOAD DATA
# ---------------------------------------------------------
print("Descargando precios...")
prices = yf.download(assets, start=start_date, progress=False)["Close"]

print("Descargando crédito (HYG)...")
hyg_data = yf.download("HYG", start=credit_start, progress=False)
if isinstance(hyg_data, pd.DataFrame) and "Close" in hyg_data.columns:
    hyg = hyg_data["Close"]
else:
    hyg = pd.Series(dtype=float)

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------
def compute_sync(ret):
    roc5  = ret.rolling(5).sum()
    roc21 = ret.rolling(21).sum()
    roc63 = ret.rolling(63).sum()
    m5  = (roc5  - roc5.rolling(252).mean())  / roc5.rolling(252).std()
    m21 = (roc21 - roc21.rolling(252).mean()) / roc21.rolling(252).std()
    m63 = (roc63 - roc63.rolling(252).mean()) / roc63.rolling(252).std()

    # Compute rolling correlations - ensure we get Series, not DataFrame
    corr_5_21 = m5.rolling(21).corr(m21)
    corr_5_63 = m5.rolling(21).corr(m63)
    corr_21_63 = m21.rolling(21).corr(m63)
    
    sync = (corr_5_21 + corr_5_63 + corr_21_63) / 3
    return (sync + 1) / 2

def compute_energy(ret, credit_vol=None):
    v5  = ret.rolling(5).std()*np.sqrt(252)
    v21 = ret.rolling(21).std()*np.sqrt(252)
    v63 = ret.rolling(63).std()*np.sqrt(252)

    if credit_vol is not None and credit_vol.notna().sum() > 0:
        # Use credit_vol where available, fallback to base formula where not
        E_with_credit = 0.20*v5 + 0.30*v21 + 0.25*v63 + 0.25*credit_vol
        E_base = 0.30*v5 + 0.40*v21 + 0.30*v63
        E = E_with_credit.fillna(E_base)
    else:
        E = 0.30*v5 + 0.40*v21 + 0.30*v63  # HAR-like base

    return E.rolling(252).rank(pct=True)

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
results = []

for t in assets:
    print(f"\nProcesando {t}...")
    px = prices[t].dropna()
    ret = px.pct_change().dropna()

    sync = compute_sync(ret)

    credit_vol = None
    # Use credit vol if available (HYG data starts 2007)
    if isinstance(hyg, pd.Series) and len(hyg) > 0:
        hyg_ret = hyg.pct_change()
        cv = hyg_ret.rolling(42).std()*np.sqrt(252)
        credit_vol = cv.reindex(ret.index)

    E = compute_energy(ret, credit_vol)

    common = sync.dropna().index.intersection(E.dropna().index)
    sync, E, ret = sync.loc[common], E.loc[common], ret.loc[common]

    # Ensure we have Series (not DataFrame)
    if isinstance(sync, pd.DataFrame):
        sync = sync.iloc[:, 0] if sync.shape[1] > 0 else pd.Series(dtype=float, index=sync.index)
    if isinstance(E, pd.DataFrame):
        E = E.iloc[:, 0] if E.shape[1] > 0 else pd.Series(dtype=float, index=E.index)

    # Exogenous fragile state
    state = ((sync > sync.quantile(0.9)) & (E > E.quantile(0.9))).astype(int)

    SR = (E*(1+sync)).rolling(252).rank(pct=True)
    
    # Ensure SR is a Series
    if isinstance(SR, pd.DataFrame):
        SR = SR.iloc[:, 0] if SR.shape[1] > 0 else pd.Series(dtype=float, index=SR.index)
    
    # Align all series to common index before creating DataFrame
    common_idx = SR.dropna().index.intersection(state.dropna().index).intersection(ret.dropna().index)
    SR = SR.loc[common_idx]
    state = state.loc[common_idx]
    ret = ret.loc[common_idx]

    df = pd.DataFrame({"SR":SR,"state":state,"ret":ret}).dropna()
    num_fragile = df["state"].sum()
    if num_fragile < 30:
        print(f"  (muy pocos estados frágiles: {num_fragile} de {len(df)} observaciones)")
        continue

    auc = roc_auc_score(df["state"], df["SR"])

    future = df["ret"].rolling(10).sum().shift(-10)
    m0 = future[df["state"]==0].mean()
    m1 = future[df["state"]==1].mean()

    results.append([t, auc, m0, m1])

# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------
res = pd.DataFrame(results, columns=["Asset","AUC","Future_Normal","Future_Fragile"])
res["Δ"] = res["Future_Fragile"] - res["Future_Normal"]

print("\n================= RESULTADOS =================")
print(res.round(4))

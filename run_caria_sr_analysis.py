import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
assets = ["SPY","QQQ","IWM","DIA","XLF","XLE","XLK","EFA","EEM","GLD"]
start_date = "1990-01-01"
q_sync = 0.90
q_E = 0.90
fwd_window = 10

results = []

# ===============================
# FUNCTIONS
# ===============================
def build_caria_sr(px):
    ret = px.pct_change()

    # Momentum
    m5  = ret.rolling(5).sum()
    m21 = ret.rolling(21).sum()
    m63 = ret.rolling(63).sum()

    m5  = (m5 - m5.rolling(252).mean()) / m5.rolling(252).std()
    m21 = (m21 - m21.rolling(252).mean()) / m21.rolling(252).std()
    m63 = (m63 - m63.rolling(252).mean()) / m63.rolling(252).std()

    # Synchrony - use pd.concat to ensure proper Series alignment
    # Calculate correlations separately and then combine
    corr_5_21 = m5.rolling(21).corr(m21)
    corr_5_63 = m5.rolling(21).corr(m63)
    corr_21_63 = m21.rolling(21).corr(m63)
    
    sync = (corr_5_21 + corr_5_63 + corr_21_63) / 3

    # Energy (equity-only, HAR-style)
    v5  = ret.rolling(5).std()  * np.sqrt(252)
    v21 = ret.rolling(21).std() * np.sqrt(252)
    v63 = ret.rolling(63).std() * np.sqrt(252)

    E3 = 0.30*v5 + 0.40*v21 + 0.30*v63
    E3 = E3.rolling(252).rank(pct=True)

    # CARIA-SR
    SR = (E3 * (1 + sync)).rolling(252).rank(pct=True)

    # State definition (exógena) - ensure we have valid data before quantile
    sync_valid = sync.dropna()
    E3_valid = E3.dropna()
    
    if len(sync_valid) == 0 or len(E3_valid) == 0:
        return pd.DataFrame()
    
    q_sync_val = sync_valid.quantile(q_sync)
    q_E_val = E3_valid.quantile(q_E)
    
    fragile = (
        (sync > q_sync_val) &
        (E3  > q_E_val)
    ).astype(int)

    # Future returns
    future_ret = ret.rolling(fwd_window).sum().shift(-fwd_window)

    # Combine into DataFrame - ensure all are Series (1D)
    # Convert to Series if they're DataFrames
    if isinstance(SR, pd.DataFrame):
        SR = SR.squeeze()
    if isinstance(fragile, pd.DataFrame):
        fragile = fragile.squeeze()
    if isinstance(future_ret, pd.DataFrame):
        future_ret = future_ret.squeeze()
    
    df = pd.DataFrame({
        "SR": SR,
        "Fragile": fragile,
        "FutureRet": future_ret
    }).dropna()

    return df

# ===============================
# RUN
# ===============================
print("Descargando datos desde 1990...\n")

for t in assets:
    px_data = yf.download(t, start=start_date, progress=False)
    if isinstance(px_data, pd.DataFrame):
        px = px_data["Close"].squeeze().dropna()
    else:
        px = px_data.dropna()
    
    if len(px) < 2000:
        continue

    df = build_caria_sr(px)

    auc = roc_auc_score(df["Fragile"], df["SR"])
    m0 = df.loc[df["Fragile"]==0,"FutureRet"].mean()
    m1 = df.loc[df["Fragile"]==1,"FutureRet"].mean()

    results.append([t, auc, m0, m1, m1-m0])

# ===============================
# RESULTS
# ===============================
res = pd.DataFrame(results,
                   columns=["Asset","AUC(State)","MeanRet Normal","MeanRet Fragile","Δ (Fragile-Normal)"])

print("\n================ RESULTADOS =================")
print(res.round(4))

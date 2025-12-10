import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import roc_auc_score
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

ASSETS = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "EFA", "EEM", "GLD"]
START_DATE = "1990-01-01"   # ampliamos muestra
FORWARD_HORIZON = 10        # días para retornos futuros
STATE_Q = 0.90              # percentil para Sync y E4

print("Descargando HYG (crédito) ...")
hyg = yf.download("HYG", start="2007-01-01", progress=False)["Close"].dropna()
ret_hyg = hyg.pct_change().dropna()
print(f"Crédito cargado: {len(ret_hyg)} retornos\n")

def compute_caria_sr_for_asset(ticker, ret_asset, vol_credit, state_q=0.90, fwd_h=10):
    """
    Dado un asset y la serie de vol_credit (HYG), construye:
      - E4
      - Sync (proxy)
      - SR
      - estado frágil exógeno (Sync & E4 en top state_q)
      - retornos futuros a fwd_h días

    Devuelve: df, auc_state, future_normal, future_fragile, delta
    """
    # Vols equity
    vol_5  = ret_asset.rolling(5).std() * np.sqrt(252)
    vol_21 = ret_asset.rolling(21).std() * np.sqrt(252)
    vol_63 = ret_asset.rolling(63).std() * np.sqrt(252)

    # 4-Scale Energy (basado en HAR + crédito)
    E4 = (0.20 * vol_5 +
          0.30 * vol_21 +
          0.25 * vol_63 +
          0.25 * vol_credit)

    # Sincronía proxy: correlación vol_21 vs vol_63
    sync = (vol_21.rolling(21).corr(vol_63) + 1) / 2

    # Filtrar NaN tempranos
    df = pd.DataFrame({
        "ret": ret_asset,
        "E4": E4,
        "sync": sync
    }).dropna()

    if len(df) < 500:
        # muy pocos datos, salimos
        return None, np.nan, np.nan, np.nan, np.nan

    # Percentiles GLOBALES (no rolling) sobre toda la muestra del activo
    q_sync = df["sync"].quantile(state_q)
    q_E4   = df["E4"].quantile(state_q)

    # Estado frágil exógeno
    df["fragile"] = ((df["sync"] > q_sync) & (df["E4"] > q_E4)).astype(int)

    # Construcción de SR (igual para todos los assets)
    # Primero rank global de E4
    df["E4_rank"] = df["E4"].rank(pct=True)
    F = df["E4_rank"] * (1 + df["sync"])         # vulnerabilidad estructural
    F_rank = F.rank(pct=True)
    alpha, beta, w = 1.5, 1.0, 0.5
    S = (F_rank ** alpha) * (df["sync"] ** beta) # modulación no lineal
    SR = w * df["E4_rank"] + (1 - w) * S
    df["SR"] = SR.rank(pct=True)

    # AUC: SR vs estado frágil
    y = df["fragile"].values
    scores = df["SR"].values
    if y.sum() == 0 or y.sum() == len(y):
        auc_state = np.nan
    else:
        auc_state = roc_auc_score(y, scores)

    # Retornos futuros (FORWARD_HORIZON días)
    future_loss = df["ret"].rolling(fwd_h).sum().shift(-fwd_h)
    df["fwd_loss"] = future_loss

    # dividir por estado
    df_valid = df.dropna(subset=["fwd_loss"])
    normal = df_valid[df_valid["fragile"] == 0]["fwd_loss"]
    fragil = df_valid[df_valid["fragile"] == 1]["fwd_loss"]

    if len(normal) < 50 or len(fragil) < 20:
        return df_valid, auc_state, np.nan, np.nan, np.nan

    future_normal  = normal.mean()
    future_fragile = fragil.mean()
    delta = future_fragile - future_normal

    return df_valid, auc_state, future_normal, future_fragile, delta


# ============================================================
# 2. LOOP CROSS-ASSET
# ============================================================

results = []

for ticker in ASSETS:
    print(f"Procesando {ticker} ...")

    # Descargar precios
    px = yf.download(ticker, start=START_DATE, progress=False)["Close"].dropna()
    ret = px.pct_change().dropna()

    # Alinear con crédito
    common = ret.index.intersection(ret_hyg.index)
    ret_aligned = ret.reindex(common)
    vol_credit = ret_hyg.reindex(common).rolling(42).std() * np.sqrt(252)

    df_asset, auc_state, m_norm, m_frag, delta = compute_caria_sr_for_asset(
        ticker, ret_aligned, vol_credit, state_q=STATE_Q, fwd_h=FORWARD_HORIZON
    )

    results.append({
        "Asset": ticker,
        "AUC": auc_state,
        "Future_Normal": m_norm,
        "Future_Fragile": m_frag,
        "Δ": delta
    })

print("\n================= RESULTADOS =================")
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

print("\nTotal evaluado:", res_df["AUC"].notna().sum())

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
    # Asegurar que ret_asset es una Series
    if isinstance(ret_asset, pd.DataFrame):
        ret_asset = ret_asset.iloc[:, 0] if len(ret_asset.columns) > 0 else ret_asset.squeeze()
    if not isinstance(ret_asset, pd.Series):
        ret_asset = pd.Series(ret_asset).squeeze()
    
    # Asegurar que vol_credit es una Series
    if isinstance(vol_credit, pd.DataFrame):
        vol_credit = vol_credit.iloc[:, 0] if len(vol_credit.columns) > 0 else vol_credit.squeeze()
    if not isinstance(vol_credit, pd.Series):
        vol_credit = pd.Series(vol_credit).squeeze()
    
    # Alinear vol_credit con ret_asset
    common_idx = ret_asset.index.intersection(vol_credit.index)
    if len(common_idx) == 0:
        return None, np.nan, np.nan, np.nan, np.nan
    
    ret_asset = ret_asset.reindex(common_idx).dropna()
    vol_credit = vol_credit.reindex(common_idx).dropna()
    
    # Re-alinear después de dropna
    common_idx = ret_asset.index.intersection(vol_credit.index)
    ret_asset = ret_asset.reindex(common_idx)
    vol_credit = vol_credit.reindex(common_idx)
    
    # Vols equity - asegurar que son Series
    vol_5  = pd.Series(ret_asset.rolling(5).std() * np.sqrt(252), index=ret_asset.index)
    vol_21 = pd.Series(ret_asset.rolling(21).std() * np.sqrt(252), index=ret_asset.index)
    vol_63 = pd.Series(ret_asset.rolling(63).std() * np.sqrt(252), index=ret_asset.index)

    # 4-Scale Energy (basado en HAR + crédito) - alinear todas las series
    # Asegurar que vol_credit está alineado con las otras vols
    vol_credit_aligned = vol_credit.reindex(vol_5.index)
    E4 = pd.Series(0.20 * vol_5 + 0.30 * vol_21 + 0.25 * vol_63 + 0.25 * vol_credit_aligned, index=vol_5.index)

    # Sincronía proxy: correlación vol_21 vs vol_63
    # Asegurar que ambas Series están alineadas
    common_vol_idx = vol_21.index.intersection(vol_63.index)
    vol_21_aligned = vol_21.reindex(common_vol_idx)
    vol_63_aligned = vol_63.reindex(common_vol_idx)
    
    # Calcular correlación rolling manualmente
    sync_raw = pd.Series(index=vol_21_aligned.index, dtype=float)
    for i in range(20, len(vol_21_aligned)):
        window_21 = vol_21_aligned.iloc[i-20:i+1]
        window_63 = vol_63_aligned.iloc[i-20:i+1]
        if len(window_21) == 21 and len(window_63) == 21:
            corr_val = window_21.corr(window_63)
            if not np.isnan(corr_val):
                sync_raw.iloc[i] = corr_val
    
    sync = (sync_raw + 1) / 2

    # Asegurar que todas las Series son 1D y están alineadas
    # Encontrar el índice común de todas las series
    all_idx = ret_asset.index.intersection(E4.index).intersection(sync.index)
    
    # Convertir a Series 1D si es necesario y reindexar
    ret_asset_clean = pd.Series(ret_asset.reindex(all_idx), index=all_idx) if isinstance(ret_asset, pd.Series) else pd.Series(ret_asset.reindex(all_idx).squeeze(), index=all_idx)
    E4_clean = pd.Series(E4.reindex(all_idx), index=all_idx) if isinstance(E4, pd.Series) else pd.Series(E4.reindex(all_idx).squeeze(), index=all_idx)
    sync_clean = pd.Series(sync.reindex(all_idx), index=all_idx) if isinstance(sync, pd.Series) else pd.Series(sync.reindex(all_idx).squeeze(), index=all_idx)
    
    # Filtrar NaN tempranos - construir DataFrame con Series alineadas
    df = pd.DataFrame({
        "ret": ret_asset_clean,
        "E4": E4_clean,
        "sync": sync_clean
    })
    df = df.dropna()

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

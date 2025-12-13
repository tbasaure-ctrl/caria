import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import roc_auc_score
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN
# ============================================================
START_DATE = "1990-01-01"   # Muestra ampliada (se recorta sola por HYG)
ASSETS = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "EFA", "EEM", "GLD"]
CREDIT_TICKER = "HYG"       # Proxy de crédito (común a todos)
ROLL_RANK_WINDOW = 252      # Ventana para rank percentil
FRAG_Q = 0.80               # Umbral percentil para Sync y E4
FWD_HORIZON = 10            # Retorno futuro a 10 días
SEED = 42

print("Descargando datos de crédito (HYG) desde", START_DATE, "...")
hyg = yf.download(CREDIT_TICKER, start=START_DATE, progress=False)["Close"].dropna()
ret_hyg = hyg.pct_change().dropna()
vol_credit = ret_hyg.rolling(42).std() * np.sqrt(252)
print(f"Crédito cargado: {len(vol_credit)} observaciones útiles\n")

np.random.seed(SEED)

# ============================================================
# FUNCIONES NÚCLEO
# ============================================================

def build_caria_sr_for_asset(price, vol_credit,
                             roll_rank_window=ROLL_RANK_WINDOW,
                             alpha=1.5, beta=1.0, w=0.5):
    """
    Construye CARIA-SR para un activo dado:
      - price: Serie de precios (pd.Series)
      - vol_credit: Serie de vol anualizada de crédito (HYG), ya calculada
    Devuelve:
      df: DataFrame con Close, ret, SR, E4, Sync, state, future_loss
      auc_state: AUC(SR vs estado frágil)
      m0: media futuro 10d en estado normal
      m1: media futuro 10d en estado frágil
    """
    # Alinear con crédito
    common = price.index.intersection(vol_credit.index)
    price = price.reindex(common).dropna()
    vol_cred = vol_credit.reindex(common)

    if len(price) < 400:
        print(f"    Debug: precio tiene solo {len(price)} datos después de alinear")
        return None, np.nan, np.nan, np.nan  # muy poca data
    
    print(f"    Debug: precio={len(price)}, vol_cred={len(vol_cred.dropna())}")

    # Retornos
    ret = price.pct_change().dropna()

    # Volatilidades equity
    vol_5 = ret.rolling(5).std() * np.sqrt(252)
    vol_21 = ret.rolling(21).std() * np.sqrt(252)
    vol_63 = ret.rolling(63).std() * np.sqrt(252)

    # 4-Scale Energy E4(t)
    E4_raw = 0.20 * vol_5 + 0.30 * vol_21 + 0.25 * vol_63 + 0.25 * vol_cred
    E4 = E4_raw.rolling(roll_rank_window).rank(pct=True)

    # Momentum normalizado (para sincronía)
    roc_5 = ret.rolling(5).sum()
    roc_21 = ret.rolling(21).sum()
    roc_63 = ret.rolling(63).sum()

    def zscore(x, win=ROLL_RANK_WINDOW):
        m = x.rolling(win).mean()
        s = x.rolling(win).std()
        return (x - m) / (s + 1e-12)

    m5 = zscore(roc_5)
    m21 = zscore(roc_21)
    m63 = zscore(roc_63)

    # Sincronía: correlación entre momentum normalizado y volatilidad de crédito
    # Usamos la versión más simple y robusta (como en otros archivos)
    roc_21 = ret.rolling(21).sum()
    mom_norm = (roc_21 - roc_21.rolling(roll_rank_window).mean()) / (roc_21.rolling(roll_rank_window).std() + 1e-12)
    
    # Alinear mom_norm y vol_cred antes de correlación
    common_sync = mom_norm.index.intersection(vol_cred.index)
    mom_norm_aligned = mom_norm.reindex(common_sync)
    vol_cred_aligned = vol_cred.reindex(common_sync)
    
    # Crear DataFrame temporal para calcular correlación móvil
    temp_df = pd.DataFrame({
        'mom': mom_norm_aligned,
        'vol': vol_cred_aligned
    }).dropna()
    
    # Correlación móvil usando el método correcto
    sync_raw = temp_df['mom'].rolling(21).corr(temp_df['vol'])
    sync = (sync_raw + 1) / 2.0  # Normalizar de [-1,1] a [0,1]
    sync = sync.reindex(common_sync)  # Reindexar al índice común

    # Alinear E4 y sync antes de calcular SR
    common_sr = E4.index.intersection(sync.index)
    E4_aligned = E4.reindex(common_sr)
    sync_aligned = sync.reindex(common_sr)
    
    if len(E4_aligned.dropna()) < 100 or len(sync_aligned.dropna()) < 100:
        return None, np.nan, np.nan, np.nan
    
    # CARIA-SR: E4 rank × (1 + sync) luego rank percentil
    SR_raw = E4_aligned * (1 + sync_aligned)
    SR = SR_raw.rolling(roll_rank_window).rank(pct=True)

    # Estado estructural frágil: High Sync & High E4 (percentil FRAG_Q)
    # Calcular quantiles solo en datos válidos
    valid_sync = sync_aligned.dropna()
    valid_E4 = E4_aligned.dropna()
    
    if len(valid_sync) < 100 or len(valid_E4) < 100:
        return None, np.nan, np.nan, np.nan
    
    q_sync = valid_sync.quantile(FRAG_Q)
    q_E4 = valid_E4.quantile(FRAG_Q)
    state = ((sync_aligned > q_sync) & (E4_aligned > q_E4)).fillna(False).astype(int)

    # Retorno futuro a FWD_HORIZON días - alinear con ret
    future_loss = ret.rolling(FWD_HORIZON).sum().shift(-FWD_HORIZON)

    # Encontrar índice común de todas las series necesarias
    common_idx = common_sr.intersection(ret.index).intersection(price.index).intersection(future_loss.index)
    
    if len(common_idx) < 100:
        return None, np.nan, np.nan, np.nan
    
    # Debug: verificar tamaños antes de crear DataFrame
    print(f"    Debug: common_idx={len(common_idx)}, E4={len(E4_aligned)}, sync={len(sync_aligned)}, ret={len(ret)}, price={len(price)}")
    
    # Construir DataFrame con índices alineados
    df = pd.DataFrame({
        'Close': price.reindex(common_idx),
        'ret': ret.reindex(common_idx),
        'SR': SR.reindex(common_idx),
        'E4': E4_aligned.reindex(common_idx),
        'Sync': sync_aligned.reindex(common_idx),
        'state': state.reindex(common_idx),
        'future_loss': future_loss.reindex(common_idx)
    }).dropna()

    # Verificar que hay suficientes eventos frágiles
    n_fragile = df['state'].sum()
    if n_fragile < 5:
        return None, np.nan, np.nan, np.nan
    
    # Debug: mostrar algunos estadísticos
    if len(df) > 0:
        print(f"    Datos válidos: {len(df)}, Fragiles: {n_fragile}, Sync range: [{df['Sync'].min():.3f}, {df['Sync'].max():.3f}], E4 range: [{df['E4'].min():.3f}, {df['E4'].max():.3f}]")

    # AUC: qué tan bien SR separa Normal vs Frágil
    try:
        auc_state = roc_auc_score(df['state'], df['SR'])
    except ValueError:
        auc_state = np.nan

    # Medias de retorno futuro por estado
    m0 = df[df['state'] == 0]['future_loss'].mean()
    m1 = df[df['state'] == 1]['future_loss'].mean()

    return df, auc_state, m0, m1


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":
    results = []
    
    print("Evaluando activos...")
    for asset in ASSETS:
        print(f"\nProcesando {asset}...")
        try:
            asset_data = yf.download(asset, start=START_DATE, progress=False)["Close"].dropna()
            if len(asset_data) < 100:
                print(f"  {asset}: datos insuficientes")
                continue
            
            df, auc, m0, m1 = build_caria_sr_for_asset(asset_data, vol_credit)
            
            if df is None:
                print(f"  {asset}: no se pudo calcular (datos insuficientes o sin eventos frágiles)")
                continue
            
            results.append({
                'Asset': asset,
                'AUC': auc,
                'Mean_Normal': m0,
                'Mean_Fragile': m1,
                'N_Fragile': df['state'].sum(),
                'N_Total': len(df)
            })
            
            print(f"  {asset}: AUC={auc:.3f}, Normal={m0:.4f}, Fragile={m1:.4f}")
            
        except Exception as e:
            print(f"  {asset}: Error - {e}")
            continue
    
    # Mostrar resultados
    print("\n" + "="*70)
    print("RESULTADOS CARIA-SR")
    print("="*70)
    print(f"{'Asset':<8} | {'AUC':>6} | {'Normal (10d)':>12} | {'Fragile (10d)':>14} | {'N_Fragile':>10}")
    print("-"*70)
    
    for r in results:
        print(f"{r['Asset']:<8} | {r['AUC']:6.3f} | {r['Mean_Normal']:>12.4f} | {r['Mean_Fragile']:>14.4f} | {r['N_Fragile']:>10}")
    
    print(f"\nTotal evaluado: {len(results)} activos")

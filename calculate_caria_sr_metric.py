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
FRAG_Q = 0.90               # Umbral percentil para Sync y E4
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
        return None, np.nan, np.nan, np.nan  # muy poca data

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

    # Sincronía multiescala: correlaciones entre escalas de momentum
    # Usamos ventana móvil para correlaciones
    corr_5_21 = m5.rolling(21).corr(m21)
    corr_21_63 = m21.rolling(21).corr(m63)
    corr_5_63 = m5.rolling(21).corr(m63)
    
    # También correlación con crédito (proxy de sincronía sistémica)
    corr_credit = m21.rolling(21).corr(vol_cred)
    
    # Sincronía combinada: promedio de correlaciones normalizado a [0,1]
    sync_raw = (corr_5_21 + corr_21_63 + corr_5_63 + corr_credit) / 4.0
    sync = (sync_raw + 1) / 2.0  # Normalizar de [-1,1] a [0,1]
    
    # Alternativa más simple: usar solo correlación momentum-credito (como en otros archivos)
    # Descomentar si prefieres esta versión:
    # roc_21 = ret.rolling(21).sum()
    # mom_norm = (roc_21 - roc_21.rolling(roll_rank_window).mean()) / (roc_21.rolling(roll_rank_window).std() + 1e-12)
    # sync = (mom_norm.rolling(21).corr(vol_cred) + 1) / 2.0

    # CARIA-SR: E4 rank × (1 + sync) luego rank percentil
    SR_raw = E4 * (1 + sync)
    SR = SR_raw.rolling(roll_rank_window).rank(pct=True)

    # Estado estructural frágil: High Sync & High E4 (percentil FRAG_Q)
    q_sync = sync.quantile(FRAG_Q)
    q_E4 = E4.quantile(FRAG_Q)
    state = ((sync > q_sync) & (E4 > q_E4)).fillna(False).astype(int)

    # Retorno futuro a FWD_HORIZON días
    future_loss = ret.rolling(FWD_HORIZON).sum().shift(-FWD_HORIZON)

    # Construir DataFrame
    df = pd.DataFrame({
        'Close': price,
        'ret': ret,
        'SR': SR,
        'E4': E4,
        'Sync': sync,
        'state': state,
        'future_loss': future_loss
    }).dropna()

    # Verificar que hay suficientes eventos frágiles
    if df['state'].sum() < 5:
        return None, np.nan, np.nan, np.nan

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

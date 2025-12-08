import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. Descarga de crédito (HYG) y retornos
# ============================================================
print("Descargando HYG (crédito) ...")
hyg = yf.download("HYG", start="2007-01-01", progress=False)["Close"].dropna()
ret_hyg = hyg.pct_change().dropna()
vol_credit = ret_hyg.rolling(42).std() * np.sqrt(252)
print(f"Crédito cargado: {len(hyg)} muestras\n")

# ============================================================
# 2. Función para un activo: CARIA-SR, estado estructural, AUC, retornos futuros
# ============================================================
def evaluate_asset(ticker, vol_credit_global):
    print(f"Procesando {ticker} ...")

    # Precio y retornos del activo
    px_data = yf.download(ticker, start="2007-01-01", progress=False)["Close"]
    px = px_data.squeeze().dropna() if isinstance(px_data, pd.DataFrame) else px_data.dropna()
    ret = px.pct_change().dropna()

    # Alineamos con crédito
    common_idx = ret.index.intersection(vol_credit_global.index)
    if len(common_idx) < 300:
        print(f"  {ticker}: muy pocos datos tras alinear con HYG → skip\n")
        return None, None, None, None

    ret = ret.loc[common_idx]
    vol_cred = vol_credit_global.loc[common_idx]
    # Asegurar que vol_cred es una Serie
    if isinstance(vol_cred, pd.DataFrame):
        vol_cred = vol_cred.squeeze()

    # Volatilidades equity
    vol_5  = ret.rolling(5).std() * np.sqrt(252)
    vol_21 = ret.rolling(21).std() * np.sqrt(252)
    vol_63 = ret.rolling(63).std() * np.sqrt(252)

    # 4-Scale Energy (E4) - asegurar que todas las Series son 1D
    energy_raw = 0.20 * vol_5 + 0.30 * vol_21 + 0.25 * vol_63 + 0.25 * vol_cred
    # Asegurar que energy_raw es una Serie 1D
    if isinstance(energy_raw, pd.DataFrame):
        energy_raw = energy_raw.squeeze()
    E4 = energy_raw.rolling(252).rank(pct=True)
    if isinstance(E4, pd.DataFrame):
        E4 = E4.squeeze()

    # Proxy de sincronía: correlación vol_21 vs vol_63
    sync = (vol_21.rolling(21).corr(vol_63) + 1) / 2
    if isinstance(sync, pd.DataFrame):
        sync = sync.squeeze()

    # CARIA-SR: energía × (1 + sincronía) normalizado
    SR_raw = E4 * (1 + sync)
    SR = SR_raw.rolling(252).rank(pct=True)
    if isinstance(SR, pd.DataFrame):
        SR = SR.squeeze()

    # Estado estructural: High Sync & High Energy (percentil 80–100 en ambas)
    q_sync = sync.quantile(0.8)
    q_E4   = E4.quantile(0.8)
    state = ((sync > q_sync) & (E4 > q_E4)).fillna(False).astype(int)

    # Crear DataFrame usando concat - asegurar que todas son Series primero
    px_series = px.loc[common_idx]
    if isinstance(px_series, pd.DataFrame):
        px_series = px_series.squeeze()
    
    # Asegurar que todas las Series son realmente Series (no DataFrames)
    def ensure_series(s, name):
        if isinstance(s, pd.DataFrame):
            s = s.squeeze()
        if not isinstance(s, pd.Series):
            s = pd.Series(s, name=name)
        return s
    
    ret_ser = ensure_series(ret, "ret")
    E4_ser = ensure_series(E4, "E4")
    sync_ser = ensure_series(sync, "sync")
    SR_ser = ensure_series(SR, "SR")
    state_ser = ensure_series(state, "state")
    
    # Convertir cada Serie a DataFrame de una columna antes de concatenar
    df = pd.concat([
        px_series.to_frame("price"),
        ret_ser.to_frame("ret"),
        E4_ser.to_frame("E4"),
        sync_ser.to_frame("sync"),
        SR_ser.to_frame("SR"),
        state_ser.to_frame("state"),
    ], axis=1, join='inner').dropna()

    # Si casi no hay estado=1, no sirve para análisis
    if df["state"].sum() < 5:
        print(f"  {ticker}: muy pocos eventos estructurales → skip\n")
        return None, None, None, None

    # AUC: qué tan bien SR separa Normal vs Frágil (state=0/1)
    auc_state = roc_auc_score(df["state"], df["SR"])

    # Retorno futuro 10 días
    df["future_loss_10d"] = df["ret"].rolling(10).sum().shift(-10)
    m_by_state = df.groupby("state")["future_loss_10d"].mean()

    m0 = m_by_state.get(0, np.nan)  # normal
    m1 = m_by_state.get(1, np.nan)  # frágil

    return df, auc_state, m0, m1

# ============================================================
# 3. Evaluación cross-asset
# ============================================================
assets = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "EFA", "EEM", "GLD"]

results = []
dfs = {}  # para guardar SPY y eventualmente otros

for t in assets:
    df_t, auc_state, m0, m1 = evaluate_asset(t, vol_credit)
    if df_t is None:
        continue
    dfs[t] = df_t
    results.append((t, auc_state, m0, m1))

# ============================================================
# 4. Imprimir tabla de resultados
# ============================================================
print("\n================= RESULTADOS =================")
print("Ticker |   AUC   | FutureLoss Normal | FutureLoss Fragile")
print("---------------------------------------------------------")
for t, auc_s, m0, m1 in results:
    print(f"{t:<5} | {auc_s:6.3f} | {m0:+9.4f} | {m1:+9.4f}")
print(f"\nTotal evaluado: {len(results)}\n")

# ============================================================
# 5. Minsky Chart para SPY: precio + fragilidad estructural
# ============================================================
if "SPY" in dfs:
    df_spy = dfs["SPY"].copy()

    # Figura estilo "Minsky Chart"
    import matplotlib.dates as mdates
    from matplotlib.colors import Normalize
    from matplotlib.cm import get_cmap

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)

    ax_price = fig.add_subplot(gs[0])
    ax_regime = fig.add_subplot(gs[1], sharex=ax_price)

    dates = df_spy.index
    prices = df_spy["price"]
    sr_vals = df_spy["SR"].values

    # Top: precio en log + color por SR (fragilidad)
    ax_price.set_title("The Minsky Chart: High Returns in Fragile Zones (Red)")
    ax_price.set_ylabel("S&P 500 (Log)")
    ax_price.set_yscale("log")

    cmap = get_cmap("coolwarm")
    norm = Normalize(vmin=0, vmax=1)

    for i in range(len(dates) - 1):
        ax_price.plot(dates[i:i+2],
                      prices.iloc[i:i+2],
                      color=cmap(norm(sr_vals[i])),
                      linewidth=1.3)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax_price, pad=0.01
    )
    cbar.set_label("Structural Fragility (SR)")

    # Bottom: barras de régimen frágil
    ax_regime.set_ylabel("Regime Active")
    ax_regime.set_yticks([0, 1])
    ax_regime.set_yticklabels(["Normal", "Fragile"])

    # Rellenar donde state=1
    fragile = df_spy["state"] == 1
    for i in range(len(dates) - 1):
        if fragile.iloc[i]:
            ax_regime.axvspan(dates[i], dates[i+1],
                              color="red", alpha=0.3)

    ax_regime.set_xlabel("")
    ax_regime.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_regime.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.setp(ax_price.get_xticklabels(), visible=False)

    plt.tight_layout()
    fname = "SPY_caria_sr_minsky.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Gráfico guardado en: {fname}")
else:
    print("SPY no se pudo evaluar, no se genera Minsky Chart.")

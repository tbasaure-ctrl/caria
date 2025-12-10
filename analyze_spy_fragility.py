import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import roc_auc_score
from scipy import stats
import matplotlib.pyplot as plt

plt.style.use("default")
np.random.seed(42)

print("="*70)
print("CARIA-SR – Paso 1: Validación estadística en SPY (con HYG como crédito)")
print("="*70)

# ============================================================
# 1. Descarga y preparación de datos
# ============================================================

start_date = "2007-01-01"
tickers = ["SPY", "HYG"]

print(f"\nDescargando datos de {tickers} desde {start_date} ...")
px = yf.download(tickers, start=start_date, progress=False)["Close"].dropna()

if isinstance(px, pd.Series):
    raise SystemExit("Descarga no produjo DataFrame de precios. Revisa tickers/fecha.")

spy = px["SPY"].dropna()
hyg = px["HYG"].dropna()

# Alineamos ambos a un índice común
common_index = spy.index.intersection(hyg.index)
spy = spy.reindex(common_index)
hyg = hyg.reindex(common_index)

ret = spy.pct_change().dropna()
ret_hyg = hyg.pct_change().dropna()

# Reajustar common_index después de los pct_change
common_index = ret.index.intersection(ret_hyg.index)
ret = ret.reindex(common_index)
ret_hyg = ret_hyg.reindex(common_index)

print(f"Datos alineados: {len(common_index)} observaciones")

# ============================================================
# 2. Funciones núcleo: CARIA-SR, fragilidad, bootstrap
# ============================================================

def compute_caria_sr(returns, credit_returns, window_rank=252,
                     alpha=1.5, beta=1.0, w=0.5):
    """
    Construye:
      - volatilidades σ5, σ21, σ63 (asset)
      - σ_credit (HYG a 42d)
      - E4(t) = 0.20σ5 + 0.30σ21 + 0.25σ63 + 0.25σ_credit
      - Sync(t): promedio de correlaciones móviles entre momentos a 5,21,63 días
      - F(t)  = rank_252 [E4(t) * (1 + Sync(t))]
      - S(t)  = F(t)^alpha * Sync(t)^beta
      - SR(t) = w * E4_rank + (1 - w) * rank_252[S(t)]
    Todas las ranks son percentiles en ventana rodante.
    """

    # 1) Volatilidades (anualizadas)
    vol_5  = returns.rolling(5).std() * np.sqrt(252)
    vol_21 = returns.rolling(21).std() * np.sqrt(252)
    vol_63 = returns.rolling(63).std() * np.sqrt(252)
    vol_credit = credit_returns.rolling(42).std() * np.sqrt(252)

    # E4 tipo HAR extendido (citando a Corsi: mismo esquema 5–21–63 con crédito extra)
    E4_raw = 0.20*vol_5 + 0.30*vol_21 + 0.25*vol_63 + 0.25*vol_credit
    E4 = E4_raw.rolling(window_rank).rank(pct=True)

    # 2) Momentos normalizados para sincronía
    def zscore_rolling(x, w_mom=252):
        m = x.rolling(w_mom).mean()
        s = x.rolling(w_mom).std()
        return (x - m) / (s + 1e-12)

    roc_5  = returns.rolling(5).sum()
    roc_21 = returns.rolling(21).sum()
    roc_63 = returns.rolling(63).sum()

    m5  = zscore_rolling(roc_5)
    m21 = zscore_rolling(roc_21)
    m63 = zscore_rolling(roc_63)

    # 3) Sincronía multiescala
    corr_5_21  = m5.rolling(21).corr(m21)
    corr_5_63  = m5.rolling(21).corr(m63)
    corr_21_63 = m21.rolling(21).corr(m63)

    sync_raw = (corr_5_21 + corr_5_63 + corr_21_63) / 3
    Sync = (sync_raw + 1) / 2  # reescala a [0,1]

    # 4) Fragilidad estructural F, modulación S, índice SR
    F_raw = E4 * (1 + Sync)
    F = F_raw.rolling(window_rank).rank(pct=True)

    S = (F**alpha) * (Sync**beta)

    E4_rank = E4
    S_rank = S.rolling(window_rank).rank(pct=True)

    SR = w*E4_rank + (1 - w)*S_rank
    SR = SR.rolling(window_rank).rank(pct=True)

    df = pd.DataFrame({
        "vol_5": vol_5,
        "vol_21": vol_21,
        "vol_63": vol_63,
        "vol_credit": vol_credit,
        "E4": E4_rank,
        "Sync": Sync,
        "F": F,
        "S": S_rank,
        "SR": SR,
    }).dropna()

    return df


def define_fragility_state(df, q_sync=0.9, q_E4=0.9):
    """
    Define estado frágil de forma exógena:
      state(t) = 1 si Sync(t) > q_sync y E4(t) > q_E4
      state(t) = 0 en caso contrario
    """
    thr_sync = df["Sync"].quantile(q_sync)
    thr_E4   = df["E4"].quantile(q_E4)
    state = ((df["Sync"] > thr_sync) & (df["E4"] > thr_E4)).astype(int)
    return state, thr_sync, thr_E4


def compute_future_loss(returns, horizon=10):
    """
    Retorno acumulado a 'horizon' días hacia adelante.
    """
    fwd = returns.rolling(horizon).sum().shift(-horizon)
    return fwd


def bootstrap_auc(y, score, n_boot=2000, seed=42):
    """
    Bootstrap de AUC (IC 95%) y test contra 0.5.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    s = np.asarray(score)
    n = len(y)

    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        sb = s[idx]
        aucs.append(roc_auc_score(yb, sb))

    aucs = np.array(aucs)
    auc_hat = aucs.mean()
    ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])

    # Aproximación z contra 0.5
    se = aucs.std(ddof=1)
    z = (auc_hat - 0.5) / (se + 1e-12)
    p_two = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "auc_hat": auc_hat,
        "ci": (ci_low, ci_high),
        "z": z,
        "p_two": p_two,
        "aucs": aucs,
    }


def bootstrap_mean_diff(x0, x1, n_boot=5000, seed=42):
    """
    Bootstrap para diferencia de medias (x1 - x0) con IC 95%.
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)

    diffs = []
    for _ in range(n_boot):
        xb0 = rng.choice(x0, size=len(x0), replace=True)
        xb1 = rng.choice(x1, size=len(x1), replace=True)
        diffs.append(xb1.mean() - xb0.mean())

    diffs = np.array(diffs)
    diff_hat = diffs.mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    se = diffs.std(ddof=1)
    z = diff_hat / (se + 1e-12)
    p_two = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "diff_hat": diff_hat,
        "ci": (ci_low, ci_high),
        "z": z,
        "p_two": p_two,
        "diffs": diffs,
    }

# ============================================================
# 3. Construcción de CARIA-SR y estado frágil en SPY
# ============================================================

print("\nConstruyendo índice CARIA-SR en SPY...")

df = compute_caria_sr(ret, ret_hyg,
                      window_rank=252,
                      alpha=1.5,
                      beta=1.0,
                      w=0.5)

# Alinear retornos a df
ret_aligned = ret.reindex(df.index)

# Definir estado frágil vía percentiles (aquí 90-90 como ejemplo)
state, thr_sync, thr_E4 = define_fragility_state(df, q_sync=0.90, q_E4=0.90)
df["state"] = state
df["ret"] = ret_aligned

# Filtrar NaN potenciales
df = df.dropna(subset=["SR", "state", "ret"])

print(f"Total observaciones tras construcción: {len(df)}")
print(f"Fracción en estado frágil: {df['state'].mean():.3%}")
print(f"Umbral Sync (q90): {thr_sync:.3f} | Umbral E4 (q90): {thr_E4:.3f}")

# ============================================================
# 4. AUC de CARIA-SR para detectar régimen frágil
# ============================================================

print("\n[1] AUC: CARIA-SR vs estado frágil exógeno")

y_state = df["state"].values
score_sr = df["SR"].values

auc_raw = roc_auc_score(y_state, score_sr)
boot_auc = bootstrap_auc(y_state, score_sr, n_boot=3000)

print(f"  AUC (raw)       : {auc_raw:.4f}")
print(f"  AUC (bootstrap) : {boot_auc['auc_hat']:.4f}")
print(f"  IC 95%          : [{boot_auc['ci'][0]:.4f}, {boot_auc['ci'][1]:.4f}]")
print(f"  z vs 0.5        : {boot_auc['z']:.2f}  (p_two={boot_auc['p_two']:.4f})")

# ============================================================
# 5. Pérdida futura de retornos (10 días) en Normal vs Frágil
# ============================================================

print("\n[2] Pérdida futura (10 días) en estados Normal vs Frágil")

horizon = 10
df["future_10d"] = compute_future_loss(df["ret"], horizon=horizon)

df_losses = df.dropna(subset=["future_10d"])

loss_normal = df_losses.loc[df_losses["state"] == 0, "future_10d"].values
loss_fragile = df_losses.loc[df_losses["state"] == 1, "future_10d"].values

print(f"  Muestras Normal : {len(loss_normal)}")
print(f"  Muestras Frágil : {len(loss_fragile)}")

mu_n = loss_normal.mean()
mu_f = loss_fragile.mean()
sd_n = loss_normal.std(ddof=1)
sd_f = loss_fragile.std(ddof=1)

print(f"  Media Normal    : {mu_n:+.4f}")
print(f"  Media Frágil    : {mu_f:+.4f}")

boot_diff = bootstrap_mean_diff(loss_normal, loss_fragile, n_boot=5000)

print(f"  ΔMedia (F-N)    : {boot_diff['diff_hat']:+.4f}")
print(f"  IC 95%          : [{boot_diff['ci'][0]:+.4f}, {boot_diff['ci'][1]:+.4f}]")
print(f"  z               : {boot_diff['z']:.2f}  (p_two={boot_diff['p_two']:.4f})")

# ============================================================
# 6. Gráficos básicos (para paper)
# ============================================================

print("\nGenerando gráficos...")

# 6.1. Curva ROC aproximada (sampling de thresholds)
from sklearn.metrics import roc_curve

fpr, tpr, thr = roc_curve(y_state, score_sr)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, label=f"CARIA-SR (AUC={auc_raw:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Curva ROC – CARIA-SR vs Estado Frágil (SPY)")
ax.legend()
plt.tight_layout()
plt.savefig("spy_roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# 6.2. Distribución de SR en estados Normal vs Frágil
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df.loc[df["state"] == 0, "SR"], bins=40, alpha=0.6, label="Normal")
ax.hist(df.loc[df["state"] == 1, "SR"], bins=40, alpha=0.6, label="Frágil")
ax.set_xlabel("SR (percentil)")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución de CARIA-SR por estado (SPY)")
ax.legend()
plt.tight_layout()
plt.savefig("spy_sr_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# 6.3. Boxplot de pérdida futura en Normal vs Frágil
fig, ax = plt.subplots(figsize=(5, 4))
ax.boxplot([loss_normal, loss_fragile],
           labels=["Normal", "Frágil"],
           showfliers=False)
ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.7)
ax.set_ylabel(f"Retorno acumulado a {horizon} días")
ax.set_title("Pérdida futura en estados Normal vs Frágil (SPY)")
plt.tight_layout()
plt.savefig("spy_future_loss_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n✓ Paso 1 completado para SPY (CARIA-SR vs estado estructural + pérdidas futuras)")
print("\n✓ Gráficos guardados:")
print("  - spy_roc_curve.png")
print("  - spy_sr_distribution.png")
print("  - spy_future_loss_boxplot.png")

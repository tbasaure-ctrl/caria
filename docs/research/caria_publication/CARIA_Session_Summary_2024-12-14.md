# CARIA-SR Hysteresis Validation - Session Summary
**Fecha**: 2025-12-14

## Objetivo
Validar el índice CARIA-SR como señal de alerta temprana de crisis financieras, basado en la hipótesis de hysteresis estructural.

---

## 1. Descarga de Datos

### Código Validado
```python
# Get S&P 500 tickers
url = f"https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={FMP_API_KEY}"
resp = requests.get(url)
sp500_tickers = [x['symbol'] for x in resp.json()] if resp.status_code == 200 else []
print(f"Downloading {len(sp500_tickers)} stocks...")

# Download in batches
all_prices = []
for i in range(0, len(sp500_tickers), 50):
    batch = sp500_tickers[i:i+50]
    try:
        data = yf.download(batch, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)['Close']
        all_prices.append(data)
        print(f"   Batch {i//50 + 1}/{(len(sp500_tickers)-1)//50 + 1}")
    except Exception as e:
        print(f"   Error batch {i//50 + 1}: {e}")

prices = pd.concat(all_prices, axis=1).dropna(axis=1, how='all')
prices.to_csv(f'{WORK_DIR}/sp500_prices.csv')

# Market data (VIX, SPY, TLT, 10Y Treasury)
print("Downloading market data...")
market = yf.download(['^VIX', 'SPY', 'TLT', '^TNX'], start=START_DATE, end=END_DATE, progress=False)

market_df = pd.DataFrame({
    'volatility': market['Close']['^VIX'],
    'price': market['Close']['SPY'],
    'tlt': market['Close']['TLT'],
    'treasury_10y': market['Close']['^TNX']
}).dropna()
market_df.index.name = 'Date'
market_df.to_csv(f'{WORK_DIR}/market_validation_data.csv')
```

### Output
```
✅ Prices: 503 stocks, 6527 days
✅ Market: 5877 days
```

![Download Output](img_01_download.png)

---

## 2. Cálculo de Métricas Estructurales (AR + Entropy)

### Código Validado
```python
def cov_to_corr(S):
    d = np.sqrt(np.diag(S))
    d = np.where(d == 0, 1e-10, d)
    C = S / np.outer(d, d)
    return np.nan_to_num((C + C.T) / 2)

def eig_metrics(C, k_frac=0.2):
    w = np.sort(np.linalg.eigvalsh(C))[::-1]
    w = np.maximum(w, 1e-10)  # Avoid negative eigenvalues
    k = max(1, int(np.ceil(k_frac * len(w))))
    ar = np.sum(w[:k]) / np.sum(w)
    p = w / np.sum(w)
    ent = -np.sum(p * np.log(p + 1e-10)) / np.log(len(w)) if len(w) > 1 else 0.5
    return float(ar), float(ent)

# Calculate returns
returns = np.log(prices).diff()
good_coverage = returns.notna().mean() >= 0.9
returns = returns.loc[:, good_coverage]

# Rolling structural metrics
window = 252
step = 5
lw = LedoitWolf()

struct = pd.DataFrame(index=returns.index, columns=['absorption_ratio', 'entropy'], dtype=float)

total_steps = (len(returns) - window) // step
for idx, t in enumerate(range(window, len(returns), step)):
    W = returns.iloc[t-window:t]
    W = W.loc[:, W.notna().mean() >= 0.9]
    if W.shape[1] < 100:
        continue
    W = W.apply(lambda s: s.fillna(s.mean()))
    X = W.values - np.nanmean(W.values, axis=0)
    try:
        C = cov_to_corr(lw.fit(X).covariance_)
    except:
        C = np.corrcoef(X, rowvar=False)
        C = np.nan_to_num((C + C.T) / 2)
    ar, ent = eig_metrics(C)
    struct.iloc[t] = [ar, ent]

struct = struct.ffill().bfill()
struct.to_csv(f'{WORK_DIR}/caria_structural_metrics.csv')
```

### Definiciones
| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **Absorption Ratio** | Σ(top 20% eigenvalues) / Σ(all eigenvalues) | Alta sincronización → fragilidad |
| **Entropy** | -Σ(p·log(p)) / log(n) | Baja entropía → poca diversidad |

---

## 3. Quantile Regression (Phase 8)

### Código Validado
```python
# Merge and calculate signals
df = struct_df.join(market_df, how='inner').sort_index()

# Z-Score normalization
window_z = 252
rolling_mean = df['absorption_ratio'].rolling(window=window_z).mean()
rolling_std = df['absorption_ratio'].rolling(window=window_z).std()
df['absorp_z'] = (df['absorption_ratio'] - rolling_mean) / rolling_std

# Peak Memory - THE KEY INNOVATION
window_memory = 60
df['caria_peak'] = df['absorp_z'].rolling(window=window_memory).max()

# Regime filtering: VIX < 20
low_vol_df = df[df['volatility'] < 20].copy()
low_vol_df['future_ret_22'] = low_vol_df['price'].pct_change(22).shift(-22)

# Quantile Regression (τ = 0.05)
mod_vix = smf.quantreg('future_ret_22 ~ volatility', low_vol_df)
res_vix = mod_vix.fit(q=0.05)

mod_struct = smf.quantreg('future_ret_22 ~ volatility + caria_peak', low_vol_df)
res_struct = mod_struct.fit(q=0.05)

imp = ((res_struct.prsquared - res_vix.prsquared)/res_vix.prsquared)*100
```

### Resultados
```
Base Model (VIX Only) Pseudo R²:      0.03215
Structural Model (+Peak) Pseudo R²:   0.04400
🔥 Improvement in Low-Vol Regime:     36.8%
```

### Visualización: Peak Memory vs Original Signal
![Peak Memory COVID](C:/Users/tomas/.gemini/antigravity/brain/d187eeae-2954-419b-95e7-2a45c158cfd5/uploaded_image_1765689928956.png)

**Interpretación**: La señal original (salmon) se desvanece después de cada pico. El Peak Memory (rojo oscuro) mantiene la "cicatriz" estructural por 60 días.

---

## 4. Robustness Heatmap (Phase 9)

### Código Validado
```python
windows = [20, 40, 60, 90, 120]
vix_caps = [15, 18, 20, 22, 25]
results_matrix = np.zeros((len(windows), len(vix_caps)))

# Pre-compute peak signals for all windows
for w in windows:
    df[f'peak_{w}'] = df['absorp_z'].rolling(window=w).max()

for i, w in enumerate(windows):
    for j, v in enumerate(vix_caps):
        subset = df[df['volatility'] < v].copy()
        subset['ret_future'] = subset['price'].pct_change(22).shift(-22)
        subset = subset.dropna()
        
        if len(subset) > 500:
            mod_base = smf.quantreg('ret_future ~ volatility', subset).fit(q=0.05)
            mod_struct = smf.quantreg(f'ret_future ~ volatility + peak_{w}', subset).fit(q=0.05)
            imp = ((mod_struct.prsquared - mod_base.prsquared)/mod_base.prsquared) * 100
            results_matrix[i, j] = imp
```

### Visualización
![Robustness Heatmap](C:/Users/tomas/.gemini/antigravity/brain/d187eeae-2954-419b-95e7-2a45c158cfd5/uploaded_image_1765689967368.png)

### Interpretación
| Zona | Mejora | Significado |
|------|--------|-------------|
| VIX < 20, Memory 40-90 | **31-36%** | Zona óptima |
| VIX < 15 (muy calmo) | 0-10% | Poco poder predictivo |
| Memory 120d | 20-28% | Memoria demasiado larga |

---

## 5. Structural Alpha Landscape (AR × Entropy)

### Código Validado
```python
# Señal combinada: Alta Sync (AR) + Baja Entropy
for i, s_pct in enumerate(pcts):
    s_thresh = subset['absorption_ratio'].quantile(s_pct)
    for j, e_pct in enumerate(pcts):
        e_thresh = subset['entropy'].quantile(e_pct)
        signal = ((subset['absorption_ratio'] > s_thresh) &
                  (subset['entropy'] < e_thresh)).astype(int)
```

![Structural Alpha Landscape](C:/Users/tomas/.gemini/antigravity/brain/d187eeae-2954-419b-95e7-2a45c158cfd5/uploaded_image_1765690363228.png)

### Interpretación
- **Zona óptima (azul oscuro)**: Top 10-80% Sync + Bot 10-20% Entropy → **26-30% mejora**
- Las dos métricas son complementarias: necesitas AMBAS condiciones

---

## 6. Risk Metrics + Bootstrap (Phase 14)

### Código Validado
```python
def calc_risk_metrics(series, rf=0.04):
    rf_daily = (1 + rf)**(1/252) - 1
    excess_ret = series - rf_daily
    ann_ret = np.mean(series) * 252
    ann_vol = np.std(series) * np.sqrt(252)
    sharpe = np.mean(excess_ret) / np.std(excess_ret) * np.sqrt(252)  # Fixed: use excess_ret std
    downside = excess_ret[excess_ret < 0]
    downside_std = np.std(downside) * np.sqrt(252)
    sortino = np.mean(excess_ret) / downside_std * np.sqrt(252) if downside_std > 0 else 0
    return ann_ret, ann_vol, sharpe, sortino

# Bootstrap (1000 iterations)
for i in range(n_boot):
    sample = subset.sample(n=len(subset), replace=True)
    mod_base = smf.quantreg('ret_future ~ volatility', sample).fit(q=0.05)
    mod_struct = smf.quantreg('ret_future ~ volatility + caria_peak', sample).fit(q=0.05)
    imp = (mod_struct.prsquared - mod_base.prsquared) / mod_base.prsquared  # Fixed variable names
    improvements.append(imp)
```

### Resultados
![Risk Metrics & Bootstrap](C:/Users/tomas/.gemini/antigravity/brain/d187eeae-2954-419b-95e7-2a45c158cfd5/uploaded_image_1765690022981.png)

| Métrica | S&P 500 | Minsky Hedge | Minsky 1.5x |
|---------|---------|--------------|-------------|
| Ann. Return | 10.3% | 8.3% | 10.3% |
| Ann. Volatility | 18.7% | **16.0%** | 21.2% |
| Sharpe Ratio | 0.34 | 0.27 | 0.30 |

### Bootstrap Confidence Interval
```
Mean Improvement: 38.1%
95% CI: [7.7%, 104.6%]
P(Improvement > 0): 100.0%  ← ESTADÍSTICAMENTE SIGNIFICATIVO
```

---

## 6. Rigorous Out-of-Sample Validation

### Walk-Forward Cross-Validation
```python
# Parameters
TRAIN_YEARS = 5
TEST_YEARS = 1
PURGE_DAYS = 60  # Gap to avoid leakage
```

![Walk-Forward CV Results](C:/Users/tomas/.gemini/antigravity/brain/d187eeae-2954-419b-95e7-2a45c158cfd5/uploaded_image_1765690186130.png)

### Walk-Forward Results
```
Folds completed: 15
Mean R² (VIX only):      0.08251
Mean R² (VIX + Peak):    0.22567
Mean Improvement:        285.5%

Folds where Peak > VIX:  15/15
Win Rate:                100.0%  ← CLAVE
```

### Permutation Test (Statistical Significance)
```python
# Compare real signal vs 500 random shuffles
for i in range(500):
    test_df_perm['peak'] = np.random.permutation(test_df_perm['peak'].values)
```

![Permutation Test Results](C:/Users/tomas/.gemini/antigravity/brain/d187eeae-2954-419b-95e7-2a45c158cfd5/uploaded_image_1765690223164.png)

```
Real Improvement (R²):    0.00834
Random Mean:              0.00052
Random 95th Percentile:   0.00184

P-value:                  0.0000
Significant (p < 0.05):   ✅ YES
```

---

### Parameter Stability Test (Out-of-Sample)
![Parameter Stability](C:/Users/tomas/.gemini/antigravity/brain/d187eeae-2954-419b-95e7-2a45c158cfd5/uploaded_image_1765690256663.png)

```
Windows that beat baseline: 5/5
| VIX Entry | < 20 | Complacencia |
| VIX Exit | < 25 | Mercado calmado |

---

## Conclusiones

1. **Poder Predictivo**: Peak Memory mejora predicción de tail risk en **38%** sobre VIX-only (bootstrap p < 0.05)

2. **Out-of-Sample**: **100% Win Rate** en 15 folds de walk-forward validation (no overfitting)

3. **Robustez**: Funciona para ventanas 40-90 días y VIX < 20-22

4. **Minsky Hedge**: Reduce volatilidad (16% vs 18.7%) a costo de CAGR

5. **Limitación**: Señal requiere hysteresis asimétrica (entry fast, exit slow) para no salir durante crisis


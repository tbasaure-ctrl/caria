# ==============================================================================
# GREAT CARIA v6.0: STRUCTURAL RESONANCE (CARIA-SR)
# FIX: GLOBAL CREDIT ANCHOR DEFINITION
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("GREAT CARIA v6.0: CROSS-ASSET VALIDATION")
print("Fixing 'ret_hyg not defined' error")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1. LOAD THE GLOBAL ANCHOR (CREDIT) - DO THIS FIRST
# -----------------------------------------------------------------------------
print("[1] Loading Global Credit Anchor (HYG)...")
# Descargamos HYG una sola vez. Esto se usará para TODOS los activos.
credit_df = yf.download('HYG', start='2007-01-01', progress=False)
# Handle MultiIndex columns if present
if isinstance(credit_df.columns, pd.MultiIndex):
    credit_data = credit_df['Close']
    if isinstance(credit_data, pd.DataFrame):
        credit_data = credit_data.iloc[:, 0]  # Take first column if still DataFrame
else:
    credit_data = credit_df['Close']
ret_hyg = credit_data.pct_change().dropna()

# Pre-calculamos la volatilidad del crédito (Componente E4 fijo)
# Volatilidad de 42 días (Credit Clock)
vol_credit_global = ret_hyg.rolling(42).std() * np.sqrt(252)

print(f"    ✓ Credit Data Loaded. Samples: {len(ret_hyg)}")

# -----------------------------------------------------------------------------
# 2. DEFINE THE CARIA-SR FUNCTION
# -----------------------------------------------------------------------------
def compute_caria_sr_safe(returns, vol_credit_series, window=252):
    """
    Calcula CARIA-SR asegurando que las fechas coincidan.
    """
    # Alinear índices (El activo puede tener días que el crédito no, o viceversa)
    common_idx = returns.index.intersection(vol_credit_series.index)
    
    if len(common_idx) < 252:
        return None
        
    r = returns.loc[common_idx]
    v_cred = vol_credit_series.loc[common_idx]
    
    # --- LAYER 1: MACRO ENERGY (E4) ---
    v_fast = r.rolling(5).std() * np.sqrt(252)
    v_med  = r.rolling(21).std() * np.sqrt(252)
    v_slow = r.rolling(63).std() * np.sqrt(252)
    
    # E4 Formula (Including Credit)
    E4_raw = 0.20*v_fast + 0.30*v_med + 0.25*v_slow + 0.25*v_cred
    E4_rank = E4_raw.rolling(window).rank(pct=True)
    
    # --- LAYER 2: MICRO FRAGILITY (Structure) ---
    # Synchronization Proxy (Momentum Correlation)
    mom_fast = r.rolling(5).sum()
    mom_slow = r.rolling(63).sum()
    sync = (mom_fast.rolling(21).corr(mom_slow) + 1) / 2
    S_rank = sync.rolling(window).rank(pct=True).fillna(0.5)
    
    # Structural Energy (F)
    F_raw = E4_rank * (1 + S_rank)
    F_rank = F_raw.rolling(window).rank(pct=True)
    
    # --- FUSION: CARIA-SR ---
    # Alpha=1.5, Beta=1.0 for non-linear sensitivity
    term_structure = (F_rank ** 1.5) * (S_rank ** 1.0)
    term_structure = term_structure.rolling(window).rank(pct=True)
    
    CARIA_SR = 0.5 * E4_rank + 0.5 * term_structure
    
    # Also compute HAR-RV for comparison
    HAR_RV = (0.3*v_fast + 0.4*v_med + 0.3*v_slow).rolling(window).rank(pct=True)
    
    return pd.DataFrame({'CARIA_SR': CARIA_SR, 'HAR_RV': HAR_RV, 'Returns': r}).dropna()

# -----------------------------------------------------------------------------
# 3. RUN CROSS-ASSET LOOP
# -----------------------------------------------------------------------------
ASSETS = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'EFA', 'EEM', 'GLD']
print(f"\n[2] Testing across {len(ASSETS)} assets...")

results = []

for ticker in ASSETS:
    try:
        # Descargar data del activo
        asset_df = yf.download(ticker, start='2007-01-01', progress=False)
        # Handle MultiIndex columns if present
        if isinstance(asset_df.columns, pd.MultiIndex):
            df = asset_df['Close']
            if isinstance(df, pd.DataFrame):
                df = df.iloc[:, 0]  # Take first column if still DataFrame
        else:
            df = asset_df['Close']
        ret_asset = df.pct_change().dropna()
        
        # Calcular CARIA-SR pasando la serie de crédito global
        df_model = compute_caria_sr_safe(ret_asset, vol_credit_global)
        
        if df_model is None:
            print(f"  ⚠ {ticker}: Insufficient data alignment")
            continue
            
        # Validar (p1 Tail Risk)
        fwd = df_model['Returns'].rolling(5).sum().shift(-5)
        # Definir crisis localmente para cada activo (1% peor)
        y_p1 = (fwd < fwd.quantile(0.01)).astype(int)
        
        # Alinear
        idx = y_p1.dropna().index.intersection(df_model.index)
        
        if len(idx) > 500:
            auc_caria = roc_auc_score(y_p1.loc[idx], df_model.loc[idx, 'CARIA_SR'])
            auc_har = roc_auc_score(y_p1.loc[idx], df_model.loc[idx, 'HAR_RV'])
            
            winner = "CARIA" if auc_caria > auc_har else "HAR"
            diff = auc_caria - auc_har
            
            print(f"  {ticker:<5} | p1 AUC: {auc_caria:.3f} (HAR: {auc_har:.3f}) | Δ: {diff:+.3f} | Winner: {winner}")
            
            results.append({'Asset': ticker, 'CARIA': auc_caria, 'HAR': auc_har, 'Diff': diff})
            
    except Exception as e:
        print(f"  ✖ {ticker}: Error {e}")

# -----------------------------------------------------------------------------
# 4. SUMMARY
# -----------------------------------------------------------------------------
if results:
    df_res = pd.DataFrame(results)
    print("\n" + "="*60)
    print(f"FINAL SCOREBOARD (Tail Risk p1)")
    print("="*60)
    print(f"CARIA Wins: {len(df_res[df_res['Diff'] > 0])} / {len(df_res)}")
    print(f"Avg Advantage: {df_res['Diff'].mean():+.4f}")

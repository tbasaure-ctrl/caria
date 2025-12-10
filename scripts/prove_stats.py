
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import signal
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import mutual_info_regression

# === CONFIG ===
ASSETS = {
    'S&P 500': '^GSPC',  # The "Heavy" Object
    'Bitcoin': 'BTC-USD' # The "Light" Object
}
START_DATE = '2010-01-01'

def fetch_data(ticker):
    print(f"Fetching {ticker}...")
    df = yf.download(ticker, start=START_DATE, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df = df.xs('Close', axis=1, level=0)
    return df.iloc[:, 0].dropna() if isinstance(df, pd.DataFrame) else df.dropna()

def calculate_physics(price_series):
    # 1. Vector Physics: Raw Returns for Decomposition
    ret = price_series.pct_change().dropna()
    
    # 2. Decomposition (Institutional Physics)
    # Fast: < 1 week (Noise)
    # Medium: 1 week - 1 quarter (Resonance)
    # Slow: > 1 quarter (Structure)
    fast = ret - ret.rolling(5).mean()
    med = ret.rolling(5).mean() - ret.rolling(60).mean()
    slow = ret.rolling(60).mean() # Using 60 as base for Slow to capture quarterly cycle
    
    # 3. Phase Extraction (Hilbert)
    phases = np.angle(np.column_stack([
        signal.hilbert(fast.fillna(0).values), 
        signal.hilbert(med.fillna(0).values), 
        signal.hilbert(slow.fillna(0).values)
    ]))
    
    # 4. Synchronization (r)
    r = np.abs(np.mean(np.exp(1j * phases), axis=1))
    sync = pd.Series(r, index=ret.index).rolling(20).mean().fillna(0)
    
    # 5. Entropy (H)
    def calc_entropy(row):
        counts, _ = np.histogram(row, bins=8, range=(-np.pi, np.pi), density=True)
        counts = counts[counts > 0]
        return -np.sum(counts * np.log(counts))
    
    entropy_vals = np.apply_along_axis(calc_entropy, 1, phases)
    entropy = pd.Series(entropy_vals, index=ret.index)
    
    # 6. Signals
    # Volatility (Speed) - Aligned with Slow window for Regime detection
    vol = ret.rolling(60).std() * np.sqrt(252)
    
    # Structural Momentum (Mass * Speed)
    momentum = sync * vol
    
    return pd.DataFrame({
        'Price': price_series,
        'Returns': ret,
        'Sync': sync,
        'Entropy': entropy,
        'Vol': vol,
        'Momentum': momentum
    }).dropna()

def run_diagnostics(name, df):
    print(f"\n>>> DIAGNOSTICS FOR {name} <<<")
    
    # --- 1. THERMODYNAMICS CHECK ---
    # Mutual Information between 1/Sync and Entropy
    # We expect High Sync -> Low Entropy. So 1/Sync and Entropy should be related.
    inv_sync = 1 / (df['Sync'] + 0.01)
    mi = mutual_info_regression(inv_sync.values.reshape(-1, 1), df['Entropy'].values)
    print(f"[Thermodynamics] Mutual Information (1/r vs H): {mi[0]:.4f}")
    
    # --- 2. CLASSIFICATION CHECK ---
    # Target: Did the market crash (-5%) in the next 20 days?
    fwd_ret = df['Price'].pct_change(20).shift(-20)
    y_true = (fwd_ret < -0.05).astype(int)
    
    valid_idx = y_true.dropna().index
    y_true = y_true[valid_idx]
    df_eval = df.loc[valid_idx]
    
    # Models
    # A. Volatility Only (The VIX Model)
    vol_thresh = df_eval['Vol'].quantile(0.80)
    pred_vol = (df_eval['Vol'] > vol_thresh).astype(int)
    
    # B. Smart CARIA (Vector Model)
    # Logic: Sync > 0.8 (Mass) AND Down Trend (Direction)
    sync_thresh = 0.75 # High Consensus
    trend = df_eval['Price'].pct_change(20)
    pred_smart = ((df_eval['Sync'] > sync_thresh) & (trend < 0)).astype(int)
    
    # Metrics
    for model_name, y_pred in [('Volatility Only', pred_vol), ('Smart CARIA', pred_smart)]:
        mcc = matthews_corrcoef(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred)
        conf = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = conf.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"\n--- Model: {model_name} ---")
        print(f"MCC Score: {mcc:.4f}  ( Random=0, Perfect=1 )")
        print(f"Precision: {prec:.2%}  ( 'When I say Exit, do we crash?' )")
        print(f"Recall:    {rec:.2%}  ( 'Did I catch the crashes?' )")
        print(f"False Pos: {fpr:.2%}  ( 'Did I panic unnecessarily?' )")

# === EXECUTION ===
with open('final_proof.txt', 'w', encoding='utf-8') as f:
    for name, ticker in ASSETS.items():
        try:
            data = fetch_data(ticker)
            if data.empty: continue
            phys = calculate_physics(data)
            
            # Capture print output by redirecting stdout temporarily or just writing directly
            # For simplicity, let's redefine run_diagnostics to return string or write to f
            
            f.write(f"\n>>> DIAGNOSTICS FOR {name} <<<\n")
            f.write(f"Sync Mean: {phys['Sync'].mean():.4f}, Max: {phys['Sync'].max():.4f}\n")
            
            # MI Debug: Ensure no NaNs and align properly
            valid_mi = phys[['Sync', 'Entropy']].dropna()
            inv_sync = 1 / (valid_mi['Sync'] + 0.01)
            mi = mutual_info_regression(inv_sync.values.reshape(-1, 1), valid_mi['Entropy'].values)
            f.write(f"[Thermodynamics] Mutual Information (1/r vs H): {mi[0]:.4f}\n")
            
            fwd_ret = phys['Price'].pct_change(20).shift(-20)
            y_true = (fwd_ret < -0.05).astype(int)
            valid_idx = y_true.dropna().index
            y_true = y_true.loc[valid_idx] # Fix: use loc
            df_eval = phys.loc[valid_idx]
            
            vol_thresh = df_eval['Vol'].quantile(0.80)
            pred_vol = (df_eval['Vol'] > vol_thresh).astype(int)
            
            # Model B: Smart CARIA (Structural Momentum)
            # Logic: High Momentum (Mass * Speed) AND Down Trend
            # This is the "Anvil" theory. Sync alone (Mass) isn't enough without Speed (Vol).
            mom_thresh = phys['Momentum'].quantile(0.80)
            f.write(f"Momentum Threshold (80th pc): {mom_thresh:.4f}\n")
            
            pred_smart = ((phys['Momentum'] > mom_thresh) & (phys['Price'].pct_change(20) < 0)).astype(int)
            pred_smart = pred_smart.loc[valid_idx] # Align
            
            for model_name, y_pred in [('Volatility Only', pred_vol), ('Smart CARIA (Momentum)', pred_smart)]:
                mcc = matthews_corrcoef(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred)
                conf = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = conf.ravel()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                f.write(f"\n--- Model: {model_name} ---\n")
                f.write(f"MCC Score: {mcc:.4f}\n")
                f.write(f"Precision: {prec:.2%}\n")
                f.write(f"Recall:    {rec:.2%}\n")
                f.write(f"False Pos: {fpr:.2%}\n")
                
        except Exception as e:
            f.write(f"Error analyzing {name}: {str(e)}\n")

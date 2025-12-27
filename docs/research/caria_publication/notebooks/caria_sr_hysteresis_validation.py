#!/usr/bin/env python3
"""
CARIA-SR Hysteresis Validation Script
======================================

Complete validation pipeline for "Hysteresis in Financial Fragility:
Structural Memory and Tail Risk in Low-Volatility Regimes"

This script addresses the reviewer requirements:
1. Absorption Ratio calculation with Ledoit-Wolf shrinkage
2. Peak Memory (Hysteresis) signal: rolling(60).max()
3. Quantile Regression for tail risk (τ = 0.05)
4. Regime-conditional analysis (VIX < 15, VIX < 20)
5. Robustness heatmap (sensitivity analysis)
6. Minsky Hedge backtest with economic performance

Author: CARIA Research Team
Date: December 2025
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Optional imports with fallback
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

try:
    import statsmodels.formula.api as smf
    from statsmodels.regression.quantile_regression import QuantReg
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. Quantile regression unavailable.")

try:
    from sklearn.covariance import LedoitWolf
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not installed. Using simple covariance.")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# ==============================================================================
# CONFIGURATION
# ==============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Core parameters
PARAMS = {
    'START_DATE': "2000-01-01",
    'FWD_WINDOW': 22,           # ~1 month forward (standard)
    'CRASH_QUANTILE': 0.05,     # Bottom 5% = tail risk
    'COV_WINDOW': 252,          # 1 year rolling covariance
    'MEMORY_WINDOW': 60,        # 60-day hysteresis (quarterly cycle)
    'K_FRAC': 0.20,             # Top 20% eigenvalues for AR
    'MIN_ASSETS': 120,          # Minimum assets for covariance
}

# Sensitivity analysis grid
MEMORY_WINDOWS = [20, 40, 60, 90, 120]
VIX_THRESHOLDS = [12, 15, 18, 20, 22, 25]

# Known crisis events
CRISIS_EVENTS = {
    'Dot_Com_Bottom': pd.Timestamp('2002-10-09'),
    'GFC_Peak': pd.Timestamp('2007-10-11'),
    'Flash_Crash': pd.Timestamp('2010-05-06'),
    'Euro_Crisis': pd.Timestamp('2011-08-08'),
    'China_Crash': pd.Timestamp('2015-08-24'),
    'Volmageddon': pd.Timestamp('2018-02-05'),
    'COVID_Crash': pd.Timestamp('2020-03-16'),
    'SVB_Collapse': pd.Timestamp('2023-03-10'),
}


@dataclass
class PerformanceMetrics:
    """Performance statistics for backtest."""
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    calmar: float
    
    def __repr__(self):
        return (f"CAGR={self.cagr:.2%}, Vol={self.volatility:.2%}, "
                f"Sharpe={self.sharpe:.2f}, MaxDD={self.max_drawdown:.1%}, "
                f"Calmar={self.calmar:.2f}")


# ==============================================================================
# CORE METRICS: ABSORPTION RATIO & SPECTRAL ENTROPY
# ==============================================================================

def cov_to_corr(S: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix."""
    d = np.sqrt(np.diag(S))
    d = np.where(d == 0, np.nan, d)
    C = S / np.outer(d, d)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    return (C + C.T) / 2.0


def eigenvalue_metrics(C: np.ndarray, k_frac: float = 0.2) -> Tuple[float, float]:
    """
    Calculate Absorption Ratio and Spectral Entropy from correlation matrix.
    
    Returns:
        ar: Absorption Ratio (fraction of variance in top k eigenvectors)
        entropy: Normalized spectral entropy (disorder measure)
    """
    w = np.linalg.eigvalsh(C)
    w = np.sort(w)[::-1]
    n = len(w)
    k = max(1, int(np.ceil(k_frac * n)))
    
    # Absorption Ratio: Kritzman et al. (2010)
    ar = float(np.sum(w[:k]) / np.sum(w))
    
    # Spectral Entropy: Shannon entropy of eigenvalue distribution
    p = w / np.sum(w)
    p = p[p > 0]
    ent = -np.sum(p * np.log(p))
    ent_norm = float(ent / np.log(n)) if n > 1 else np.nan
    
    return ar, ent_norm


def rolling_structural_metrics(
    ret_mat: pd.DataFrame,
    window: int = 252,
    k_frac: float = 0.2,
    min_assets: int = 120,
    step: int = 5,
    coverage_threshold: float = 0.90,
) -> pd.DataFrame:
    """
    Calculate rolling Absorption Ratio and Spectral Entropy.
    
    Uses Ledoit-Wolf shrinkage for robust covariance estimation.
    """
    idx = ret_mat.index
    out = pd.DataFrame(index=idx, columns=["AR", "Entropy", "N_assets"], dtype=float)
    
    if HAS_SKLEARN:
        lw = LedoitWolf()
    
    for t in range(window, len(idx), step):
        W = ret_mat.iloc[t - window + 1 : t + 1]
        good = W.notna().mean() >= coverage_threshold
        W = W.loc[:, good]
        
        if W.shape[1] < min_assets:
            continue
        
        # Fill residual NaN with column mean
        W = W.apply(lambda s: s.fillna(s.mean()), axis=0)
        X = W.to_numpy()
        X = X - np.nanmean(X, axis=0, keepdims=True)
        
        try:
            if HAS_SKLEARN:
                S = lw.fit(X).covariance_
                C = cov_to_corr(S)
            else:
                C = np.corrcoef(X, rowvar=False)
                C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
                C = (C + C.T) / 2.0
        except Exception:
            C = np.corrcoef(X, rowvar=False)
            C = np.nan_to_num(C)
            C = (C + C.T) / 2.0
        
        ar, ent = eigenvalue_metrics(C, k_frac=k_frac)
        out.iloc[t] = [ar, ent, W.shape[1]]
    
    return out.ffill().bfill()


def zscore(s: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score normalization."""
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std().replace(0, np.nan)
    return (s - mu) / sd


def compute_caria_sr(struct_df: pd.DataFrame, window: int = 252) -> pd.Series:
    """
    Compute CARIA-SR index from structural metrics.
    
    Formula: CARIA_SR = Z(AR) + Z(1 - Entropy)
    
    High AR + Low Entropy = High Synchronization = Fragile
    """
    ar_z = zscore(struct_df['AR'], window)
    # Invert entropy: low entropy = high order = fragile
    ent_z = zscore(1.0 - struct_df['Entropy'], window)
    return ar_z + ent_z


def add_peak_memory(df: pd.DataFrame, signal_col: str, memory_windows: List[int] = None) -> pd.DataFrame:
    """
    Add Peak Memory (Hysteresis) signals.
    
    The "memory" captures that stress leaves structural scars that persist
    even when current volatility normalizes.
    """
    if memory_windows is None:
        memory_windows = MEMORY_WINDOWS
    
    for w in memory_windows:
        df[f'Peak{w}'] = df[signal_col].rolling(w).max()
    
    return df


# ==============================================================================
# QUANTILE REGRESSION: TAIL RISK PREDICTION
# ==============================================================================

def run_quantile_regression(
    df: pd.DataFrame,
    target_col: str = 'future_ret',
    tau: float = 0.05,
    vix_threshold: Optional[float] = None,
    memory_col: str = 'Peak60'
) -> Dict:
    """
    Run quantile regression for tail risk prediction.
    
    Model: Q_τ(future_ret) = α + β₁*VIX + β₂*Peak_Memory + ε
    
    Returns comparison of VIX-only vs VIX+Structure models.
    """
    if not HAS_STATSMODELS:
        return {"error": "statsmodels not installed"}
    
    # Filter by VIX regime if specified
    if vix_threshold is not None and 'volatility' in df.columns:
        subset = df[df['volatility'] < vix_threshold].copy()
    else:
        subset = df.copy()
    
    subset = subset.dropna(subset=[target_col, 'volatility', memory_col])
    
    if len(subset) < 500:
        return {"error": f"Insufficient data: {len(subset)} < 500"}
    
    # Model A: VIX Only (Baseline)
    mod_vix = smf.quantreg(f'{target_col} ~ volatility', subset)
    res_vix = mod_vix.fit(q=tau)
    
    # Model B: VIX + Peak Memory (Structural)
    mod_struct = smf.quantreg(f'{target_col} ~ volatility + {memory_col}', subset)
    res_struct = mod_struct.fit(q=tau)
    
    # Calculate improvement
    r2_base = res_vix.prsquared
    r2_struct = res_struct.prsquared
    improvement = ((r2_struct - r2_base) / r2_base) * 100 if r2_base > 0 else np.nan
    
    return {
        'n_obs': len(subset),
        'vix_threshold': vix_threshold,
        'tau': tau,
        'base_pseudo_r2': r2_base,
        'struct_pseudo_r2': r2_struct,
        'improvement_pct': improvement,
        'memory_coef': res_struct.params.get(memory_col, np.nan),
        'memory_pval': res_struct.pvalues.get(memory_col, np.nan),
        'vix_coef': res_struct.params.get('volatility', np.nan),
        'vix_pval': res_struct.pvalues.get('volatility', np.nan),
        'model_summary': res_struct.summary() if hasattr(res_struct, 'summary') else None
    }


# ==============================================================================
# ROBUSTNESS HEATMAP: SENSITIVITY ANALYSIS
# ==============================================================================

def run_sensitivity_analysis(
    df: pd.DataFrame,
    memory_windows: List[int] = None,
    vix_thresholds: List[float] = None,
    tau: float = 0.05
) -> pd.DataFrame:
    """
    Create robustness heatmap across parameter combinations.
    
    Tests Pseudo-R² improvement for each (Memory Window, VIX Threshold) pair.
    """
    if not HAS_STATSMODELS:
        print("Warning: statsmodels not installed. Cannot run sensitivity analysis.")
        return pd.DataFrame()
    
    if memory_windows is None:
        memory_windows = MEMORY_WINDOWS
    if vix_thresholds is None:
        vix_thresholds = VIX_THRESHOLDS
    
    results = np.zeros((len(memory_windows), len(vix_thresholds)))
    
    print("Running Sensitivity Grid Analysis...")
    
    # Pre-compute all memory signals
    signal_col = 'caria_sr' if 'caria_sr' in df.columns else 'absorp_z'
    for w in memory_windows:
        if f'peak_{w}' not in df.columns:
            df[f'peak_{w}'] = df[signal_col].rolling(window=w).max()
    
    for i, w in enumerate(memory_windows):
        for j, v in enumerate(vix_thresholds):
            subset = df[df['volatility'] < v].copy()
            
            # Prepare target
            if 'future_ret' not in subset.columns:
                subset['future_ret'] = subset['price'].pct_change(22).shift(-22)
            subset = subset.dropna()
            
            if len(subset) > 500:
                try:
                    # Base model (VIX only)
                    mod_base = smf.quantreg('future_ret ~ volatility', subset)
                    res_base = mod_base.fit(q=tau)
                    
                    # Structural model (VIX + Memory)
                    mod_struct = smf.quantreg(f'future_ret ~ volatility + peak_{w}', subset)
                    res_struct = mod_struct.fit(q=tau)
                    
                    # Calculate improvement
                    imp = ((res_struct.prsquared - res_base.prsquared) / 
                           res_base.prsquared) * 100 if res_base.prsquared > 0 else 0
                    results[i, j] = imp
                except Exception as e:
                    results[i, j] = 0
            else:
                results[i, j] = np.nan
    
    # Create DataFrame
    results_df = pd.DataFrame(
        results,
        index=memory_windows,
        columns=vix_thresholds
    )
    results_df.index.name = 'Memory_Window'
    results_df.columns.name = 'VIX_Threshold'
    
    return results_df


def plot_robustness_heatmap(results_df: pd.DataFrame, output_path: str = None):
    """Plot robustness heatmap."""
    if not HAS_PLOT:
        print("Matplotlib not available for plotting")
        return
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        results_df, 
        annot=True, 
        fmt=".1f", 
        cmap="RdYlGn",
        center=0,
        vmin=-50,
        vmax=100
    )
    plt.title("Robustness Check: Improvement in Tail Risk Prediction (%)\n"
              "(Y=Memory Window, X=VIX Threshold)")
    plt.xlabel("Low-Volatility Threshold (VIX < X)")
    plt.ylabel("Memory Window (Days)")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {output_path}")
    
    plt.close()


# ==============================================================================
# MINSKY HEDGE BACKTEST
# ==============================================================================

def get_max_drawdown(equity: pd.Series) -> float:
    """Calculate maximum drawdown."""
    eq = equity.dropna()
    if len(eq) == 0:
        return np.nan
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def calculate_performance(returns: pd.Series, ann: int = 252) -> PerformanceMetrics:
    """Calculate standard performance metrics."""
    r = returns.dropna()
    if len(r) < 30:
        return PerformanceMetrics(np.nan, np.nan, np.nan, np.nan, np.nan)
    
    eq = (1 + r).cumprod()
    years = len(r) / ann
    
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(r.std() * np.sqrt(ann))
    sharpe = float((r.mean() / (r.std() + 1e-12)) * np.sqrt(ann))
    mdd = get_max_drawdown(eq)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan
    
    return PerformanceMetrics(cagr, vol, sharpe, mdd, calmar)


def run_minsky_hedge_backtest(
    df: pd.DataFrame,
    vix_threshold: float = 20,
    peak_z_threshold: float = 1.5,
    memory_col: str = 'PeakZ'
) -> Dict:
    """
    Run Minsky Hedge backtest: Switch SPY -> TLT when fragile.
    
    Fragile state = (VIX < threshold) AND (Peak_Z > 1.5)
    The "Deep Calm" paradox: structure is dangerous when VIX is low.
    """
    # Define fragile state
    fragile = (df['volatility'] < vix_threshold) & (df[memory_col] > peak_z_threshold)
    
    # Strategy returns: Treasury when fragile, Equity otherwise
    if 'treasury_ret' in df.columns:
        tlt_ret = df['treasury_ret']
    else:
        # Approximate treasury return from yield
        tlt_ret = (df['treasury_10y'] / 100) / 252 if 'treasury_10y' in df.columns else 0
    
    spy_ret = df['daily_ret'] if 'daily_ret' in df.columns else df['spy_ret']
    
    strategy_ret = pd.Series(
        np.where(fragile, tlt_ret, spy_ret),
        index=df.index
    )
    
    # Calculate metrics
    perf_strategy = calculate_performance(strategy_ret)
    perf_bnh = calculate_performance(spy_ret)
    
    # Calculate cumulative equity
    eq_strategy = (1 + strategy_ret.fillna(0)).cumprod()
    eq_bnh = (1 + spy_ret.fillna(0)).cumprod()
    
    # Crisis-specific drawdowns
    crisis_drawdowns = {}
    for name, (start, end) in {
        'GFC': ('2007-10-01', '2009-06-30'),
        'COVID': ('2020-02-01', '2020-06-30')
    }.items():
        try:
            mdd_strat = get_max_drawdown(eq_strategy.loc[start:end])
            mdd_bnh = get_max_drawdown(eq_bnh.loc[start:end])
            crisis_drawdowns[name] = {
                'strategy_dd': mdd_strat,
                'benchmark_dd': mdd_bnh,
                'protection': mdd_strat - mdd_bnh
            }
        except KeyError:
            pass
    
    return {
        'vix_threshold': vix_threshold,
        'peak_z_threshold': peak_z_threshold,
        'time_in_hedge_pct': fragile.mean() * 100,
        'strategy_perf': perf_strategy,
        'benchmark_perf': perf_bnh,
        'crisis_drawdowns': crisis_drawdowns,
        'equity_strategy': eq_strategy,
        'equity_benchmark': eq_bnh,
        'fragile_mask': fragile
    }


# ==============================================================================
# PUBLICATION TABLES
# ==============================================================================

def generate_auc_table(results: List[Dict], output_path: str = None) -> pd.DataFrame:
    """Generate Table 1: AUC Comparison with regime conditioning."""
    rows = []
    for r in results:
        rows.append({
            'Regime': f"VIX < {r['vix_threshold']}" if r['vix_threshold'] else "All",
            'N_obs': r['n_obs'],
            'Baseline_R2': r['base_pseudo_r2'],
            'Structural_R2': r['struct_pseudo_r2'],
            'Improvement_%': r['improvement_pct'],
            'Memory_Coef': r['memory_coef'],
            'Memory_pval': r['memory_pval']
        })
    
    df = pd.DataFrame(rows)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved AUC table to {output_path}")
    
    return df


def generate_backtest_table(results: List[Dict], output_path: str = None) -> pd.DataFrame:
    """Generate Table 3: Minsky Hedge Performance."""
    rows = []
    for r in results:
        perf = r['strategy_perf']
        bnh = r['benchmark_perf']
        rows.append({
            'VIX_Threshold': r['vix_threshold'],
            'Time_in_Hedge_%': r['time_in_hedge_pct'],
            'Strategy_CAGR': perf.cagr,
            'Strategy_MaxDD': perf.max_drawdown,
            'Strategy_Sharpe': perf.sharpe,
            'Strategy_Calmar': perf.calmar,
            'Benchmark_CAGR': bnh.cagr,
            'Benchmark_MaxDD': bnh.max_drawdown,
            'GFC_Protection': r['crisis_drawdowns'].get('GFC', {}).get('protection', np.nan),
            'COVID_Protection': r['crisis_drawdowns'].get('COVID', {}).get('protection', np.nan)
        })
    
    df = pd.DataFrame(rows)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved backtest table to {output_path}")
    
    return df


# ==============================================================================
# MAIN VALIDATION PIPELINE
# ==============================================================================

def load_sp500_universe(data_path: str = None) -> Optional[pd.DataFrame]:
    """Load S&P 500 universe price data."""
    if data_path is None:
        data_path = os.path.join(DATA_DIR, 'sp500_universe_fmp.parquet')
    
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found")
        return None
    
    px = pd.read_parquet(data_path)
    if 'date' in px.columns:
        px['date'] = pd.to_datetime(px['date'])
        px = px.set_index('date')
    px.index = pd.to_datetime(px.index)
    return px.sort_index()


def download_market_data(start: str, end: str) -> Dict[str, pd.Series]:
    """Download VIX, SPY, TLT from Yahoo Finance."""
    if not HAS_YFINANCE:
        print("yfinance not installed. Cannot download market data.")
        return {}
    
    data = {}
    for symbol, name in [("^VIX", "volatility"), ("SPY", "price"), ("TLT", "tlt")]:
        try:
            df = yf.download(symbol, start=start, end=end, progress=False)
            if 'Adj Close' in df.columns:
                data[name] = df['Adj Close'].dropna()
            elif 'Close' in df.columns:
                data[name] = df['Close'].dropna()
        except Exception as e:
            print(f"Failed to download {symbol}: {e}")
    
    return data


def run_full_validation(data_path: str = None) -> Dict:
    """
    Run complete CARIA-SR validation pipeline.
    
    Returns all results for publication tables and figures.
    """
    print("=" * 70)
    print("CARIA-SR HYSTERESIS VALIDATION PIPELINE")
    print("=" * 70)
    
    results = {}
    
    # --- 1. LOAD DATA ---
    print("\n--- Phase 1: Loading Data ---")
    px = load_sp500_universe(data_path)
    
    if px is None:
        print("No price universe found. Using synthetic data for demonstration.")
        # Create minimal synthetic data for testing
        dates = pd.date_range("2000-01-01", "2024-12-31", freq='B')
        px = pd.DataFrame(
            np.exp(np.cumsum(np.random.randn(len(dates), 100) * 0.02, axis=0)),
            index=dates,
            columns=[f"STOCK_{i}" for i in range(100)]
        )
    
    px = px.loc[PARAMS['START_DATE']:]
    
    # Calculate log returns
    ret = np.log(px).diff()
    coverage = 1.0 - ret.isna().mean()
    keep = coverage[coverage >= 0.90].index.tolist()
    ret = ret[keep]
    
    print(f"Universe: {ret.shape[1]} assets, {len(ret)} days")
    print(f"Date range: {ret.index.min().date()} to {ret.index.max().date()}")
    
    # --- 2. CALCULATE STRUCTURAL METRICS ---
    print("\n--- Phase 2: Calculating Structural Metrics ---")
    struct = rolling_structural_metrics(
        ret,
        window=PARAMS['COV_WINDOW'],
        k_frac=PARAMS['K_FRAC'],
        min_assets=PARAMS['MIN_ASSETS'],
        step=5
    )
    
    struct['CARIA_SR'] = compute_caria_sr(struct, PARAMS['COV_WINDOW'])
    struct = add_peak_memory(struct, 'CARIA_SR', MEMORY_WINDOWS)
    struct['PeakZ'] = zscore(struct['Peak60'], PARAMS['COV_WINDOW'])
    
    print(f"Absorption Ratio: mean={struct['AR'].mean():.4f}, std={struct['AR'].std():.4f}")
    print(f"CARIA-SR: mean={struct['CARIA_SR'].mean():.4f}, std={struct['CARIA_SR'].std():.4f}")
    
    # --- 3. DOWNLOAD MARKET DATA ---
    print("\n--- Phase 3: Loading Market Data ---")
    start_dl = str(ret.index.min().date())
    end_dl = str((ret.index.max() + pd.Timedelta(days=1)).date())
    
    market_data = download_market_data(start_dl, end_dl)
    
    # Align all data
    if market_data:
        common_idx = struct.index
        for name, series in market_data.items():
            common_idx = common_idx.intersection(series.index)
        
        df = struct.loc[common_idx].copy()
        for name, series in market_data.items():
            df[name] = series.loc[common_idx]
        
        df['daily_ret'] = np.log(df['price']).diff()
        df['future_ret'] = df['price'].pct_change(PARAMS['FWD_WINDOW']).shift(-PARAMS['FWD_WINDOW'])
    else:
        df = struct.copy()
        df['volatility'] = 15 + np.random.randn(len(df)) * 5  # Synthetic VIX
        df['daily_ret'] = np.random.randn(len(df)) * 0.01
        df['future_ret'] = df['daily_ret'].shift(-22)
    
    df = df.dropna()
    print(f"Aligned dataset: {len(df)} observations")
    
    results['df'] = df
    results['struct'] = struct
    
    # --- 4. QUANTILE REGRESSION TESTS ---
    print("\n--- Phase 4: Quantile Regression (Tail Risk) ---")
    qr_results = []
    
    for vix_thr in [None, 25, 20, 18, 15, 12]:
        label = f"VIX < {vix_thr}" if vix_thr else "All periods"
        print(f"  Testing {label}...")
        
        qr = run_quantile_regression(
            df, 
            target_col='future_ret',
            tau=PARAMS['CRASH_QUANTILE'],
            vix_threshold=vix_thr,
            memory_col='Peak60'
        )
        
        if 'error' not in qr:
            qr_results.append(qr)
            print(f"    N={qr['n_obs']}, Base R²={qr['base_pseudo_r2']:.5f}, "
                  f"Struct R²={qr['struct_pseudo_r2']:.5f}, Δ={qr['improvement_pct']:.1f}%")
    
    results['quantile_regression'] = qr_results
    
    # --- 5. ROBUSTNESS HEATMAP ---
    print("\n--- Phase 5: Robustness Analysis ---")
    sensitivity_df = run_sensitivity_analysis(df, MEMORY_WINDOWS, VIX_THRESHOLDS)
    results['sensitivity'] = sensitivity_df
    
    if len(sensitivity_df) > 0:
        heatmap_path = os.path.join(OUTPUT_DIR, 'robustness_heatmap.png')
        plot_robustness_heatmap(sensitivity_df, heatmap_path)
        print(f"\nSensitivity Grid (Pseudo-R² Improvement %):")
        print(sensitivity_df.to_string())
    
    # --- 6. MINSKY HEDGE BACKTEST ---
    print("\n--- Phase 6: Minsky Hedge Backtest ---")
    backtest_results = []
    
    for vix_thr in [15, 18, 20]:
        print(f"\n  Testing VIX < {vix_thr} threshold...")
        bt = run_minsky_hedge_backtest(df, vix_threshold=vix_thr)
        backtest_results.append(bt)
        
        print(f"    Time in hedge: {bt['time_in_hedge_pct']:.1f}%")
        print(f"    Strategy: {bt['strategy_perf']}")
        print(f"    Benchmark: {bt['benchmark_perf']}")
        
        for crisis, dd in bt['crisis_drawdowns'].items():
            print(f"    {crisis}: Strategy DD={dd['strategy_dd']:.1%} vs "
                  f"Benchmark DD={dd['benchmark_dd']:.1%}")
    
    results['backtest'] = backtest_results
    
    # --- 7. GENERATE PUBLICATION TABLES ---
    print("\n--- Phase 7: Generating Publication Tables ---")
    
    tables_dir = os.path.join(OUTPUT_DIR, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    auc_table = generate_auc_table(
        qr_results, 
        os.path.join(tables_dir, 'Table_1_Quantile_Regression.csv')
    )
    
    backtest_table = generate_backtest_table(
        backtest_results,
        os.path.join(tables_dir, 'Table_3_Minsky_Hedge.csv')
    )
    
    sensitivity_df.to_csv(os.path.join(tables_dir, 'Table_2_Sensitivity.csv'))
    
    results['tables'] = {
        'auc': auc_table,
        'backtest': backtest_table,
        'sensitivity': sensitivity_df
    }
    
    # --- SUMMARY ---
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    # Key findings
    if qr_results:
        deep_calm = [r for r in qr_results if r.get('vix_threshold') == 15]
        if deep_calm:
            dc = deep_calm[0]
            print(f"\n** KEY FINDING: Deep Calm Regime (VIX < 15) **")
            print(f"   Observations: {dc['n_obs']}")
            print(f"   Baseline (VIX only) Pseudo-R²: {dc['base_pseudo_r2']:.5f}")
            print(f"   Structural (+Peak60) Pseudo-R²: {dc['struct_pseudo_r2']:.5f}")
            print(f"   IMPROVEMENT: {dc['improvement_pct']:.1f}%")
            print(f"   Peak Memory coefficient: {dc['memory_coef']:.5f} (p={dc['memory_pval']:.4f})")
    
    return results


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    results = run_full_validation()
    print("\nDone!")

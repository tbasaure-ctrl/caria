"""
CARIA-SR Benchmark Comparison
==============================

Rigorous comparison of CARIA-SR against established benchmarks:

1. HAR-RV (Heterogeneous AutoRegressive Realized Volatility)
   - Corsi (2009) model
   - Industry standard for volatility forecasting

2. VIX (CBOE Volatility Index)
   - Market-implied volatility
   - "Fear gauge"

3. Rolling Volatility (Simple Baseline)
   - 21-day rolling standard deviation
   - Naive benchmark

Statistical Tests:
- Bootstrap AUC comparison with confidence intervals
- Diebold-Mariano test for predictive accuracy
- McNemar test for classification performance
- Permutation tests for signal authenticity

Author: Tomás Basaure
Date: December 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==============================================================================
# BENCHMARK MODELS
# ==============================================================================

class BenchmarkModels:
    """
    Collection of benchmark models for comparison.
    """
    
    @staticmethod
    def har_rv(returns, windows=(5, 21, 63), rank_window=252):
        """
        HAR-RV: Heterogeneous AutoRegressive Realized Volatility
        
        Model: RV_t+h = β0 + β1*RV_d + β2*RV_w + β3*RV_m + ε
        
        Reference: Corsi, F. (2009). "A Simple Approximate Long-Memory 
        Model of Realized Volatility." Journal of Financial Econometrics.
        
        For our purposes, we use a percentile-ranked version.
        """
        v_d = returns.rolling(windows[0]).std() * np.sqrt(252)
        v_w = returns.rolling(windows[1]).std() * np.sqrt(252)
        v_m = returns.rolling(windows[2]).std() * np.sqrt(252)
        
        # Standard HAR weights
        har_raw = 0.3 * v_d + 0.4 * v_w + 0.3 * v_m
        har = har_raw.rolling(rank_window).rank(pct=True)
        
        return har
    
    @staticmethod
    def vix_percentile(vix_series, rank_window=252):
        """
        VIX percentile ranking.
        
        Higher percentile = higher implied volatility = higher risk
        """
        return vix_series.rolling(rank_window).rank(pct=True)
    
    @staticmethod
    def rolling_volatility(returns, window=21, rank_window=252):
        """
        Simple rolling volatility baseline.
        """
        vol = returns.rolling(window).std() * np.sqrt(252)
        return vol.rolling(rank_window).rank(pct=True)
    
    @staticmethod
    def garch_proxy(returns, window=21, rank_window=252):
        """
        GARCH(1,1) proxy using exponentially weighted volatility.
        
        Not a full GARCH model, but captures volatility clustering.
        """
        # Exponential weights
        ewm_vol = returns.ewm(span=window, min_periods=window).std() * np.sqrt(252)
        return ewm_vol.rolling(rank_window).rank(pct=True)
    
    @staticmethod
    def caria_sr(returns, vol_credit, windows=(5, 21, 63, 42), 
                 weights=(0.20, 0.30, 0.25, 0.25), rank_window=252):
        """
        CARIA-SR: Structural Resonance Index
        
        Multi-scale volatility + momentum synchronization + credit
        """
        # Volatilities
        v_fast = returns.rolling(windows[0]).std() * np.sqrt(252)
        v_med = returns.rolling(windows[1]).std() * np.sqrt(252)
        v_slow = returns.rolling(windows[2]).std() * np.sqrt(252)
        
        # E4: Multi-scale energy with credit
        E4_raw = (weights[0] * v_fast + 
                  weights[1] * v_med + 
                  weights[2] * v_slow + 
                  weights[3] * vol_credit)
        E4 = E4_raw.rolling(rank_window).rank(pct=True)
        
        # Sync: Momentum correlation
        m_fast = returns.rolling(windows[0]).sum()
        m_slow = returns.rolling(windows[2]).sum()
        sync_raw = m_fast.rolling(windows[1]).corr(m_slow)
        sync = ((sync_raw + 1) / 2).rolling(rank_window).rank(pct=True)
        
        # CARIA-SR
        SR_raw = E4 * (1 + sync)
        SR = SR_raw.rolling(rank_window).rank(pct=True)
        
        return SR


# ==============================================================================
# STATISTICAL TESTS
# ==============================================================================

def bootstrap_auc_comparison(y_true, scores1, scores2, n_bootstrap=1000, 
                              model1_name='Model 1', model2_name='Model 2'):
    """
    Compare two models using bootstrap AUC.
    
    Returns:
    --------
    dict : AUC for each model, difference CI, and p-value
    """
    np.random.seed(RANDOM_SEED)
    
    y_true = np.asarray(y_true).flatten()
    scores1 = np.asarray(scores1).flatten()
    scores2 = np.asarray(scores2).flatten()
    n = len(y_true)
    
    # Point estimates
    auc1 = roc_auc_score(y_true, scores1)
    auc2 = roc_auc_score(y_true, scores2)
    delta = auc1 - auc2
    
    # Bootstrap
    auc1_boot = []
    auc2_boot = []
    delta_boot = []
    
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        y_boot = y_true[idx]
        s1_boot = scores1[idx]
        s2_boot = scores2[idx]
        
        if len(np.unique(y_boot)) < 2:
            continue
        
        try:
            a1 = roc_auc_score(y_boot, s1_boot)
            a2 = roc_auc_score(y_boot, s2_boot)
            auc1_boot.append(a1)
            auc2_boot.append(a2)
            delta_boot.append(a1 - a2)
        except:
            pass
    
    auc1_boot = np.array(auc1_boot)
    auc2_boot = np.array(auc2_boot)
    delta_boot = np.array(delta_boot)
    
    # Confidence intervals
    ci_auc1 = (np.percentile(auc1_boot, 2.5), np.percentile(auc1_boot, 97.5))
    ci_auc2 = (np.percentile(auc2_boot, 2.5), np.percentile(auc2_boot, 97.5))
    ci_delta = (np.percentile(delta_boot, 2.5), np.percentile(delta_boot, 97.5))
    
    # P-value: proportion of bootstrap deltas <= 0 (one-sided)
    # For two-sided: 2 * min(p, 1-p)
    p_one_sided = (delta_boot <= 0).mean()
    p_two_sided = 2 * min(p_one_sided, 1 - p_one_sided)
    
    # Significant if CI doesn't contain 0
    significant = ci_delta[0] > 0 or ci_delta[1] < 0
    
    return {
        model1_name: {'auc': auc1, 'ci': ci_auc1, 'se': np.std(auc1_boot)},
        model2_name: {'auc': auc2, 'ci': ci_auc2, 'se': np.std(auc2_boot)},
        'delta': delta,
        'delta_ci': ci_delta,
        'delta_se': np.std(delta_boot),
        'p_value': p_two_sided,
        'significant': significant,
        'winner': model1_name if delta > 0 else model2_name
    }


def diebold_mariano_test(y_true, pred1, pred2, h=1):
    """
    Diebold-Mariano test for comparing predictive accuracy.
    
    H0: Both models have equal predictive accuracy
    H1: Models differ in predictive accuracy
    """
    y = np.asarray(y_true).flatten()
    p1 = np.asarray(pred1).flatten()
    p2 = np.asarray(pred2).flatten()
    
    # Loss differentials (squared error)
    e1 = (y - p1) ** 2
    e2 = (y - p2) ** 2
    d = e1 - e2
    
    d_bar = np.mean(d)
    n = len(d)
    
    # HAC variance estimate
    gamma_0 = np.var(d, ddof=1)
    
    if h > 1:
        for k in range(1, h):
            if k < len(d):
                gamma_k = np.cov(d[k:], d[:-k])[0, 1]
                gamma_0 += 2 * (1 - k / h) * gamma_k
    
    var_d_bar = gamma_0 / n
    dm_stat = d_bar / np.sqrt(var_d_bar) if var_d_bar > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return {
        'dm_statistic': dm_stat,
        'p_value': p_value,
        'mean_loss_diff': d_bar,
        'significant': p_value < 0.05,
        'better_model': 'Model 1' if d_bar < 0 else 'Model 2'
    }


def mcnemar_test(y_true, pred1, pred2, threshold=0.5):
    """
    McNemar's test for comparing classification performance.
    
    Compares the error rates of two classifiers on the same data.
    """
    y = np.asarray(y_true).flatten()
    p1 = (np.asarray(pred1).flatten() > threshold).astype(int)
    p2 = (np.asarray(pred2).flatten() > threshold).astype(int)
    
    # Classification results
    correct1 = (p1 == y)
    correct2 = (p2 == y)
    
    # Contingency table
    n01 = np.sum(correct1 & ~correct2)  # Model 1 right, Model 2 wrong
    n10 = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 right
    
    # McNemar statistic (with continuity correction)
    if n01 + n10 == 0:
        return {'statistic': 0, 'p_value': 1.0, 'significant': False}
    
    stat = (np.abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = 1 - stats.chi2.cdf(stat, df=1)
    
    return {
        'statistic': stat,
        'p_value': p_value,
        'n_model1_better': n01,
        'n_model2_better': n10,
        'significant': p_value < 0.05,
        'better_model': 'Model 1' if n01 > n10 else 'Model 2'
    }


def permutation_test_auc(y_true, scores, n_permutations=1000):
    """
    Permutation test for AUC significance.
    
    Tests H0: The model's AUC is due to chance (random labels)
    """
    np.random.seed(RANDOM_SEED)
    
    y = np.asarray(y_true).flatten()
    s = np.asarray(scores).flatten()
    
    # Observed AUC
    observed_auc = roc_auc_score(y, s)
    
    # Null distribution (permuted labels)
    null_aucs = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        null_aucs.append(roc_auc_score(y_perm, s))
    
    null_aucs = np.array(null_aucs)
    
    # P-value: proportion of null AUCs >= observed
    p_value = (null_aucs >= observed_auc).mean()
    
    return {
        'observed_auc': observed_auc,
        'null_mean': null_aucs.mean(),
        'null_std': null_aucs.std(),
        'p_value': p_value,
        'significant': p_value < 0.05
    }


# ==============================================================================
# DATA LOADING & COMPUTATION
# ==============================================================================

def load_data_for_comparison():
    """
    Load all data needed for benchmark comparison.
    """
    print("Loading data for benchmark comparison...")
    
    # Assets
    spy = yf.download("SPY", start="2000-01-01", progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy_close = spy["Close"].iloc[:, 0]
    else:
        spy_close = spy["Close"]
    ret_spy = spy_close.pct_change().dropna()
    
    # Credit
    hyg = yf.download("HYG", start="2005-01-01", progress=False)
    if isinstance(hyg.columns, pd.MultiIndex):
        hyg_close = hyg["Close"].iloc[:, 0]
    else:
        hyg_close = hyg["Close"]
    ret_hyg = hyg_close.pct_change().dropna()
    vol_credit = ret_hyg.rolling(42).std() * np.sqrt(252)
    
    # VIX
    vix = yf.download("^VIX", start="2000-01-01", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix_close = vix["Close"].iloc[:, 0]
    else:
        vix_close = vix["Close"]
    
    # Align all data
    common_idx = ret_spy.index.intersection(vol_credit.index).intersection(vix_close.index)
    
    ret_spy = ret_spy.loc[common_idx]
    vol_credit = vol_credit.loc[common_idx]
    vix = vix_close.loc[common_idx]
    
    print(f"  Data loaded: {len(ret_spy)} observations")
    print(f"  Period: {ret_spy.index.min().strftime('%Y-%m-%d')} to {ret_spy.index.max().strftime('%Y-%m-%d')}")
    
    return ret_spy, vol_credit, vix


def compute_all_models(ret_spy, vol_credit, vix):
    """
    Compute all benchmark models.
    """
    print("\nComputing all models...")
    
    models = {}
    
    # CARIA-SR (our model)
    models['CARIA-SR'] = BenchmarkModels.caria_sr(ret_spy, vol_credit)
    
    # HAR-RV
    models['HAR-RV'] = BenchmarkModels.har_rv(ret_spy)
    
    # VIX percentile
    models['VIX'] = BenchmarkModels.vix_percentile(vix)
    
    # Rolling volatility
    models['Rolling Vol'] = BenchmarkModels.rolling_volatility(ret_spy)
    
    # GARCH proxy
    models['GARCH Proxy'] = BenchmarkModels.garch_proxy(ret_spy)
    
    # Create target (exogenous crashes)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)
    fwd_ret = ret_spy.rolling(window=indexer).sum()
    crash_threshold = fwd_ret.quantile(0.05)
    target = (fwd_ret < crash_threshold).astype(int)
    
    # Build DataFrame
    df = pd.DataFrame(models)
    df['Target'] = target
    df['Fwd_Ret'] = fwd_ret
    df = df.dropna()
    
    print(f"  Models computed: {list(models.keys())}")
    print(f"  Final observations: {len(df)}")
    
    return df


# ==============================================================================
# COMPARISON ANALYSIS
# ==============================================================================

def run_full_comparison(df, n_bootstrap=1000):
    """
    Run complete benchmark comparison.
    """
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON ANALYSIS")
    print("=" * 80)
    
    y = df['Target'].values
    model_cols = ['CARIA-SR', 'HAR-RV', 'VIX', 'Rolling Vol', 'GARCH Proxy']
    
    results = {}
    
    # --- 1. Individual AUC with CI ---
    print("\n[1] AUC Analysis (with Bootstrap CI)")
    print("-" * 60)
    print(f"{'Model':<15} | {'AUC':>8} | {'95% CI':>20} | {'SE':>8}")
    print("-" * 60)
    
    auc_results = {}
    for model in model_cols:
        scores = df[model].values
        
        # Bootstrap
        aucs = []
        for _ in range(n_bootstrap):
            idx = np.random.randint(0, len(y), size=len(y))
            y_boot = y[idx]
            s_boot = scores[idx]
            if len(np.unique(y_boot)) == 2:
                try:
                    aucs.append(roc_auc_score(y_boot, s_boot))
                except:
                    pass
        
        aucs = np.array(aucs)
        auc_point = roc_auc_score(y, scores)
        ci = (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5))
        se = np.std(aucs)
        
        auc_results[model] = {
            'auc': auc_point,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'se': se
        }
        
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        print(f"{model:<15} | {auc_point:>8.4f} | {ci_str:>20} | {se:>8.4f}")
    
    results['auc'] = auc_results
    
    # --- 2. Pairwise Comparisons vs CARIA-SR ---
    print("\n[2] Pairwise Comparisons vs CARIA-SR")
    print("-" * 60)
    
    caria_scores = df['CARIA-SR'].values
    pairwise_results = {}
    
    for model in model_cols:
        if model == 'CARIA-SR':
            continue
        
        other_scores = df[model].values
        
        # Bootstrap comparison
        comp = bootstrap_auc_comparison(
            y, caria_scores, other_scores, n_bootstrap,
            'CARIA-SR', model
        )
        
        # McNemar test
        mcn = mcnemar_test(y, caria_scores, other_scores)
        
        pairwise_results[model] = {
            'delta_auc': comp['delta'],
            'delta_ci': comp['delta_ci'],
            'p_value': comp['p_value'],
            'significant': comp['significant'],
            'mcnemar_p': mcn['p_value']
        }
        
        sig_str = "***" if comp['p_value'] < 0.01 else "**" if comp['p_value'] < 0.05 else "*" if comp['p_value'] < 0.10 else ""
        delta_ci_str = f"[{comp['delta_ci'][0]:.4f}, {comp['delta_ci'][1]:.4f}]"
        
        print(f"CARIA-SR vs {model:<12}: ΔAUC = {comp['delta']:+.4f} {delta_ci_str} (p={comp['p_value']:.4f}{sig_str})")
    
    results['pairwise'] = pairwise_results
    
    # --- 3. Permutation Test for Signal Authenticity ---
    print("\n[3] Permutation Test (Signal Authenticity)")
    print("-" * 60)
    
    perm = permutation_test_auc(y, caria_scores, n_permutations=1000)
    results['permutation'] = perm
    
    print(f"Observed AUC: {perm['observed_auc']:.4f}")
    print(f"Null distribution: mean={perm['null_mean']:.4f}, std={perm['null_std']:.4f}")
    print(f"P-value: {perm['p_value']:.4f}")
    if perm['significant']:
        print("✓ SIGNIFICANT: CARIA-SR captures a real signal, not noise")
    else:
        print("⚠ NOT SIGNIFICANT: Signal may be due to chance")
    
    # --- 4. Summary Rankings ---
    print("\n[4] Model Rankings")
    print("-" * 60)
    
    ranking = pd.DataFrame([
        {'Model': m, 'AUC': auc_results[m]['auc']}
        for m in model_cols
    ]).sort_values('AUC', ascending=False)
    
    for i, row in ranking.iterrows():
        rank = ranking.index.get_loc(i) + 1
        print(f"  {rank}. {row['Model']:<15}: AUC = {row['AUC']:.4f}")
    
    results['ranking'] = ranking
    
    return results, df


# ==============================================================================
# VISUALIZATIONS
# ==============================================================================

def plot_roc_comparison(df, output_path=None):
    """
    Plot ROC curves for all models.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y = df['Target'].values
    model_cols = ['CARIA-SR', 'HAR-RV', 'VIX', 'Rolling Vol', 'GARCH Proxy']
    
    colors = ['#3b82f6', '#10b981', '#f97316', '#6b7280', '#8b5cf6']
    linewidths = [2.5, 1.5, 1.5, 1.5, 1.5]
    
    for model, color, lw in zip(model_cols, colors, linewidths):
        scores = df[model].values
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)
        
        label = f'{model} (AUC={auc:.3f})'
        ax.plot(fpr, tpr, color=color, linewidth=lw, label=label)
    
    # Random baseline
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.500)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison: CARIA-SR vs Benchmarks', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.close()


def plot_auc_comparison_bars(results, output_path=None):
    """
    Bar chart comparing AUC across models.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    auc_data = results['auc']
    models = list(auc_data.keys())
    aucs = [auc_data[m]['auc'] for m in models]
    errors = [(auc_data[m]['auc'] - auc_data[m]['ci_lower'],
               auc_data[m]['ci_upper'] - auc_data[m]['auc']) for m in models]
    errors = np.array(errors).T
    
    colors = ['#3b82f6' if m == 'CARIA-SR' else '#6b7280' for m in models]
    
    bars = ax.barh(models, aucs, xerr=errors, color=colors, capsize=5, 
                   error_kw={'linewidth': 1.5})
    
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title('Model Comparison: AUC with 95% Confidence Intervals', fontsize=14)
    ax.set_xlim(0.45, 0.75)
    
    # Add value labels
    for bar, val in zip(bars, aucs):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.close()


def plot_delta_auc_forest(results, output_path=None):
    """
    Forest plot of AUC differences vs CARIA-SR.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    pairwise = results['pairwise']
    models = list(pairwise.keys())
    
    y_pos = np.arange(len(models))
    deltas = [pairwise[m]['delta_auc'] for m in models]
    ci_lower = [pairwise[m]['delta_ci'][0] for m in models]
    ci_upper = [pairwise[m]['delta_ci'][1] for m in models]
    
    # Error bars
    errors = [[d - l for d, l in zip(deltas, ci_lower)],
              [u - d for d, u in zip(deltas, ci_upper)]]
    
    colors = ['#10b981' if d > 0 else '#ef4444' for d in deltas]
    
    ax.errorbar(deltas, y_pos, xerr=errors, fmt='o', markersize=8,
                capsize=5, capthick=2, color='black', elinewidth=2)
    ax.scatter(deltas, y_pos, c=colors, s=100, zorder=5)
    
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'vs {m}' for m in models])
    ax.set_xlabel('ΔAUC (CARIA-SR - Benchmark)', fontsize=12)
    ax.set_title('CARIA-SR Advantage Over Benchmarks', fontsize=14)
    
    # Add significance markers
    for i, m in enumerate(models):
        p = pairwise[m]['p_value']
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
        ax.text(deltas[i] + 0.01, y_pos[i], sig, va='center', fontsize=12)
    
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.close()


def create_comparison_table(results, output_path):
    """
    Create publication-ready comparison table.
    """
    auc_data = results['auc']
    pairwise = results['pairwise']
    
    rows = []
    
    # CARIA-SR first
    rows.append({
        'Model': 'CARIA-SR',
        'AUC': f"{auc_data['CARIA-SR']['auc']:.4f}",
        '95% CI': f"[{auc_data['CARIA-SR']['ci_lower']:.4f}, {auc_data['CARIA-SR']['ci_upper']:.4f}]",
        'ΔAUC': '-',
        'p-value': '-'
    })
    
    # Other models
    for model in ['HAR-RV', 'VIX', 'Rolling Vol', 'GARCH Proxy']:
        delta = pairwise[model]['delta_auc']
        p = pairwise[model]['p_value']
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
        
        rows.append({
            'Model': model,
            'AUC': f"{auc_data[model]['auc']:.4f}",
            '95% CI': f"[{auc_data[model]['ci_lower']:.4f}, {auc_data[model]['ci_upper']:.4f}]",
            'ΔAUC': f"{delta:+.4f}",
            'p-value': f"{p:.4f}{sig}"
        })
    
    table = pd.DataFrame(rows)
    table.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")
    
    return table


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_benchmark_comparison():
    """
    Run complete benchmark comparison analysis.
    """
    print("=" * 80)
    print("CARIA-SR BENCHMARK COMPARISON")
    print("=" * 80)
    
    # Load data
    ret_spy, vol_credit, vix = load_data_for_comparison()
    
    # Compute models
    df = compute_all_models(ret_spy, vol_credit, vix)
    
    # Run comparison
    results, df = run_full_comparison(df, n_bootstrap=1000)
    
    # Generate outputs
    print("\n" + "=" * 80)
    print("Generating Outputs")
    print("=" * 80)
    
    # ROC curves
    plot_roc_comparison(df, os.path.join(OUTPUT_DIR, 'Benchmark_ROC_Curves.png'))
    
    # AUC bars
    plot_auc_comparison_bars(results, os.path.join(OUTPUT_DIR, 'Benchmark_AUC_Comparison.png'))
    
    # Forest plot
    plot_delta_auc_forest(results, os.path.join(OUTPUT_DIR, 'Benchmark_Delta_AUC_Forest.png'))
    
    # Table
    table = create_comparison_table(results, os.path.join(OUTPUT_DIR, 'Table_Benchmark_Comparison.csv'))
    
    # Print final summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    caria_auc = results['auc']['CARIA-SR']['auc']
    print(f"\nCARIA-SR AUC: {caria_auc:.4f}")
    
    wins = sum(1 for m in results['pairwise'] 
               if results['pairwise'][m]['delta_auc'] > 0)
    total = len(results['pairwise'])
    print(f"CARIA-SR outperforms: {wins}/{total} benchmarks")
    
    sig_wins = sum(1 for m in results['pairwise'] 
                   if results['pairwise'][m]['significant'] and 
                   results['pairwise'][m]['delta_auc'] > 0)
    print(f"Statistically significant wins: {sig_wins}/{total}")
    
    if results['permutation']['significant']:
        print(f"\n✓ CARIA-SR signal is statistically authentic (p={results['permutation']['p_value']:.4f})")
    
    return results, df


if __name__ == "__main__":
    results, df = run_benchmark_comparison()
    print("\n✓ Benchmark comparison complete!")


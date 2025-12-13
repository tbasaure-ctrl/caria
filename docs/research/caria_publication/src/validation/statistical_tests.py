"""
Statistical Tests for Model Comparison and Validation
======================================================

This module implements statistical tests required for academic publication
in quantitative finance journals.

Tests Implemented:
------------------
1. Diebold-Mariano Test: Compare predictive accuracy between models
2. McNemar's Test: Compare classification error rates
3. Bootstrap Confidence Intervals: CI for MCC, AUC, Precision, Recall
4. Model Confidence Set (MCS): Hansen's approach for multiple comparisons

Metrics Calculated:
-------------------
- Matthews Correlation Coefficient (MCC)
- Area Under ROC Curve (AUC)
- Precision, Recall, F1
- Brier Score (probabilistic forecasts)

References:
-----------
[1] Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy"
[2] McNemar, Q. (1947). "Note on the sampling error of the difference between
    correlated proportions or percentages"
[3] Hansen, P.R. et al. (2011). "The Model Confidence Set"

Author: Tomás Basaure
Date: December 2025
License: MIT
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Tuple, Optional, Dict, List, Callable
from dataclasses import dataclass
import warnings
import statsmodels.api as sm


# ==============================================================================
# CONFIGURATION
# ==============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


@dataclass
class TestResult:
    """Container for statistical test results."""
    statistic: float
    p_value: float
    significant: bool
    confidence_level: float
    interpretation: str
    details: Dict
    
    def __repr__(self):
        sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else ""
        return f"TestResult(stat={self.statistic:.4f}, p={self.p_value:.4f}{sig})"


@dataclass 
class MetricsResult:
    """Container for classification metrics with confidence intervals."""
    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int
    
    def __repr__(self):
        return f"{self.metric_name}: {self.point_estimate:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"


# ==============================================================================
# CLASSIFICATION METRICS
# ==============================================================================

def matthews_correlation_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Calculate Matthews Correlation Coefficient.
    
    MCC is considered the best metric for binary classification with
    imbalanced classes (like crisis prediction).
    
    Formula:
    ========
    MCC = (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    
    Range: [-1, 1]
    - 1: Perfect prediction
    - 0: Random prediction
    - -1: Total disagreement
    
    Parameters:
    ===========
    y_true : array
        True binary labels
    y_pred : array
        Predicted binary labels
        
    Returns:
    ========
    float : MCC value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate Precision, Recall, and F1 Score.
    
    Parameters:
    ===========
    y_true : array
        True labels
    y_pred : array
        Predicted labels
        
    Returns:
    ========
    Tuple[float, float, float] : (precision, recall, f1)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, int]:
    """
    Calculate confusion matrix components.
    
    Returns:
    ========
    Dict with keys: 'tp', 'tn', 'fp', 'fn'
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    return {
        'tp': int(np.sum((y_true == 1) & (y_pred == 1))),
        'tn': int(np.sum((y_true == 0) & (y_pred == 0))),
        'fp': int(np.sum((y_true == 0) & (y_pred == 1))),
        'fn': int(np.sum((y_true == 1) & (y_pred == 0)))
    }


def brier_score(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> float:
    """
    Calculate Brier Score for probabilistic forecasts.
    
    Formula:
    ========
    BS = (1/N) Σ (p_i - y_i)²
    
    Lower is better. Range: [0, 1]
    """
    y_true = np.asarray(y_true).flatten()
    y_proba = np.asarray(y_proba).flatten()
    
    return np.mean((y_proba - y_true) ** 2)


def calculate_auc(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> float:
    """
    Calculate Area Under ROC Curve.
    
    Uses trapezoidal rule for numerical integration.
    """
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()
    
    # Sort by score
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_score_sorted = y_score[desc_score_indices]
    
    # Calculate TPR and FPR at each threshold
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    tpr = np.cumsum(y_true_sorted == 1) / n_pos
    fpr = np.cumsum(y_true_sorted == 0) / n_neg
    
    # Add (0, 0) point
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    
    # Trapezoidal integration
    auc = np.trapz(tpr, fpr)
    
    return auc


# ==============================================================================
# DIEBOLD-MARIANO TEST
# ==============================================================================

def diebold_mariano_test(
    actual: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    loss_func: str = 'squared',
    h: int = 1,
    alternative: str = 'two-sided'
) -> TestResult:
    """
    Diebold-Mariano test for comparing predictive accuracy.
    
    Tests H₀: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t)
    
    Parameters:
    ===========
    actual : array
        True values
    pred1 : array
        Predictions from model 1
    pred2 : array
        Predictions from model 2
    loss_func : str
        'squared' for MSE, 'absolute' for MAE
    h : int
        Forecast horizon (for HAC variance estimation)
    alternative : str
        'two-sided', 'less', or 'greater'
        
    Returns:
    ========
    TestResult : Test statistic, p-value, and interpretation
    
    Reference:
    ----------
    Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy"
    Journal of Business & Economic Statistics, 13(3), 253-263.
    """
    actual = np.asarray(actual).flatten()
    pred1 = np.asarray(pred1).flatten()
    pred2 = np.asarray(pred2).flatten()
    
    # Calculate errors
    e1 = actual - pred1
    e2 = actual - pred2
    
    # Loss differential
    if loss_func == 'squared':
        d = e1 ** 2 - e2 ** 2
    elif loss_func == 'absolute':
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss function: {loss_func}")
    
    # Mean loss differential
    d_bar = np.mean(d)
    n = len(d)
    
    # HAC variance estimator (Newey-West with h-1 lags)
    gamma_0 = np.var(d, ddof=1)
    
    if h > 1:
        # Add autocovariances
        for k in range(1, h):
            gamma_k = np.cov(d[k:], d[:-k])[0, 1]
            gamma_0 += 2 * (1 - k / h) * gamma_k
    
    # DM statistic
    var_d_bar = gamma_0 / n
    dm_stat = d_bar / np.sqrt(var_d_bar) if var_d_bar > 0 else 0
    
    # P-value
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    elif alternative == 'less':
        p_value = stats.norm.cdf(dm_stat)
    else:  # greater
        p_value = 1 - stats.norm.cdf(dm_stat)
    
    # Interpretation
    if p_value < 0.05:
        if d_bar < 0:
            interp = "Model 1 significantly outperforms Model 2"
        else:
            interp = "Model 2 significantly outperforms Model 1"
    else:
        interp = "No significant difference between models"
    
    return TestResult(
        statistic=dm_stat,
        p_value=p_value,
        significant=p_value < 0.05,
        confidence_level=0.95,
        interpretation=interp,
        details={
            'mean_loss_diff': d_bar,
            'variance': var_d_bar,
            'n_observations': n,
            'loss_function': loss_func
        }
    )

def ols_newey_west(
    y: Union[pd.Series, np.ndarray],
    x: Union[pd.Series, np.ndarray, pd.DataFrame],
    lags: int,
    add_const: bool = True,
) -> Dict[str, float]:
    """
    OLS regression with HAC (Newey–West) standard errors.

    This is the minimum-correct inference when the dependent variable is built
    from *overlapping* forward returns (e.g., 22d forward returns on daily data),
    which induces mechanical autocorrelation in residuals.

    Parameters
    ----------
    y:
        Dependent variable.
    x:
        Regressors. If Series/array, treated as a single regressor.
    lags:
        HAC maxlags. For h-day overlapping returns on daily data, a standard
        choice is maxlags = h-1.
    add_const:
        If True, include an intercept.

    Returns
    -------
    Dict with:
      - beta (coefficient on the first regressor)
      - t (HAC t-stat for beta)
      - p (two-sided HAC p-value for beta)
      - n (observations used)
    """
    y_ser = pd.Series(y).astype(float)
    if isinstance(x, pd.DataFrame):
        X = x.copy()
    else:
        X = pd.DataFrame({"x": pd.Series(x)})

    df = pd.concat([y_ser.rename("y"), X], axis=1).dropna()
    if len(df) < 30:
        return {"beta": np.nan, "t": np.nan, "p": np.nan, "n": float(len(df))}

    yv = df["y"].values
    Xv = df.drop(columns=["y"]).values
    if add_const:
        Xv = sm.add_constant(Xv, has_constant="add")

    model = sm.OLS(yv, Xv).fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)})
    # coefficient on first regressor (after constant if present)
    beta_idx = 1 if add_const else 0
    beta = float(model.params[beta_idx])
    t = float(model.tvalues[beta_idx])
    p = float(model.pvalues[beta_idx])
    return {"beta": beta, "t": t, "p": p, "n": float(len(df))}


def holm_bonferroni(p_values: Union[pd.Series, List[float], np.ndarray]) -> np.ndarray:
    """
    Holm–Bonferroni adjustment for multiple hypothesis testing.

    Use this when scanning many windows/thresholds and reporting best results.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    prev = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        val = min(1.0, p[idx] * factor)
        # enforce monotonicity
        val = max(val, prev)
        adj[idx] = val
        prev = val
    return adj


# ==============================================================================
# McNEMAR'S TEST
# ==============================================================================

def mcnemar_test(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    exact: bool = True
) -> TestResult:
    """
    McNemar's test for comparing two classifiers.
    
    Tests whether the two classifiers have the same error rate.
    
    Contingency table:
                   Model 2 Correct  Model 2 Wrong
    Model 1 Correct      n00            n01
    Model 1 Wrong        n10            n11
    
    Test statistic (exact): Based on binomial distribution
    Test statistic (approx): χ² = (n01 - n10)² / (n01 + n10)
    
    Parameters:
    ===========
    y_true : array
        True labels
    pred1, pred2 : arrays
        Predictions from each model
    exact : bool
        Use exact binomial test (recommended for small samples)
        
    Returns:
    ========
    TestResult : Test results and interpretation
    
    Reference:
    ----------
    McNemar, Q. (1947). "Note on the sampling error of the difference between
    correlated proportions or percentages"
    """
    y_true = np.asarray(y_true).flatten()
    pred1 = np.asarray(pred1).flatten()
    pred2 = np.asarray(pred2).flatten()
    
    # Correct/incorrect classifications
    correct1 = (pred1 == y_true)
    correct2 = (pred2 == y_true)
    
    # Contingency table
    n00 = np.sum(correct1 & correct2)      # Both correct
    n01 = np.sum(correct1 & ~correct2)     # Model 1 correct, Model 2 wrong
    n10 = np.sum(~correct1 & correct2)     # Model 1 wrong, Model 2 correct
    n11 = np.sum(~correct1 & ~correct2)    # Both wrong
    
    # Discordant pairs
    b = n01  # Model 1 better
    c = n10  # Model 2 better
    
    if exact:
        # Exact binomial test
        n = b + c
        if n == 0:
            p_value = 1.0
            statistic = 0.0
        else:
            # Two-sided p-value
            k = min(b, c)
            p_value = 2 * stats.binom.cdf(k, n, 0.5)
            p_value = min(p_value, 1.0)
            statistic = (b - c) / np.sqrt(n) if n > 0 else 0
    else:
        # Chi-squared approximation
        if b + c == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            # With continuity correction
            statistic = (np.abs(b - c) - 1) ** 2 / (b + c)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    # Interpretation
    if p_value < 0.05:
        if b > c:
            interp = "Model 1 significantly more accurate"
        else:
            interp = "Model 2 significantly more accurate"
    else:
        interp = "No significant difference in accuracy"
    
    return TestResult(
        statistic=statistic,
        p_value=p_value,
        significant=p_value < 0.05,
        confidence_level=0.95,
        interpretation=interp,
        details={
            'n_both_correct': int(n00),
            'n_model1_better': int(b),
            'n_model2_better': int(c),
            'n_both_wrong': int(n11),
            'method': 'exact' if exact else 'chi-squared'
        }
    )


# ==============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ==============================================================================

def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = RANDOM_SEED
) -> MetricsResult:
    """
    Calculate bootstrap confidence interval for any metric.
    
    Parameters:
    ===========
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    metric_func : callable
        Function that takes (y_true, y_pred) and returns metric value
    n_bootstrap : int
        Number of bootstrap iterations
    confidence : float
        Confidence level (default: 0.95 for 95% CI)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    ========
    MetricsResult : Point estimate and confidence interval
    """
    np.random.seed(random_state)
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = len(y_true)
    
    # Point estimate
    point_estimate = metric_func(y_true, y_pred)
    
    # Bootstrap
    bootstrap_values = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.randint(0, n, size=n)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        try:
            value = metric_func(y_true_boot, y_pred_boot)
            bootstrap_values.append(value)
        except Exception:
            pass
    
    bootstrap_values = np.array(bootstrap_values)
    
    # Confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_values, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
    
    return MetricsResult(
        metric_name=metric_func.__name__,
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence,
        n_bootstrap=n_bootstrap
    )


def bootstrap_mcc_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> MetricsResult:
    """
    Bootstrap confidence interval for MCC.
    
    This is the primary metric for crisis prediction validation.
    
    Parameters:
    ===========
    y_true, y_pred : arrays
        True and predicted labels
    n_bootstrap : int
        Bootstrap iterations
    confidence : float
        Confidence level
        
    Returns:
    ========
    MetricsResult : MCC with 95% CI
    """
    return bootstrap_metric_ci(
        y_true, y_pred,
        matthews_correlation_coefficient,
        n_bootstrap,
        confidence
    )


def calculate_all_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000
) -> Dict[str, MetricsResult]:
    """
    Calculate all metrics with bootstrap confidence intervals.
    
    Parameters:
    ===========
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    y_proba : array, optional
        Prediction probabilities (for AUC, Brier)
    n_bootstrap : int
        Bootstrap iterations
        
    Returns:
    ========
    Dict[str, MetricsResult] : All metrics with CIs
    """
    results = {}
    
    # MCC
    results['mcc'] = bootstrap_mcc_ci(y_true, y_pred, n_bootstrap)
    
    # Precision
    results['precision'] = bootstrap_metric_ci(
        y_true, y_pred,
        lambda yt, yp: precision_recall_f1(yt, yp)[0],
        n_bootstrap
    )
    results['precision'].metric_name = 'precision'
    
    # Recall
    results['recall'] = bootstrap_metric_ci(
        y_true, y_pred,
        lambda yt, yp: precision_recall_f1(yt, yp)[1],
        n_bootstrap
    )
    results['recall'].metric_name = 'recall'
    
    # F1
    results['f1'] = bootstrap_metric_ci(
        y_true, y_pred,
        lambda yt, yp: precision_recall_f1(yt, yp)[2],
        n_bootstrap
    )
    results['f1'].metric_name = 'f1'
    
    # AUC (if probabilities available)
    if y_proba is not None:
        results['auc'] = bootstrap_metric_ci(
            y_true, y_proba,
            calculate_auc,
            n_bootstrap
        )
        results['auc'].metric_name = 'auc'
        
        results['brier'] = bootstrap_metric_ci(
            y_true, y_proba,
            brier_score,
            n_bootstrap
        )
        results['brier'].metric_name = 'brier_score'
    
    return results


# ==============================================================================
# COMPREHENSIVE COMPARISON REPORT
# ==============================================================================

def compare_models_statistically(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    reference_model: str = None,
    n_bootstrap: int = 1000
) -> pd.DataFrame:
    """
    Comprehensive statistical comparison of multiple models.
    
    Parameters:
    ===========
    y_true : array
        True labels
    predictions : dict
        Dictionary mapping model name to predictions
    reference_model : str, optional
        Name of reference model for pairwise comparisons
    n_bootstrap : int
        Bootstrap iterations
        
    Returns:
    ========
    pd.DataFrame : Comparison results with p-values
    """
    results = []
    model_names = list(predictions.keys())
    
    for name in model_names:
        pred = predictions[name]
        
        # Calculate metrics
        mcc = matthews_correlation_coefficient(y_true, pred)
        prec, rec, f1 = precision_recall_f1(y_true, pred)
        
        # Bootstrap CI for MCC
        mcc_ci = bootstrap_mcc_ci(y_true, pred, n_bootstrap)
        
        results.append({
            'model': name,
            'mcc': mcc,
            'mcc_ci_lower': mcc_ci.ci_lower,
            'mcc_ci_upper': mcc_ci.ci_upper,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
    
    df = pd.DataFrame(results)
    
    # Add pairwise comparisons if reference model specified
    if reference_model and reference_model in predictions:
        ref_pred = predictions[reference_model]
        
        mcnemar_pvalues = []
        for name in model_names:
            if name == reference_model:
                mcnemar_pvalues.append(np.nan)
            else:
                test = mcnemar_test(y_true, ref_pred, predictions[name])
                mcnemar_pvalues.append(test.p_value)
        
        df['mcnemar_pvalue_vs_' + reference_model] = mcnemar_pvalues
    
    return df.sort_values('mcc', ascending=False)


def generate_statistical_report(
    comparison_df: pd.DataFrame,
    reference_model: str = None
) -> str:
    """
    Generate markdown report for statistical comparison.
    
    Parameters:
    ===========
    comparison_df : pd.DataFrame
        Output from compare_models_statistically
    reference_model : str
        Name of reference model
        
    Returns:
    ========
    str : Markdown formatted report
    """
    report = []
    report.append("# Statistical Comparison Report")
    report.append("")
    report.append("## Model Performance Metrics")
    report.append("")
    report.append("| Model | MCC | 95% CI | Precision | Recall | F1 |")
    report.append("|-------|-----|--------|-----------|--------|-----|")
    
    for _, row in comparison_df.iterrows():
        report.append(
            f"| {row['model']} | {row['mcc']:.4f} | "
            f"[{row['mcc_ci_lower']:.4f}, {row['mcc_ci_upper']:.4f}] | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )
    
    if reference_model:
        report.append("")
        report.append(f"## Pairwise Comparisons vs {reference_model}")
        report.append("")
        report.append("| Model | McNemar p-value | Significant? |")
        report.append("|-------|-----------------|--------------|")
        
        pval_col = f'mcnemar_pvalue_vs_{reference_model}'
        if pval_col in comparison_df.columns:
            for _, row in comparison_df.iterrows():
                if row['model'] != reference_model:
                    pval = row[pval_col]
                    sig = "Yes***" if pval < 0.001 else "Yes**" if pval < 0.01 else "Yes*" if pval < 0.05 else "No"
                    report.append(f"| {row['model']} | {pval:.4f} | {sig} |")
    
    report.append("")
    report.append("*p<0.05, **p<0.01, ***p<0.001")
    
    return "\n".join(report)


# ==============================================================================
# TESTS
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("STATISTICAL TESTS MODULE VALIDATION")
    print("=" * 60)
    
    np.random.seed(RANDOM_SEED)
    n = 500
    
    # Generate test data
    y_true = (np.random.rand(n) > 0.85).astype(int)  # ~15% positive
    
    # Model 1: Good predictions
    noise1 = np.random.rand(n) * 0.2
    y_pred1 = ((y_true + noise1) > 0.5).astype(int)
    
    # Model 2: Worse predictions
    noise2 = np.random.rand(n) * 0.4
    y_pred2 = ((y_true + noise2) > 0.5).astype(int)
    
    # Model 3: Random
    y_pred3 = (np.random.rand(n) > 0.85).astype(int)
    
    print("\n1. Classification Metrics Test:")
    print("-" * 40)
    
    for name, pred in [("Good Model", y_pred1), ("Worse Model", y_pred2), ("Random", y_pred3)]:
        mcc = matthews_correlation_coefficient(y_true, pred)
        prec, rec, f1 = precision_recall_f1(y_true, pred)
        print(f"   {name}:")
        print(f"      MCC={mcc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    
    print("\n2. McNemar's Test:")
    print("-" * 40)
    
    test = mcnemar_test(y_true, y_pred1, y_pred2)
    print(f"   Good vs Worse: stat={test.statistic:.4f}, p={test.p_value:.4f}")
    print(f"   {test.interpretation}")
    
    test = mcnemar_test(y_true, y_pred1, y_pred3)
    print(f"   Good vs Random: stat={test.statistic:.4f}, p={test.p_value:.4f}")
    print(f"   {test.interpretation}")
    
    print("\n3. Bootstrap CI for MCC:")
    print("-" * 40)
    
    mcc_result = bootstrap_mcc_ci(y_true, y_pred1, n_bootstrap=500)
    print(f"   Good Model: {mcc_result}")
    
    mcc_result = bootstrap_mcc_ci(y_true, y_pred3, n_bootstrap=500)
    print(f"   Random: {mcc_result}")
    
    print("\n4. Comprehensive Comparison:")
    print("-" * 40)
    
    predictions = {
        'Good': y_pred1,
        'Worse': y_pred2,
        'Random': y_pred3
    }
    
    comparison = compare_models_statistically(y_true, predictions, 'Good', n_bootstrap=500)
    print(comparison.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Statistical tests completed!")

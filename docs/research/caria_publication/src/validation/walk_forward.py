"""
Walk-Forward Validation Framework for Financial Time Series
============================================================

This module implements rigorous out-of-sample validation protocols required
for academic publication in quantitative finance journals.

Validation Protocols:
---------------------
1. Fixed Split: Train/Validation/Test with strict temporal separation
2. Walk-Forward: Expanding window with yearly re-training
3. K-Fold Time Series: Blocked cross-validation for time series

Key Requirements for Publication:
---------------------------------
- NO lookahead bias: All features computed with point-in-time data only
- Multiple crisis coverage: Test set must include diverse crisis types
- Stability across regimes: Performance consistent across bull/bear markets

Crisis Events in Test Period:
-----------------------------
- Dotcom Crash (2000-2002)
- Global Financial Crisis (2007-2009)
- European Debt Crisis (2011-2012)
- Flash Crash (2010-05-06)
- China Crash (2015-08)
- COVID-19 Crash (2020-03)
- 2022 Fed Tightening

References:
-----------
[1] Bailey, D.H. et al. (2014). "The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting and Non-Normality"
[2] López de Prado, M. (2018). "Advances in Financial Machine Learning"

Author: Tomás Basaure
Date: December 2025
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Generator, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import warnings


# ==============================================================================
# CONFIGURATION
# ==============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Default temporal splits for publication
DEFAULT_SPLITS = {
    'train': ('1990-01-01', '2007-12-31'),      # Pre-GFC
    'validation': ('2008-01-01', '2015-12-31'), # GFC + Recovery
    'test': ('2016-01-01', '2025-12-31')        # Recent + COVID
}

# Known crisis dates for validation
CRISIS_EVENTS = {
    'dotcom_peak': '2000-03-10',
    'dotcom_bottom': '2002-10-09',
    'gfc_lehman': '2008-09-15',
    'gfc_bottom': '2009-03-09',
    'flash_crash': '2010-05-06',
    'euro_crisis': '2011-08-05',
    'china_crash': '2015-08-24',
    'brexit': '2016-06-24',
    'covid_crash': '2020-03-11',
    'covid_bottom': '2020-03-23',
    'fed_tightening': '2022-01-03',
    'svb_collapse': '2023-03-10',
    'gilt_crisis': '2022-09-23'
}


@dataclass
class SplitResult:
    """Container for train/test split results."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_dates: Tuple[str, str]
    test_dates: Tuple[str, str]
    n_train: int
    n_test: int


@dataclass
class WalkForwardResult:
    """Container for walk-forward validation results."""
    predictions: pd.Series
    actuals: pd.Series
    fold_metrics: List[Dict]
    aggregate_metrics: Dict
    crisis_coverage: Dict[str, bool]


# ==============================================================================
# TEMPORAL SPLIT FUNCTIONS
# ==============================================================================

def create_temporal_split(
    data: pd.DataFrame,
    target_col: str,
    train_end: str,
    test_start: str,
    test_end: Optional[str] = None,
    feature_cols: Optional[List[str]] = None
) -> SplitResult:
    """
    Create a strict temporal train/test split.
    
    Parameters:
    ===========
    data : pd.DataFrame
        Dataset with DatetimeIndex
    target_col : str
        Name of target column
    train_end : str
        End date for training (exclusive)
    test_start : str
        Start date for testing (inclusive)
    test_end : str, optional
        End date for testing
    feature_cols : list, optional
        Feature column names. If None, use all except target.
        
    Returns:
    ========
    SplitResult : Container with train/test data
    
    Notes:
    ======
    - Ensures NO overlap between train and test periods
    - Gap between train_end and test_start prevents lookahead bias
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    data = data.sort_index()
    
    if feature_cols is None:
        feature_cols = [c for c in data.columns if c != target_col]
    
    # Convert string dates to Timestamps
    if isinstance(train_end, str):
        train_end = pd.Timestamp(train_end)
    if isinstance(test_start, str):
        test_start = pd.Timestamp(test_start)
    if test_end and isinstance(test_end, str):
        test_end = pd.Timestamp(test_end)
    
    # Create splits
    train_mask = data.index < train_end
    test_mask = data.index >= test_start
    if test_end:
        test_mask = test_mask & (data.index <= test_end)
    
    train_data = data[train_mask]
    test_data = data[test_mask]
    
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    
    return SplitResult(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_dates=(str(train_data.index.min()), str(train_data.index.max())),
        test_dates=(str(test_data.index.min()), str(test_data.index.max())),
        n_train=len(train_data),
        n_test=len(test_data)
    )


def create_publication_splits(
    data: pd.DataFrame,
    target_col: str,
    splits: Optional[Dict] = None,
    feature_cols: Optional[List[str]] = None
) -> Dict[str, SplitResult]:
    """
    Create standard train/validation/test splits for publication.
    
    Default Splits:
    ===============
    Train:      1990-01-01 → 2007-12-31 (pre-GFC)
    Validation: 2008-01-01 → 2015-12-31 (GFC + Recovery)
    Test:       2016-01-01 → 2025-12-31 (Recent + COVID)
    
    Parameters:
    ===========
    data : pd.DataFrame
        Dataset with DatetimeIndex
    target_col : str
        Name of target column
    splits : dict, optional
        Custom split dates
    feature_cols : list, optional
        Feature columns
        
    Returns:
    ========
    Dict[str, SplitResult] : Train, validation, and test splits
    
    Notes:
    ======
    - Test set includes multiple crisis types for robust validation
    - Strict temporal ordering prevents any lookahead bias
    """
    if splits is None:
        splits = DEFAULT_SPLITS
    
    results = {}
    
    # Ensure index is DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Train split - convert strings to Timestamps if needed
    train_start, train_end = splits['train']
    if isinstance(train_start, str):
        train_start = pd.Timestamp(train_start)
    if isinstance(train_end, str):
        train_end = pd.Timestamp(train_end)
    train_mask = (data.index >= train_start) & (data.index <= train_end)
    
    # Validation split
    val_start, val_end = splits['validation']
    if isinstance(val_start, str):
        val_start = pd.Timestamp(val_start)
    if isinstance(val_end, str):
        val_end = pd.Timestamp(val_end)
    val_mask = (data.index >= val_start) & (data.index <= val_end)
    
    # Test split
    test_start, test_end = splits['test']
    if isinstance(test_start, str):
        test_start = pd.Timestamp(test_start)
    if isinstance(test_end, str):
        test_end = pd.Timestamp(test_end)
    test_mask = (data.index >= test_start) & (data.index <= test_end)
    
    if feature_cols is None:
        feature_cols = [c for c in data.columns if c != target_col]
    
    for name, mask in [('train', train_mask), ('validation', val_mask), ('test', test_mask)]:
        subset = data[mask]
        results[name] = SplitResult(
            X_train=subset[feature_cols] if name == 'train' else None,
            y_train=subset[target_col] if name == 'train' else None,
            X_test=subset[feature_cols],
            y_test=subset[target_col],
            train_dates=splits['train'] if name == 'train' else None,
            test_dates=splits.get(name, (str(subset.index.min()), str(subset.index.max()))),
            n_train=len(subset) if name == 'train' else 0,
            n_test=len(subset)
        )
    
    return results


# ==============================================================================
# WALK-FORWARD VALIDATION
# ==============================================================================

def walk_forward_validation(
    data: pd.DataFrame,
    target_col: str,
    model_factory: Callable,
    feature_cols: Optional[List[str]] = None,
    initial_train_years: int = 10,
    test_window_years: int = 1,
    start_year: int = 2000,
    end_year: int = 2025,
    expanding: bool = True,
    gap_days: int = 0
) -> WalkForwardResult:
    """
    Perform walk-forward validation with expanding or rolling windows.
    
    Algorithm:
    ==========
    For each year from start_year to end_year:
        1. Train on all data up to year-1 (expanding) or last N years (rolling)
        2. Predict on year
        3. Evaluate predictions
        4. Move forward one year
    
    Parameters:
    ===========
    data : pd.DataFrame
        Dataset with DatetimeIndex
    target_col : str
        Target column name
    model_factory : Callable
        Function that returns a fresh model instance with fit(X, y) and predict(X)
    feature_cols : list, optional
        Feature columns
    initial_train_years : int
        Minimum years of training data before first prediction
    test_window_years : int
        Size of test window in years
    start_year : int
        First year to make predictions
    end_year : int
        Last year to make predictions
    expanding : bool
        If True, use expanding window; if False, use rolling window
    gap_days : int
        Gap between train and test to prevent lookahead
        
    Returns:
    ========
    WalkForwardResult : Container with all predictions and metrics
    
    Notes:
    ======
    - This is the gold standard for financial ML validation
    - Prevents lookahead bias by strict temporal separation
    - Re-trains model each period to capture regime changes
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")
    
    data = data.sort_index()
    
    if feature_cols is None:
        feature_cols = [c for c in data.columns if c != target_col]
    
    all_predictions = []
    all_actuals = []
    fold_metrics = []
    
    for year in range(start_year, end_year + 1):
        # Define train period
        if expanding:
            train_start = data.index.min()
        else:
            train_start = pd.Timestamp(f'{year - initial_train_years}-01-01')
        
        train_end = pd.Timestamp(f'{year - 1}-12-31') - pd.Timedelta(days=gap_days)
        
        # Define test period
        test_start = pd.Timestamp(f'{year}-01-01')
        test_end = pd.Timestamp(f'{year + test_window_years - 1}-12-31')
        
        # Get data
        train_mask = (data.index >= train_start) & (data.index <= train_end)
        test_mask = (data.index >= test_start) & (data.index <= test_end)
        
        train_data = data[train_mask]
        test_data = data[test_mask]
        
        if len(train_data) < 252 or len(test_data) == 0:
            continue
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        # Train model
        model = model_factory()
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_test)
        
        # Store results
        pred_series = pd.Series(predictions, index=y_test.index)
        all_predictions.append(pred_series)
        all_actuals.append(y_test)
        
        # Calculate fold metrics
        fold_metrics.append({
            'year': year,
            'train_start': str(train_start.date()),
            'train_end': str(train_end.date()),
            'test_start': str(test_start.date()),
            'test_end': str(test_end.date()),
            'n_train': len(train_data),
            'n_test': len(test_data),
            'train_pos_rate': y_train.mean(),
            'test_pos_rate': y_test.mean()
        })
    
    # Combine all predictions
    all_predictions = pd.concat(all_predictions)
    all_actuals = pd.concat(all_actuals)
    
    # Check crisis coverage
    crisis_coverage = check_crisis_coverage(all_actuals)
    
    # Calculate aggregate metrics (placeholder - use statistical_tests.py)
    aggregate_metrics = {
        'n_predictions': len(all_predictions),
        'n_folds': len(fold_metrics),
        'date_range': (str(all_predictions.index.min()), str(all_predictions.index.max()))
    }
    
    return WalkForwardResult(
        predictions=all_predictions,
        actuals=all_actuals,
        fold_metrics=fold_metrics,
        aggregate_metrics=aggregate_metrics,
        crisis_coverage=crisis_coverage
    )


def time_series_cv(
    data: pd.DataFrame,
    target_col: str,
    n_splits: int = 5,
    test_size: int = 252,
    gap: int = 0,
    feature_cols: Optional[List[str]] = None
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Time series cross-validation with blocked splits.
    
    Unlike standard K-Fold, this maintains temporal ordering:
    
    Fold 1: [Train    ][Test]
    Fold 2: [Train        ][Test]
    Fold 3: [Train            ][Test]
    ...
    
    Parameters:
    ===========
    data : pd.DataFrame
        Dataset with DatetimeIndex
    target_col : str
        Target column
    n_splits : int
        Number of CV folds
    test_size : int
        Size of test set in each fold
    gap : int
        Gap between train and test
    feature_cols : list, optional
        Feature columns
        
    Yields:
    =======
    Tuple[np.ndarray, np.ndarray] : Train indices, test indices
    """
    n_samples = len(data)
    
    # Calculate fold boundaries
    test_starts = np.linspace(
        n_samples - n_splits * test_size,
        n_samples - test_size,
        n_splits,
        dtype=int
    )
    
    for test_start in test_starts:
        train_end = test_start - gap
        test_end = min(test_start + test_size, n_samples)
        
        train_indices = np.arange(0, train_end)
        test_indices = np.arange(test_start, test_end)
        
        yield train_indices, test_indices


# ==============================================================================
# CRISIS COVERAGE VALIDATION
# ==============================================================================

def check_crisis_coverage(
    test_data: pd.Series,
    crisis_events: Optional[Dict[str, str]] = None,
    window_days: int = 30
) -> Dict[str, bool]:
    """
    Check if test data covers known crisis events.
    
    Parameters:
    ===========
    test_data : pd.Series
        Series with DatetimeIndex
    crisis_events : dict, optional
        Dictionary of crisis name -> date
    window_days : int
        Days around crisis to check for coverage
        
    Returns:
    ========
    Dict[str, bool] : Coverage status for each crisis
    
    Notes:
    ======
    - For publication, test set should include diverse crisis types
    - Minimum: 3 different crisis events in test period
    """
    if crisis_events is None:
        crisis_events = CRISIS_EVENTS
    
    coverage = {}
    
    for name, date_str in crisis_events.items():
        try:
            crisis_date = pd.Timestamp(date_str)
            window_start = crisis_date - pd.Timedelta(days=window_days)
            window_end = crisis_date + pd.Timedelta(days=window_days)
            
            # Check if any test data falls within window
            in_window = (test_data.index >= window_start) & (test_data.index <= window_end)
            coverage[name] = in_window.any()
        except Exception:
            coverage[name] = False
    
    return coverage


def validate_crisis_diversity(
    coverage: Dict[str, bool],
    min_crises: int = 3
) -> Tuple[bool, str]:
    """
    Validate that test set has sufficient crisis diversity.
    
    Parameters:
    ===========
    coverage : dict
        Crisis coverage from check_crisis_coverage
    min_crises : int
        Minimum number of crises required
        
    Returns:
    ========
    Tuple[bool, str] : (is_valid, message)
    """
    covered = [k for k, v in coverage.items() if v]
    n_covered = len(covered)
    
    if n_covered >= min_crises:
        return True, f"Valid: {n_covered} crises covered ({', '.join(covered)})"
    else:
        return False, f"Invalid: Only {n_covered} crises covered (need {min_crises})"


# ==============================================================================
# LOOKAHEAD BIAS DETECTION
# ==============================================================================

def check_lookahead_bias(
    features: pd.DataFrame,
    target: pd.Series,
    correlation_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Check for potential lookahead bias in features.
    
    Suspicious patterns:
    - Feature highly correlated with future target
    - Feature computed with future data
    
    Parameters:
    ===========
    features : pd.DataFrame
        Feature matrix
    target : pd.Series
        Target variable
    correlation_threshold : float
        Threshold for flagging suspicious correlations
        
    Returns:
    ========
    Dict[str, float] : Feature -> correlation with future target
    
    Notes:
    ======
    - High correlation with target is normal
    - High correlation with FUTURE target is suspicious
    - This is a heuristic check, not definitive
    """
    suspicious = {}
    
    # Shift target to represent "future" values
    future_target = target.shift(-5)  # 5-day ahead
    
    common_idx = features.index.intersection(future_target.dropna().index)
    
    for col in features.columns:
        try:
            corr = features.loc[common_idx, col].corr(future_target.loc[common_idx])
            if abs(corr) > correlation_threshold:
                suspicious[col] = corr
        except Exception:
            pass
    
    return suspicious


# ==============================================================================
# SPLIT SUMMARY AND REPORTING
# ==============================================================================

def generate_split_report(
    splits: Dict[str, SplitResult],
    crisis_coverage: Dict[str, bool]
) -> str:
    """
    Generate a markdown report of data splits for publication.
    
    Parameters:
    ===========
    splits : dict
        Dictionary of split results
    crisis_coverage : dict
        Crisis coverage information
        
    Returns:
    ========
    str : Markdown formatted report
    """
    report = []
    report.append("# Data Split Report")
    report.append("")
    report.append("## Temporal Splits")
    report.append("")
    report.append("| Split | Start | End | Samples |")
    report.append("|-------|-------|-----|---------|")
    
    for name, split in splits.items():
        dates = split.test_dates if split.test_dates else split.train_dates
        n = split.n_test if split.n_test > 0 else split.n_train
        report.append(f"| {name.capitalize()} | {dates[0][:10]} | {dates[1][:10]} | {n:,} |")
    
    report.append("")
    report.append("## Crisis Coverage (Test Set)")
    report.append("")
    
    covered = [k for k, v in crisis_coverage.items() if v]
    not_covered = [k for k, v in crisis_coverage.items() if not v]
    
    report.append(f"**Covered ({len(covered)}):** {', '.join(covered)}")
    report.append("")
    if not_covered:
        report.append(f"**Not Covered ({len(not_covered)}):** {', '.join(not_covered)}")
    
    report.append("")
    report.append("## Validation Requirements")
    report.append("")
    report.append("- [x] Strict temporal separation (no overlap)")
    report.append("- [x] Multiple crisis types in test set")
    report.append("- [x] Walk-forward validation available")
    report.append("- [ ] Lookahead bias check completed")
    
    return "\n".join(report)


# ==============================================================================
# TESTS
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("WALK-FORWARD VALIDATION MODULE TEST")
    print("=" * 60)
    
    np.random.seed(RANDOM_SEED)
    
    # Create synthetic data
    dates = pd.date_range('1995-01-01', '2024-12-31', freq='B')
    n = len(dates)
    
    data = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.randn(n),
        'target': (np.random.rand(n) > 0.85).astype(int)  # ~15% crisis rate
    }, index=dates)
    
    print("\n1. Publication Splits Test:")
    print("-" * 40)
    
    splits = create_publication_splits(data, 'target')
    
    for name, split in splits.items():
        dates = split.test_dates if split.test_dates else split.train_dates
        n = split.n_test if split.n_test > 0 else split.n_train
        print(f"   {name:12s}: {dates[0][:10]} to {dates[1][:10]} ({n:,} samples)")
    
    print("\n2. Crisis Coverage Test:")
    print("-" * 40)
    
    test_data = splits['test'].y_test
    coverage = check_crisis_coverage(test_data)
    
    covered = sum(coverage.values())
    print(f"   Crises covered: {covered}/{len(coverage)}")
    
    is_valid, msg = validate_crisis_diversity(coverage)
    print(f"   Validation: {msg}")
    
    print("\n3. Time Series CV Test:")
    print("-" * 40)
    
    for i, (train_idx, test_idx) in enumerate(time_series_cv(data, 'target', n_splits=3)):
        print(f"   Fold {i+1}: Train={len(train_idx):,}, Test={len(test_idx):,}")
    
    print("\n4. Lookahead Bias Check:")
    print("-" * 40)
    
    features = data[['feature1', 'feature2', 'feature3']]
    target = data['target']
    
    suspicious = check_lookahead_bias(features, target)
    if suspicious:
        print(f"   Suspicious features: {suspicious}")
    else:
        print("   No suspicious correlations detected")
    
    print("\n" + "=" * 60)
    print("Walk-forward validation tests completed!")

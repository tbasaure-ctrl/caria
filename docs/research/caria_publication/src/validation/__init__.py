"""
Validation Module
=================

Contains walk-forward validation and statistical tests for publication.
"""

from .walk_forward import (
    create_temporal_split,
    create_publication_splits,
    walk_forward_validation,
    time_series_cv,
    check_crisis_coverage,
    validate_crisis_diversity,
    check_lookahead_bias,
    generate_split_report,
    SplitResult,
    WalkForwardResult,
    CRISIS_EVENTS,
    DEFAULT_SPLITS
)

from .statistical_tests import (
    matthews_correlation_coefficient,
    precision_recall_f1,
    confusion_matrix,
    brier_score,
    calculate_auc,
    diebold_mariano_test,
    mcnemar_test,
    bootstrap_metric_ci,
    bootstrap_mcc_ci,
    calculate_all_metrics_with_ci,
    compare_models_statistically,
    generate_statistical_report,
    TestResult,
    MetricsResult
)

__all__ = [
    # Walk-forward
    'create_temporal_split',
    'create_publication_splits',
    'walk_forward_validation',
    'time_series_cv',
    'check_crisis_coverage',
    'validate_crisis_diversity',
    'check_lookahead_bias',
    'generate_split_report',
    'SplitResult',
    'WalkForwardResult',
    'CRISIS_EVENTS',
    'DEFAULT_SPLITS',
    
    # Statistical tests
    'matthews_correlation_coefficient',
    'precision_recall_f1',
    'confusion_matrix',
    'brier_score',
    'calculate_auc',
    'diebold_mariano_test',
    'mcnemar_test',
    'bootstrap_metric_ci',
    'bootstrap_mcc_ci',
    'calculate_all_metrics_with_ci',
    'compare_models_statistically',
    'generate_statistical_report',
    'TestResult',
    'MetricsResult'
]

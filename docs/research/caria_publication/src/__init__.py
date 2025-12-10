"""
Caria Risk Engine - Academic Publication Code
=============================================

"Entropic Resonance and Volatility Compression as Precursors to Systemic Failure"
Basaure, T. (2025)

This package provides the complete implementation for the paper,
including feature engineering, model training, validation, and analysis.

Modules:
--------
- features: Entropy, synchronization, and volatility calculations
- models: Caria Risk Engine and benchmark models
- validation: Walk-forward validation and statistical tests

Usage:
------
>>> from caria_publication.src.models import CariaRiskEngine
>>> from caria_publication.src.features import rolling_shannon_entropy
>>> from caria_publication.src.validation import bootstrap_mcc_ci
"""

__version__ = "1.0.0"
__author__ = "Tomás Basaure"
__license__ = "MIT"

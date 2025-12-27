# GAMLSS Implementation for Phase Transition Analysis

This directory contains an experimental implementation of Generalized Additive Models for Location, Scale and Shape (GAMLSS) as an alternative to threshold regression for modeling the regime-dependent relationship between Accumulated Spectral Fragility (ASF) and tail risk.

## Overview

The GAMLSS framework allows modeling not just the mean (location) but also the variance (scale) and higher moments (shape) of the response distribution as functions of covariates. This implementation uses a Beta distribution to model bounded drawdown data [0,1] and compares performance against the threshold regression baseline.

## Files

- `gamlss_model.py`: Main GAMLSS implementation with Beta distribution
  - Data preparation functions
  - BetaGAMLSS class
  - Three model specifications (linear interaction, threshold-like, simple)

- `gamlss_comparison.py`: Comparison framework
  - Threshold regression baseline
  - Linear OLS baselines
  - In-sample and out-of-sample metrics
  - Model comparison function

- `gamlss_results.py`: Results generation
  - Comparison tables (CSV and LaTeX)
  - Predicted vs actual plots
  - Regime-specific marginal effects plots
  - Residual diagnostics

- `requirements_gamlss.txt`: Python dependencies

## Usage

### Basic Usage

```python
from gamlss_model import prepare_data, model_1_linear_interaction

# Prepare data
df = prepare_data()

# Fit model
model, result = model_1_linear_interaction(df)
print(f"AIC: {model.aic():.2f}")
print(f"BIC: {model.bic():.2f}")
```

### Full Comparison

```python
from gamlss_comparison import compare_models, prepare_data

# Load data
df = prepare_data()

# Compare all models
results = compare_models(df, test_start_date='2020-01-01')

# Access results
print(results['comparison'])  # Comparison DataFrame
```

### Generate All Results

```python
from gamlss_results import generate_all_results

# Generate tables and figures
results = generate_all_results()
```

## Model Specifications

### Model 1: Linear Interaction
```
logit(μ) = β₀ + β₁·ASF + β₂·Connectivity + β₃·(ASF × Connectivity)
log(σ) = γ₀ + γ₁·ASF + γ₂·Connectivity
```

### Model 2: Threshold-like with Smooth Transition
```
logit(μ) = β₀ + β₁·ASF·I(C ≤ τ) + β₂·ASF·I(C > τ) + β₃·smooth_transition(C, τ)
log(σ) = γ₀ + γ₁·ASF + γ₂·Connectivity
```

### Model 3: Simple Location
```
logit(μ) = β₀ + β₁·ASF + β₂·Connectivity
log(σ) = γ₀ (constant scale)
```

## Data Requirements

Data should be in `outputs/coodination_data/` directory with CSV files containing:
- Date column: `date`, `Date`, or `DATE`
- Price column: `adjClose`, `close`, `Close`, or `Adj Close`

## Outputs

Running the comparison generates:
- `gamlss_data.csv`: Prepared dataset
- `gamlss_comparison_results.csv`: Model comparison metrics
- `gamlss_comparison_table.csv`: Formatted comparison table
- `gamlss_comparison_table.tex`: LaTeX table
- `gamlss_predicted_vs_actual.png`: Scatter plots
- `gamlss_regime_effects.png`: Marginal effects plots
- `gamlss_residual_diagnostics.png`: Diagnostic plots

## Notes

- This is experimental - original threshold regression code remains unchanged
- Beta distribution naturally handles bounded [0,1] data
- Smooth functions may better capture gradual regime transitions
- Results should be compared against threshold regression baseline



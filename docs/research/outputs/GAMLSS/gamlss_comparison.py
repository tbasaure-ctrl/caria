"""
GAMLSS Comparison Framework
Compares GAMLSS models against threshold regression baseline
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from gamlss_model import (
    prepare_data, BetaGAMLSS, 
    model_1_linear_interaction, model_2_threshold_like, model_3_simple_location
)


# ============================================================================
# BASELINE MODELS
# ============================================================================

def threshold_regression_baseline(df, tau=0.14):
    """
    Threshold regression baseline (Hansen 2000 style).
    
    Risk = α_L + θ_L·ASF + φ_L·C + ε  if C ≤ τ
    Risk = α_H + θ_H·ASF + φ_H·C + ε  if C > τ
    """
    C = df['Connectivity'].values
    ASF = df['ASF'].values
    y = df['Future_Drawdown'].values
    
    # Split by threshold
    mask_low = C <= tau
    mask_high = C > tau
    
    # Construct design matrix
    X = pd.DataFrame(index=df.index)
    X['Low_Const'] = 1 * mask_low
    X['High_Const'] = 1 * mask_high
    X['Low_ASF'] = ASF * mask_low
    X['High_ASF'] = ASF * mask_high
    X['Low_Connectivity'] = C * mask_low
    X['High_Connectivity'] = C * mask_high
    
    # Fit OLS with HAC standard errors
    try:
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    except:
        model = sm.OLS(y, X).fit()
    
    return model


def linear_ols_baseline(df):
    """Simple linear OLS baseline."""
    X = sm.add_constant(df[['ASF', 'Connectivity']])
    y = df['Future_Drawdown'].values
    
    model = sm.OLS(y, X).fit()
    return model


def linear_interaction_baseline(df):
    """Linear interaction baseline."""
    df_temp = df.copy()
    df_temp['ASF_x_Connectivity'] = df_temp['ASF'] * df_temp['Connectivity']
    
    X = sm.add_constant(df_temp[['ASF', 'Connectivity', 'ASF_x_Connectivity']])
    y = df_temp['Future_Drawdown'].values
    
    model = sm.OLS(y, X).fit()
    return model


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(model, y_true, y_pred=None, model_type='gamlss'):
    """
    Compute model comparison metrics.
    
    Args:
        model: Fitted model
        y_true: True values
        y_pred: Predicted values (optional, computed if None)
        model_type: 'gamlss', 'ols', or 'threshold'
    
    Returns:
        dict: Dictionary of metrics
    """
    if y_pred is None:
        if model_type == 'gamlss':
            # X_mu and X_sigma already have intercept included, so pass without adding another
            X_sigma_for_pred = model.X_sigma if hasattr(model, 'X_sigma') and model.X_sigma is not None else None
            # Debug: check dimensions
            if model.X_mu.shape[1] != len(model.beta_mu):
                print(f"WARNING: Dimension mismatch for {name}: X_mu.shape={model.X_mu.shape}, beta_mu.shape={model.beta_mu.shape}")
            y_pred = model.predict(model.X_mu, X_sigma_for_pred, add_intercept=False)
        elif model_type == 'ols' or model_type == 'threshold':
            y_pred = model.fittedvalues
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    # Prediction metrics
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    # Tail prediction accuracy (top 5% drawdowns)
    threshold = np.percentile(y_true, 95)
    top_5_mask = y_true >= threshold
    if top_5_mask.sum() > 0:
        tail_rmse = np.sqrt(np.mean((y_true[top_5_mask] - y_pred[top_5_mask])**2))
        tail_mae = np.mean(np.abs(y_true[top_5_mask] - y_pred[top_5_mask]))
    else:
        tail_rmse = np.nan
        tail_mae = np.nan
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Tail_RMSE': tail_rmse,
        'Tail_MAE': tail_mae
    }
    
    # Model-specific metrics
    if model_type == 'gamlss':
        metrics['LogLikelihood'] = model.loglikelihood()
        metrics['AIC'] = model.aic()
        metrics['BIC'] = model.bic()
        n_params = len(model.beta_mu) + len(model.beta_sigma)
        metrics['N_Params'] = n_params
    elif model_type == 'ols' or model_type == 'threshold':
        metrics['LogLikelihood'] = model.llf
        metrics['AIC'] = model.aic
        metrics['BIC'] = model.bic
        metrics['N_Params'] = model.df_model + 1  # +1 for error variance
    
    return metrics


def out_of_sample_metrics(models_dict, df_train, df_test):
    """
    Compute out-of-sample metrics for all models.
    
    Args:
        models_dict: Dictionary of fitted models {name: model}
        df_train: Training data
        df_test: Test data
    
    Returns:
        DataFrame: Out-of-sample metrics
    """
    results = []
    
    for name, model_info in models_dict.items():
        model = model_info['model']
        model_type = model_info['type']
        
        # Predict on test set
        if model_type == 'gamlss':
            # Build design matrices based on model specification
            # For Model 1: ASF, Connectivity, ASF*Connectivity
            # For Model 2: ASF*I_low, ASF*I_high, smooth_trans
            # For Model 3: ASF, Connectivity
            
            # Check which model by looking at X_mu shape
            n_features = model.X_mu.shape[1] - 1  # Subtract intercept
            
            if n_features == 3:  # Model 1: Linear Interaction
                X_mu_test = np.column_stack([
                    df_test['ASF'].values,
                    df_test['Connectivity'].values,
                    (df_test['ASF'] * df_test['Connectivity']).values
                ])
                X_sigma_test = np.column_stack([
                    df_test['ASF'].values,
                    df_test['Connectivity'].values
                ])
            elif n_features == 3:  # Model 2: Threshold-like (also 3 features)
                C_test = df_test['Connectivity'].values
                ASF_test = df_test['ASF'].values
                I_low = (C_test <= 0.14).astype(float)
                I_high = (C_test > 0.14).astype(float)
                k = 50
                smooth_trans = 1 / (1 + np.exp(-k * (C_test - 0.14)))
                X_mu_test = np.column_stack([
                    ASF_test * I_low,
                    ASF_test * I_high,
                    smooth_trans
                ])
                X_sigma_test = np.column_stack([
                    ASF_test,
                    C_test
                ])
            else:  # Model 3: Simple (2 features)
                X_mu_test = np.column_stack([
                    df_test['ASF'].values,
                    df_test['Connectivity'].values
                ])
                X_sigma_test = None
            
            y_pred = model.predict(X_mu_test, X_sigma_test, add_intercept=True)
        elif model_type == 'threshold':
            C_test = df_test['Connectivity'].values
            ASF_test = df_test['ASF'].values
            mask_low = C_test <= 0.14
            mask_high = C_test > 0.14
            
            # Build X_test with same structure as training
            X_test = pd.DataFrame(index=df_test.index)
            X_test['Low_Const'] = 1 * mask_low
            X_test['High_Const'] = 1 * mask_high
            X_test['Low_ASF'] = ASF_test * mask_low
            X_test['High_ASF'] = ASF_test * mask_high
            X_test['Low_Connectivity'] = C_test * mask_low
            X_test['High_Connectivity'] = C_test * mask_high
            
            # Get expected column names from model
            expected_cols = list(model.model.exog_names)
            
            # Ensure all expected columns exist
            for col in expected_cols:
                if col not in X_test.columns:
                    X_test[col] = 0
            
            # Reorder columns to match training exactly
            X_test = X_test[expected_cols]
            y_pred = model.predict(X_test)
        else:  # OLS
            # Check if it's Linear_Interaction model
            if 'Interaction' in name:
                df_temp = df_test.copy()
                df_temp['ASF_x_Connectivity'] = df_temp['ASF'] * df_temp['Connectivity']
                X_test = sm.add_constant(df_temp[['ASF', 'Connectivity', 'ASF_x_Connectivity']])
            else:
                X_test = sm.add_constant(df_test[['ASF', 'Connectivity']])
            y_pred = model.predict(X_test)
        
        y_true = df_test['Future_Drawdown'].values
        
        metrics = compute_metrics(model, y_true, y_pred, model_type)
        metrics['Model'] = name
        metrics['Sample'] = 'OOS'
        results.append(metrics)
    
    return pd.DataFrame(results)


# ============================================================================
# MAIN COMPARISON FUNCTION
# ============================================================================

def compare_models(df, test_split=0.2, test_start_date=None):
    """
    Compare GAMLSS models against baseline models.
    
    Args:
        df: Full dataset
        test_split: Fraction for test set (if test_start_date is None)
        test_start_date: Start date for test set (e.g., '2020-01-01')
    
    Returns:
        dict: Dictionary with comparison results
    """
    print("="*70)
    print("MODEL COMPARISON: GAMLSS vs BASELINE")
    print("="*70)
    
    # Split data
    if test_start_date is not None:
        df_train = df.loc[df.index < test_start_date].copy()
        df_test = df.loc[df.index >= test_start_date].copy()
    else:
        split_idx = int(len(df) * (1 - test_split))
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()
    
    print(f"\nTraining set: {len(df_train)} observations ({df_train.index[0].date()} to {df_train.index[-1].date()})")
    print(f"Test set: {len(df_test)} observations ({df_test.index[0].date()} to {df_test.index[-1].date()})")
    
    # Fit models on training data
    models = {}
    
    print("\n" + "-"*70)
    print("FITTING MODELS ON TRAINING DATA")
    print("-"*70)
    
    # GAMLSS Models
    print("\n1. GAMLSS Model 1: Linear Interaction")
    model1, result1 = model_1_linear_interaction(df_train)
    models['GAMLSS_Linear_Interaction'] = {'model': model1, 'type': 'gamlss'}
    
    print("2. GAMLSS Model 2: Threshold-like")
    model2, result2 = model_2_threshold_like(df_train, tau=0.14)
    models['GAMLSS_Threshold_Like'] = {'model': model2, 'type': 'gamlss'}
    
    print("3. GAMLSS Model 3: Simple Location")
    model3, result3 = model_3_simple_location(df_train)
    models['GAMLSS_Simple'] = {'model': model3, 'type': 'gamlss'}
    
    # Baseline Models
    print("\n4. Baseline: Threshold Regression")
    threshold_model = threshold_regression_baseline(df_train, tau=0.14)
    models['Threshold_Regression'] = {'model': threshold_model, 'type': 'threshold'}
    
    print("5. Baseline: Linear OLS")
    ols_model = linear_ols_baseline(df_train)
    models['Linear_OLS'] = {'model': ols_model, 'type': 'ols'}
    
    print("6. Baseline: Linear Interaction")
    interaction_model = linear_interaction_baseline(df_train)
    models['Linear_Interaction'] = {'model': interaction_model, 'type': 'ols'}
    
    # Compute in-sample metrics
    print("\n" + "-"*70)
    print("IN-SAMPLE METRICS")
    print("-"*70)
    
    is_results = []
    for name, model_info in models.items():
        model = model_info['model']
        model_type = model_info['type']
        
        try:
            if model_type == 'gamlss':
                # X_mu and X_sigma already have intercept, so use add_intercept=False
                X_sigma_for_pred = model.X_sigma if hasattr(model, 'X_sigma') and model.X_sigma is not None else None
                # Check dimensions
                if model.X_mu.shape[1] != len(model.beta_mu):
                    print(f"WARNING {name}: X_mu has {model.X_mu.shape[1]} cols, beta_mu has {len(model.beta_mu)} elements")
                    # Use covariates only and add intercept
                    y_pred = model.predict(model.X_mu[:, 1:], X_sigma_for_pred[:, 1:] if X_sigma_for_pred is not None and X_sigma_for_pred.shape[1] > 1 else None, add_intercept=True)
                else:
                    y_pred = model.predict(model.X_mu, X_sigma_for_pred, add_intercept=False)
            elif model_type == 'threshold':
                y_pred = model.fittedvalues
            else:
                y_pred = model.fittedvalues
            
            y_true = df_train['Future_Drawdown'].values
            metrics = compute_metrics(model, y_true, y_pred, model_type)
            metrics['Model'] = name
            metrics['Sample'] = 'In-Sample'
            is_results.append(metrics)
            
            print(f"\n{name}:")
            print(f"  AIC: {metrics['AIC']:.2f}")
            print(f"  BIC: {metrics['BIC']:.2f}")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  R²: {metrics['R2']:.4f}")
        except Exception as e:
            print(f"\nERROR with {name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    is_df = pd.DataFrame(is_results)
    
    # Compute out-of-sample metrics
    print("\n" + "-"*70)
    print("OUT-OF-SAMPLE METRICS")
    print("-"*70)
    
    oos_df = out_of_sample_metrics(models, df_train, df_test)
    
    for _, row in oos_df.iterrows():
        print(f"\n{row['Model']}:")
        print(f"  RMSE: {row['RMSE']:.4f}")
        print(f"  MAE: {row['MAE']:.4f}")
        print(f"  R²: {row['R2']:.4f}")
        print(f"  Tail RMSE: {row['Tail_RMSE']:.4f}")
    
    # Combine results
    comparison_df = pd.concat([is_df, oos_df], ignore_index=True)
    
    return {
        'models': models,
        'comparison': comparison_df,
        'train_data': df_train,
        'test_data': df_test
    }


if __name__ == "__main__":
    # Load data
    df = prepare_data()
    
    # Compare models
    results = compare_models(df, test_start_date='2020-01-01')
    
    # Save results
    output_dir = os.path.dirname(__file__)
    results['comparison'].to_csv(
        os.path.join(output_dir, 'gamlss_comparison_results.csv'),
        index=False
    )
    print(f"\nResults saved to {os.path.join(output_dir, 'gamlss_comparison_results.csv')}")


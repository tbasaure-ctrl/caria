"""
GAMLSS Results Generation
Generate tables and figures comparing GAMLSS vs threshold regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from gamlss_comparison import compare_models
from gamlss_model import prepare_data

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def generate_comparison_table(results, output_dir):
    """Generate LaTeX table comparing all models."""
    df = results['comparison']
    
    # Pivot to show in-sample vs out-of-sample
    metrics = ['AIC', 'BIC', 'RMSE', 'MAE', 'R2', 'Tail_RMSE']
    
    table_data = []
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        row = {'Model': model}
        
        for metric in metrics:
            is_val = model_data[model_data['Sample'] == 'In-Sample'][metric].values
            oos_val = model_data[model_data['Sample'] == 'OOS'][metric].values
            
            if len(is_val) > 0 and not np.isnan(is_val[0]):
                row[f'{metric}_IS'] = is_val[0]
            else:
                row[f'{metric}_IS'] = np.nan
            
            if len(oos_val) > 0 and not np.isnan(oos_val[0]):
                row[f'{metric}_OOS'] = oos_val[0]
            else:
                row[f'{metric}_OOS'] = np.nan
        
        table_data.append(row)
    
    table_df = pd.DataFrame(table_data)
    
    # Save CSV
    table_df.to_csv(os.path.join(output_dir, 'gamlss_comparison_table.csv'), index=False)
    
    # Generate LaTeX
    latex_str = "\\begin{table}[H]\n\\centering\n\\caption{GAMLSS vs Baseline Models Comparison}\n"
    latex_str += "\\begin{tabular}{l" + "c" * (len(metrics) * 2) + "}\n\\toprule\n"
    latex_str += "\\textbf{Model} & "
    latex_str += " & ".join([f"\\textbf{{{m}}} (IS)" for m in metrics])
    latex_str += " & " + " & ".join([f"\\textbf{{{m}}} (OOS)" for m in metrics])
    latex_str += " \\\\\n\\midrule\n"
    
    for _, row in table_df.iterrows():
        latex_str += f"{row['Model']} & "
        values = []
        for metric in metrics:
            val_is = row[f'{metric}_IS']
            val_oos = row[f'{metric}_OOS']
            if not np.isnan(val_is):
                values.append(f"{val_is:.4f}")
            else:
                values.append("---")
            if not np.isnan(val_oos):
                values.append(f"{val_oos:.4f}")
            else:
                values.append("---")
        latex_str += " & ".join(values) + " \\\\\n"
    
    latex_str += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    
    with open(os.path.join(output_dir, 'gamlss_comparison_table.tex'), 'w') as f:
        f.write(latex_str)
    
    print(f"Comparison table saved to {os.path.join(output_dir, 'gamlss_comparison_table.csv')}")
    print(f"LaTeX table saved to {os.path.join(output_dir, 'gamlss_comparison_table.tex')}")
    
    return table_df


def plot_predicted_vs_actual(results, output_dir):
    """Plot predicted vs actual drawdowns."""
    models = results['models']
    df_test = results['test_data']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (name, model_info) in enumerate(models.items()):
        if idx >= len(axes):
            break
        
        model = model_info['model']
        model_type = model_info['type']
        
        # Predict
        if model_type == 'gamlss':
            # Build design matrices based on model specification
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
            elif name == 'GAMLSS_Threshold_Like':  # Model 2: Threshold-like
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
            
            X_test = pd.DataFrame(index=df_test.index)
            X_test['Low_Const'] = 1 * mask_low
            X_test['High_Const'] = 1 * mask_high
            X_test['Low_ASF'] = ASF_test * mask_low
            X_test['High_ASF'] = ASF_test * mask_high
            X_test['Low_Connectivity'] = C_test * mask_low
            X_test['High_Connectivity'] = C_test * mask_high
            
            # Get expected column names from model
            expected_cols = list(model.model.exog_names)
            for col in expected_cols:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_test = X_test[expected_cols]
            y_pred = model.predict(X_test)
        else:  # OLS
            import statsmodels.api as sm
            # Check if it's Linear_Interaction model
            if 'Interaction' in name:
                df_temp = df_test.copy()
                df_temp['ASF_x_Connectivity'] = df_temp['ASF'] * df_temp['Connectivity']
                X_test = sm.add_constant(df_temp[['ASF', 'Connectivity', 'ASF_x_Connectivity']])
            else:
                X_test = sm.add_constant(df_test[['ASF', 'Connectivity']])
            y_pred = model.predict(X_test)
        
        y_true = df_test['Future_Drawdown'].values
        
        # Scatter plot
        ax = axes[idx]
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Drawdown')
        ax.set_ylabel('Predicted Drawdown')
        ax.set_title(f'{name}\nR² = {np.corrcoef(y_true, y_pred)[0,1]**2:.3f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gamlss_predicted_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Predicted vs actual plot saved to {os.path.join(output_dir, 'gamlss_predicted_vs_actual.png')}")


def plot_regime_effects(results, output_dir):
    """Plot regime-specific marginal effects (similar to threshold regression)."""
    models = results['models']
    df_train = results['train_data']
    
    # Focus on GAMLSS models
    gamlss_models = {k: v for k, v in models.items() if v['type'] == 'gamlss'}
    
    if len(gamlss_models) == 0:
        print("No GAMLSS models to plot")
        return
    
    fig, axes = plt.subplots(1, len(gamlss_models), figsize=(6*len(gamlss_models), 6))
    if len(gamlss_models) == 1:
        axes = [axes]
    
    connectivity_range = np.linspace(df_train['Connectivity'].min(), 
                                     df_train['Connectivity'].max(), 100)
    asf_fixed = df_train['ASF'].median()
    
    for idx, (name, model_info) in enumerate(gamlss_models.items()):
        model = model_info['model']
        
        # Compute marginal effect: dE[Drawdown]/dASF as function of Connectivity
        marginal_effects = []
        
        for C in connectivity_range:
            # Small perturbation in ASF
            ASF_low = asf_fixed - 0.01
            ASF_high = asf_fixed + 0.01
            
            # Predictions
            if name == 'GAMLSS_Linear_Interaction':
                X_mu_low = np.array([[ASF_low, C, ASF_low * C]])
                X_mu_high = np.array([[ASF_high, C, ASF_high * C]])
                X_sigma = np.array([[asf_fixed, C]])
            elif name == 'GAMLSS_Threshold_Like':
                I_low = 1 if C <= 0.14 else 0
                I_high = 1 if C > 0.14 else 0
                k = 50
                smooth_trans = 1 / (1 + np.exp(-k * (C - 0.14)))
                X_mu_low = np.array([[ASF_low * I_low, ASF_low * I_high, smooth_trans]])
                X_mu_high = np.array([[ASF_high * I_low, ASF_high * I_high, smooth_trans]])
                X_sigma = np.array([[asf_fixed, C]])
            else:  # Simple
                X_mu_low = np.array([[ASF_low, C]])
                X_mu_high = np.array([[ASF_high, C]])
                X_sigma = None
            
            mu_low = model.predict(X_mu_low, X_sigma, add_intercept=True)
            mu_high = model.predict(X_mu_high, X_sigma, add_intercept=True)
            
            marginal_effect = (mu_high[0] - mu_low[0]) / (ASF_high - ASF_low)
            marginal_effects.append(marginal_effect)
        
        # Plot
        ax = axes[idx]
        ax.plot(connectivity_range, marginal_effects, 'b-', lw=2, label='Marginal Effect')
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax.axvline(0.14, color='g', linestyle='--', alpha=0.5, label='Threshold (0.14)')
        ax.set_xlabel('Connectivity')
        ax.set_ylabel('∂E[Drawdown]/∂ASF')
        ax.set_title(f'{name}\nMarginal Effect of ASF')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gamlss_regime_effects.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Regime effects plot saved to {os.path.join(output_dir, 'gamlss_regime_effects.png')}")


def plot_residual_diagnostics(results, output_dir):
    """Plot residual diagnostics (Q-Q plots, residuals vs fitted)."""
    models = results['models']
    df_test = results['test_data']
    
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 4*len(models)))
    
    for idx, (name, model_info) in enumerate(models.items()):
        model = model_info['model']
        model_type = model_info['type']
        
        # Predict
        if model_type == 'gamlss':
            # Build design matrices based on model specification
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
            elif name == 'GAMLSS_Threshold_Like':  # Model 2: Threshold-like
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
            
            X_test = pd.DataFrame(index=df_test.index)
            X_test['Low_Const'] = 1 * mask_low
            X_test['High_Const'] = 1 * mask_high
            X_test['Low_ASF'] = ASF_test * mask_low
            X_test['High_ASF'] = ASF_test * mask_high
            X_test['Low_Connectivity'] = C_test * mask_low
            X_test['High_Connectivity'] = C_test * mask_high
            
            # Get expected column names from model
            expected_cols = list(model.model.exog_names)
            for col in expected_cols:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_test = X_test[expected_cols]
            y_pred = model.predict(X_test)
        else:  # OLS
            import statsmodels.api as sm
            # Check if it's Linear_Interaction model
            if 'Interaction' in name:
                df_temp = df_test.copy()
                df_temp['ASF_x_Connectivity'] = df_temp['ASF'] * df_temp['Connectivity']
                X_test = sm.add_constant(df_temp[['ASF', 'Connectivity', 'ASF_x_Connectivity']])
            else:
                X_test = sm.add_constant(df_test[['ASF', 'Connectivity']])
            y_pred = model.predict(X_test)
        
        y_true = df_test['Future_Drawdown'].values
        residuals = y_true - y_pred
        
        # Residuals vs Fitted
        ax1 = axes[idx, 0]
        ax1.scatter(y_pred, residuals, alpha=0.5, s=20)
        ax1.axhline(0, color='r', linestyle='--')
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title(f'{name}: Residuals vs Fitted')
        ax1.grid(True, alpha=0.3)
        
        # Q-Q Plot
        ax2 = axes[idx, 1]
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title(f'{name}: Q-Q Plot')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gamlss_residual_diagnostics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Residual diagnostics saved to {os.path.join(output_dir, 'gamlss_residual_diagnostics.png')}")


def generate_all_results(output_dir=None):
    """Generate all results tables and figures."""
    if output_dir is None:
        output_dir = os.path.dirname(__file__)
    
    print("="*70)
    print("GENERATING GAMLSS RESULTS")
    print("="*70)
    
    # Load data and compare models
    df = prepare_data()
    results = compare_models(df, test_start_date='2020-01-01')
    
    # Generate tables
    print("\nGenerating comparison table...")
    table_df = generate_comparison_table(results, output_dir)
    
    # Generate figures
    print("\nGenerating figures...")
    plot_predicted_vs_actual(results, output_dir)
    plot_regime_effects(results, output_dir)
    plot_residual_diagnostics(results, output_dir)
    
    print("\n" + "="*70)
    print("RESULTS GENERATION COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}")
    print("\nFiles generated:")
    print("  - gamlss_comparison_table.csv")
    print("  - gamlss_comparison_table.tex")
    print("  - gamlss_predicted_vs_actual.png")
    print("  - gamlss_regime_effects.png")
    print("  - gamlss_residual_diagnostics.png")
    
    return results


if __name__ == "__main__":
    generate_all_results()


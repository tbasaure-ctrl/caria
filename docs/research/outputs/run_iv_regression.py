"""
Instrumental Variables Regression - Statsmodels Only Version

Uses manual 2SLS implementation that works without linearmodels.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
ASF_FILE = 'Table_Theory_Data.csv'
VIX_FILE = 'coodination_data/CBOE_Volatility_Index.csv'
SPX_FILE = 'coodination_data/S&P_500.csv'
TLT_FILE = 'coodination_data/Treasuries_20Y.csv'
GLD_FILE = 'coodination_data/Gold.csv'

def load_data():
    """Load and merge all data sources."""
    asf = pd.read_csv(ASF_FILE)
    asf.rename(columns={asf.columns[0]: 'Date'}, inplace=True)
    asf['Date'] = pd.to_datetime(asf['Date'])
    asf = asf.set_index('Date').sort_index()
    
    df = asf[['ASF']].copy()
    
    files = {
        'VIX': (VIX_FILE, True),
        'SPX': (SPX_FILE, False),
        'TLT': (TLT_FILE, False),
        'GLD': (GLD_FILE, False),
    }
    
    for name, (path, scale) in files.items():
        if os.path.exists(path):
            temp = pd.read_csv(path)
            d_col = 'date' if 'date' in temp.columns else 'Date'
            temp[d_col] = pd.to_datetime(temp[d_col])
            temp = temp.set_index(d_col).sort_index()
            c_col = 'adjClose' if 'adjClose' in temp.columns else 'close'
            if scale:
                temp[name] = temp[c_col] / 100.0
            else:
                temp[name] = temp[c_col]
            how = 'inner' if name in ['SPX', 'VIX'] else 'left'
            df = df.join(temp[[name]], how=how)
    
    return df

def calculate_forward_drawdown(series, window=21):
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
    rolling_min = series.rolling(window=indexer).min()
    fwd_worst_return = (rolling_min / series) - 1
    return -fwd_worst_return

def compute_mean_correlation(df, assets, window=63):
    rets = df[assets].pct_change()
    rolling_corrs = []
    
    for i in range(len(assets)):
        for j in range(i+1, len(assets)):
            col1, col2 = assets[i], assets[j]
            rc = rets[col1].rolling(window).corr(rets[col2])
            rolling_corrs.append(rc)
    
    if rolling_corrs:
        return pd.concat(rolling_corrs, axis=1).mean(axis=1)
    else:
        return pd.Series(np.nan, index=df.index)

def manual_2sls(y, X_exog, X_endog, Z_instruments):
    """
    Manual 2SLS implementation.
    
    y: dependent variable (Series)
    X_exog: exogenous regressors including constant (DataFrame)
    X_endog: endogenous regressor (Series)
    Z_instruments: instruments for X_endog (DataFrame)
    """
    n = len(y)
    
    # First stage: regress endogenous on all instruments + exogenous
    Z_all = X_exog.join(Z_instruments)
    first_stage = sm.OLS(X_endog, Z_all).fit()
    X_endog_fitted = first_stage.fittedvalues
    
    # F-test for instrument relevance (partial F on excluded instruments)
    model_reduced = sm.OLS(X_endog, X_exog).fit()
    ssr_reduced = model_reduced.ssr
    ssr_full = first_stage.ssr
    k_instr = Z_instruments.shape[1]
    df_resid = first_stage.df_resid
    
    partial_f = ((ssr_reduced - ssr_full) / k_instr) / (ssr_full / df_resid)
    partial_f_pval = 1 - stats.f.cdf(partial_f, k_instr, df_resid)
    
    # Second stage: regress y on fitted values + exogenous
    X_second = X_exog.copy()
    X_second['Connectivity_IV'] = X_endog_fitted.values
    
    second_stage = sm.OLS(y, X_second).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    
    return {
        'first_stage': first_stage,
        'second_stage': second_stage,
        'partial_f': partial_f,
        'partial_f_pval': partial_f_pval,
        'fitted_endog': X_endog_fitted
    }

def hausman_test(ols_model, iv_second_stage, endog_name='Connectivity'):
    """
    Wu-Hausman test for endogeneity.
    Compare OLS and IV estimates.
    """
    # Get coefficients and covariances
    b_ols = ols_model.params
    b_iv = iv_second_stage.params
    
    # Find common parameters
    common = [p for p in b_ols.index if p in b_iv.index and p != 'const']
    
    diff = b_iv[common] - b_ols[common]
    
    # Variance of difference (simplified - assumes independence)
    var_diff = np.diag(iv_second_stage.cov_params().loc[common, common]) - np.diag(ols_model.cov_params().loc[common, common])
    var_diff = np.abs(var_diff)  # Ensure positive
    
    # Hausman statistic
    hausman_stat = np.sum(diff**2 / var_diff)
    hausman_pval = 1 - stats.chi2.cdf(hausman_stat, len(common))
    
    return hausman_stat, hausman_pval

def run_iv_regression():
    print("\n" + "="*70)
    print("INSTRUMENTAL VARIABLES REGRESSION")
    print("Addressing Endogeneity in Connectivity-Risk Relationship")
    print("="*70 + "\n")
    
    # Load data
    df = load_data()
    df['Fwd_MaxDD'] = calculate_forward_drawdown(df['SPX'], window=21)
    
    # Compute connectivity
    assets = [c for c in ['SPX', 'TLT', 'GLD'] if c in df.columns]
    if len(assets) >= 2:
        df['Connectivity'] = compute_mean_correlation(df, assets)
    
    # Create lagged instruments
    for lag in [4, 8, 12]:
        df[f'Connectivity_L{lag}'] = df['Connectivity'].shift(lag)
    
    # Clean data
    analysis_cols = ['ASF', 'Connectivity', 'Fwd_MaxDD', 
                     'Connectivity_L4', 'Connectivity_L8', 'Connectivity_L12']
    df_clean = df.dropna(subset=analysis_cols)
    
    print(f"Sample size: {len(df_clean)} observations")
    print(f"Date range: {df_clean.index.min().date()} to {df_clean.index.max().date()}\n")
    
    # =========================================================================
    # OLS BASELINE
    # =========================================================================
    print("-"*70)
    print("BASELINE: OLS Regression")
    print("-"*70)
    
    y = df_clean['Fwd_MaxDD']
    X_ols = sm.add_constant(df_clean[['ASF', 'Connectivity']])
    
    model_ols = sm.OLS(y, X_ols).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    
    print(f"\n{'Variable':<20} {'Coefficient':>12} {'t-stat':>10} {'p-value':>10}")
    print("-"*55)
    for var in model_ols.params.index:
        print(f"{var:<20} {model_ols.params[var]:>12.4f} {model_ols.tvalues[var]:>10.2f} {model_ols.pvalues[var]:>10.4f}")
    print(f"\nR²: {model_ols.rsquared:.4f}")
    
    # =========================================================================
    # IV REGRESSION (2SLS)
    # =========================================================================
    print("\n" + "-"*70)
    print("2SLS IV REGRESSION")
    print("Instruments: Lagged Connectivity (4, 8, 12 weeks)")
    print("-"*70)
    
    X_exog = sm.add_constant(df_clean[['ASF']])
    X_endog = df_clean['Connectivity']
    Z_instruments = df_clean[['Connectivity_L4', 'Connectivity_L8', 'Connectivity_L12']]
    
    iv_results = manual_2sls(y, X_exog, X_endog, Z_instruments)
    
    # First stage results
    print("\n--- First Stage ---")
    print(f"Partial F-statistic (instruments): {iv_results['partial_f']:.2f}")
    print(f"Partial F p-value: {iv_results['partial_f_pval']:.4f}")
    
    if iv_results['partial_f'] > 10:
        print("✓ Instruments are STRONG (F > 10)")
    else:
        print("⚠ Instruments may be WEAK (F < 10)")
    
    print(f"\nFirst-stage R²: {iv_results['first_stage'].rsquared:.4f}")
    
    # Second stage results
    print("\n--- Second Stage (IV Estimates) ---")
    iv_model = iv_results['second_stage']
    
    print(f"\n{'Variable':<20} {'Coefficient':>12} {'t-stat':>10} {'p-value':>10}")
    print("-"*55)
    for var in iv_model.params.index:
        display_var = var.replace('_IV', '')
        print(f"{display_var:<20} {iv_model.params[var]:>12.4f} {iv_model.tvalues[var]:>10.2f} {iv_model.pvalues[var]:>10.4f}")
    
    # =========================================================================
    # HAUSMAN TEST
    # =========================================================================
    print("\n" + "-"*70)
    print("ENDOGENEITY TEST (Wu-Hausman)")
    print("-"*70)
    
    hausman_stat, hausman_pval = hausman_test(model_ols, iv_model)
    print(f"\nHausman statistic: {hausman_stat:.3f}")
    print(f"Hausman p-value: {hausman_pval:.4f}")
    
    if hausman_pval < 0.05:
        print("✓ Reject null of exogeneity (p < 0.05) - IV estimates preferred")
    else:
        print("Cannot reject exogeneity - OLS may be consistent")
    
    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print("\n" + "-"*70)
    print("COMPARISON: OLS vs 2SLS IV")
    print("-"*70)
    
    print(f"\n{'Variable':<20} {'OLS':>12} {'2SLS IV':>12} {'Difference':>12}")
    print("-"*60)
    
    ols_asf = model_ols.params['ASF']
    iv_asf = iv_model.params['ASF']
    print(f"{'ASF':<20} {ols_asf:>12.4f} {iv_asf:>12.4f} {iv_asf - ols_asf:>12.4f}")
    
    ols_conn = model_ols.params['Connectivity']
    iv_conn = iv_model.params['Connectivity_IV']
    print(f"{'Connectivity':<20} {ols_conn:>12.4f} {iv_conn:>12.4f} {iv_conn - ols_conn:>12.4f}")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results_df = pd.DataFrame({
        'Test': ['First-Stage Partial F', 'Hausman Statistic', 
                 'OLS: ASF', 'OLS: Connectivity',
                 'IV: ASF', 'IV: Connectivity'],
        'Value': [iv_results['partial_f'], hausman_stat,
                  ols_asf, ols_conn, iv_asf, iv_conn],
        'P-value': [iv_results['partial_f_pval'], hausman_pval,
                    model_ols.pvalues['ASF'], model_ols.pvalues['Connectivity'],
                    iv_model.pvalues['ASF'], iv_model.pvalues['Connectivity_IV']],
        'Interpretation': [
            'Strong' if iv_results['partial_f'] > 10 else 'Weak',
            'Endogenous' if hausman_pval < 0.05 else 'Exogenous',
            'Significant' if model_ols.pvalues['ASF'] < 0.05 else 'Not significant',
            'Significant' if model_ols.pvalues['Connectivity'] < 0.05 else 'Not significant',
            'Significant' if iv_model.pvalues['ASF'] < 0.05 else 'Not significant',
            'Significant' if iv_model.pvalues['Connectivity_IV'] < 0.05 else 'Not significant'
        ]
    })
    
    results_df.to_csv('IV_Regression_Results.csv', index=False)
    print("\n✓ Results saved to IV_Regression_Results.csv")
    
    # LaTeX table
    latex = f"""
\\begin{{table}}[H]
\\centering
\\begin{{threeparttable}}
\\caption{{\\textbf{{Instrumental Variables Regression}}}}
\\label{{tab:iv}}
\\begin{{tabular}}{{lcccc}}
\\toprule
& \\multicolumn{{2}}{{c}}{{\\textbf{{OLS}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{2SLS IV}}}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
\\textbf{{Variable}} & Coefficient & $t$-stat & Coefficient & $t$-stat \\\\
\\midrule
ASF & {ols_asf:.3f} & {model_ols.tvalues['ASF']:.2f} & {iv_asf:.3f} & {iv_model.tvalues['ASF']:.2f} \\\\
Connectivity & {ols_conn:.3f} & {model_ols.tvalues['Connectivity']:.2f} & {iv_conn:.3f} & {iv_model.tvalues['Connectivity_IV']:.2f} \\\\
\\midrule
\\multicolumn{{5}}{{l}}{{\\textit{{Diagnostic Tests}}}} \\\\
First-Stage Partial $F$ & \\multicolumn{{4}}{{c}}{{{iv_results['partial_f']:.1f}}} \\\\
Hausman Test & \\multicolumn{{4}}{{c}}{{{hausman_stat:.2f} (p = {hausman_pval:.3f})}} \\\\
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}[flushleft]
\\footnotesize
\\item \\textit{{Notes:}} Dependent variable is forward 21-day maximum drawdown. Instruments are lagged connectivity (4, 8, and 12 weeks). First-Stage $F > 10$ indicates strong instruments. Hausman test evaluates exogeneity; rejection supports IV.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""
    
    with open('IV_Table_LaTeX.tex', 'w') as f:
        f.write(latex)
    print("✓ LaTeX table saved to IV_Table_LaTeX.tex")
    
    print("\n" + "="*70)
    print("IV REGRESSION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    run_iv_regression()

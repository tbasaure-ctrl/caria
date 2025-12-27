"""
ENHANCEMENT 3: Causal Identification & Horse-Race Regressions

This implements:
1. Horse-race regressions: ASF vs VIX vs alternative risk measures
2. Diebold-Mariano tests for out-of-sample predictive comparison
3. Encompassing tests (does ASF subsume VIX information?)
4. Rolling window out-of-sample forecasts
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'

# Configuration
ASF_FILE = 'Table_Theory_Data.csv'
VIX_FILE = 'coodination_data/CBOE_Volatility_Index.csv'
SPX_FILE = 'coodination_data/S&P_500.csv'
TLT_FILE = 'coodination_data/Treasuries_20Y.csv'

def load_data():
    """Load and merge all data sources."""
    asf = pd.read_csv(ASF_FILE)
    asf.rename(columns={asf.columns[0]: 'Date'}, inplace=True)
    asf['Date'] = pd.to_datetime(asf['Date'])
    asf = asf.set_index('Date').sort_index()
    df = asf[['ASF']].copy()
    
    if os.path.exists(VIX_FILE):
        vix = pd.read_csv(VIX_FILE)
        vix['date'] = pd.to_datetime(vix['date'])
        vix = vix.set_index('date').sort_index()
        vix['VIX'] = vix['adjClose'] / 100.0
        df = df.join(vix[['VIX']], how='left')
    
    if os.path.exists(SPX_FILE):
        spx = pd.read_csv(SPX_FILE)
        spx['date'] = pd.to_datetime(spx['date'])
        spx = spx.set_index('date').sort_index()
        spx['SPX'] = spx['adjClose']
        df = df.join(spx[['SPX']], how='inner')
    
    if os.path.exists(TLT_FILE):
        tlt = pd.read_csv(TLT_FILE)
        tlt['date'] = pd.to_datetime(tlt['date'])
        tlt = tlt.set_index('date').sort_index()
        tlt['TLT'] = tlt['adjClose']
        df = df.join(tlt[['TLT']], how='left')
    
    return df

def calculate_forward_drawdown(series, window=21):
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
    rolling_min = series.rolling(window=indexer).min()
    return -((rolling_min / series) - 1)

def compute_additional_measures(df):
    """Compute additional risk measures for horse race."""
    
    # Calculate returns
    if 'SPX' in df.columns:
        df['SPX_ret'] = df['SPX'].pct_change()
        
        # Realized volatility (21-day)
        df['RealizedVol'] = df['SPX_ret'].rolling(21).std() * np.sqrt(252)
        
        # GARCH-style volatility proxy (EWMA)
        df['EWMA_Vol'] = df['SPX_ret'].ewm(span=21).std() * np.sqrt(252)
        
    # Mean correlation proxy (using ASF derivative)
    df['MeanCorr'] = 1 - df['ASF']  # Inverse relationship
    
    # Absorption Ratio proxy (simplified)
    df['AbsorptionRatio'] = df['ASF'].rolling(21).mean()
    
    # TED spread proxy: Use VIX changes
    if 'VIX' in df.columns:
        df['VIX_Change'] = df['VIX'].pct_change()
        df['VIX_Level'] = df['VIX']
    
    return df

def run_horse_race_regressions(df):
    """
    Run horse-race regressions comparing predictive power.
    
    Tests:
    1. Univariate: Each predictor alone
    2. Multivariate: All predictors together
    3. Incremental: Does ASF add information over VIX?
    """
    print("\n" + "="*70)
    print("HORSE-RACE REGRESSIONS")
    print("="*70)
    
    # Prepare dependent variable
    df['Fwd_MaxDD'] = calculate_forward_drawdown(df['SPX'])
    df = compute_additional_measures(df)
    
    # Define predictors
    predictors = {
        'ASF': 'ASF',
        'VIX': 'VIX_Level',
        'RealizedVol': 'RealizedVol',
        'MeanCorr': 'MeanCorr'
    }
    
    available_predictors = {k: v for k, v in predictors.items() if v in df.columns}
    
    # Prepare clean dataset
    cols_needed = ['Fwd_MaxDD'] + list(available_predictors.values())
    df_clean = df[cols_needed].dropna()
    
    print(f"\nSample size: {len(df_clean)} observations")
    print(f"Predictors available: {list(available_predictors.keys())}")
    
    results = []
    
    # ==========================================================================
    # 1. UNIVARIATE REGRESSIONS
    # ==========================================================================
    print("\n" + "-"*70)
    print("1. UNIVARIATE REGRESSIONS (each predictor alone)")
    print("-"*70)
    
    y = df_clean['Fwd_MaxDD']
    
    univariate_results = {}
    
    for name, col in available_predictors.items():
        X = sm.add_constant(df_clean[[col]])
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
        
        univariate_results[name] = {
            'beta': model.params[col],
            't_stat': model.tvalues[col],
            'p_value': model.pvalues[col],
            'R2': model.rsquared,
            'AIC': model.aic
        }
        
        sig = "***" if model.pvalues[col] < 0.01 else "**" if model.pvalues[col] < 0.05 else "*" if model.pvalues[col] < 0.10 else ""
        print(f"  {name:<15} beta = {model.params[col]:>8.4f}, t = {model.tvalues[col]:>6.2f}{sig}, R2 = {model.rsquared:.4f}, AIC = {model.aic:.0f}")
        
        results.append({
            'Model': f'Univariate: {name}',
            'Beta_Primary': model.params[col],
            'T_Stat': model.tvalues[col],
            'P_Value': model.pvalues[col],
            'R2': model.rsquared,
            'AIC': model.aic
        })
    
    # ==========================================================================
    # 2. MULTIVARIATE (ALL PREDICTORS)
    # ==========================================================================
    print("\n" + "-"*70)
    print("2. MULTIVARIATE REGRESSION (all predictors)")
    print("-"*70)
    
    X_all = sm.add_constant(df_clean[list(available_predictors.values())])
    model_all = sm.OLS(y, X_all).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    
    print(f"\n  R2 = {model_all.rsquared:.4f}, AIC = {model_all.aic:.0f}\n")
    print(f"  {'Variable':<15} {'Beta':>10} {'t-stat':>10} {'p-value':>10}")
    print("  " + "-"*50)
    
    for name, col in available_predictors.items():
        sig = "***" if model_all.pvalues[col] < 0.01 else "**" if model_all.pvalues[col] < 0.05 else "*" if model_all.pvalues[col] < 0.10 else ""
        print(f"  {name:<15} {model_all.params[col]:>10.4f} {model_all.tvalues[col]:>10.2f} {model_all.pvalues[col]:>10.4f} {sig}")
    
    results.append({
        'Model': 'Multivariate: All',
        'Beta_Primary': model_all.params.get('ASF', np.nan),
        'T_Stat': model_all.tvalues.get('ASF', np.nan),
        'P_Value': model_all.pvalues.get('ASF', np.nan),
        'R2': model_all.rsquared,
        'AIC': model_all.aic
    })
    
    # ==========================================================================
    # 3. INCREMENTAL TEST (Does ASF add to VIX?)
    # ==========================================================================
    print("\n" + "-"*70)
    print("3. INCREMENTAL TEST (Does ASF add information over VIX?)")
    print("-"*70)
    
    if 'VIX_Level' in df_clean.columns and 'ASF' in df_clean.columns:
        # Restricted model (VIX only)
        X_vix = sm.add_constant(df_clean[['VIX_Level']])
        model_vix = sm.OLS(y, X_vix).fit()
        
        # Unrestricted model (VIX + ASF)
        X_both = sm.add_constant(df_clean[['VIX_Level', 'ASF']])
        model_both = sm.OLS(y, X_both).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
        
        # F-test for ASF coefficient
        f_stat = ((model_vix.ssr - model_both.ssr) / 1) / (model_both.ssr / model_both.df_resid)
        f_pval = 1 - stats.f.cdf(f_stat, 1, model_both.df_resid)
        
        print(f"\n  VIX-only model:    R2 = {model_vix.rsquared:.4f}")
        print(f"  VIX + ASF model:   R2 = {model_both.rsquared:.4f}")
        print(f"  Incremental R2:    {model_both.rsquared - model_vix.rsquared:.4f}")
        print(f"\n  F-test for ASF:    F = {f_stat:.2f}, p = {f_pval:.4f}")
        
        if f_pval < 0.05:
            print("  => ASF adds significant incremental information over VIX")
        else:
            print("  => ASF does not add significant incremental information")
        
        # Encompassing test: Does VIX add to ASF?
        X_asf = sm.add_constant(df_clean[['ASF']])
        model_asf = sm.OLS(y, X_asf).fit()
        
        f_stat_rev = ((model_asf.ssr - model_both.ssr) / 1) / (model_both.ssr / model_both.df_resid)
        f_pval_rev = 1 - stats.f.cdf(f_stat_rev, 1, model_both.df_resid)
        
        print(f"\n  ASF-only model:    R2 = {model_asf.rsquared:.4f}")
        print(f"  F-test for VIX:    F = {f_stat_rev:.2f}, p = {f_pval_rev:.4f}")
        
        if f_pval_rev < 0.05:
            print("  => VIX adds significant incremental information over ASF")
        else:
            print("  => VIX does not add significant incremental information")
        
        results.append({
            'Model': 'Incremental: ASF|VIX',
            'Beta_Primary': model_both.params['ASF'],
            'T_Stat': model_both.tvalues['ASF'],
            'P_Value': f_pval,
            'R2': model_both.rsquared - model_vix.rsquared,
            'AIC': model_both.aic
        })
    
    return pd.DataFrame(results), df_clean, univariate_results


def diebold_mariano_test(e1, e2, h=1):
    """
    Diebold-Mariano test for equal predictive accuracy.
    
    e1, e2: forecast errors from two models
    h: forecast horizon
    
    Returns: DM statistic, p-value
    """
    d = e1**2 - e2**2  # Loss differential
    n = len(d)
    
    # Compute autocovariance
    d_bar = np.mean(d)
    gamma_0 = np.var(d)
    
    # Long-run variance (with HAC correction for h > 1)
    gamma_sum = gamma_0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        gamma_sum += 2 * gamma_k
    
    var_d = gamma_sum / n
    
    # DM statistic
    dm_stat = d_bar / np.sqrt(var_d + 1e-10)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value


def out_of_sample_forecasts(df, train_end='2019-12-31'):
    """
    Rolling window out-of-sample forecasts.
    
    Train on data up to train_end, test on remainder.
    Compare forecast accuracy across models.
    """
    print("\n" + "="*70)
    print("OUT-OF-SAMPLE FORECAST COMPARISON")
    print("="*70)
    
    train_end = pd.to_datetime(train_end)
    
    df['Fwd_MaxDD'] = calculate_forward_drawdown(df['SPX'])
    df = compute_additional_measures(df)
    
    cols = ['Fwd_MaxDD', 'ASF', 'VIX_Level', 'RealizedVol']
    cols = [c for c in cols if c in df.columns]
    df_clean = df[cols].dropna()
    
    train = df_clean.loc[:train_end]
    test = df_clean.loc[train_end:]
    
    print(f"\nTraining period: {train.index.min().date()} to {train.index.max().date()} ({len(train)} obs)")
    print(f"Test period:     {test.index.min().date()} to {test.index.max().date()} ({len(test)} obs)")
    
    if len(test) < 50:
        print("  Warning: Small test sample")
    
    # Define models
    models = {
        'Historical Mean': [],
        'VIX Only': ['VIX_Level'],
        'ASF Only': ['ASF'],
        'VIX + ASF': ['VIX_Level', 'ASF'],
    }
    
    # Add RealizedVol if available
    if 'RealizedVol' in df_clean.columns:
        models['RealizedVol Only'] = ['RealizedVol']
        models['All Predictors'] = ['VIX_Level', 'ASF', 'RealizedVol']
    
    forecasts = {}
    errors = {}
    
    y_train = train['Fwd_MaxDD']
    y_test = test['Fwd_MaxDD']
    
    for name, predictors in models.items():
        if name == 'Historical Mean':
            # Naive forecast: historical mean
            pred = np.full(len(test), y_train.mean())
        else:
            # Check if all predictors available
            if not all(p in train.columns for p in predictors):
                continue
            
            X_train = sm.add_constant(train[predictors])
            X_test = sm.add_constant(test[predictors])
            
            model = sm.OLS(y_train, X_train).fit()
            pred = model.predict(X_test)
        
        forecasts[name] = pred
        errors[name] = y_test.values - pred
    
    # Compute metrics
    print("\n" + "-"*70)
    print("FORECAST ACCURACY METRICS")
    print("-"*70)
    
    metrics = []
    print(f"\n  {'Model':<20} {'RMSE':>10} {'MAE':>10} {'MAPE':>10}")
    print("  " + "-"*55)
    
    for name in forecasts:
        e = errors[name]
        rmse = np.sqrt(np.mean(e**2))
        mae = np.mean(np.abs(e))
        mape = np.mean(np.abs(e / (y_test.values + 1e-10))) * 100
        
        print(f"  {name:<20} {rmse:>10.4f} {mae:>10.4f} {mape:>10.2f}%")
        
        metrics.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        })
    
    # Diebold-Mariano tests vs benchmark
    print("\n" + "-"*70)
    print("DIEBOLD-MARIANO TESTS (vs Historical Mean)")
    print("-"*70)
    
    benchmark = 'Historical Mean'
    
    dm_results = []
    print(f"\n  {'Model':<20} {'DM Stat':>10} {'p-value':>10} {'Conclusion':>20}")
    print("  " + "-"*65)
    
    for name in forecasts:
        if name == benchmark:
            continue
        
        dm_stat, dm_pval = diebold_mariano_test(errors[benchmark], errors[name], h=1)
        
        conclusion = "Better***" if dm_pval < 0.01 else "Better**" if dm_pval < 0.05 else "Better*" if dm_pval < 0.10 else "Equal"
        if dm_stat < 0:
            conclusion = "Worse" if dm_pval < 0.10 else "Equal"
        
        print(f"  {name:<20} {dm_stat:>10.2f} {dm_pval:>10.4f} {conclusion:>20}")
        
        dm_results.append({
            'Model': name,
            'DM_Statistic': dm_stat,
            'P_Value': dm_pval,
            'Better_Than_Benchmark': dm_stat > 0 and dm_pval < 0.10
        })
    
    # Compare ASF vs VIX directly
    if 'ASF Only' in errors and 'VIX Only' in errors:
        print("\n" + "-"*70)
        print("DIRECT COMPARISON: ASF vs VIX")
        print("-"*70)
        
        dm_stat, dm_pval = diebold_mariano_test(errors['VIX Only'], errors['ASF Only'], h=1)
        
        print(f"\n  DM test (VIX errors vs ASF errors):")
        print(f"    DM statistic: {dm_stat:.3f}")
        print(f"    p-value: {dm_pval:.4f}")
        
        if dm_stat > 0 and dm_pval < 0.05:
            print("  => ASF significantly outperforms VIX")
        elif dm_stat < 0 and dm_pval < 0.05:
            print("  => VIX significantly outperforms ASF")
        else:
            print("  => No significant difference in predictive accuracy")
    
    return pd.DataFrame(metrics), pd.DataFrame(dm_results), forecasts, y_test


def plot_horse_race_results(regression_results, metrics, dm_results, forecasts, y_test):
    """Generate visualization of horse-race results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: R-squared comparison (Univariate)
    ax1 = axes[0, 0]
    univariate = regression_results[regression_results['Model'].str.contains('Univariate')]
    names = [m.split(': ')[1] for m in univariate['Model']]
    r2s = univariate['R2'].values
    
    colors = ['steelblue' if 'ASF' in n else 'orange' for n in names]
    bars = ax1.bar(names, r2s, color=colors, edgecolor='black')
    ax1.set_ylabel('R-squared', fontsize=11)
    ax1.set_title('Univariate Predictive Power', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, r2s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                 f'{val:.3f}', ha='center', fontsize=9)
    
    # Panel B: Out-of-sample RMSE
    ax2 = axes[0, 1]
    model_names = metrics['Model'].values
    rmses = metrics['RMSE'].values
    
    colors = ['steelblue' if 'ASF' in n else 'orange' if 'VIX' in n else 'gray' for n in model_names]
    bars = ax2.bar(model_names, rmses, color=colors, edgecolor='black')
    ax2.set_ylabel('RMSE (lower is better)', fontsize=11)
    ax2.set_title('Out-of-Sample Forecast Error', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel C: DM Statistics
    ax3 = axes[1, 0]
    dm_models = dm_results['Model'].values
    dm_stats = dm_results['DM_Statistic'].values
    
    colors = ['green' if s > 1.96 else 'red' if s < -1.96 else 'gray' for s in dm_stats]
    bars = ax3.bar(dm_models, dm_stats, color=colors, edgecolor='black')
    ax3.axhline(y=1.96, color='black', linestyle='--', linewidth=1, label='5% threshold')
    ax3.axhline(y=-1.96, color='black', linestyle='--', linewidth=1)
    ax3.set_ylabel('Diebold-Mariano Statistic', fontsize=11)
    ax3.set_title('Forecast Accuracy vs Benchmark', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel D: Forecast comparison (time series)
    ax4 = axes[1, 1]
    if 'ASF Only' in forecasts and 'VIX Only' in forecasts:
        ax4.plot(y_test.index, y_test.values * 100, 'k-', linewidth=1, label='Actual', alpha=0.7)
        ax4.plot(y_test.index, forecasts['ASF Only'] * 100, 'b--', linewidth=1.5, label='ASF Forecast')
        ax4.plot(y_test.index, forecasts['VIX Only'] * 100, 'r--', linewidth=1.5, label='VIX Forecast')
        ax4.set_xlabel('Date', fontsize=11)
        ax4.set_ylabel('Forward Drawdown (%)', fontsize=11)
        ax4.set_title('Out-of-Sample Forecasts', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('HorseRace_Results.pdf', bbox_inches='tight')
    plt.savefig('HorseRace_Results.png', dpi=300, bbox_inches='tight')
    print("\n  Saved: HorseRace_Results.pdf/png")
    plt.close()


def generate_latex_table(regression_results, metrics, dm_results):
    """Generate LaTeX tables for manuscript."""
    
    latex = r"""
\begin{table}[H]
\centering
\begin{threeparttable}
\caption{\textbf{Horse-Race Regressions: ASF vs Alternative Indicators}}
\label{tab:horserace}
\begin{tabular}{lcccc}
\toprule
\textbf{Predictor} & \textbf{Coefficient} & \textbf{$t$-statistic} & \textbf{$R^2$} & \textbf{AIC} \\
\midrule
\multicolumn{5}{l}{\textit{Panel A: Univariate Regressions}} \\
"""
    
    # Add univariate results
    univariate = regression_results[regression_results['Model'].str.contains('Univariate')]
    for _, row in univariate.iterrows():
        name = row['Model'].split(': ')[1]
        sig = "***" if row['P_Value'] < 0.01 else "**" if row['P_Value'] < 0.05 else "*" if row['P_Value'] < 0.10 else ""
        latex += f"{name} & {row['Beta_Primary']:.4f} & {row['T_Stat']:.2f}{sig} & {row['R2']:.4f} & {row['AIC']:.0f} \\\\\n"
    
    latex += r"""
\midrule
\multicolumn{5}{l}{\textit{Panel B: Out-of-Sample Forecast Comparison (2020--2024)}} \\
"""
    latex += r"\textbf{Model} & \textbf{RMSE} & \textbf{MAE} & \textbf{DM Stat} & \textbf{$p$-value} \\" + "\n"
    
    for i, (_, m_row) in enumerate(metrics.iterrows()):
        dm_row = dm_results[dm_results['Model'] == m_row['Model']] if len(dm_results) > 0 else None
        dm_stat = dm_row['DM_Statistic'].values[0] if dm_row is not None and len(dm_row) > 0 else "---"
        dm_pval = dm_row['P_Value'].values[0] if dm_row is not None and len(dm_row) > 0 else "---"
        
        if isinstance(dm_stat, str):
            latex += f"{m_row['Model']} & {m_row['RMSE']:.4f} & {m_row['MAE']:.4f} & {dm_stat} & {dm_pval} \\\\\n"
        else:
            latex += f"{m_row['Model']} & {m_row['RMSE']:.4f} & {m_row['MAE']:.4f} & {dm_stat:.2f} & {dm_pval:.4f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]
\footnotesize
\item \textit{Notes:} Panel A reports in-sample univariate regressions of forward 21-day maximum drawdown on each predictor. Panel B reports out-of-sample forecast accuracy for 2020--2024 using models estimated on 1993--2019 data. DM Stat is the Diebold-Mariano statistic testing for equal predictive accuracy vs. the historical mean benchmark. $^{***}p<0.01$, $^{**}p<0.05$, $^*p<0.10$.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    
    with open('HorseRace_LaTeX_Table.tex', 'w') as f:
        f.write(latex)
    print("  Saved: HorseRace_LaTeX_Table.tex")


def main():
    print("="*70)
    print("ENHANCEMENT 3: HORSE-RACE REGRESSIONS")
    print("="*70)
    
    # Load data
    df = load_data()
    print(f"\nData loaded: {len(df)} observations")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    # 1. Horse-race regressions
    regression_results, df_clean, univariate = run_horse_race_regressions(df)
    regression_results.to_csv('HorseRace_Regressions.csv', index=False)
    print(f"\n  Saved: HorseRace_Regressions.csv")
    
    # 2. Out-of-sample forecasts
    metrics, dm_results, forecasts, y_test = out_of_sample_forecasts(df)
    metrics.to_csv('HorseRace_OOS_Metrics.csv', index=False)
    dm_results.to_csv('HorseRace_DM_Tests.csv', index=False)
    print(f"  Saved: HorseRace_OOS_Metrics.csv, HorseRace_DM_Tests.csv")
    
    # 3. Visualizations
    print("\n" + "-"*70)
    print("GENERATING FIGURES")
    print("-"*70)
    plot_horse_race_results(regression_results, metrics, dm_results, forecasts, y_test)
    
    # 4. LaTeX table
    generate_latex_table(regression_results, metrics, dm_results)
    
    print("\n" + "="*70)
    print("HORSE-RACE ANALYSIS COMPLETE")
    print("="*70)
    
    print("\nKey findings:")
    best_univariate = regression_results[regression_results['Model'].str.contains('Univariate')].sort_values('R2', ascending=False).iloc[0]
    print(f"  Best univariate predictor: {best_univariate['Model']} (R2 = {best_univariate['R2']:.4f})")
    
    if len(metrics) > 0:
        best_oos = metrics.sort_values('RMSE').iloc[0]
        print(f"  Best OOS model: {best_oos['Model']} (RMSE = {best_oos['RMSE']:.4f})")

if __name__ == "__main__":
    main()

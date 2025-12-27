"""
ENHANCEMENT 4: Practical Implications - Strategy Backtest

This implements:
1. Regime-conditional trading strategy using ASF signals
2. VIX-based benchmark strategy for comparison
3. Sharpe ratio, max drawdown, and performance attribution
4. Welfare analysis: economic cost of missing transitions
"""

import pandas as pd
import numpy as np
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

THRESHOLD = 0.14
TRANSACTION_COST = 0.001  # 10 bps per trade

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


def compute_connectivity(df, window=63):
    """Compute mean correlation as connectivity proxy."""
    # Using ASF as inverse proxy
    df['Connectivity'] = 1 - df['ASF']
    return df


def compute_regime(df):
    """Assign regime based on connectivity threshold."""
    df['Regime'] = np.where(df['Connectivity'] > THRESHOLD, 'Coordination', 'Contagion')
    return df


class StrategyBacktester:
    """Backtest trading strategies with transaction costs."""
    
    def __init__(self, df, start_date='2000-01-01', end_date=None):
        self.df = df.copy()
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else df.index.max()
        
        # Calculate returns
        self.df['SPX_ret'] = self.df['SPX'].pct_change()
        if 'TLT' in self.df.columns:
            self.df['TLT_ret'] = self.df['TLT'].pct_change()
        
        # Filter date range
        self.df = self.df.loc[self.start_date:self.end_date]
        
        self.results = {}
    
    def run_buy_and_hold(self):
        """Benchmark: 100% SPX buy-and-hold."""
        equity_curve = (1 + self.df['SPX_ret']).cumprod()
        returns = self.df['SPX_ret']
        
        self.results['Buy & Hold SPX'] = {
            'returns': returns,
            'equity': equity_curve,
            'positions': pd.Series(1.0, index=self.df.index)
        }
        return equity_curve
    
    def run_60_40(self):
        """Benchmark: Classic 60/40 portfolio."""
        if 'TLT_ret' not in self.df.columns:
            return None
        
        returns = 0.6 * self.df['SPX_ret'] + 0.4 * self.df['TLT_ret']
        equity_curve = (1 + returns).cumprod()
        
        self.results['60/40 Portfolio'] = {
            'returns': returns,
            'equity': equity_curve,
            'positions': pd.Series(0.6, index=self.df.index)
        }
        return equity_curve
    
    def run_asf_strategy(self, high_exposure=1.0, low_exposure=0.3, smooth_window=5):
        """
        ASF-based regime-conditional strategy.
        
        Logic:
        - In Coordination regime (high connectivity): full exposure (structure stable)
        - In Contagion regime (low connectivity): reduced exposure (transition risk)
        - Use smoothed signals to reduce turnover
        """
        df = self.df.copy()
        
        # Smooth ASF signal
        df['ASF_smooth'] = df['ASF'].rolling(smooth_window).mean()
        
        # Compute fragility signal
        # Low ASF = low entropy = fragile
        fragility_threshold = df['ASF_smooth'].quantile(0.25)
        
        # Position sizing
        # When ASF is low (fragile) + high connectivity: most dangerous (coordination failure imminent)
        # When ASF is high (stable) + low connectivity: safest (diversification works)
        
        positions = []
        prev_position = high_exposure
        
        for idx, row in df.iterrows():
            asf = row['ASF_smooth'] if not pd.isna(row['ASF_smooth']) else row['ASF']
            conn = row.get('Connectivity', 1 - asf)
            
            if conn > THRESHOLD:
                # Coordination regime
                if asf < fragility_threshold:
                    # High connectivity + low entropy = imminent danger
                    target_pos = low_exposure
                else:
                    # High connectivity + high entropy = stable
                    target_pos = high_exposure
            else:
                # Contagion regime
                if asf < fragility_threshold:
                    # Low connectivity + low entropy = building fragility
                    target_pos = 0.5 * (high_exposure + low_exposure)
                else:
                    # Low connectivity + high entropy = normal
                    target_pos = 0.8 * high_exposure
            
            # Smooth position changes
            position = 0.8 * prev_position + 0.2 * target_pos
            positions.append(position)
            prev_position = position
        
        df['Position'] = positions
        
        # Calculate returns with transaction costs
        df['Position_Change'] = df['Position'].diff().abs()
        df['Transaction_Cost'] = df['Position_Change'] * TRANSACTION_COST
        
        df['Strategy_Return'] = df['Position'].shift(1) * df['SPX_ret'] - df['Transaction_Cost']
        df['Strategy_Return'] = df['Strategy_Return'].fillna(0)
        
        equity_curve = (1 + df['Strategy_Return']).cumprod()
        
        self.results['ASF Strategy'] = {
            'returns': df['Strategy_Return'],
            'equity': equity_curve,
            'positions': df['Position'],
            'turnover': df['Position_Change'].sum()
        }
        
        return equity_curve
    
    def run_vix_strategy(self, vix_threshold=0.20, high_exposure=1.0, low_exposure=0.3):
        """
        VIX-based benchmark strategy.
        
        Logic:
        - When VIX < threshold: full exposure
        - When VIX > threshold: reduced exposure
        """
        if 'VIX' not in self.df.columns:
            return None
        
        df = self.df.copy()
        
        # Position based on VIX
        df['Position'] = np.where(df['VIX'] > vix_threshold, low_exposure, high_exposure)
        
        # Smooth to reduce whipsaw
        df['Position'] = df['Position'].rolling(5).mean()
        df['Position'] = df['Position'].fillna(high_exposure)
        
        # Transaction costs
        df['Position_Change'] = df['Position'].diff().abs()
        df['Transaction_Cost'] = df['Position_Change'] * TRANSACTION_COST
        
        df['Strategy_Return'] = df['Position'].shift(1) * df['SPX_ret'] - df['Transaction_Cost']
        df['Strategy_Return'] = df['Strategy_Return'].fillna(0)
        
        equity_curve = (1 + df['Strategy_Return']).cumprod()
        
        self.results['VIX Strategy'] = {
            'returns': df['Strategy_Return'],
            'equity': equity_curve,
            'positions': df['Position'],
            'turnover': df['Position_Change'].sum()
        }
        
        return equity_curve
    
    def compute_metrics(self):
        """Compute performance metrics for all strategies."""
        metrics = []
        
        for name, data in self.results.items():
            returns = data['returns'].dropna()
            equity = data['equity'].dropna()
            
            # Annualized return
            n_years = len(returns) / 252
            total_return = equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) > 1 else 0
            cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
            
            # Volatility
            vol = returns.std() * np.sqrt(252)
            
            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe = cagr / vol if vol > 0 else 0
            
            # Max drawdown
            rolling_max = equity.expanding().max()
            drawdowns = equity / rolling_max - 1
            max_dd = drawdowns.min()
            
            # Calmar ratio
            calmar = cagr / abs(max_dd) if max_dd != 0 else 0
            
            # Turnover (if available)
            turnover = data.get('turnover', 0)
            
            metrics.append({
                'Strategy': name,
                'CAGR': cagr,
                'Volatility': vol,
                'Sharpe': sharpe,
                'Max_Drawdown': max_dd,
                'Calmar': calmar,
                'Turnover': turnover,
                'Total_Return': total_return
            })
        
        return pd.DataFrame(metrics)


def welfare_analysis(df, backtester_results):
    """
    Compute economic cost of missing regime transitions.
    
    Estimates the dollar value of avoiding drawdowns during
    coordination failures.
    """
    print("\n" + "="*70)
    print("WELFARE ANALYSIS: Economic Value of Regime Detection")
    print("="*70)
    
    # Identify major crisis periods (coordination failures)
    crises = [
        ('2008-09-01', '2009-03-31', 'Global Financial Crisis'),
        ('2020-02-15', '2020-03-31', 'COVID Crash'),
        ('2022-01-01', '2022-10-31', 'Rate Shock'),
    ]
    
    crisis_analysis = []
    
    for start, end, name in crises:
        try:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            
            # Check if period exists in data
            mask = (df.index >= start_dt) & (df.index <= end_dt)
            
            if mask.sum() == 0:
                continue
            
            crisis_df = df.loc[mask]
            
            # SPX drawdown during crisis
            spx_start = crisis_df['SPX'].iloc[0]
            spx_min = crisis_df['SPX'].min()
            crisis_dd = (spx_min / spx_start - 1) * 100
            
            # ASF level before crisis (average 30 days prior)
            pre_crisis = df.loc[:start_dt].tail(30)
            asf_before = pre_crisis['ASF'].mean() if len(pre_crisis) > 0 else np.nan
            
            crisis_analysis.append({
                'Crisis': name,
                'Period': f"{start} to {end}",
                'SPX_Drawdown': f"{crisis_dd:.1f}%",
                'ASF_Pre_Crisis': f"{asf_before:.3f}" if not np.isna(asf_before) else "N/A"
            })
            
            print(f"\n  {name}:")
            print(f"    Period: {start} to {end}")
            print(f"    Max SPX Drawdown: {crisis_dd:.1f}%")
            print(f"    ASF before crisis: {asf_before:.3f}" if not np.isna(asf_before) else "    ASF before crisis: N/A")
            
        except Exception as e:
            print(f"  Error analyzing {name}: {e}")
    
    # Compute value added by ASF strategy
    if 'ASF Strategy' in backtester_results and 'Buy & Hold SPX' in backtester_results:
        asf_ret = backtester_results['ASF Strategy']['returns']
        bh_ret = backtester_results['Buy & Hold SPX']['returns']
        
        # Excess returns
        excess = asf_ret - bh_ret
        annual_excess = excess.mean() * 252
        
        # Value of $1B portfolio
        portfolio_size = 1e9  # $1 billion
        annual_value_added = portfolio_size * annual_excess
        
        print(f"\n  Value Added (vs Buy & Hold):")
        print(f"    Annual excess return: {annual_excess*100:.2f}%")
        print(f"    Annual value on $1B: ${annual_value_added/1e6:.1f}M")
    
    return crisis_analysis


def plot_backtest_results(backtester_metrics, backtester_results, df):
    """Generate backtest visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Equity curves
    ax1 = axes[0, 0]
    for name, data in backtester_results.items():
        equity = data['equity']
        ax1.plot(equity.index, equity.values, linewidth=1.5, label=name)
    
    ax1.set_ylabel('Cumulative Return', fontsize=11)
    ax1.set_title('Strategy Equity Curves', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Panel B: Sharpe ratio comparison
    ax2 = axes[0, 1]
    strategies = backtester_metrics['Strategy'].values
    sharpes = backtester_metrics['Sharpe'].values
    
    colors = ['steelblue' if 'ASF' in s else 'orange' if 'VIX' in s else 'gray' for s in strategies]
    bars = ax2.bar(strategies, sharpes, color=colors, edgecolor='black')
    ax2.set_ylabel('Sharpe Ratio', fontsize=11)
    ax2.set_title('Risk-Adjusted Performance', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, sharpes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{val:.2f}', ha='center', fontsize=9)
    
    # Panel C: Max Drawdown comparison
    ax3 = axes[1, 0]
    max_dds = backtester_metrics['Max_Drawdown'].values * 100
    
    bars = ax3.bar(strategies, max_dds, color=colors, edgecolor='black')
    ax3.set_ylabel('Max Drawdown (%)', fontsize=11)
    ax3.set_title('Drawdown Risk', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel D: Rolling positions for ASF strategy
    ax4 = axes[1, 1]
    if 'ASF Strategy' in backtester_results:
        positions = backtester_results['ASF Strategy']['positions']
        ax4.fill_between(positions.index, 0, positions.values, alpha=0.6, color='steelblue')
        ax4.set_ylabel('Equity Exposure', fontsize=11)
        ax4.set_xlabel('Date', fontsize=11)
        ax4.set_title('ASF Strategy Position Over Time', fontsize=12, fontweight='bold')
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3)
        
        # Mark crisis periods
        crises = [('2008-09-15', 'GFC'), ('2020-03-12', 'COVID')]
        for date_str, label in crises:
            try:
                date = pd.to_datetime(date_str)
                if date >= positions.index.min() and date <= positions.index.max():
                    ax4.axvline(x=date, color='red', linestyle='--', alpha=0.7)
            except:
                pass
    
    plt.tight_layout()
    plt.savefig('Strategy_Backtest.pdf', bbox_inches='tight')
    plt.savefig('Strategy_Backtest.png', dpi=300, bbox_inches='tight')
    print("\n  Saved: Strategy_Backtest.pdf/png")
    plt.close()


def generate_latex_table(metrics):
    """Generate LaTeX table for strategy comparison."""
    
    latex = r"""
\begin{table}[H]
\centering
\begin{threeparttable}
\caption{\textbf{Strategy Backtest: ASF Signals vs Benchmarks}}
\label{tab:strategy}
\begin{tabular}{lcccccc}
\toprule
\textbf{Strategy} & \textbf{CAGR} & \textbf{Volatility} & \textbf{Sharpe} & \textbf{Max DD} & \textbf{Calmar} \\
\midrule
"""
    
    for _, row in metrics.iterrows():
        latex += f"{row['Strategy']} & {row['CAGR']*100:.1f}\\% & {row['Volatility']*100:.1f}\\% & {row['Sharpe']:.2f} & {row['Max_Drawdown']*100:.1f}\\% & {row['Calmar']:.2f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]
\footnotesize
\item \textit{Notes:} Backtest period 2000--2024. CAGR = Compound Annual Growth Rate. Max DD = Maximum Drawdown. Transaction costs of 10 bps per trade included. ASF Strategy adjusts equity exposure based on structural fragility signals; VIX Strategy uses volatility-based timing.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    
    with open('Strategy_LaTeX_Table.tex', 'w') as f:
        f.write(latex)
    print("  Saved: Strategy_LaTeX_Table.tex")


def main():
    print("="*70)
    print("ENHANCEMENT 4: PRACTICAL IMPLICATIONS - STRATEGY BACKTEST")
    print("="*70)
    
    # Load data
    df = load_data()
    df = compute_connectivity(df)
    df = compute_regime(df)
    
    print(f"\nData loaded: {len(df)} observations")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    # Initialize backtester
    backtester = StrategyBacktester(df, start_date='2000-01-01')
    
    # Run strategies
    print("\n" + "-"*70)
    print("RUNNING STRATEGY BACKTESTS")
    print("-"*70)
    
    print("\n  Running: Buy & Hold SPX...")
    backtester.run_buy_and_hold()
    
    print("  Running: 60/40 Portfolio...")
    backtester.run_60_40()
    
    print("  Running: ASF Strategy...")
    backtester.run_asf_strategy()
    
    print("  Running: VIX Strategy...")
    backtester.run_vix_strategy()
    
    # Compute metrics
    metrics = backtester.compute_metrics()
    
    print("\n" + "-"*70)
    print("PERFORMANCE SUMMARY")
    print("-"*70)
    
    print(f"\n  {'Strategy':<20} {'CAGR':>10} {'Vol':>10} {'Sharpe':>10} {'Max DD':>10}")
    print("  " + "-"*60)
    
    for _, row in metrics.iterrows():
        print(f"  {row['Strategy']:<20} {row['CAGR']*100:>9.1f}% {row['Volatility']*100:>9.1f}% {row['Sharpe']:>10.2f} {row['Max_Drawdown']*100:>9.1f}%")
    
    # Save metrics
    metrics.to_csv('Strategy_Backtest_Metrics.csv', index=False)
    print(f"\n  Saved: Strategy_Backtest_Metrics.csv")
    
    # Welfare analysis
    crisis_analysis = welfare_analysis(df, backtester.results)
    
    # Generate visualizations
    print("\n" + "-"*70)
    print("GENERATING FIGURES")
    print("-"*70)
    plot_backtest_results(metrics, backtester.results, df)
    
    # Generate LaTeX
    generate_latex_table(metrics)
    
    print("\n" + "="*70)
    print("STRATEGY BACKTEST COMPLETE")
    print("="*70)
    
    # Key findings
    asf_row = metrics[metrics['Strategy'] == 'ASF Strategy']
    bh_row = metrics[metrics['Strategy'] == 'Buy & Hold SPX']
    
    if len(asf_row) > 0 and len(bh_row) > 0:
        print("\nKey findings:")
        print(f"  ASF Strategy Sharpe:     {asf_row['Sharpe'].values[0]:.2f}")
        print(f"  Buy & Hold Sharpe:       {bh_row['Sharpe'].values[0]:.2f}")
        print(f"  Sharpe improvement:      {asf_row['Sharpe'].values[0] - bh_row['Sharpe'].values[0]:.2f}")
        print(f"  Max DD reduction:        {(asf_row['Max_Drawdown'].values[0] - bh_row['Max_Drawdown'].values[0])*100:.1f}%")

if __name__ == "__main__":
    main()

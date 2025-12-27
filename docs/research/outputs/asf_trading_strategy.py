"""
ASF Trading Strategy v2 - Evidence-Based Kelly Allocation
==========================================================
Uses Kelly Criterion and empirical regime-conditional returns to optimize allocations

Key improvements:
1. Estimates expected returns and volatility PER REGIME and ASF level
2. Uses fractional Kelly (half-Kelly for safety)
3. Computes regime-conditional Sharpe ratios
4. Dynamic allocation based on measured edge, not heuristics
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_dir': 'coodination_data',
    'output_dir': 'paper_trading',
    
    # ASF Parameters
    'entropy_window': 63,
    'theta': 0.995,
    'regime_threshold': 0.14,
    
    # Kelly Parameters
    'kelly_fraction': 0.5,  # Half-Kelly for safety
    'max_leverage': 1.0,    # No leverage allowed
    'min_weight': 0.05,     # Minimum allocation per asset class
    'risk_free_rate': 0.04, # Annual risk-free rate
    
    # Portfolio
    'initial_capital': 100000,
    'lookback_years': 5,    # Years of data for parameter estimation
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_prices(data_dir):
    """Load all available price data"""
    prices_dict = {}
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for f in files:
        try:
            filepath = os.path.join(data_dir, f)
            df = pd.read_csv(filepath, parse_dates=['date'])
            df.set_index('date', inplace=True)
            df = df.sort_index()
            col = [c for c in df.columns if c != 'date'][0]
            prices_dict[col] = df[col]
        except:
            pass
    
    prices = pd.DataFrame(prices_dict).sort_index()
    prices = prices.loc['2007-01-01':]
    prices = prices.dropna(axis=1, thresh=int(len(prices)*0.5))
    prices = prices.ffill().dropna()
    
    return prices


def categorize_assets(columns):
    """Categorize assets into classes"""
    equity = ['S&P_500', 'Dow_Jones', 'NASDAQ', 'Russell_2000']
    bonds = ['AGG', 'BND', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'MBB', 'MUB', 'TIP', 'Treasuries_20Y']
    commodities = ['GLD', 'SLV', 'Gold', 'Oil', 'DBC', 'DBA', 'DBB', 'USO', 'UNG', 'PALL', 'PPLT']
    crypto = ['BTC_USD']
    fx = ['Euro_USD']
    
    categories = {}
    for col in columns:
        if col in equity:
            categories[col] = 'equity'
        elif col in bonds:
            categories[col] = 'bonds'
        elif col in commodities:
            categories[col] = 'commodities'
        elif col in crypto:
            categories[col] = 'crypto'
        elif col in fx:
            categories[col] = 'fx'
        else:
            categories[col] = 'other'
    
    return categories


# ============================================================================
# SIGNAL COMPUTATION
# ============================================================================

def compute_signals(prices, window=63, theta=0.995, threshold=0.14):
    """Compute ASF, connectivity, and regime signals"""
    returns = prices.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], 0)
    
    entropy_list, connectivity_list, dates = [], [], []
    
    for i in range(window, len(returns)):
        w = returns.iloc[i-window:i]
        try:
            corr = w.corr().values
            if np.any(np.isnan(corr)):
                entropy_list.append(np.nan)
                connectivity_list.append(np.nan)
            else:
                eig = np.linalg.eigvalsh(corr)
                eig = eig[eig > 1e-10]
                p = eig / eig.sum()
                H = -np.sum(p * np.log(p)) / np.log(len(p))
                entropy_list.append(H)
                
                mask = ~np.eye(len(corr), dtype=bool)
                connectivity_list.append(corr[mask].mean())
        except:
            entropy_list.append(np.nan)
            connectivity_list.append(np.nan)
        dates.append(returns.index[i])
    
    entropy = pd.Series(entropy_list, index=dates)
    connectivity = pd.Series(connectivity_list, index=dates)
    asf = (1 - entropy).ewm(alpha=1-theta, adjust=False).mean()
    regime = pd.Series(np.where(connectivity < threshold, 'contagion', 'coordination'), index=dates)
    
    return pd.DataFrame({
        'entropy': entropy,
        'connectivity': connectivity,
        'asf': asf,
        'regime': regime
    }).dropna()


# ============================================================================
# KELLY CRITERION COMPUTATION
# ============================================================================

def compute_kelly_weights(returns, risk_free=0.04/252):
    """
    Compute Kelly-optimal weights using mean-variance approximation
    
    For a single asset: f* = (mu - rf) / sigma^2
    For portfolio: f* = Sigma^{-1} * (mu - rf)
    
    We use fractional Kelly for safety
    """
    mu = returns.mean()
    sigma = returns.std()
    excess_return = mu - risk_free
    
    # Kelly weight for each asset (simplified single-asset Kelly)
    kelly_raw = excess_return / (sigma ** 2)
    
    # Clip to reasonable range
    kelly_raw = kelly_raw.clip(-2, 2)
    
    return kelly_raw


def estimate_regime_parameters(returns, signals, asset_categories):
    """
    Estimate expected returns and volatility for each regime and ASF bucket
    Returns evidence-based parameters for Kelly allocation
    """
    # Align data
    common_idx = returns.index.intersection(signals.index)
    returns = returns.loc[common_idx]
    signals = signals.loc[common_idx]
    
    # Create ASF buckets
    signals['asf_bucket'] = pd.cut(signals['asf'], 
                                    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                    labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    # Compute returns by asset class
    class_returns = {}
    for asset_class in ['equity', 'bonds', 'commodities']:
        cols = [c for c, cat in asset_categories.items() if cat == asset_class and c in returns.columns]
        if cols:
            class_returns[asset_class] = returns[cols].mean(axis=1)
    
    class_returns_df = pd.DataFrame(class_returns)
    
    # Estimate parameters by regime
    results = {}
    
    for regime in ['contagion', 'coordination']:
        regime_mask = signals['regime'] == regime
        results[regime] = {}
        
        for asf_bucket in ['very_low', 'low', 'medium', 'high', 'very_high']:
            bucket_mask = signals['asf_bucket'] == asf_bucket
            combined_mask = regime_mask & bucket_mask
            
            if combined_mask.sum() < 30:  # Need minimum data
                continue
            
            bucket_returns = class_returns_df.loc[combined_mask]
            
            results[regime][asf_bucket] = {}
            for asset_class in class_returns_df.columns:
                r = bucket_returns[asset_class]
                
                # Annualized stats
                mu = r.mean() * 252
                sigma = r.std() * np.sqrt(252)
                sharpe = mu / sigma if sigma > 0 else 0
                
                # Forward returns (predictive power)
                fwd_5d = class_returns_df[asset_class].shift(-5).loc[combined_mask].mean() * 252
                fwd_21d = class_returns_df[asset_class].shift(-21).loc[combined_mask].mean() * 252
                
                results[regime][asf_bucket][asset_class] = {
                    'n_obs': len(r),
                    'mean_annual': mu,
                    'volatility': sigma,
                    'sharpe': sharpe,
                    'fwd_5d_return': fwd_5d,
                    'fwd_21d_return': fwd_21d,
                }
    
    return results


def compute_optimal_kelly_allocation(regime_params, current_regime, current_asf_bucket, 
                                      kelly_fraction=0.5, max_leverage=1.0, rf=0.04):
    """
    Compute optimal Kelly allocation based on empirical regime parameters
    """
    if current_regime not in regime_params:
        return None
    
    if current_asf_bucket not in regime_params[current_regime]:
        # Fall back to nearest bucket
        available = list(regime_params[current_regime].keys())
        if not available:
            return None
        current_asf_bucket = available[0]
    
    params = regime_params[current_regime][current_asf_bucket]
    
    # Calculate Kelly weight for each asset class
    kelly_weights = {}
    for asset_class, p in params.items():
        mu = p['mean_annual']
        sigma = p['volatility']
        
        if sigma > 0:
            # Kelly criterion: f* = (mu - rf) / sigma^2
            f_star = (mu - rf) / (sigma ** 2)
            
            # Apply fractional Kelly
            f_kelly = f_star * kelly_fraction
            
            # Clip to reasonable bounds
            f_kelly = max(0, min(f_kelly, 1.0))  # Long only, max 100%
            
            kelly_weights[asset_class] = f_kelly
        else:
            kelly_weights[asset_class] = 0
    
    # Normalize to sum to max_leverage
    total = sum(kelly_weights.values())
    if total > max_leverage:
        factor = max_leverage / total
        kelly_weights = {k: v * factor for k, v in kelly_weights.items()}
    
    # Add cash as remainder
    invested = sum(kelly_weights.values())
    kelly_weights['cash'] = max(0, 1.0 - invested)
    
    return kelly_weights


# ============================================================================
# MAIN STRATEGY
# ============================================================================

def run_kelly_strategy():
    """Main strategy execution with Kelly-optimal allocations"""
    
    print("=" * 80)
    print("ASF TRADING STRATEGY v2 - EVIDENCE-BASED KELLY ALLOCATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    
    # Load data
    print("\n[1] Loading price data...")
    prices = load_all_prices(CONFIG['data_dir'])
    asset_categories = categorize_assets(prices.columns)
    print(f"    Loaded {len(prices.columns)} assets")
    
    # Compute signals
    print("\n[2] Computing signals...")
    signals = compute_signals(prices, CONFIG['entropy_window'], 
                               CONFIG['theta'], CONFIG['regime_threshold'])
    
    latest = signals.iloc[-1]
    print(f"    Connectivity: {latest['connectivity']:.4f}")
    print(f"    ASF:          {latest['asf']:.4f}")
    print(f"    Regime:       {latest['regime'].upper()}")
    
    # Estimate regime parameters from historical data
    print("\n[3] Estimating regime-conditional parameters...")
    returns = prices.pct_change().dropna()
    regime_params = estimate_regime_parameters(returns, signals, asset_categories)
    
    # Display evidence
    print("\n" + "=" * 80)
    print("EMPIRICAL EVIDENCE BY REGIME")
    print("=" * 80)
    
    for regime in ['contagion', 'coordination']:
        print(f"\n{regime.upper()} REGIME:")
        print("-" * 60)
        
        if regime not in regime_params:
            print("  Insufficient data")
            continue
        
        for bucket, assets in regime_params[regime].items():
            print(f"\n  ASF Bucket: {bucket}")
            print(f"  {'Asset Class':<15} {'N Obs':>8} {'Ann.Ret':>10} {'Vol':>10} {'Sharpe':>8} {'Fwd21d':>10}")
            print(f"  {'-'*60}")
            
            for asset_class, p in assets.items():
                print(f"  {asset_class:<15} {p['n_obs']:>8} {p['mean_annual']*100:>9.1f}% {p['volatility']*100:>9.1f}% {p['sharpe']:>8.2f} {p['fwd_21d_return']*100:>9.1f}%")
    
    # Compute Kelly allocation
    print("\n" + "=" * 80)
    print("KELLY-OPTIMAL ALLOCATION")
    print("=" * 80)
    
    # Map current ASF to bucket
    current_asf = latest['asf']
    if current_asf < 0.2:
        asf_bucket = 'very_low'
    elif current_asf < 0.4:
        asf_bucket = 'low'
    elif current_asf < 0.6:
        asf_bucket = 'medium'
    elif current_asf < 0.8:
        asf_bucket = 'high'
    else:
        asf_bucket = 'very_high'
    
    print(f"\nCurrent State:")
    print(f"  Regime: {latest['regime'].upper()}")
    print(f"  ASF: {current_asf:.4f} (bucket: {asf_bucket})")
    
    kelly_weights = compute_optimal_kelly_allocation(
        regime_params, 
        latest['regime'], 
        asf_bucket,
        kelly_fraction=CONFIG['kelly_fraction'],
        max_leverage=CONFIG['max_leverage'],
        rf=CONFIG['risk_free_rate']
    )
    
    if kelly_weights:
        print(f"\nKelly-Optimal Weights (using {CONFIG['kelly_fraction']*100:.0f}% Kelly):")
        print(f"  {'Asset Class':<15} {'Weight':>10}")
        print(f"  {'-'*25}")
        for asset_class, weight in sorted(kelly_weights.items(), key=lambda x: -x[1]):
            print(f"  {asset_class:<15} {weight*100:>9.1f}%")
        
        # Show the reasoning
        print(f"\nRationale (from empirical data):")
        if latest['regime'] in regime_params and asf_bucket in regime_params[latest['regime']]:
            for asset_class, p in regime_params[latest['regime']][asf_bucket].items():
                edge = p['mean_annual'] - CONFIG['risk_free_rate']
                print(f"  {asset_class}: Expected excess return = {edge*100:.1f}%, Vol = {p['volatility']*100:.1f}%")
                print(f"           Kelly = ({edge:.3f}) / ({p['volatility']:.3f})^2 = {edge/(p['volatility']**2) if p['volatility'] > 0 else 0:.2f}")
    
    # Save state
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    state = {
        'date': datetime.now().isoformat(),
        'data_date': str(prices.index[-1].date()),
        'signals': {
            'entropy': float(latest['entropy']),
            'connectivity': float(latest['connectivity']),
            'asf': float(latest['asf']),
            'regime': latest['regime'],
            'asf_bucket': asf_bucket
        },
        'kelly_allocation': kelly_weights,
        'methodology': {
            'kelly_fraction': CONFIG['kelly_fraction'],
            'max_leverage': CONFIG['max_leverage'],
            'risk_free_rate': CONFIG['risk_free_rate'],
            'description': 'Half-Kelly based on regime-conditional expected returns and volatility'
        }
    }
    
    with open(os.path.join(CONFIG['output_dir'], 'current_state.json'), 'w') as f:
        json.dump(state, f, indent=2)
    
    # Backtest Kelly strategy
    print("\n" + "=" * 80)
    print("BACKTEST: KELLY STRATEGY vs STATIC 60/40")
    print("=" * 80)
    
    backtest_kelly_strategy(prices, signals, regime_params, asset_categories)
    
    return state


def backtest_kelly_strategy(prices, signals, regime_params, asset_categories):
    """Backtest the Kelly strategy vs benchmark"""
    
    returns = prices.pct_change().dropna()
    
    # Get class returns
    class_returns = {}
    for asset_class in ['equity', 'bonds', 'commodities']:
        cols = [c for c, cat in asset_categories.items() if cat == asset_class and c in returns.columns]
        if cols:
            class_returns[asset_class] = returns[cols].mean(axis=1)
    
    class_returns_df = pd.DataFrame(class_returns)
    
    # Align
    common_idx = class_returns_df.index.intersection(signals.index)
    class_returns_df = class_returns_df.loc[common_idx]
    signals = signals.loc[common_idx]
    
    # Strategy returns
    strategy_rets = []
    benchmark_rets = []
    
    # Map ASF to bucket
    def asf_to_bucket(asf):
        if asf < 0.2: return 'very_low'
        elif asf < 0.4: return 'low'
        elif asf < 0.6: return 'medium'
        elif asf < 0.8: return 'high'
        else: return 'very_high'
    
    for date in signals.index:
        regime = signals.loc[date, 'regime']
        asf = signals.loc[date, 'asf']
        bucket = asf_to_bucket(asf)
        
        # Get Kelly weights
        weights = compute_optimal_kelly_allocation(
            regime_params, regime, bucket,
            kelly_fraction=CONFIG['kelly_fraction'],
            max_leverage=CONFIG['max_leverage'],
            rf=CONFIG['risk_free_rate']
        )
        
        if weights is None:
            weights = {'equity': 0.6, 'bonds': 0.4, 'commodities': 0.0, 'cash': 0.0}
        
        # Compute strategy return
        strat_ret = 0
        for asset_class in class_returns_df.columns:
            if asset_class in weights:
                strat_ret += weights.get(asset_class, 0) * class_returns_df.loc[date, asset_class]
        strategy_rets.append(strat_ret)
        
        # Benchmark (60/40)
        eq_ret = class_returns_df.loc[date, 'equity'] if 'equity' in class_returns_df.columns else 0
        bd_ret = class_returns_df.loc[date, 'bonds'] if 'bonds' in class_returns_df.columns else 0
        bench_ret = 0.6 * eq_ret + 0.4 * bd_ret
        benchmark_rets.append(bench_ret)
    
    strategy_rets = pd.Series(strategy_rets, index=signals.index)
    benchmark_rets = pd.Series(benchmark_rets, index=signals.index)
    
    # Metrics
    strat_cum = (1 + strategy_rets).cumprod()
    bench_cum = (1 + benchmark_rets).cumprod()
    
    strat_total = (strat_cum.iloc[-1] - 1) * 100
    bench_total = (bench_cum.iloc[-1] - 1) * 100
    
    strat_annual = (strat_cum.iloc[-1] ** (252 / len(strat_cum)) - 1) * 100
    bench_annual = (bench_cum.iloc[-1] ** (252 / len(bench_cum)) - 1) * 100
    
    strat_vol = strategy_rets.std() * np.sqrt(252) * 100
    bench_vol = benchmark_rets.std() * np.sqrt(252) * 100
    
    strat_sharpe = (strat_annual/100 - CONFIG['risk_free_rate']) / (strat_vol/100) if strat_vol > 0 else 0
    bench_sharpe = (bench_annual/100 - CONFIG['risk_free_rate']) / (bench_vol/100) if bench_vol > 0 else 0
    
    strat_dd = ((strat_cum.expanding().max() - strat_cum) / strat_cum.expanding().max()).max() * 100
    bench_dd = ((bench_cum.expanding().max() - bench_cum) / bench_cum.expanding().max()).max() * 100
    
    print(f"""
Period: {signals.index[0].date()} to {signals.index[-1].date()} ({len(signals)} days)

                        Kelly Strategy    60/40 Benchmark
                        --------------    ---------------
Total Return:           {strat_total:>10.1f}%       {bench_total:>10.1f}%
Annualized Return:      {strat_annual:>10.1f}%       {bench_annual:>10.1f}%
Volatility:             {strat_vol:>10.1f}%       {bench_vol:>10.1f}%
Sharpe Ratio:           {strat_sharpe:>10.2f}        {bench_sharpe:>10.2f}
Max Drawdown:           {strat_dd:>10.1f}%       {bench_dd:>10.1f}%
""")
    
    if strat_sharpe > bench_sharpe:
        print(">>> KELLY STRATEGY HAS HIGHER SHARPE RATIO <<<")
    if strat_dd < bench_dd:
        print(">>> KELLY STRATEGY HAS LOWER MAX DRAWDOWN <<<")
    
    return strategy_rets, benchmark_rets


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    state = run_kelly_strategy()
    
    print("\n" + "=" * 80)
    print("EVIDENCE-BASED ALLOCATION COMPLETE")
    print("=" * 80)
    print(f"""
The allocations above are computed using:
1. Historical returns BY REGIME (contagion vs coordination)
2. Historical returns BY ASF LEVEL (very_low to very_high)
3. Kelly Criterion: f* = (expected excess return) / variance
4. Half-Kelly for safety ({CONFIG['kelly_fraction']*100:.0f}% of optimal)

This is not heuristic - it's based on measured edge in each state.
""")

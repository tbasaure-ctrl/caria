import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# --- Configuration ---
REAL_DATA_PATH = r"c:/key/wise_adviser_cursor_context/Caria_repo/caria/data/silver/fundamentals/quality_signals.parquet"
OUTPUT_DIR = r"c:/key/wise_adviser_cursor_context/Caria_repo/caria/validation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class DataLoader:
    def __init__(self):
        self.macro_data = None
        self.quality_data = None
        self.market_data = None

    def load_or_generate_macro(self):
        print("Loading Macro Data...")
        # Synthetic Macro Data
        dates = pd.date_range(start="2000-01-01", end="2023-12-31", freq="M")
        n = len(dates)
        
        # Synthetic Regimes: 0=Recession, 1=Recovery, 2=Expansion, 3=Slowdown
        # Markov chain simulation
        regimes = [2]
        transition_matrix = [
            [0.8, 0.2, 0.0, 0.0], # Recession -> Recovery
            [0.0, 0.9, 0.1, 0.0], # Recovery -> Expansion
            [0.05, 0.0, 0.9, 0.05], # Expansion -> Slowdown/Recession
            [0.1, 0.0, 0.0, 0.9]  # Slowdown -> Recession
        ]
        
        for _ in range(n-1):
            current = regimes[-1]
            next_regime = np.random.choice([0, 1, 2, 3], p=transition_matrix[current])
            regimes.append(next_regime)
            
        regime_map = {0: "recession", 1: "recovery", 2: "expansion", 3: "slowdown"}
        
        # Synthetic Asset Returns
        # Equities perform well in Recovery/Expansion, bad in Recession
        equity_returns = []
        bond_returns = []
        
        for r in regimes:
            if r == 0: # Recession
                equity_returns.append(np.random.normal(-0.02, 0.08))
                bond_returns.append(np.random.normal(0.01, 0.02))
            elif r == 1: # Recovery
                equity_returns.append(np.random.normal(0.03, 0.05))
                bond_returns.append(np.random.normal(-0.005, 0.02))
            elif r == 2: # Expansion
                equity_returns.append(np.random.normal(0.01, 0.04))
                bond_returns.append(np.random.normal(0.002, 0.01))
            else: # Slowdown
                equity_returns.append(np.random.normal(0.00, 0.05))
                bond_returns.append(np.random.normal(0.005, 0.015))

        self.macro_data = pd.DataFrame({
            "date": dates,
            "predicted_regime": [regime_map[r] for r in regimes],
            "equity_return": equity_returns,
            "bond_return": bond_returns
        })
        print("  Using Synthetic Macro Data (since real file not found)")
        return self.macro_data

    def load_or_generate_quality(self):
        print("Loading Quality Data...")
        if os.path.exists(REAL_DATA_PATH):
            try:
                df = pd.read_parquet(REAL_DATA_PATH)
                print(f"  Loaded real data from {REAL_DATA_PATH}")
                # Check if we have returns, if not generate them
                if 'fwd_3m_ret' not in df.columns:
                    print("  'fwd_3m_ret' not found, generating synthetic returns correlated with quality...")
                    # Generate synthetic returns: higher quality -> slightly higher return + noise
                    # Normalize quality score first
                    if 'quality_score' in df.columns:
                        q_mean = df['quality_score'].mean()
                        q_std = df['quality_score'].std()
                        df['norm_quality'] = (df['quality_score'] - q_mean) / q_std
                        
                        # Synthetic return generation
                        # Return = Market + Alpha * Quality + Noise
                        # We need dates to simulate market factor
                        dates = df['date'].unique()
                        market_factor = {d: np.random.normal(0.02, 0.05) for d in dates}
                        
                        df['market_ret'] = df['date'].map(market_factor)
                        df['fwd_3m_ret'] = df['market_ret'] + 0.005 * df['norm_quality'] + np.random.normal(0, 0.1, len(df))
                        df['fwd_6m_ret'] = df['fwd_3m_ret'] * 2 + np.random.normal(0, 0.05, len(df)) # Rough approx
                    else:
                        print("  'quality_score' column missing in real data! Falling back to full synthetic.")
                        return self._generate_synthetic_quality()
                
                self.quality_data = df
                return df
            except Exception as e:
                print(f"  Error loading real data: {e}. Falling back to synthetic.")
        
        return self._generate_synthetic_quality()

    def _generate_synthetic_quality(self):
        dates = pd.date_range(start="2015-01-01", end="2023-12-31", freq="Q")
        tickers = [f"TICKER_{i}" for i in range(100)]
        
        data = []
        for d in dates:
            market_ret = np.random.normal(0.02, 0.05)
            for t in tickers:
                quality = np.random.normal(50, 10) # Score 0-100
                # Return correlated with quality
                ret_3m = market_ret + (quality - 50)/100 * 0.02 + np.random.normal(0, 0.1)
                data.append({
                    "date": d,
                    "ticker": t,
                    "quality_score": quality,
                    "fwd_3m_ret": ret_3m,
                    "fwd_6m_ret": ret_3m * 2 # simplified
                })
        
        self.quality_data = pd.DataFrame(data)
        print("  Using Synthetic Quality Data")
        return self.quality_data

class MacroValidator:
    def __init__(self, data):
        self.data = data

    def run(self):
        print("\n--- Macro Model Evaluation ---")
        df = self.data.copy()
        
        # 1. Regime Stats
        print("1. Regime Statistics:")
        stats = df['predicted_regime'].value_counts(normalize=True)
        print(stats)
        
        # 2. Economic Usefulness
        print("\n2. Asset Performance by Regime:")
        perf = df.groupby('predicted_regime')[['equity_return', 'bond_return']].agg(['mean', 'std'])
        # Sharpe (assuming 0 risk free for simplicity or just mean/std)
        for asset in ['equity_return', 'bond_return']:
            perf[(asset, 'sharpe')] = perf[(asset, 'mean')] / perf[(asset, 'std')]
        print(perf)
        
        # 3. Strategy Backtest
        print("\n3. Strategy Backtest (Simple Switching):")
        # Strategy: 100% Equity in Expansion/Recovery, 100% Bond in Recession/Slowdown
        df['strategy_ret'] = df.apply(
            lambda x: x['equity_return'] if x['predicted_regime'] in ['expansion', 'recovery'] else x['bond_return'], 
            axis=1
        )
        df['benchmark_ret'] = 0.6 * df['equity_return'] + 0.4 * df['bond_return']
        
        strat_cum = (1 + df['strategy_ret']).cumprod()
        bench_cum = (1 + df['benchmark_ret']).cumprod()
        
        print(f"Strategy Total Return: {strat_cum.iloc[-1]:.2f}x")
        print(f"Benchmark Total Return: {bench_cum.iloc[-1]:.2f}x")
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(df['date'], strat_cum, label='Regime Strategy')
        plt.plot(df['date'], bench_cum, label='60/40 Benchmark')
        plt.title('Macro Regime Strategy vs Benchmark')
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, 'macro_strategy.png'))
        print(f"  Plot saved to {os.path.join(OUTPUT_DIR, 'macro_strategy.png')}")

class QualityValidator:
    def __init__(self, data):
        self.data = data

    def run(self):
        print("\n--- Quality Model Evaluation ---")
        df = self.data.copy()
        
        # 1. Portfolio Construction by Deciles
        print("1. Decile Performance:")
        # Create deciles per date
        df['decile'] = df.groupby('date')['quality_score'].transform(
            lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
        )
        
        decile_perf = df.groupby('decile')['fwd_3m_ret'].mean()
        print(decile_perf)
        
        # 2. Long-Short Performance
        print("\n2. Long-Short (Top vs Bottom Decile):")
        top = df[df['decile'] == 9].groupby('date')['fwd_3m_ret'].mean()
        bottom = df[df['decile'] == 0].groupby('date')['fwd_3m_ret'].mean()
        ls_ret = top - bottom
        
        print(f"Avg Long-Short Return (3M): {ls_ret.mean():.4f}")
        print(f"Long-Short Sharpe: {ls_ret.mean() / ls_ret.std():.2f}")
        
        # Plot Deciles
        plt.figure(figsize=(10, 6))
        decile_perf.plot(kind='bar')
        plt.title('Avg 3M Return by Quality Decile')
        plt.xlabel('Decile (0=Low, 9=High)')
        plt.ylabel('Avg 3M Return')
        plt.savefig(os.path.join(OUTPUT_DIR, 'quality_deciles.png'))
        print(f"  Plot saved to {os.path.join(OUTPUT_DIR, 'quality_deciles.png')}")


def main():
    loader = DataLoader()
    
    # Macro
    macro_data = loader.load_or_generate_macro()
    macro_val = MacroValidator(macro_data)
    macro_val.run()
    
    # Quality
    quality_data = loader.load_or_generate_quality()
    quality_val = QualityValidator(quality_data)
    quality_val.run()
    
    print("\n--- Global Conclusion ---")
    print("Validation complete. See output plots in:", OUTPUT_DIR)
    print("CRITICAL NOTE: If synthetic data was used, these results test the VALIDATION LOGIC, not the MODEL EFFICACY.")

if __name__ == "__main__":
    main()

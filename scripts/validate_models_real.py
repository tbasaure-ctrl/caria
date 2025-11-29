import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle
import gc
from datetime import datetime, timedelta

# Add caria-lib to path
sys.path.append(r"c:/key/wise_adviser_cursor_context/Caria_repo/caria/caria-lib")
try:
    from caria.models.regime.hmm_regime_detector import HMMRegimeDetector
except ImportError:
    print("Could not import HMMRegimeDetector. Make sure hmmlearn is installed.")
    pass

# --- Configuration ---
REPO_ROOT = r"c:/key/wise_adviser_cursor_context/Caria_repo/caria"
MACRO_DATA_PATH = os.path.join(REPO_ROOT, "silver/macro/fred_us.parquet")
MODEL_PATH = os.path.join(REPO_ROOT, "caria_data/models/regime_hmm_model.pkl")
QUALITY_DATA_PATH = os.path.join(REPO_ROOT, "data/silver/fundamentals/quality_signals.parquet")
VALUE_DATA_PATH = os.path.join(REPO_ROOT, "data/silver/fundamentals/value_signals.parquet")
PRICES_DATA_PATH = os.path.join(REPO_ROOT, "silver/market/stock_prices_daily.parquet")
OUTPUT_DIR = os.path.join(REPO_ROOT, "validation_results_real")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class DataLoader:
    def load_macro_and_predict(self):
        print("Loading Macro Data & Model...")
        if not os.path.exists(MACRO_DATA_PATH):
            print(f"Macro data not found at {MACRO_DATA_PATH}")
            return None
        
        try:
            df_macro = pd.read_parquet(MACRO_DATA_PATH)
            print(f"  Loaded {len(df_macro)} macro records.")
            
            if not os.path.exists(MODEL_PATH):
                print(f"Model not found at {MODEL_PATH}")
                return None
                
            detector = HMMRegimeDetector.load(MODEL_PATH)
            print("  HMM Model loaded.")
            
            if 'date' not in df_macro.columns and isinstance(df_macro.index, pd.DatetimeIndex):
                df_macro = df_macro.reset_index()
                df_macro.rename(columns={'index': 'date'}, inplace=True)
            
            regimes = detector.predict_historical_regimes(df_macro)
            print("  Regimes predicted.")
            
            result = pd.merge(df_macro, regimes, on='date', how='inner')
            return result
        except Exception as e:
            print(f"  Error in macro prediction: {e}")
            return None

    def load_quality_and_returns(self):
        print("Loading Quality & Price Data...")
        if not os.path.exists(QUALITY_DATA_PATH):
            print("Quality data not found.")
            return None, None
        
        q_df = pd.read_parquet(QUALITY_DATA_PATH)
        print(f"  Loaded {len(q_df)} quality signals.")
        print(f"  Quality columns: {q_df.columns.tolist()}")
        
        # Normalize ticker col
        if 'symbol' in q_df.columns: q_df.rename(columns={'symbol': 'ticker'}, inplace=True)
        
        # Calculate quality_score if missing
        if 'quality_score' not in q_df.columns:
            print("  'quality_score' missing. Calculating from signals...")
            # Use available signals: roic, returnOnEquity, freeCashFlowYield
            signals = []
            if 'roic' in q_df.columns: signals.append('roic')
            if 'returnOnEquity' in q_df.columns: signals.append('returnOnEquity')
            if 'freeCashFlowYield' in q_df.columns: signals.append('freeCashFlowYield')
            
            if not signals:
                print("  ERROR: No quality signals found to calculate score!")
                return None, None
            
            # Simple Z-score average
            for col in signals:
                # Winsorize to handle outliers
                q_df[col] = q_df[col].clip(lower=q_df[col].quantile(0.01), upper=q_df[col].quantile(0.99))
                # Z-score per date (cross-sectional)
                q_df[f'z_{col}'] = q_df.groupby('date')[col].transform(lambda x: (x - x.mean()) / x.std())
            
            q_df['quality_score'] = q_df[[f'z_{s}' for s in signals]].mean(axis=1)
            # Scale to 0-100
            q_df['quality_score'] = (q_df['quality_score'] - q_df['quality_score'].min()) / (q_df['quality_score'].max() - q_df['quality_score'].min()) * 100
            print("  Calculated quality_score.")

        unique_tickers = q_df['ticker'].unique()
        print(f"  Unique tickers in quality data: {len(unique_tickers)}")
        
        if not os.path.exists(PRICES_DATA_PATH):
            print("Price data not found.")
            return None, None
            
        print("  Loading prices (filtered)...")
        # Read prices
        p_df = pd.read_parquet(PRICES_DATA_PATH)
        
        # Normalize ticker col
        if 'symbol' in p_df.columns: p_df.rename(columns={'symbol': 'ticker'}, inplace=True)
        
        if 'ticker' not in p_df.columns:
            print("  ERROR: 'ticker' column not found in prices!")
            return None, None

        # Filter immediately
        p_df = p_df[p_df['ticker'].isin(unique_tickers)]
        print(f"  Filtered prices: {len(p_df)}")
        
        # Garbage collect
        gc.collect()
        
        p_df['date'] = pd.to_datetime(p_df['date'])
        q_df['date'] = pd.to_datetime(q_df['date'])
        
        # Sort
        p_df = p_df.sort_values(['ticker', 'date'])
        
        # Calculate returns per ticker without pivot (memory efficient)
        print("  Calculating returns...")
        # Ensure adj_close exists
        price_col = 'adj_close' if 'adj_close' in p_df.columns else 'close'
        p_df['fwd_3m_ret'] = p_df.groupby('ticker')[price_col].transform(lambda x: x.shift(-63) / x - 1)
        
        # Drop NaN returns
        p_df = p_df.dropna(subset=['fwd_3m_ret'])
        
        print("  Merging quality and returns...")
        merged = pd.merge(q_df, p_df[['date', 'ticker', 'fwd_3m_ret']], on=['date', 'ticker'], how='inner')
        print(f"  Merged data shape: {merged.shape}")
        
        return merged, p_df
    
    def load_quality_value_and_returns(self):
        print("Loading Quality, Value & Price Data...")
        if not os.path.exists(QUALITY_DATA_PATH):
            print("Quality data not found.")
            return None, None
        
        q_df = pd.read_parquet(QUALITY_DATA_PATH)
        print(f"  Loaded {len(q_df)} quality signals.")
        
        # Load value signals
        if not os.path.exists(VALUE_DATA_PATH):
            print("Value data not found.")
            return None, None
        
        v_df = pd.read_parquet(VALUE_DATA_PATH)
        print(f"  Loaded {len(v_df)} value signals.")
        print(f"  Value columns: {v_df.columns.tolist()}")
        
        # Normalize ticker col
        if 'symbol' in q_df.columns: q_df.rename(columns={'symbol': 'ticker'}, inplace=True)
        if 'symbol' in v_df.columns: v_df.rename(columns={'symbol': 'ticker'}, inplace=True)
        
        # Calculate quality_score if missing
        if 'quality_score' not in q_df.columns:
            print("  'quality_score' missing. Calculating from signals...")
            signals = []
            if 'roic' in q_df.columns: signals.append('roic')
            if 'returnOnEquity' in q_df.columns: signals.append('returnOnEquity')
            if 'freeCashFlowYield' in q_df.columns: signals.append('freeCashFlowYield')
            
            if not signals:
                print("  ERROR: No quality signals found to calculate score!")
                return None, None
            
            # Simple Z-score average
            for col in signals:
                q_df[col] = q_df[col].clip(lower=q_df[col].quantile(0.01), upper=q_df[col].quantile(0.99))
                q_df[f'z_{col}'] = q_df.groupby('date')[col].transform(lambda x: (x - x.mean()) / x.std())
            
            q_df['quality_score'] = q_df[[f'z_{s}' for s in signals]].mean(axis=1)
            q_df['quality_score'] = (q_df['quality_score'] - q_df['quality_score'].min()) / (q_df['quality_score'].max() - q_df['quality_score'].min()) * 100
            print("  Calculated quality_score.")
        
        # Calculate value_score
        print("  Calculating value_score...")
        value_signals = []
        # Lower is better for valuation ratios (cheaper = better value)
        if 'priceToBookRatio' in v_df.columns: value_signals.append('priceToBookRatio')
        if 'priceToSalesRatio' in v_df.columns: value_signals.append('priceToSalesRatio')
        # Higher is better for these
        if 'freeCashFlowYield' in v_df.columns: value_signals.append('freeCashFlowYield')
        
        if not value_signals:
            print("  WARNING: No value signals found. Using raw signals if available...")
            # Try to use any numeric columns as value signals
            numeric_cols = v_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in ['date', 'ticker']]
            if numeric_cols:
                value_signals = numeric_cols[:3]  # Use first 3 numeric columns
                print(f"  Using numeric columns as value signals: {value_signals}")
        
        if value_signals:
            for col in value_signals:
                # Handle infinite and NaN values
                v_df[col] = v_df[col].replace([np.inf, -np.inf], np.nan)
                # Winsorize
                v_df[col] = v_df[col].clip(lower=v_df[col].quantile(0.01), upper=v_df[col].quantile(0.99))
                # For valuation ratios (lower is better), invert the Z-score
                if 'priceToBook' in col or 'priceToSales' in col:
                    v_df[f'z_{col}'] = v_df.groupby('date')[col].transform(lambda x: -(x - x.mean()) / x.std())
                else:
                    v_df[f'z_{col}'] = v_df.groupby('date')[col].transform(lambda x: (x - x.mean()) / x.std())
            
            v_df['value_score'] = v_df[[f'z_{s}' for s in value_signals]].mean(axis=1)
            # Scale to 0-100
            v_df['value_score'] = (v_df['value_score'] - v_df['value_score'].min()) / (v_df['value_score'].max() - v_df['value_score'].min()) * 100
            print("  Calculated value_score.")
        else:
            print("  ERROR: Could not calculate value_score!")
            return None, None
        
        # Merge quality and value
        print("  Merging quality and value signals...")
        q_df['date'] = pd.to_datetime(q_df['date'])
        v_df['date'] = pd.to_datetime(v_df['date'])
        merged_signals = pd.merge(q_df[['date', 'ticker', 'quality_score']], 
                                  v_df[['date', 'ticker', 'value_score']], 
                                  on=['date', 'ticker'], how='inner')
        print(f"  Merged signals shape: {merged_signals.shape}")
        
        # Compute combined_score
        print("  Computing combined_score = 0.5 * Quality + 0.5 * Value...")
        # Normalize scores to 0-1 range first
        merged_signals['quality_norm'] = (merged_signals['quality_score'] - merged_signals['quality_score'].min()) / (merged_signals['quality_score'].max() - merged_signals['quality_score'].min())
        merged_signals['value_norm'] = (merged_signals['value_score'] - merged_signals['value_score'].min()) / (merged_signals['value_score'].max() - merged_signals['value_score'].min())
        merged_signals['combined_score'] = 0.5 * merged_signals['quality_norm'] + 0.5 * merged_signals['value_norm']
        merged_signals['combined_score'] = merged_signals['combined_score'] * 100  # Scale to 0-100
        print("  Calculated combined_score.")
        
        unique_tickers = merged_signals['ticker'].unique()
        print(f"  Unique tickers: {len(unique_tickers)}")
        
        if not os.path.exists(PRICES_DATA_PATH):
            print("Price data not found.")
            return None, None
            
        print("  Loading prices (filtered)...")
        p_df = pd.read_parquet(PRICES_DATA_PATH)
        
        if 'symbol' in p_df.columns: p_df.rename(columns={'symbol': 'ticker'}, inplace=True)
        
        if 'ticker' not in p_df.columns:
            print("  ERROR: 'ticker' column not found in prices!")
            return None, None

        p_df = p_df[p_df['ticker'].isin(unique_tickers)]
        print(f"  Filtered prices: {len(p_df)}")
        
        gc.collect()
        
        p_df['date'] = pd.to_datetime(p_df['date'])
        p_df = p_df.sort_values(['ticker', 'date'])
        
        print("  Calculating returns...")
        price_col = 'adj_close' if 'adj_close' in p_df.columns else 'close'
        p_df['fwd_3m_ret'] = p_df.groupby('ticker')[price_col].transform(lambda x: x.shift(-63) / x - 1)
        p_df = p_df.dropna(subset=['fwd_3m_ret'])
        
        print("  Merging signals and returns...")
        merged = pd.merge(merged_signals, p_df[['date', 'ticker', 'fwd_3m_ret']], on=['date', 'ticker'], how='inner')
        print(f"  Final merged data shape: {merged.shape}")
        
        return merged, p_df

class Validator:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def validate_macro(self, df, prices):
        if df is None: return
        print("\n--- Macro Validation ---")
        
        # Market proxy - calculate first to get date range
        print("  Calculating market proxy...")
        price_col = 'adj_close' if 'adj_close' in prices.columns else 'close'
        prices['date'] = pd.to_datetime(prices['date'])
        market_ret = prices.groupby('date')[price_col].sum().pct_change().reset_index()
        market_ret.columns = ['date', 'market_ret']
        market_ret = market_ret.dropna(subset=['market_ret'])
        market_ret['date'] = pd.to_datetime(market_ret['date']).dt.normalize()
        
        # Get market date range
        market_min_date = market_ret['date'].min()
        market_max_date = market_ret['date'].max()
        print(f"  Market data date range: {market_min_date} to {market_max_date}")
        
        # Inspect and prepare macro data
        print("  Inspecting date columns...")
        df['date'] = pd.to_datetime(df['date'])
        df_tz = df['date'].dt.tz
        
        print(f"  df_macro date column type: {df['date'].dtype}")
        print(f"  df_macro date timezone: {df_tz if df_tz is not None else 'naive'}")
        print(f"  df_macro date range (before filter): {df['date'].min()} to {df['date'].max()}")
        
        # Ensure timezone naiveness
        if df_tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
            print("  Removed timezone from df_macro dates")
        
        # Filter macro data to market date range (with some buffer for forward fill)
        print(f"  Filtering macro data to market date range...")
        df_filtered = df[(df['date'] >= market_min_date - pd.Timedelta(days=30)) & 
                        (df['date'] <= market_max_date)].copy()
        print(f"  Filtered from {len(df)} to {len(df_filtered)} records")
        
        if len(df_filtered) == 0:
            print("  ⚠️  WARNING: No macro data in market date range!")
            print("  The macro data file (fred_us.parquet) only contains data from 1980,")
            print("  but market data is from 2015-2025. Macro validation cannot proceed.")
            print("  Please update the macro data file with recent data to enable macro validation.")
            print("\n  Skipping macro validation...")
            return
        
        print("Regime Frequency (filtered):")
        print(df_filtered['regime'].value_counts(normalize=True))
        
        # Set date as index for resampling
        df_indexed = df_filtered.set_index('date').sort_index()
        
        # Resample df_macro to daily frequency using forward fill
        print("  Resampling df_macro to daily frequency (forward fill)...")
        print(f"  Original frequency: {df_indexed.index.to_series().diff().mode()[0] if len(df_indexed) > 1 else 'unknown'}")
        
        # Create daily date range from market min to max
        date_range = pd.date_range(start=market_min_date, end=market_max_date, freq='D')
        df_daily = df_indexed.reindex(date_range).ffill()
        df_daily = df_daily.reset_index()
        df_daily.rename(columns={'index': 'date'}, inplace=True)
        
        print(f"  Resampled to {len(df_daily)} daily records")
        
        # Normalize dates to date-only (remove time component if any)
        df_daily['date'] = pd.to_datetime(df_daily['date']).dt.normalize()
        
        # Debug: show date ranges
        print(f"  df_daily date range: {df_daily['date'].min()} to {df_daily['date'].max()}")
        print(f"  market_ret date range: {market_ret['date'].min()} to {market_ret['date'].max()}")
        print(f"  df_daily unique dates: {df_daily['date'].nunique()}")
        print(f"  market_ret unique dates: {market_ret['date'].nunique()}")
        
        # Merge with market returns
        data = pd.merge(df_daily, market_ret, on='date', how='inner')
        print(f"  After merge with market_ret: {len(data)} records")
        
        print("\nAvg Daily Market Return by Regime:")
        regime_stats = data.groupby('regime')['market_ret'].agg(['mean', 'std', 'count'])
        print(regime_stats)
        
        # Save to output file for verification
        with open(os.path.join(self.output_dir, 'output_real.txt'), 'w') as f:
            f.write("Avg Daily Market Return by Regime:\n")
            f.write(str(regime_stats))
            f.write("\n\n")
        
        # Regime Compass (use resampled data)
        try:
            if 'DGS10' in data.columns and 'DGS2' in data.columns and 'VIXCLS' in data.columns:
                 data['slope'] = data['DGS10'] - data['DGS2']
                 # Filter out NaN values for plotting
                 compass_data = data[['slope', 'VIXCLS', 'regime']].dropna()
                 if len(compass_data) > 0:
                     plt.figure(figsize=(10, 8))
                     sns.scatterplot(data=compass_data, x='slope', y='VIXCLS', hue='regime', alpha=0.6)
                     plt.title('Regime Compass: Yield Slope vs VIX')
                     plt.xlabel('Yield Slope (10Y - 2Y)')
                     plt.ylabel('VIX')
                     plt.grid(True, alpha=0.3)
                     plt.tight_layout()
                     plt.savefig(os.path.join(self.output_dir, 'regime_compass.png'), dpi=150)
                     print("  Saved regime_compass.png")
                 else:
                     print("  WARNING: No valid data for Regime Compass plot (missing DGS10, DGS2, or VIXCLS)")
        except Exception as e:
            print(f"Error plotting compass: {e}")
            import traceback
            traceback.print_exc()

        # Win Rate
        data['positive_day'] = data['market_ret'] > 0
        print("\nWin Rate by Regime:")
        print(data.groupby('regime')['positive_day'].mean())

    def validate_quality(self, df, prices):
        if df is None: return
        print("\n--- Quality Validation ---")
        
        df['decile'] = df.groupby('date')['quality_score'].transform(
            lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
        )
        
        perf = df.groupby('decile')['fwd_3m_ret'].mean()
        print("Avg 3M Return by Decile:")
        print(perf)
        
        plt.figure(figsize=(10, 6))
        perf.plot(kind='bar', color='skyblue')
        plt.title('Real Data: Quality Decile Performance (3M Fwd)')
        plt.ylabel('Avg Return')
        plt.savefig(os.path.join(self.output_dir, 'quality_deciles_real.png'))
        print("  Saved quality_deciles_real.png")
        
        # Quality Shield
        # Market 3M return
        print("  Calculating market 3M return...")
        price_col = 'adj_close' if 'adj_close' in prices.columns else 'close'
        market_prices = prices.groupby('date')[price_col].sum()
        market_ret_3m = (market_prices.shift(-63) / market_prices - 1).reset_index()
        market_ret_3m.columns = ['date', 'market_ret_3m']
        
        df_merged = pd.merge(df, market_ret_3m, on='date', how='inner')
        
        crashes = df_merged[df_merged['market_ret_3m'] < -0.10]
        if len(crashes) > 0:
            print("\n--- Quality Shield (During Market Crashes < -10%) ---")
            crash_perf = crashes.groupby('decile')['fwd_3m_ret'].mean()
            print(crash_perf)
            
            plt.figure(figsize=(10, 6))
            crash_perf.plot(kind='bar', color='salmon')
            plt.title('Quality Performance During Market Crashes')
            plt.ylabel('Avg Return')
            plt.savefig(os.path.join(self.output_dir, 'quality_shield.png'))
            print("  Saved quality_shield.png")
        else:
            print("No crash periods found.")
    
    def validate_quality_value(self, df, prices):
        if df is None: return
        print("\n--- Quality + Value Validation ---")
        
        # Quality-Value Matrix plot
        print("  Creating Quality-Value Matrix plot...")
        # Sample data if too large for plotting
        plot_df = df.copy()
        if len(plot_df) > 10000:
            print(f"  Sampling {len(plot_df)} records to 10000 for plotting...")
            plot_df = plot_df.sample(n=10000, random_state=42)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(plot_df['value_score'], plot_df['quality_score'], 
                             c=plot_df['fwd_3m_ret'], cmap='RdYlGn', 
                             alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, label='Forward 3M Return')
        plt.xlabel('Value Score', fontsize=12)
        plt.ylabel('Quality Score', fontsize=12)
        plt.title('Quality-Value Matrix: Stock Performance Map', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'quality_value_matrix.png'), dpi=150)
        print("  Saved quality_value_matrix.png")
        plt.close()
        
        # Combined score deciles
        print("\n  Analyzing combined_score performance...")
        df['decile'] = df.groupby('date')['combined_score'].transform(
            lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
        )
        
        perf = df.groupby('decile')['fwd_3m_ret'].agg(['mean', 'std', 'count'])
        print("Avg 3M Return by Combined Score Decile:")
        print(perf)
        
        plt.figure(figsize=(10, 6))
        perf['mean'].plot(kind='bar', color='steelblue')
        plt.title('Real Data: Quality+Value Combined Score Decile Performance (3M Fwd)')
        plt.ylabel('Avg Return')
        plt.xlabel('Decile (1=Lowest, 10=Highest)')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'quality_value_combined_deciles.png'), dpi=150)
        print("  Saved quality_value_combined_deciles.png")
        plt.close()
        
        # Individual quality and value deciles for comparison
        print("\n  Quality Score Deciles:")
        df['quality_decile'] = df.groupby('date')['quality_score'].transform(
            lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
        )
        quality_perf = df.groupby('quality_decile')['fwd_3m_ret'].mean()
        print(quality_perf)
        
        print("\n  Value Score Deciles:")
        df['value_decile'] = df.groupby('date')['value_score'].transform(
            lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
        )
        value_perf = df.groupby('value_decile')['fwd_3m_ret'].mean()
        print(value_perf)

def main():
    try:
        loader = DataLoader()
        validator = Validator(OUTPUT_DIR)
        
        macro_data = loader.load_macro_and_predict()
        quality_data, prices = loader.load_quality_and_returns()
        quality_value_data, prices_qv = loader.load_quality_value_and_returns()
        
        if prices is not None:
            validator.validate_macro(macro_data, prices)
            validator.validate_quality(quality_data, prices)
        
        if prices_qv is not None:
            validator.validate_quality_value(quality_value_data, prices_qv)
        else:
            print("Cannot proceed with quality+value validation without price data.")
            
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

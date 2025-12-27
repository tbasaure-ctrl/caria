"""
Matrix Entropy Research: Risk != Volatility
===========================================

Hypothesis:
    Volatility (Energy) measures the magnitude of fluctuations.
    Entropy (Structure) measures the diversity of drivers.

    Crisis Warning = Low Entropy (High Correlation/Low Diversity) + Low Volatility (Complacency)
    The "Minsky Moment" is when the system is fragile (Low Entropy) but appears calm (Low Val).

Methodology:
    1. Universe: Global Multi-Asset ETFs (Equity, Bonds, EM, Commodities)
    2. Rolling Window: 63 days (Quarterly)
    3. Metric: Von Neumann Entropy of the Correlation Matrix
       S = - sum(lambda_i * log(lambda_i))    (Normalized)
    4. Comparison: vs Realized Volatility

Author: Antigravity Agent
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import os
import warnings

warnings.filterwarnings("ignore")

# Configuration
UNIVERSE = [
    # US Equities
    "SPY", "QQQ", "IWM", "RSP",
    # Global Equities
    "EFA", "EEM", "ACWI",
    # Sectors (Cyclical vs Defensive)
    "XLE", "XLF", "XLK", "XLV", "XLP", "XLU",
    # Rates & Credit
    "TLT", "LQD", "HYG",
    # Commodities
    "GLD", "DBC"
]

START_DATE = "2005-01-01"
WINDOW = 63  # 3 Months

def load_data(tickers, start_date):
    """Load Adjusted Close prices."""
    print(f"Loading data for {len(tickers)} assets from {start_date}...")
    try:
        data = yf.download(tickers, start=start_date, progress=False)['Adj Close']
    except KeyError:
        # Fallback for yfinance versions where structure might differ
        data = yf.download(tickers, start=start_date, progress=False)['Close']
    
    # Clean data
    data = data.dropna(how='all')
    return data

def calculate_matrix_entropy(returns, window):
    """
    Calculate Von Neumann Entropy of the Correlation Matrix over a rolling window.
    
    Entropy S = - Sum (p_i * log(p_i)) / log(N)
    Where p_i = lambda_i / N
    lambda_i are eigenvalues of the Correlation Matrix.
    N is number of assets.
    """
    entropy_series = {}
    dates = []
    
    print("Calculating Matrix Entropy...")
    
    # Efficient rolling iteration
    for i in range(window, len(returns)):
        window_data = returns.iloc[i-window : i]
        
        # 1. Correlation Matrix
        corr_matrix = window_data.corr()
        
        # Handle NaNs (if any asset is flat/nan in window)
        corr_matrix = corr_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
        if corr_matrix.empty:
            continue

        # 2. Eigenvalues
        try:
            eigvals = np.linalg.eigvalsh(corr_matrix)
        except np.linalg.LinAlgError:
            continue
            
        # Filter small numerical noise
        eigvals = eigvals[eigvals > 1e-10]
        
        # 3. Normalize Eigenvalues (Probabilities)
        # Sum of eigenvalues of correlation matrix = N (trace)
        # But we reduced dimensions if we dropped NaNs, so normalize by sum
        eig_sum = np.sum(eigvals)
        probs = eigvals / eig_sum
        
        # 4. Von Neumann Entropy
        # S = - sum(p * log(p))
        S = -np.sum(probs * np.log(probs))
        
        # 5. Normalize Entropy (0 to 1)
        # Max entropy is log(N) where N is number of effective components
        N = len(probs)
        S_norm = S / np.log(N)
        
        dates.append(returns.index[i])
        entropy_series[returns.index[i]] = S_norm

    return pd.Series(entropy_series)

def analyze_market_regime(data):
    """Compute Entropy, Volatility and Quadrants."""
    
    # 1. Returns
    daily_returns = data.pct_change().dropna()
    
    # 2. Market Reference (SPY or average)
    if 'SPY' in daily_returns.columns:
        market_ret = daily_returns['SPY']
    else:
        market_ret = daily_returns.mean(axis=1)
        
    # 3. Entropy (Structure)
    entropy = calculate_matrix_entropy(daily_returns, WINDOW)
    
    # 4. Volatility (Energy) - Annualized
    volatility = market_ret.rolling(WINDOW).std() * np.sqrt(252)
    
    # Align
    df = pd.DataFrame({
        'Entropy': entropy,
        'Volatility': volatility,
        'Price': data['SPY'] if 'SPY' in data.columns else data.iloc[:,0]
    }).dropna()
    
    # 5. Future Returns (Forward 1 month) for Crash Detection
    df['Fwd_Ret_21d'] = df['Price'].shift(-21) / df['Price'] - 1
    
    return df

def plot_fresh_perspective(df, output_dir):
    """Generate the 'Fresh Air' visualizations."""
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # --- Plot 1: The decoupling ---
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    ax1.plot(df.index, df['Price'], color='black', alpha=0.6, label='S&P 500')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.set_yscale('log')
    
    ax2 = ax1.twinx()
    # Invert Entropy: Low Entropy = High Fragility
    ax2.plot(df.index, df['Entropy'], color='#2ca02c', alpha=0.8, linewidth=1.5, label='Market Entropy (Diversity)')
    ax2.set_ylabel('Entropy (Higher = Safer)', fontsize=12, color='#2ca02c')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    
    # Highlight crashes
    crashes = df[df['Fwd_Ret_21d'] < -0.10] # >10% drop next month
    ax1.scatter(crashes.index, crashes['Price'], color='red', s=30, zorder=5, label='Pre-Crash')
    
    plt.title("The Structure of Risk: Market Entropy vs Price", fontsize=16)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.savefig(os.path.join(output_dir, "Fresh_Perspective_1_Entropy_Time.png"))
    plt.close()

    # --- Plot 2: Orthogonality (Vol vs Entropy) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color points by future return
    sc = ax.scatter(df['Volatility'], df['Entropy'], 
                    c=df['Fwd_Ret_21d'], cmap='RdYlGn', 
                    alpha=0.6, s=15, vmin=-0.1, vmax=0.05)
    
    ax.set_xlabel('Volatility (Energy)', fontsize=12)
    ax.set_ylabel('Entropy (Diversity)', fontsize=12)
    ax.set_title('Risk Quadrants: The "Silent Risk" Zone', fontsize=16)
    
    # Add quadrants
    vol_mid = df['Volatility'].median()
    ent_mid = df['Entropy'].median()
    
    ax.axvline(vol_mid, color='grey', linestyle='--')
    ax.axhline(ent_mid, color='grey', linestyle='--')
    
    # Annotate Minsky Zone
    # Low Vol, Low Entropy
    ax.text(df['Volatility'].min(), df['Entropy'].min() + 0.02, 
            "MINSKY ZONE\n(Fragile + Complacent)\nSilent Risk", 
            color='red', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.colorbar(sc, label='Next Month Return')
    plt.savefig(os.path.join(output_dir, "Fresh_Perspective_2_Quadrants.png"))
    plt.close()

    print(f"Artifacts saved in {output_dir}")

def main():
    print("--- Starting Matrix Entropy Research ---")
    data = load_data(UNIVERSE, START_DATE)
    
    if data.empty:
        print("Error: No data loaded.")
        return

    df = analyze_market_regime(data)
    
    # Output stats
    print("\nResults Summary:")
    print(df.describe()[['Entropy', 'Volatility', 'Fwd_Ret_21d']])
    
    # Correlation Check
    corr = df['Entropy'].corr(df['Volatility'])
    print(f"\nCorrelation(Entropy, Volatility) = {corr:.2f}")
    if abs(corr) < 0.5:
        print("SUCCESS: Metric is distinct from Volatility (Orthogonal).")
    else:
        print("WARNING: Metric is highly correlated with Volatility.")

    # Save data
    output_dir = r"c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "matrix_entropy_results.csv"))
    
    plot_fresh_perspective(df, output_dir)
    print("--- Research Complete ---")

if __name__ == "__main__":
    main()

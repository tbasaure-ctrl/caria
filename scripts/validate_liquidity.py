import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add repo root to path
# Script is in caria/scripts/
# Package is in caria/caria-lib/
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../caria-lib'))
sys.path.append(lib_path)

from caria.models.liquidity.liquidity_engine import LiquidityEngine

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../validation_results_real')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def validate_liquidity():
    print("--- Validating Liquidity Engine ---")
    engine = LiquidityEngine()
    
    try:
        df = engine.fetch_data()
        if df is None:
            print("Failed to fetch data.")
            return

        df = engine.calculate_signals(df)
        if df is None:
            print("Failed to calculate signals.")
            return
            
        print(f"Generated {len(df)} liquidity records.")
        print("Last 5 records:")
        print(df[['net_liquidity', 'hydraulic_score', 'liquidity_state']].tail())
        
        # Plot Net Liquidity and Score
        plt.figure(figsize=(12, 8))
        
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(df.index, df['net_liquidity'], label='Net Liquidity (Billions)', color='blue')
        ax1.set_title('Net Liquidity (Fed Assets - TGA - RRP)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        # Color code the score
        points = plt.scatter(df.index, df['hydraulic_score'], c=df['hydraulic_score'], cmap='RdYlGn', s=10)
        plt.colorbar(points, label='Hydraulic Score')
        ax2.axhline(60, color='green', linestyle='--', alpha=0.5, label='Expansion Threshold')
        ax2.axhline(40, color='red', linestyle='--', alpha=0.5, label='Contraction Threshold')
        ax2.set_title('Hydraulic Score (0-100)')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'liquidity_hydraulic_score.png'))
        print("Saved liquidity_hydraulic_score.png")
        
        # Save data for integration
        df.to_parquet(os.path.join(OUTPUT_DIR, 'liquidity_signals.parquet'))
        print("Saved liquidity_signals.parquet")
        
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    validate_liquidity()

"""
Script to generate sample historical data for model validation.
Creates realistic predictions vs actuals for backtesting and statistical analysis.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Database connection
def get_db_connection():
    password = os.getenv("POSTGRES_PASSWORD")
    if not password:
        raise ValueError("POSTGRES_PASSWORD environment variable not set")
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),  # Use 'postgres' as default for Docker
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "caria_user"),
        password=password,
        database=os.getenv("POSTGRES_DB", "caria"),
    )


def create_predictions_table(conn):
    """Create table to store historical predictions if it doesn't exist."""
    with conn.cursor() as cursor:
        # Try gen_random_uuid() first (PostgreSQL 13+), fallback to uuid_generate_v4()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    date DATE NOT NULL UNIQUE,
                    predicted_regime VARCHAR(20),
                    predicted_confidence DECIMAL(5, 4),
                    predicted_return DECIMAL(10, 6),
                    actual_regime VARCHAR(20),
                    actual_return DECIMAL(10, 6),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                );
                
                CREATE INDEX IF NOT EXISTS idx_model_predictions_date 
                ON model_predictions(date);
            """)
        except Exception:
            # Fallback: try to create extension and use uuid_generate_v4
            try:
                cursor.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
                conn.commit()
            except:
                pass
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    date DATE NOT NULL UNIQUE,
                    predicted_regime VARCHAR(20),
                    predicted_confidence DECIMAL(5, 4),
                    predicted_return DECIMAL(10, 6),
                    actual_regime VARCHAR(20),
                    actual_return DECIMAL(10, 6),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                );
                
                CREATE INDEX IF NOT EXISTS idx_model_predictions_date 
                ON model_predictions(date);
            """)
        conn.commit()
        print("✓ Predictions table created/verified")


def generate_realistic_data(start_date: str, end_date: str, n_days: int = 100) -> pd.DataFrame:
    """
    Generate realistic historical predictions vs actuals.
    
    Simulates:
    - Model predictions with some accuracy (70-80%)
    - Realistic regime transitions
    - Returns correlated with regimes
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[:n_days]  # Limit to n_days
    
    regimes = ["expansion", "slowdown", "recession", "stress"]
    
    # Generate actual regimes with realistic transitions
    np.random.seed(42)  # For reproducibility
    actual_regimes = []
    current_regime_idx = 0
    
    for i in range(len(dates)):
        # Regime persistence (70% chance to stay, 30% to transition)
        if np.random.random() < 0.7:
            regime = regimes[current_regime_idx]
        else:
            # Transition to adjacent regime or stress
            if np.random.random() < 0.3:
                current_regime_idx = 3  # Stress
            else:
                current_regime_idx = (current_regime_idx + np.random.choice([-1, 1])) % 3
            regime = regimes[current_regime_idx]
        
        actual_regimes.append(regime)
    
    # Generate actual returns based on regime
    regime_returns = {
        "expansion": (0.0008, 0.015),  # Mean, std
        "slowdown": (0.0002, 0.012),
        "recession": (-0.001, 0.018),
        "stress": (-0.002, 0.025),
    }
    
    actual_returns = []
    for regime in actual_regimes:
        mean, std = regime_returns[regime]
        ret = np.random.normal(mean, std)
        actual_returns.append(ret)
    
    # Generate predictions with 75% accuracy
    predicted_regimes = []
    predicted_confidences = []
    predicted_returns = []
    
    for i, actual_regime in enumerate(actual_regimes):
        if np.random.random() < 0.75:  # 75% accuracy
            predicted_regime = actual_regime
            confidence = np.random.uniform(0.6, 0.95)
        else:
            # Wrong prediction - pick nearby regime
            actual_idx = regimes.index(actual_regime)
            if actual_idx == 3:  # Stress
                predicted_idx = np.random.choice([0, 1, 2])
            else:
                predicted_idx = (actual_idx + np.random.choice([-1, 1])) % 3
            predicted_regime = regimes[predicted_idx]
            confidence = np.random.uniform(0.3, 0.6)  # Lower confidence when wrong
        
        predicted_regimes.append(predicted_regime)
        predicted_confidences.append(confidence)
        
        # Predicted return (close to actual but with noise)
        mean, std = regime_returns[predicted_regime]
        pred_ret = actual_returns[i] + np.random.normal(0, std * 0.3)
        predicted_returns.append(pred_ret)
    
    df = pd.DataFrame({
        "date": dates,
        "predicted_regime": predicted_regimes,
        "predicted_confidence": predicted_confidences,
        "predicted_return": predicted_returns,
        "actual_regime": actual_regimes,
        "actual_return": actual_returns,
    })
    
    return df


def load_data_to_db(conn, df: pd.DataFrame):
    """Load generated data into database."""
    with conn.cursor() as cursor:
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO model_predictions 
                (date, predicted_regime, predicted_confidence, predicted_return, 
                 actual_regime, actual_return)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (date) DO UPDATE SET
                    predicted_regime = EXCLUDED.predicted_regime,
                    predicted_confidence = EXCLUDED.predicted_confidence,
                    predicted_return = EXCLUDED.predicted_return,
                    actual_regime = EXCLUDED.actual_regime,
                    actual_return = EXCLUDED.actual_return
            """, (
                row["date"],
                row["predicted_regime"],
                float(row["predicted_confidence"]),
                float(row["predicted_return"]),
                row["actual_regime"],
                float(row["actual_return"]),
            ))
        conn.commit()
        print(f"✓ Loaded {len(df)} predictions into database")


def main():
    """Generate and load validation data."""
    print("Generating model validation data...")
    
    # Generate data for last 6 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    df = generate_realistic_data(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        n_days=100  # ~100 trading days
    )
    
    print(f"Generated {len(df)} data points")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nRegime distribution:")
    print(df["actual_regime"].value_counts())
    print(f"\nPrediction accuracy: {(df['predicted_regime'] == df['actual_regime']).mean():.1%}")
    
    # Load to database
    conn = get_db_connection()
    try:
        create_predictions_table(conn)
        load_data_to_db(conn, df)
        print("\n✓ Validation data ready!")
        print(f"\nYou can now test endpoints:")
        print(f"  POST /api/model/validation/backtest")
        print(f"  POST /api/model/validation/statistics")
        print(f"  GET  /api/model/validation/benchmark")
    finally:
        conn.close()


if __name__ == "__main__":
    main()



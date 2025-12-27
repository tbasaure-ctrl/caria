"""
Logistic Regression Analysis: Gold, Silver, Dollar, S&P 500, Oil, Bitcoin, Ethereum
Data desde 2010 usando FMP API

Autor: Auto-generated
Fecha: 2024-12-26
"""

import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
API_KEY = "79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq"
BASE_URL = "https://financialmodelingprep.com/api/v3"
START_DATE = "2010-01-01"
OUTPUT_DIR = "c:/key/wise_adviser_cursor_context/Caria_repo/caria/docs/research/outputs"

# Assets to fetch - FMP compatible symbols
ASSETS = {
    "GCUSD": "Gold",           # Gold futures
    "SIUSD": "Silver",         # Silver futures  
    "DX-Y.NYB": "Dollar",      # US Dollar Index (DXY alternative)
    "^GSPC": "SP500",          # S&P 500 Index
    "CLUSD": "Oil",            # Crude Oil WTI
    "BTCUSD": "Bitcoin",       # Bitcoin
    "ETHUSD": "Ethereum"       # Ethereum
}

# Alternative symbols to try if primary fails
ALTERNATIVE_SYMBOLS = {
    "Gold": ["GC=F", "XAUUSD", "GLD"],
    "Silver": ["SI=F", "XAGUSD", "SLV"],
    "Dollar": ["UUP", "DXY"],
    "SP500": ["SPY", "^SPX"],
    "Oil": ["CL=F", "USO", "WTIUSD"],
    "Bitcoin": ["BTC-USD", "BTCUSD"],
    "Ethereum": ["ETH-USD", "ETHUSD"]
}

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================
def fetch_historical_data(symbol, name):
    """Fetch historical price data from FMP API"""
    print(f"Fetching {name} ({symbol})...")
    
    url = f"{BASE_URL}/historical-price-full/{symbol}?from={START_DATE}&apikey={API_KEY}"
    
    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        
        if "historical" in data and len(data["historical"]) > 0:
            df = pd.DataFrame(data["historical"])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df = df[['date', 'close']].rename(columns={'close': name})
            print(f"  -> Success: {len(df)} records from {df['date'].min().date()} to {df['date'].max().date()}")
            return df
        else:
            print(f"  -> No data for {symbol}")
            return None
            
    except Exception as e:
        print(f"  -> Error: {e}")
        return None


def try_alternative_symbols(name, alternatives):
    """Try alternative symbols if primary fails"""
    for alt_symbol in alternatives:
        df = fetch_historical_data(alt_symbol, name)
        if df is not None:
            return df
    return None


def fetch_commodity_data(symbol, name):
    """Fetch commodity data from FMP commodities endpoint"""
    print(f"Fetching commodity {name} ({symbol})...")
    
    url = f"{BASE_URL}/historical-price-full/commodity/{symbol}?apikey={API_KEY}"
    
    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            if 'date' in df.columns and 'close' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                df = df[df['date'] >= START_DATE]
                df = df[['date', 'close']].rename(columns={'close': name})
                print(f"  -> Success: {len(df)} records")
                return df
                
        # Try historical endpoint variation
        url2 = f"{BASE_URL}/historical-price-full/{symbol}?from={START_DATE}&apikey={API_KEY}"
        response = requests.get(url2, timeout=30)
        data = response.json()
        
        if "historical" in data and len(data["historical"]) > 0:
            df = pd.DataFrame(data["historical"])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df = df[['date', 'close']].rename(columns={'close': name})
            print(f"  -> Success (alt): {len(df)} records")
            return df
            
        print(f"  -> No commodity data for {symbol}")
        return None
        
    except Exception as e:
        print(f"  -> Error: {e}")
        return None


def fetch_forex_data(symbol, name):
    """Fetch forex data from FMP"""
    print(f"Fetching forex {name} ({symbol})...")
    
    url = f"{BASE_URL}/historical-price-full/{symbol}?from={START_DATE}&apikey={API_KEY}"
    
    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        
        if "historical" in data and len(data["historical"]) > 0:
            df = pd.DataFrame(data["historical"])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df = df[['date', 'close']].rename(columns={'close': name})
            print(f"  -> Success: {len(df)} records")
            return df
            
        print(f"  -> No forex data for {symbol}")
        return None
        
    except Exception as e:
        print(f"  -> Error: {e}")
        return None


def fetch_all_assets():
    """Fetch all required asset data"""
    print("="*60)
    print("FETCHING ASSET DATA FROM FMP API")
    print("="*60)
    
    dataframes = {}
    
    # Gold - Try commodity symbols
    df = fetch_historical_data("GCUSD", "Gold")
    if df is None:
        df = fetch_commodity_data("GC", "Gold")
    if df is None:
        df = fetch_historical_data("XAUUSD", "Gold")
    if df is None:
        df = fetch_historical_data("GLD", "Gold")  # Gold ETF as fallback
    if df is not None:
        dataframes["Gold"] = df
    
    # Silver - Try commodity symbols
    df = fetch_historical_data("SIUSD", "Silver")
    if df is None:
        df = fetch_commodity_data("SI", "Silver")
    if df is None:
        df = fetch_historical_data("XAGUSD", "Silver")
    if df is None:
        df = fetch_historical_data("SLV", "Silver")  # Silver ETF as fallback
    if df is not None:
        dataframes["Silver"] = df
    
    # Dollar Index
    df = fetch_historical_data("DX-Y.NYB", "Dollar")
    if df is None:
        df = fetch_historical_data("USDX", "Dollar")
    if df is None:
        df = fetch_forex_data("EURUSD", "Dollar")  # Inverse for dollar strength
        if df is not None:
            df["Dollar"] = 1 / df["Dollar"]  # Invert EUR/USD to get dollar strength
    if df is None:
        df = fetch_historical_data("UUP", "Dollar")  # Dollar ETF as fallback
    if df is not None:
        dataframes["Dollar"] = df
    
    # S&P 500
    df = fetch_historical_data("^GSPC", "SP500")
    if df is None:
        df = fetch_historical_data("SPY", "SP500")  # S&P 500 ETF as fallback
    if df is not None:
        dataframes["SP500"] = df
    
    # Oil (WTI Crude)
    df = fetch_historical_data("CLUSD", "Oil")
    if df is None:
        df = fetch_commodity_data("CL", "Oil")
    if df is None:
        df = fetch_historical_data("WTIUSD", "Oil")
    if df is None:
        df = fetch_historical_data("USO", "Oil")  # Oil ETF as fallback
    if df is not None:
        dataframes["Oil"] = df
    
    # Bitcoin
    df = fetch_historical_data("BTCUSD", "Bitcoin")
    if df is None:
        df = fetch_forex_data("BTCUSD", "Bitcoin")
    if df is not None:
        dataframes["Bitcoin"] = df
    
    # Ethereum
    df = fetch_historical_data("ETHUSD", "Ethereum")
    if df is None:
        df = fetch_forex_data("ETHUSD", "Ethereum")
    if df is not None:
        dataframes["Ethereum"] = df
    
    print("\n" + "="*60)
    print(f"SUMMARY: Fetched {len(dataframes)} out of 7 assets")
    print("Available assets:", list(dataframes.keys()))
    print("="*60 + "\n")
    
    return dataframes


def merge_dataframes(dataframes):
    """Merge all dataframes on date"""
    if len(dataframes) == 0:
        print("ERROR: No data to merge!")
        return None
    
    # Start with first dataframe
    keys = list(dataframes.keys())
    merged = dataframes[keys[0]].copy()
    
    # Merge remaining dataframes
    for key in keys[1:]:
        merged = pd.merge(merged, dataframes[key], on='date', how='outer')
    
    # Sort by date
    merged = merged.sort_values('date')
    
    # Filter from 2010
    merged = merged[merged['date'] >= START_DATE]
    
    print(f"Merged dataset: {len(merged)} rows, {len(merged.columns)} columns")
    print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    print(f"Columns: {list(merged.columns)}")
    
    return merged


# ============================================================================
# LOGISTIC REGRESSION ANALYSIS
# ============================================================================
def prepare_data_for_regression(df, target_col='SP500', horizon=5):
    """
    Prepare data for logistic regression.
    
    Target: Whether S&P 500 will be UP (1) or DOWN (0) in 'horizon' days
    Features: Lagged returns of all assets
    """
    print("\n" + "="*60)
    print("PREPARING DATA FOR LOGISTIC REGRESSION")
    print("="*60)
    
    # Calculate returns for all numeric columns
    df_returns = df.copy()
    df_returns = df_returns.set_index('date')
    
    # Calculate daily returns
    for col in df_returns.columns:
        df_returns[f'{col}_ret'] = df_returns[col].pct_change()
    
    # Create target: Future return of S&P 500 (positive = 1, negative = 0)
    if f'{target_col}_ret' in df_returns.columns:
        df_returns['target'] = (df_returns[f'{target_col}_ret'].shift(-horizon) > 0).astype(int)
    elif target_col in df_returns.columns:
        df_returns['target'] = (df_returns[target_col].pct_change().shift(-horizon) > 0).astype(int)
    else:
        print(f"ERROR: Target column '{target_col}' not found!")
        return None, None, None
    
    # Create lagged features
    feature_cols = [col for col in df_returns.columns if col.endswith('_ret')]
    
    lag_features = []
    for lag in [1, 5, 10, 21]:  # 1 day, 1 week, 2 weeks, 1 month
        for col in feature_cols:
            new_col = f'{col}_lag{lag}'
            df_returns[new_col] = df_returns[col].shift(lag)
            lag_features.append(new_col)
    
    # Add rolling volatility features
    for col in feature_cols:
        df_returns[f'{col}_vol5'] = df_returns[col].rolling(5).std()
        df_returns[f'{col}_vol21'] = df_returns[col].rolling(21).std()
        lag_features.extend([f'{col}_vol5', f'{col}_vol21'])
    
    # Drop NaN rows
    df_clean = df_returns[lag_features + ['target']].dropna()
    
    print(f"Features created: {len(lag_features)}")
    print(f"Samples after cleaning: {len(df_clean)}")
    print(f"Date range for regression: {df_clean.index.min()} to {df_clean.index.max()}")
    print(f"Target distribution: {df_clean['target'].value_counts().to_dict()}")
    
    X = df_clean[lag_features]
    y = df_clean['target']
    dates = df_clean.index
    
    return X, y, dates


def run_logistic_regression(X, y, dates):
    """Run logistic regression and report results"""
    print("\n" + "="*60)
    print("RUNNING LOGISTIC REGRESSION")
    print("="*60)
    
    # Split data (time-series aware - no shuffling)
    split_idx = int(len(X) * 0.7)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]
    
    print(f"Training period: {dates_train.min()} to {dates_train.max()} ({len(X_train)} samples)")
    print(f"Testing period: {dates_test.min()} to {dates_test.max()} ({len(X_test)} samples)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit logistic regression
    model = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        penalty='l2',
        C=1.0,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Results
    print("\n" + "-"*40)
    print("IN-SAMPLE RESULTS (Training)")
    print("-"*40)
    print(f"Accuracy: {(y_train_pred == y_train).mean():.4f}")
    
    print("\n" + "-"*40)
    print("OUT-OF-SAMPLE RESULTS (Testing)")
    print("-"*40)
    print(f"Accuracy: {(y_test_pred == y_test).mean():.4f}")
    try:
        auc = roc_auc_score(y_test, y_test_prob)
        print(f"ROC AUC: {auc:.4f}")
    except:
        auc = None
        print("ROC AUC: Could not calculate")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Down', 'Up']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(pd.DataFrame(cm, index=['Actual Down', 'Actual Up'], 
                       columns=['Pred Down', 'Pred Up']))
    
    # Feature importance
    print("\n" + "-"*40)
    print("TOP 20 FEATURE COEFFICIENTS")
    print("-"*40)
    
    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_[0]
    })
    coef_df['abs_coef'] = coef_df['coefficient'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    
    print(coef_df.head(20).to_string(index=False))
    
    return {
        'model': model,
        'scaler': scaler,
        'train_accuracy': (y_train_pred == y_train).mean(),
        'test_accuracy': (y_test_pred == y_test).mean(),
        'auc': auc,
        'coefficients': coef_df,
        'confusion_matrix': cm
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*60)
    print("LOGISTIC REGRESSION ANALYSIS")
    print("Assets: Gold, Silver, Dollar, S&P500, Oil, Bitcoin, Ethereum")
    print("Period: 2010 - Present")
    print("="*60 + "\n")
    
    # 1. Fetch all asset data
    dataframes = fetch_all_assets()
    
    if len(dataframes) < 3:
        print("\nWARNING: Less than 3 assets fetched. Analysis may be limited.")
    
    if len(dataframes) == 0:
        print("\nERROR: No data fetched. Please check API connectivity.")
        exit(1)
    
    # 2. Merge into single dataframe
    merged_df = merge_dataframes(dataframes)
    
    if merged_df is None or len(merged_df) < 100:
        print("\nERROR: Insufficient data for analysis.")
        exit(1)
    
    # 3. Save merged dataset
    output_file = f"{OUTPUT_DIR}/logistic_regression_data.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"\nData saved to: {output_file}")
    
    # 4. Prepare data for regression
    # Determine target column (use SP500 if available, otherwise first available)
    target_col = 'SP500' if 'SP500' in merged_df.columns else list(dataframes.keys())[0]
    
    X, y, dates = prepare_data_for_regression(merged_df, target_col=target_col, horizon=5)
    
    if X is None or len(X) < 100:
        print("\nERROR: Insufficient data after preprocessing.")
        exit(1)
    
    # 5. Run logistic regression
    results = run_logistic_regression(X, y, dates)
    
    # 6. Save results
    results_file = f"{OUTPUT_DIR}/logistic_regression_results.csv"
    results['coefficients'].to_csv(results_file, index=False)
    print(f"\nCoefficients saved to: {results_file}")
    
    # 7. Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"Testing Accuracy: {results['test_accuracy']:.4f}")
    if results['auc']:
        print(f"ROC AUC Score: {results['auc']:.4f}")
    
    print("\nInterpretation:")
    print("- Positive coefficients indicate the feature predicts UPWARD movement")
    print("- Negative coefficients indicate the feature predicts DOWNWARD movement")
    print("- Larger absolute values indicate stronger predictive power")
    
    print("\nFiles generated:")
    print(f"  1. {output_file}")
    print(f"  2. {results_file}")

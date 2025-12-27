
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import os

def load_data():
    path = r'c:\key\wise_adviser_cursor_context\Caria_repo\caria\docs\research\outputs\Table_Theory_Data.csv'
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # Ensure column names map correctly
        if 'Basal_Corr' in df.columns:
            df = df.rename(columns={'Basal_Corr': 'Connectivity'})
        if 'Risk_Metric' in df.columns:
             df = df.rename(columns={'Risk_Metric': 'Future_DD_Mag'})
        return df.dropna()
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def run_causality_tests(df, maxit=12):
    """
    Run Granger Causality tests for ASF -> Risk and Risk -> ASF
    """
    # 1. Prepare Data
    # Risk is forward looking, so we need to align current ASF with *future* Risk?
    # No, Granger tests if X lags predict Y.
    # So we use time series: ASF_t and Realized_Risk_t (contemporaneous or lagged?)
    # "Future_DD_Mag" in the CSV is likely calculated at t for t+h. 
    # Let's use the realized drawdown at t (actually Future_DD_Mag is shifted).
    # Correct approach:
    # Does ASF_t Granger Cause Realized_Risk_{t}?
    # We want to see if past ASF predicts current Risk.
    
    data = df[['Future_DD_Mag', 'ASF']].dropna()
    
    print("\n--- Granger Causality: Does ASF cause Risk? (Risk ~ ASF_lags) ---")
    # input order: x2 (predictand), x1 (predictor)
    gc_res_1 = grangercausalitytests(data[['Future_DD_Mag', 'ASF']], maxit, verbose=True)
    
    print("\n--- Granger Causality: Does Risk cause ASF? (Reverse Causality) ---")
    gc_res_2 = grangercausalitytests(data[['ASF', 'Future_DD_Mag']], maxit, verbose=True)
    
    return gc_res_1, gc_res_2

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        run_causality_tests(df)

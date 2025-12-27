import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

# Load Data
try:
    df = pd.read_csv('outputs/Table_Theory_Data.csv', index_col=0, parse_dates=True)
except:
    df = pd.read_csv('docs/research/outputs/Table_Theory_Data.csv', index_col=0, parse_dates=True)

# Define Variables
# Risk = Future_DD_Mag
# Fragility = ASF
# Connectivity = Basal_Corr
# Interaction = ASF * Basal_Corr

if 'Future_DD_Mag' not in df.columns:
    if 'Next_1M_DD' in df.columns:
        df['Future_DD_Mag'] = df['Next_1M_DD']
    else:
        raise ValueError("Target variable 'Future_DD_Mag' not found")

# Interaction term
df['Interaction'] = df['ASF'] * df['Basal_Corr']

# Variables for regression
y = df['Future_DD_Mag']
X = df[['ASF', 'Basal_Corr', 'Interaction']]
X = sm.add_constant(X)

# Rolling Regression (10-year window approx 520 weeks)
window = 520
rolling_betas = []
dates = []

print(f"Starting rolling regression with window {window}...")

for i in range(window, len(df)):
    y_window = y.iloc[i-window:i]
    X_window = X.iloc[i-window:i]
    
    # Require decent coverage
    if len(y_window.dropna()) < window * 0.8:
        continue
        
    try:
        model = sm.OLS(y_window, X_window, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 12})
        # Beta_3 is the coefficient for 'Interaction'
        beta_3 = model.params['Interaction']
        # t_stat = model.tvalues['Interaction']
        rolling_betas.append(beta_3)
        dates.append(df.index[i])
    except Exception as e:
        continue

# Create Series
beta_series = pd.Series(rolling_betas, index=dates)

# Plotting
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

plt.plot(beta_series, color='#1f77b4', linewidth=2, label=r'Interaction Coefficient $\beta_3$')
plt.axhline(0, color='red', linestyle='--', alpha=0.8, label='Zero Threshold')

# Highlight Regimes if needed
# The user mentioned "mid-2000s" shift.
# Let's add the shaded region mentioned in the caption: "post-2005 period"
# We can shade from 2005 onwards
plt.axvspan(pd.Timestamp("2005-01-01"), beta_series.index[-1], color='gray', alpha=0.1, label='Disintegration Regime Era')

plt.title(r'Rolling 10-Year Estimation of Fragility-Risk Interaction ($\beta_3$)', fontsize=14)
plt.ylabel(r'Interaction Coefficient $\beta_3$', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/research/outputs/Figure_Rolling_Beta3.png', dpi=300)
print("Generated Figure_Rolling_Beta3.png")

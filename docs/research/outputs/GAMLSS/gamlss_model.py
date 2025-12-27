"""
GAMLSS Model Implementation for Phase Transition Analysis
Implements Beta distribution GAMLSS models as alternative to threshold regression
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit, logit
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_price_data(data_dir='coodination_data'):
    """
    Load all asset price data from CSV files in coodination_data directory.
    
    Returns:
        prices: DataFrame with dates as index, assets as columns
    """
    # GAMLSS folder is in outputs/, so go up one level then into coodination_data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), data_dir)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    prices_dict = {}
    files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    
    print(f"Loading {len(files)} asset files...")
    
    for filename in files:
        try:
            filepath = os.path.join(data_path, filename)
            df = pd.read_csv(filepath)
            
            # Handle different date column names
            date_col = None
            for col in ['date', 'Date', 'DATE']:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col is None:
                continue
            
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            df = df.sort_index()
            
            # Extract price column
            price_col = None
            for col in ['adjClose', 'close', 'Close', 'Adj Close']:
                if col in df.columns:
                    price_col = col
                    break
            
            if price_col is None:
                continue
            
            asset_name = filename.replace('.csv', '')
            prices_dict[asset_name] = df[price_col]
            
        except Exception as e:
            print(f"  Warning: Could not load {filename}: {e}")
            continue
    
    if len(prices_dict) == 0:
        raise ValueError("No price data loaded!")
    
    prices = pd.DataFrame(prices_dict).sort_index()
    
    # Filter to common date range (2007-2024 for ETF data)
    prices = prices.loc['2007-01-01':]
    
    # Drop assets with too much missing data
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.5))
    
    # Forward fill missing values
    prices = prices.ffill().dropna()
    
    print(f"Loaded {len(prices.columns)} assets, {len(prices)} dates")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    return prices


def compute_spectral_entropy(returns, window=63):
    """
    Compute spectral entropy from rolling correlation matrices.
    
    Args:
        returns: DataFrame of asset returns
        window: Rolling window size (default 63 days)
    
    Returns:
        entropy: Series of spectral entropy values
    """
    entropy_list = []
    dates = []
    
    for i in range(window, len(returns)):
        window_data = returns.iloc[i-window:i]
        
        try:
            corr_matrix = window_data.corr().values
            
            if np.any(np.isnan(corr_matrix)):
                entropy_list.append(np.nan)
                dates.append(returns.index[i])
                continue
            
            # Compute eigenvalues
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zero
            
            if len(eigenvalues) == 0:
                entropy_list.append(np.nan)
                dates.append(returns.index[i])
                continue
            
            # Normalize eigenvalues
            p = eigenvalues / eigenvalues.sum()
            
            # Compute spectral entropy
            H = -np.sum(p * np.log(p)) / np.log(len(p))
            entropy_list.append(H)
            dates.append(returns.index[i])
            
        except Exception as e:
            entropy_list.append(np.nan)
            dates.append(returns.index[i])
    
    return pd.Series(entropy_list, index=dates, name='Entropy')


def compute_connectivity(returns, window=63):
    """
    Compute connectivity as mean pairwise correlation.
    
    Args:
        returns: DataFrame of asset returns
        window: Rolling window size
    
    Returns:
        connectivity: Series of mean pairwise correlations
    """
    connectivity_list = []
    dates = []
    
    for i in range(window, len(returns)):
        window_data = returns.iloc[i-window:i]
        
        try:
            corr_matrix = window_data.corr().values
            
            if np.any(np.isnan(corr_matrix)):
                connectivity_list.append(np.nan)
                dates.append(returns.index[i])
                continue
            
            # Mean off-diagonal correlation
            mask = ~np.eye(len(corr_matrix), dtype=bool)
            mean_corr = corr_matrix[mask].mean()
            connectivity_list.append(mean_corr)
            dates.append(returns.index[i])
            
        except Exception:
            connectivity_list.append(np.nan)
            dates.append(returns.index[i])
    
    return pd.Series(connectivity_list, index=dates, name='Connectivity')


def compute_asf(entropy, theta=0.995):
    """
    Compute Accumulated Spectral Fragility (ASF).
    
    Args:
        entropy: Series of spectral entropy values
        theta: Persistence parameter (default 0.995, half-life ≈ 139 days)
    
    Returns:
        asf: Series of ASF values
    """
    fragility = 1 - entropy
    asf = fragility.ewm(alpha=1-theta, adjust=False).mean()
    return asf.rename('ASF')


def compute_future_drawdown(prices, lookahead=21):
    """
    Compute forward-looking maximum drawdown.
    
    Args:
        prices: DataFrame of asset prices
        lookahead: Number of days ahead to look (default 21 ≈ 1 month)
    
    Returns:
        future_drawdown: Series of maximum drawdowns
    """
    # Equal-weighted portfolio
    portfolio_price = prices.mean(axis=1)
    
    drawdown_list = []
    dates = []
    
    for i in range(len(portfolio_price) - lookahead):
        current_date = portfolio_price.index[i]
        future_window = portfolio_price.iloc[i+1:i+1+lookahead]
        
        if len(future_window) == 0:
            continue
        
        # Compute running maximum
        peak = future_window.expanding().max()
        
        # Compute drawdown
        drawdown = (peak - future_window) / peak
        
        # Maximum drawdown in the window
        max_dd = drawdown.max()
        
        drawdown_list.append(max_dd)
        dates.append(current_date)
    
    return pd.Series(drawdown_list, index=dates, name='Future_Drawdown')


def prepare_data(data_dir='coodination_data', window=63, theta=0.995, lookahead=21):
    """
    Prepare complete dataset for GAMLSS modeling.
    
    Returns:
        df: DataFrame with columns: ASF, Connectivity, Future_Drawdown
    """
    print("="*70)
    print("GAMLSS DATA PREPARATION")
    print("="*70)
    
    # Load prices
    prices = load_price_data(data_dir)
    
    # Compute returns
    print("\nComputing returns...")
    returns = prices.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], 0)
    print(f"Returns: {len(returns)} observations, {len(returns.columns)} assets")
    
    # Compute spectral entropy
    print("\nComputing spectral entropy...")
    entropy = compute_spectral_entropy(returns, window=window)
    print(f"Entropy: mean={entropy.mean():.4f}, std={entropy.std():.4f}")
    
    # Compute ASF
    print("\nComputing ASF...")
    asf = compute_asf(entropy, theta=theta)
    print(f"ASF: mean={asf.mean():.4f}, std={asf.std():.4f}")
    
    # Compute connectivity
    print("\nComputing connectivity...")
    connectivity = compute_connectivity(returns, window=window)
    print(f"Connectivity: mean={connectivity.mean():.4f}, std={connectivity.std():.4f}")
    
    # Compute future drawdown
    print("\nComputing future drawdown...")
    future_dd = compute_future_drawdown(prices, lookahead=lookahead)
    print(f"Future Drawdown: mean={future_dd.mean():.4f}, std={future_dd.std():.4f}")
    
    # Combine into DataFrame
    df = pd.DataFrame({
        'ASF': asf,
        'Connectivity': connectivity,
        'Future_Drawdown': future_dd
    }).dropna()
    
    # Transform drawdowns for Beta distribution (avoid exact 0/1)
    df['Future_Drawdown'] = np.clip(df['Future_Drawdown'], 0.0001, 0.9999)
    
    print(f"\nFinal dataset: {len(df)} observations")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    return df


# ============================================================================
# GAMLSS MODEL IMPLEMENTATION
# ============================================================================

class BetaGAMLSS:
    """
    Beta distribution GAMLSS model.
    
    Models:
    - Location (μ): logit(μ) = X_mu @ beta_mu
    - Scale (σ): log(σ) = X_sigma @ beta_sigma
    
    Beta distribution parameterization:
    - α = μ·φ, β = (1-μ)·φ
    - φ = 1/σ² (precision parameter)
    """
    
    def __init__(self):
        self.beta_mu = None
        self.beta_sigma = None
        self.X_mu = None
        self.X_sigma = None
        self.y = None
        self.fitted = False
    
    def _prepare_design_matrix(self, X_mu, X_sigma=None):
        """Prepare design matrices with intercept."""
        # Add intercept to location model
        if X_mu.ndim == 1:
            X_mu = X_mu.reshape(-1, 1)
        X_mu = np.column_stack([np.ones(len(X_mu)), X_mu])
        
        # Add intercept to scale model (use same if not specified)
        if X_sigma is None:
            X_sigma = X_mu.copy()
        else:
            if X_sigma.ndim == 1:
                X_sigma = X_sigma.reshape(-1, 1)
            X_sigma = np.column_stack([np.ones(len(X_sigma)), X_sigma])
        
        return X_mu, X_sigma
    
    def _beta_loglik(self, params, y, X_mu, X_sigma):
        """Negative log-likelihood for Beta distribution."""
        n_mu = X_mu.shape[1]
        beta_mu = params[:n_mu]
        beta_sigma = params[n_mu:]
        
        # Transform to natural parameters
        mu = expit(X_mu @ beta_mu)  # logit inverse
        sigma = np.exp(X_sigma @ beta_sigma)  # log link
        
        # Compute Beta parameters
        phi = 1 / (sigma**2)
        alpha = mu * phi
        beta = (1 - mu) * phi
        
        # Ensure parameters are valid
        alpha = np.clip(alpha, 1e-6, 1e6)
        beta = np.clip(beta, 1e-6, 1e6)
        
        # Log-likelihood
        loglik = np.sum(stats.beta.logpdf(y, alpha, beta))
        
        return -loglik  # Return negative for minimization
    
    def fit(self, y, X_mu, X_sigma=None, method='BFGS'):
        """
        Fit Beta GAMLSS model.
        
        Args:
            y: Target variable (drawdowns, bounded [0,1])
            X_mu: Design matrix for location model
            X_sigma: Design matrix for scale model (optional)
            method: Optimization method
        """
        X_mu, X_sigma = self._prepare_design_matrix(X_mu, X_sigma)
        
        # Initial parameters (zeros for intercepts, small values for others)
        n_mu = X_mu.shape[1]
        n_sigma = X_sigma.shape[1]
        initial_params = np.zeros(n_mu + n_sigma)
        initial_params[0] = logit(np.mean(y))  # Intercept for mu
        initial_params[n_mu] = np.log(np.std(y))  # Intercept for sigma
        
        # Optimize
        result = minimize(
            self._beta_loglik,
            initial_params,
            args=(y, X_mu, X_sigma),
            method=method,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        # Store results
        self.beta_mu = result.x[:n_mu]
        self.beta_sigma = result.x[n_mu:]
        self.X_mu = X_mu
        self.X_sigma = X_sigma
        self.y = y
        self.fitted = True
        
        return result
    
    def predict(self, X_mu, X_sigma=None, return_params=False, add_intercept=True):
        """
        Predict mean and scale parameters.
        
        Args:
            X_mu: Design matrix for location (without intercept if add_intercept=True)
            X_sigma: Design matrix for scale (without intercept if add_intercept=True)
            return_params: If True, return (mu, sigma) tuple
            add_intercept: If True, add intercept column (default True)
        
        Returns:
            mu: Predicted mean (or tuple (mu, sigma) if return_params=True)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if add_intercept:
            X_mu, X_sigma = self._prepare_design_matrix(X_mu, X_sigma)
        else:
            # If intercept already included, just ensure correct shape
            if X_mu.ndim == 1:
                X_mu = X_mu.reshape(-1, 1)
            if X_sigma is None:
                X_sigma = X_mu.copy()
            elif X_sigma.ndim == 1:
                X_sigma = X_sigma.reshape(-1, 1)
            
            # Verify dimensions match
            if X_mu.shape[1] != len(self.beta_mu):
                raise ValueError(f"Dimension mismatch: X_mu has {X_mu.shape[1]} columns but beta_mu has {len(self.beta_mu)} elements")
            if X_sigma.shape[1] != len(self.beta_sigma):
                raise ValueError(f"Dimension mismatch: X_sigma has {X_sigma.shape[1]} columns but beta_sigma has {len(self.beta_sigma)} elements")
        
        mu = expit(X_mu @ self.beta_mu)
        sigma = np.exp(X_sigma @ self.beta_sigma)
        
        if return_params:
            return mu, sigma
        return mu
    
    def loglikelihood(self, y=None):
        """Compute log-likelihood."""
        if y is None:
            y = self.y
        if not self.fitted:
            raise ValueError("Model must be fitted")
        
        mu = expit(self.X_mu @ self.beta_mu)
        sigma = np.exp(self.X_sigma @ self.beta_sigma)
        phi = 1 / (sigma**2)
        alpha = mu * phi
        beta = (1 - mu) * phi
        
        return np.sum(stats.beta.logpdf(y, alpha, beta))
    
    def aic(self, y=None):
        """Compute AIC."""
        if y is None:
            y = self.y
        n_params = len(self.beta_mu) + len(self.beta_sigma)
        return 2 * n_params - 2 * self.loglikelihood(y)
    
    def bic(self, y=None):
        """Compute BIC."""
        if y is None:
            y = self.y
        n_params = len(self.beta_mu) + len(self.beta_sigma)
        n_obs = len(y)
        return n_params * np.log(n_obs) - 2 * self.loglikelihood(y)


# ============================================================================
# MODEL SPECIFICATIONS
# ============================================================================

def model_1_linear_interaction(df):
    """
    Model 1: Linear Interaction
    logit(μ) = β₀ + β₁·ASF + β₂·Connectivity + β₃·(ASF × Connectivity)
    log(σ) = γ₀ + γ₁·ASF + γ₂·Connectivity
    """
    X_mu = np.column_stack([
        df['ASF'].values,
        df['Connectivity'].values,
        (df['ASF'] * df['Connectivity']).values
    ])
    
    X_sigma = np.column_stack([
        df['ASF'].values,
        df['Connectivity'].values
    ])
    
    y = df['Future_Drawdown'].values
    
    model = BetaGAMLSS()
    result = model.fit(y, X_mu, X_sigma)
    
    return model, result


def model_2_threshold_like(df, tau=0.14):
    """
    Model 2: Threshold-like with Smooth Transition
    logit(μ) = β₀ + β₁·ASF·I(C ≤ τ) + β₂·ASF·I(C > τ) + β₃·smooth_transition(C, τ)
    log(σ) = γ₀ + γ₁·ASF + γ₂·Connectivity
    
    smooth_transition uses logistic function: 1 / (1 + exp(-k*(C - τ)))
    """
    C = df['Connectivity'].values
    ASF = df['ASF'].values
    
    # Indicator functions
    I_low = (C <= tau).astype(float)
    I_high = (C > tau).astype(float)
    
    # Smooth transition (k=50 for sharp but smooth transition)
    k = 50
    smooth_trans = 1 / (1 + np.exp(-k * (C - tau)))
    
    X_mu = np.column_stack([
        ASF * I_low,
        ASF * I_high,
        smooth_trans
    ])
    
    X_sigma = np.column_stack([
        ASF,
        C
    ])
    
    y = df['Future_Drawdown'].values
    
    model = BetaGAMLSS()
    result = model.fit(y, X_mu, X_sigma)
    
    return model, result


def model_3_simple_location(df):
    """
    Model 3: Simple location model (for comparison)
    logit(μ) = β₀ + β₁·ASF + β₂·Connectivity
    log(σ) = γ₀ (constant scale)
    """
    X_mu = np.column_stack([
        df['ASF'].values,
        df['Connectivity'].values
    ])
    
    # Constant scale (no covariates)
    X_sigma = None
    
    y = df['Future_Drawdown'].values
    
    model = BetaGAMLSS()
    result = model.fit(y, X_mu, X_sigma)
    
    return model, result


if __name__ == "__main__":
    # Prepare data
    df = prepare_data()
    
    # Fit models
    print("\n" + "="*70)
    print("FITTING GAMLSS MODELS")
    print("="*70)
    
    print("\nModel 1: Linear Interaction")
    model1, result1 = model_1_linear_interaction(df)
    print(f"  Log-likelihood: {model1.loglikelihood():.2f}")
    print(f"  AIC: {model1.aic():.2f}")
    print(f"  BIC: {model1.bic():.2f}")
    
    print("\nModel 2: Threshold-like with Smooth Transition")
    model2, result2 = model_2_threshold_like(df, tau=0.14)
    print(f"  Log-likelihood: {model2.loglikelihood():.2f}")
    print(f"  AIC: {model2.aic():.2f}")
    print(f"  BIC: {model2.bic():.2f}")
    
    print("\nModel 3: Simple Location")
    model3, result3 = model_3_simple_location(df)
    print(f"  Log-likelihood: {model3.loglikelihood():.2f}")
    print(f"  AIC: {model3.aic():.2f}")
    print(f"  BIC: {model3.bic():.2f}")
    
    # Save data
    output_dir = os.path.dirname(__file__)
    df.to_csv(os.path.join(output_dir, 'gamlss_data.csv'))
    print(f"\nData saved to {os.path.join(output_dir, 'gamlss_data.csv')}")


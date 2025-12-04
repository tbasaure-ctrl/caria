"""
CARIA V22 Monthly Prediction Generator
Fetches market data and generates direction predictions

Strategy:
- Yahoo Finance (FREE): Stock indices, FX, commodities, VIX - unlimited calls
- FRED (FREE): US economic indicators - unlimited calls  
- Trading Economics: Only if TE_API_KEY is set, minimal calls for macro data

Run locally: python generate_weekly_predictions.py
Run in CI: GitHub Actions workflow (update-predictions.yml) - monthly
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import shutil

# ============================================================
# Configuration
# ============================================================

COUNTRIES = ['USA', 'CHN', 'JPN', 'DEU', 'GBR', 'FRA', 'IND', 'BRA',
             'CAN', 'KOR', 'AUS', 'MEX', 'IDN', 'ZAF', 'CHL',
             'SGP', 'NLD', 'HKG', 'CHE', 'TWN', 'VNM', 'NOR']

COUNTRY_NAMES = {
    'USA': 'United States', 'CHN': 'China', 'JPN': 'Japan',
    'DEU': 'Germany', 'GBR': 'United Kingdom', 'FRA': 'France',
    'IND': 'India', 'BRA': 'Brazil', 'CAN': 'Canada',
    'KOR': 'South Korea', 'AUS': 'Australia', 'MEX': 'Mexico',
    'IDN': 'Indonesia', 'ZAF': 'South Africa', 'CHL': 'Chile',
    'SGP': 'Singapore', 'NLD': 'Netherlands', 'HKG': 'Hong Kong',
    'CHE': 'Switzerland', 'TWN': 'Taiwan', 'VNM': 'Vietnam', 'NOR': 'Norway'
}

# Yahoo Finance tickers (FREE - unlimited)
INDEX_TICKERS = {
    'USA': '^GSPC', 'CHN': '000001.SS', 'JPN': '^N225', 'DEU': '^GDAXI',
    'GBR': '^FTSE', 'FRA': '^FCHI', 'IND': '^BSESN', 'BRA': '^BVSP',
    'CAN': '^GSPTSE', 'KOR': '^KS11', 'AUS': '^AXJO', 'MEX': '^MXX',
    'IDN': '^JKSE', 'ZAF': '^J203.JO', 'CHL': '^SPIPSA',
    'SGP': '^STI', 'NLD': '^AEX', 'HKG': '^HSI', 'CHE': '^SSMI',
    'TWN': '^TWII', 'VNM': '^VNINDEX', 'NOR': 'OSEBX.OL'
}

FX_TICKERS = {
    'EUR': 'EURUSD=X', 'JPY': 'JPY=X', 'GBP': 'GBPUSD=X', 'CNY': 'CNY=X',
    'INR': 'INR=X', 'BRL': 'BRL=X', 'CAD': 'CAD=X', 'KRW': 'KRW=X',
    'AUD': 'AUDUSD=X', 'MXN': 'MXN=X', 'IDR': 'IDR=X', 'ZAR': 'ZAR=X',
    'CLP': 'CLP=X', 'SGD': 'SGD=X', 'HKD': 'HKD=X', 'CHF': 'CHF=X',
    'TWD': 'TWD=X', 'VND': 'VND=X', 'NOK': 'NOK=X'
}

COMMODITY_TICKERS = {
    'Oil': 'CL=F', 'Gold': 'GC=F', 'Copper': 'HG=F', 'Silver': 'SI=F'
}

GLOBAL_TICKERS = {
    'VIX': '^VIX', 'DXY': 'DX-Y.NYB', 'US10Y': '^TNX', 'US2Y': '^IRX'
}

# FRED Series (FREE - unlimited with API key)
FRED_SERIES = {
    'UNRATE': 'US Unemployment',
    'CPIAUCSL': 'US CPI',
    'FEDFUNDS': 'Fed Funds Rate',
    'T10Y2Y': 'Yield Curve 10Y-2Y',
    'UMCSENT': 'Consumer Sentiment',
    'ICSA': 'Initial Claims'
}


# ============================================================
# Model Architecture (must match checkpoint)
# ============================================================

class EconomicRelationshipDiscoverer(nn.Module):
    def __init__(self, num_nodes, in_feats, d_model=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        
        self.input_proj = nn.Sequential(
            nn.Linear(in_feats, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.node_embed = nn.Parameter(torch.randn(num_nodes, 64) * 0.1)
        self.from_embed = nn.Parameter(torch.randn(num_nodes, 32) * 0.1)
        self.to_embed = nn.Parameter(torch.randn(num_nodes, 32) * 0.1)
        self.adj_temperature = 0.5
        
        self.graph_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model)
            ) for _ in range(n_layers)
        ])
        
        self.temporal_short = nn.GRU(d_model * num_nodes, d_model, batch_first=True)
        self.temporal_mid = nn.GRU(d_model * num_nodes, d_model, batch_first=True)
        self.temporal_long = nn.GRU(d_model * num_nodes, d_model, batch_first=True)
        
        self.direction_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_nodes)
        )
        
        self.relationship_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
    
    def get_adjacency(self):
        adj = torch.mm(self.from_embed, self.to_embed.t())
        return F.softmax(adj / self.adj_temperature, dim=-1)
    
    def forward(self, x, return_embeddings=False):
        b, s, n, f = x.shape
        x = self.input_proj(x)
        adj = self.get_adjacency()
        
        for layer in self.graph_layers:
            x_agg = torch.einsum('bsnd,nm->bsmd', x, adj)
            x = layer(x + x_agg)
        
        d = x.shape[-1]
        x_flat = x.view(b, s, n * d)
        
        _, h_short = self.temporal_short(x_flat[:, -10:, :])
        _, h_mid = self.temporal_mid(x_flat[:, -20:, :])
        _, h_long = self.temporal_long(x_flat)
        
        h = torch.cat([h_short.squeeze(0), h_mid.squeeze(0), h_long.squeeze(0)], dim=-1)
        direction = self.direction_head(h)
        
        if return_embeddings:
            return direction, self.node_embed, adj
        return direction


# ============================================================
# Data Fetching (Optimized for minimal API costs)
# ============================================================

def fetch_yahoo_data(seq_len=60):
    """Fetch market data from Yahoo Finance (FREE - unlimited)"""
    import yfinance as yf
    
    print("📊 Fetching from Yahoo Finance (FREE)...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=seq_len * 2)
    
    all_data = {}
    
    # Fetch indices
    print("  - Stock indices (22 countries)...")
    for country, ticker in INDEX_TICKERS.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) > 0:
                all_data[f'{country}_index'] = df['Close'].pct_change()
        except Exception as e:
            print(f"    ⚠ {ticker}: {e}")
    
    # Fetch FX
    print("  - FX rates...")
    for currency, ticker in FX_TICKERS.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) > 0:
                all_data[f'{currency}_fx'] = df['Close'].pct_change()
        except:
            pass
    
    # Fetch commodities
    print("  - Commodities...")
    for name, ticker in COMMODITY_TICKERS.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) > 0:
                all_data[f'{name}'] = df['Close'].pct_change()
        except:
            pass
    
    # Fetch global indicators
    print("  - Global indicators (VIX, DXY, yields)...")
    for name, ticker in GLOBAL_TICKERS.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) > 0:
                all_data[f'{name}'] = df['Close'].pct_change()
        except:
            pass
    
    combined = pd.DataFrame(all_data)
    combined = combined.dropna(how='all').fillna(0)
    print(f"  ✓ Yahoo: {len(combined.columns)} series, {len(combined)} days")
    return combined


def fetch_fred_data(seq_len=60):
    """Fetch US economic data from FRED (FREE with API key)"""
    fred_key = os.environ.get('FRED_API_KEY')
    if not fred_key:
        print("  ⚠ FRED_API_KEY not set, skipping FRED data")
        return pd.DataFrame()
    
    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_key)
        
        print("📊 Fetching from FRED (FREE)...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=seq_len * 7)  # Monthly data needs more history
        
        all_data = {}
        for series_id, name in FRED_SERIES.items():
            try:
                data = fred.get_series(series_id, start_date, end_date)
                if len(data) > 0:
                    all_data[series_id] = data
                    print(f"    ✓ {name}")
            except Exception as e:
                print(f"    ⚠ {series_id}: {e}")
        
        if all_data:
            combined = pd.DataFrame(all_data)
            combined = combined.ffill().bfill()  # Forward/backward fill for monthly data
            print(f"  ✓ FRED: {len(combined.columns)} series")
            return combined
    except ImportError:
        print("  ⚠ fredapi not installed")
    except Exception as e:
        print(f"  ⚠ FRED error: {e}")
    
    return pd.DataFrame()


def fetch_market_data(seq_len=60):
    """Fetch all market data from free sources"""
    
    # Primary: Yahoo Finance (FREE, unlimited)
    yahoo_data = fetch_yahoo_data(seq_len)
    
    # Secondary: FRED (FREE with API key)
    fred_data = fetch_fred_data(seq_len)
    
    # Merge if FRED data available
    if len(fred_data) > 0:
        # Resample FRED to daily and merge
        fred_daily = fred_data.resample('D').ffill()
        yahoo_data = yahoo_data.join(fred_daily, how='left')
        yahoo_data = yahoo_data.fillna(method='ffill').fillna(0)
    
    print(f"\n✓ Total: {len(yahoo_data.columns)} features, {len(yahoo_data)} days")
    return yahoo_data


def build_feature_tensor(market_data, num_nodes=22, num_features=79, seq_len=45):
    """Build the feature tensor for model input"""
    print("🔧 Building feature tensor...")
    
    # Initialize tensor
    T = len(market_data)
    if T < seq_len:
        print(f"  Warning: Only {T} days of data, padding to {seq_len}")
        # Pad with zeros
        padding = seq_len - T
        market_data = pd.concat([
            pd.DataFrame(np.zeros((padding, len(market_data.columns))), columns=market_data.columns),
            market_data
        ]).reset_index(drop=True)
        T = seq_len
    
    # Take last seq_len days
    market_data = market_data.tail(seq_len).reset_index(drop=True)
    
    # Build tensor [seq_len, num_nodes, num_features]
    tensor = np.zeros((seq_len, num_nodes, num_features), dtype=np.float32)
    
    for i, country in enumerate(COUNTRIES[:num_nodes]):
        # Index returns (feature 0)
        col = f'{country}_index'
        if col in market_data.columns:
            tensor[:, i, 0] = market_data[col].values
        
        # Fill remaining features with available data
        feature_idx = 1
        for col in market_data.columns:
            if feature_idx >= num_features:
                break
            if col != f'{country}_index':
                tensor[:, i, feature_idx] = market_data[col].values
                feature_idx += 1
    
    # Normalize
    mean = tensor.mean(axis=0, keepdims=True)
    std = tensor.std(axis=0, keepdims=True) + 1e-8
    tensor = (tensor - mean) / std
    
    print(f"  ✓ Tensor shape: {tensor.shape}")
    return tensor


# ============================================================
# Prediction Generation
# ============================================================

def load_model(checkpoint_path):
    """Load the trained model"""
    print(f"🧠 Loading model from {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    in_feats = checkpoint['input_proj.0.weight'].shape[1]
    num_nodes = checkpoint['node_embed'].shape[0]
    
    model = EconomicRelationshipDiscoverer(
        num_nodes=num_nodes,
        in_feats=in_feats,
        d_model=64,
        n_layers=2,
        dropout=0.3
    )
    
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"  ✓ Loaded model with {num_nodes} nodes, {in_feats} features")
    return model, num_nodes, in_feats


def generate_predictions(model, data_tensor, num_nodes):
    """Generate direction predictions"""
    print("🔮 Generating predictions...")
    
    x_tensor = torch.FloatTensor(data_tensor).unsqueeze(0)  # [1, seq, nodes, feats]
    
    with torch.no_grad():
        raw_output = model(x_tensor)
        probs = torch.sigmoid(raw_output).numpy()[0]
        predictions = raw_output.numpy()[0]
    
    directions = ['UP' if p > 0.5 else 'DOWN' for p in probs]
    confidences = [abs(p - 0.5) * 2 for p in probs]
    
    max_conf = max(confidences) if confidences else 1
    normalized_confidences = [c / max_conf for c in confidences]
    
    print(f"  ✓ Generated predictions for {num_nodes} countries")
    return predictions, directions, confidences, normalized_confidences


def save_predictions(predictions, directions, confidences, normalized_confidences, num_nodes):
    """Save predictions to JSON files"""
    print("💾 Saving predictions...")
    
    output = {
        "predictionDate": datetime.now().strftime('%Y-%m-%d'),
        "modelVersion": "V22-Relationships",
        "modelAccuracy": "59.6%",
        "countries": COUNTRIES[:num_nodes],
        "countryNames": [COUNTRY_NAMES.get(c, c) for c in COUNTRIES[:num_nodes]],
        "predictions": predictions.tolist(),
        "directions": directions,
        "confidences": confidences,
        "normalizedConfidences": normalized_confidences,
        "summary": {
            "totalCountries": num_nodes,
            "upPredictions": sum(1 for d in directions if d == 'UP'),
            "downPredictions": sum(1 for d in directions if d == 'DOWN'),
            "avgConfidence": float(np.mean(confidences)),
            "maxConfidence": float(max(confidences)),
            "minConfidence": float(min(confidences))
        }
    }
    
    # Save to models directory
    models_path = 'caria_direction_predictions.json'
    with open(models_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  ✓ Saved: {models_path}")
    
    # Copy to frontend
    frontend_path = '../frontend/caria-app/public/data/caria_direction_predictions.json'
    if os.path.exists(os.path.dirname(frontend_path)):
        shutil.copy(models_path, frontend_path)
        print(f"  ✓ Copied to: {frontend_path}")
    else:
        print(f"  ⚠ Frontend path not found, skipping copy")
    
    return output


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("CARIA V22 WEEKLY PREDICTION GENERATOR")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Load model
        model, num_nodes, in_feats = load_model('caria_v22_relationships.pth')
        
        # Fetch market data
        market_data = fetch_market_data(seq_len=60)
        
        # Build feature tensor
        data_tensor = build_feature_tensor(market_data, num_nodes, in_feats)
        
        # Generate predictions
        predictions, directions, confidences, normalized_confidences = generate_predictions(
            model, data_tensor, num_nodes
        )
        
        # Save predictions
        output = save_predictions(
            predictions, directions, confidences, normalized_confidences, num_nodes
        )
        
        # Print summary
        print()
        print("=" * 60)
        print("PREDICTION SUMMARY")
        print("=" * 60)
        print(f"Total Countries: {output['summary']['totalCountries']}")
        print(f"UP Predictions: {output['summary']['upPredictions']}")
        print(f"DOWN Predictions: {output['summary']['downPredictions']}")
        print(f"Avg Confidence: {output['summary']['avgConfidence']:.1%}")
        print()
        
        print("Predictions by Country:")
        for i, (code, name) in enumerate(zip(output['countries'], output['countryNames'])):
            arrow = "▲" if output['directions'][i] == 'UP' else "▼"
            conf = output['confidences'][i]
            print(f"  {code:3} ({name:15}): {arrow} {output['directions'][i]:4} | Conf: {conf:.1%}")
        
        print()
        print("✅ Predictions generated successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


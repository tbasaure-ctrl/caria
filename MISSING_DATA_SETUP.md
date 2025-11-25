# Missing Production Data Files - Setup Guide

## 📋 Summary

Your Caria application requires 4 critical data files for production deployment. Here's the current status:

### ✅ Files Found and Ready:
1. **`regime_hmm_model.pkl`** - ✅ Already in correct location
   - Location: `caria_data/models/regime_hmm_model.pkl`
   - Purpose: Market regime detection (Expansion/Recession)

2. **`historical_events_wisdom.jsonl`** - ✅ Copied to correct location
   - Location: `caria_data/raw/wisdom/historical_events_wisdom.jsonl`
   - Purpose: Timeline of events for crisis simulations
   - Source: Copied from `data/raw/wisdom/2025-11-08/historical_events_wisdom.jsonl`

### ⚠️ Files Missing (Need Generation):
3. **`quality_signals.parquet`** - ❌ Needs generation
   - Required location: `data/silver/fundamentals/quality_signals.parquet`
   - Purpose: Fundamental quality data (ROE, ROA, margins) for stock valuation

4. **`value_signals.parquet`** - ❌ Needs generation
   - Required location: `data/silver/fundamentals/value_signals.parquet`
   - Purpose: Valuation metrics (P/B, P/S, growth rates) for stock analysis

---

## 🚀 Quick Start - Generate Missing Files

### Option 1: Generate Mock Data (Fast - 5 seconds)
**Recommended for testing or if you don't have FMP API access**

```powershell
cd c:\key\wise_adviser_cursor_context\notebooks\caria
python generate_missing_data.py --mock
```

This will create sample data for 25 popular stocks with realistic values.

### Option 2: Download Real Data (Slow - 30-60 minutes)
**Required for production deployment**

1. **Get FMP API Key** (if you don't have one):
   - Sign up at https://financialmodelingprep.com/
   - Free tier gives you 250 requests/day
   - Premium recommended for full S&P 500 data

2. **Set the API key**:
   ```powershell
   $env:FMP_API_KEY='your-fmp-api-key-here'
   ```

3. **Run the download**:
   ```powershell
   cd c:\key\wise_adviser_cursor_context\notebooks\caria
   python generate_missing_data.py
   ```

   This will download fundamentals for ~500 S&P 500 companies.

### Option 3: Verify Current Status
```powershell
python generate_missing_data.py --verify
```

---

## 📂 Directory Structure Created

The following directories have been created:

```
caria/
├── data/
│   └── silver/
│       └── fundamentals/          # ← quality_signals.parquet & value_signals.parquet go here
├── caria_data/
│   ├── models/
│   │   └── regime_hmm_model.pkl   # ✅ Already present
│   └── raw/
│       └── wisdom/
│           └── historical_events_wisdom.jsonl  # ✅ Copied
```

---

## 🔧 What Was Fixed

### 1. `.gitignore` Updates
The `.gitignore` file has been updated to **allow** these critical production files while still ignoring general data files:

```gitignore
# EXCEPTION: Allow critical production data files
!data/silver/fundamentals/quality_signals.parquet
!data/silver/fundamentals/value_signals.parquet
!caria_data/raw/wisdom/historical_events_wisdom.jsonl
```

The regime model was already allowed:
```gitignore
!caria_data/models/regime_hmm_model.pkl
```

### 2. Directory Structure
Created the necessary directory tree that was missing.

---

## 📤 Push to Main Branch

Once you've generated the data files, follow these steps:

### Step 1: Verify all files exist
```powershell
python generate_missing_data.py --verify
```

### Step 2: Check git status
```powershell
git status
```

You should see the new files listed as untracked:
- `data/silver/fundamentals/quality_signals.parquet`
- `data/silver/fundamentals/value_signals.parquet`
- `caria_data/raw/wisdom/historical_events_wisdom.jsonl`
- `.gitignore` (modified)
- `generate_missing_data.py` (new helper script)

### Step 3: Stage the files
```powershell
# Add the data files
git add data/silver/fundamentals/quality_signals.parquet
git add data/silver/fundamentals/value_signals.parquet
git add caria_data/raw/wisdom/historical_events_wisdom.jsonl

# Add the helper files
git add .gitignore
git add generate_missing_data.py
git add MISSING_DATA_SETUP.md
```

### Step 4: Commit
```powershell
git commit -m "Add production data files for Valuation, Alpha Picks, and Crisis Simulator

- Added quality_signals.parquet and value_signals.parquet for fundamentals analysis
- Added historical_events_wisdom.jsonl for crisis simulations
- Updated .gitignore to allow critical production data files
- Added generate_missing_data.py helper script for future updates"
```

### Step 5: Push to main
```powershell
git push origin main
```

---

## ⚙️ For Production Deployment (Railway/Vercel)

### File Size Considerations

The parquet files may be large depending on how many stocks you download:

- **Mock data** (25 stocks): ~50 KB total
- **Real data** (500 stocks, 2 years): ~5-10 MB total

**Note**: If files are too large for git (>100 MB), you may need to:
1. Use Git LFS (Large File Storage)
2. Upload files directly to your production environment
3. Generate files during deployment using environment secrets

### Alternative: Generate Data During Deployment

If you prefer not to commit large data files, you can:

1. **Store only the script** (`generate_missing_data.py`)
2. **Set FMP_API_KEY** as an environment variable in Railway/Vercel
3. **Run during build/startup**:
   ```yaml
   # In railway.json or similar
   build:
     buildCommand: "python generate_missing_data.py"
   ```

---

## 🔍 Verification

### Quick Check
```powershell
# Verify files exist and show sizes
Get-ChildItem -Recurse -Include quality_signals.parquet,value_signals.parquet,historical_events_wisdom.jsonl,regime_hmm_model.pkl | Select-Object FullName, Length
```

### Load and Inspect Data (Python)
```python
import pandas as pd

# Check quality signals
quality = pd.read_parquet('data/silver/fundamentals/quality_signals.parquet')
print(f"Quality signals: {len(quality):,} rows, {quality['ticker'].nunique()} tickers")
print(quality.head())

# Check value signals
value = pd.read_parquet('data/silver/fundamentals/value_signals.parquet')
print(f"Value signals: {len(value):,} rows, {value['ticker'].nunique()} tickers")
print(value.head())
```

---

## 📚 What These Files Do

### 1. **quality_signals.parquet**
Used by the **Valuation & Alpha Stock Picker** modules to score stocks on:
- Return on Invested Capital (ROIC)
- Return on Equity (ROE)
- Profit margins
- Cash flow metrics

### 2. **value_signals.parquet**
Used for valuation analysis:
- Price-to-Book ratio
- Price-to-Sales ratio
- Revenue and income growth rates
- Enterprise value calculations

### 3. **historical_events_wisdom.jsonl**
Powers the **Crisis Simulator**:
- Historical market events (2008 crisis, COVID, etc.)
- News context for each period
- Used to create realistic simulation scenarios

### 4. **regime_hmm_model.pkl**
Market Regime Detection:
- Trained Hidden Markov Model
- Classifies market as Expansion/Recession
- Improves portfolio recommendations based on market conditions

---

## 🆘 Troubleshooting

### "FMP_API_KEY not set" Error
```powershell
# Set for current session
$env:FMP_API_KEY='your-key-here'

# Or edit generate_missing_data.py and add it at line 21:
FMP_API_KEY = "your-fmp-api-key-here"
```

### "Permission Denied" or "File Not Found"
Make sure you're in the correct directory:
```powershell
cd c:\key\wise_adviser_cursor_context\notebooks\caria
```

### Git Says "Files Too Large"
If parquet files exceed 100MB:
```powershell
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.parquet"

# Add and commit
git add .gitattributes
git commit -m "Enable Git LFS for parquet files"
```

### Files Already Exist (Regeneration)
```powershell
# Delete old files
Remove-Item data/silver/fundamentals/*.parquet

# Regenerate
python generate_missing_data.py --mock
# or
python generate_missing_data.py
```

---

## 📞 Next Steps

1. **Generate the data** using one of the options above
2. **Verify** all 4 files are present
3. **Test locally** that your Valuation and Alpha Picks features work
4. **Commit and push** to the main branch
5. **Deploy** to Railway/Vercel and verify in production

---

## 📝 File Manifest

```
✅ caria_data/models/regime_hmm_model.pkl (1,950 bytes)
✅ caria_data/raw/wisdom/historical_events_wisdom.jsonl (8,746 bytes)
⚠️  data/silver/fundamentals/quality_signals.parquet (NEEDS GENERATION)
⚠️  data/silver/fundamentals/value_signals.parquet (NEEDS GENERATION)
```

**Status**: 2/4 files ready, 2 need generation

---

*Last updated: 2025-11-25*
*Generated by: Antigravity AI Assistant*

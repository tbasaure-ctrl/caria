# Caria Risk Engine: Academic Publication Code

## Entropic Resonance and Volatility Compression as Precursors to Systemic Failure

**Author:** Tomás Basaure  
**Date:** December 2025  
**Status:** Ready for Peer Review

---

## Abstract

This repository contains the complete implementation and validation code for the paper:

> **"Entropic Resonance and Volatility Compression as Precursors to Systemic Failure"**

The study investigates precursors of systemic financial crises by analyzing the relationship between Shannon Entropy (H) and Temporal Synchrony (r). The key finding is that systemic failures are driven by **"Entropic Resonance"** - a state of high entropy and high synchronization combined with volatility compression.

### Key Results

- **MCC Score:** 0.5822 (Institutional Grade)
- **Recall:** 84.88% (Detects ~8.5 out of 10 crises)
- **Precision:** 46.79% (Nearly 1:1 signal-to-noise ratio)

---

## Repository Structure

```
caria_publication/
├── README.md                    # This file
├── requirements.txt             # Python dependencies (exact versions)
├── config.yaml                  # All hyperparameters
│
├── src/
│   ├── __init__.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── entropy.py           # Shannon entropy calculations
│   │   ├── synchronization.py   # Kuramoto order parameter
│   │   └── volatility.py        # Crisis detection methods
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── caria_risk_engine.py # Main model
│   │   └── benchmarks.py        # HAR-RV, GARCH, etc.
│   │
│   └── validation/
│       ├── __init__.py
│       ├── walk_forward.py      # Temporal validation
│       ├── statistical_tests.py # McNemar, DM, Bootstrap
│       └── economic_analysis.py # Lead time, utility
│
├── notebooks/
│   └── full_replication.ipynb   # Complete replication notebook
│
├── data/
│   └── README.md                # Data download instructions
│
└── results/
    ├── tables/                  # CSV results
    └── figures/                 # PNG figures
```

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/caria_publication.git
cd caria_publication

# Option A: Fresh virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Option B: Existing environment with OpenBB/other packages
# Use minimal requirements to avoid conflicts:
pip install -r requirements_minimal.txt

# Option C: If you have conflicts, install only what's missing:
pip install -r requirements.txt --upgrade-strategy only-if-needed
```

### 2. Run Replication

```bash
# Launch Jupyter
jupyter notebook notebooks/full_replication.ipynb
```

Or run directly:

```python
from src.models import CariaRiskEngine
from src.features import rolling_shannon_entropy, rolling_correlation_sync

# Load your price data
# prices = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)

# Initialize engine
engine = CariaRiskEngine(use_tree=True)

# Fit on training data
engine.fit(train_prices, train_crisis_labels)

# Predict
predictions = engine.predict(test_prices)

# Get detailed assessment
assessment = engine.assess_risk(recent_prices)
print(f"Current regime: {assessment.regime}")
print(f"Crisis probability: {assessment.probability:.1%}")
```

---

## The Caria Risk Formula

The Decision Tree model identified three distinct risk regimes:

### Regime A: Hypersynchronization (CRITICAL)
- **Condition:** Desynchronization ≤ 0.13 (i.e., |r| > 0.87)
- **Interpretation:** Total market lockstep. Immediate danger regardless of volatility.

### Regime B: Volatility Compression (ALERT)
- **Condition:** 0.13 < Desync ≤ 0.31 AND Volatility ≤ 0.08
- **Interpretation:** The "Calm Before the Storm." Synchronized but suppressed.

### Regime C: Complacency (FRAGILE)
- **Condition:** Desync > 0.31 AND Volatility ≤ 0.06
- **Interpretation:** Extreme calm indicates liquidity fragility.

---

## Reproducibility

All results can be reproduced with:

1. **Random Seed:** 42 (set in all modules)
2. **Python Version:** 3.10+
3. **Dependencies:** Exact versions in requirements.txt
4. **Data:** Publicly available via Yahoo Finance API

---

## Citation

```bibtex
@article{basaure2025entropic,
  title={Entropic Resonance and Volatility Compression as Precursors to Systemic Failure},
  author={Basaure, Tom{\'a}s},
  journal={TBD},
  year={2025}
}
```

---

## License

MIT License - See LICENSE file for details.

---

## Contact

For questions or issues, please open a GitHub issue or contact the author.

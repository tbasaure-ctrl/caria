# CARIA Publication Package

## Paper: "Crisis Anticipation via Resonance, Integration, and Asymmetry"

**Target Journals:** Journal of Financial Economics, Review of Financial Studies, Journal of Finance

---

## 📁 File Structure

```
paper/
├── CARIA_Paper.tex          # LaTeX manuscript (ready for submission)
├── CARIA_Paper_Draft.md     # Markdown version (for editing/review)
└── README.md                # This file

notebooks/
├── CARIA_Publication_Final.py  # Complete analysis script
├── CARIA_FMP_RealData_Colab.ipynb  # Original Colab notebook
└── ...

results/
├── figures/                 # Publication-quality figures
│   ├── fig1_fragility_timeseries.png/pdf
│   ├── fig2_factor_loadings.png/pdf
│   ├── fig3_hysteresis.png/pdf
│   ├── fig4_cusp_surface.png
│   ├── fig5_strategy_performance.png/pdf
│   └── fig6_model_comparison.png/pdf
└── tables/
    ├── table1_summary_statistics.csv/tex
    ├── table2_factor_loadings.csv/tex
    ├── table3_hysteresis.csv
    ├── table4_oos_results.csv
    ├── table5_strategy_performance.csv
    └── abstract.txt

data/
├── sp500_prices_alpha/      # S&P 500 constituent data (503 stocks)
├── data_fmp_1990/           # Index data (SPY, VIX, ^GSPC)
└── ...
```

---

## 🚀 Quick Start

### 1. Generate Results
```bash
cd notebooks/
python CARIA_Publication_Final.py
```

### 2. Compile Paper
```bash
cd paper/
pdflatex CARIA_Paper.tex
bibtex CARIA_Paper
pdflatex CARIA_Paper.tex
pdflatex CARIA_Paper.tex
```

---

## 📊 Key Results Summary

### Hysteresis Effect (Main Finding)
| Fragility Level | P(Tail) Rising | P(Tail) Falling | Difference |
|-----------------|----------------|-----------------|------------|
| Low (D1) | 5.2% | 6.1% | -0.9% |
| Medium (D5) | 10.2% | 9.8% | +0.4% |
| High (D10) | **22.4%** | **16.2%** | **+6.2%*** |

***Statistically significant at p < 0.01**

### Out-of-Sample Performance
| Model | AUC | Improvement |
|-------|-----|-------------|
| RV Only | 0.571 | Baseline |
| Structural | 0.612 | +7.2% |
| F_t Only | 0.628 | +10.0% |
| **F_t + Hysteresis** | **0.654** | **+14.5%** |

### Strategy Performance (OOS)
| Metric | CARIA | Buy & Hold | Improvement |
|--------|-------|------------|-------------|
| CAGR | 9.8% | 7.2% | +2.6% |
| Max DD | -32% | -54% | +22pp |
| Sharpe | 0.72 | 0.48 | +50% |

---

## 📝 Paper Sections

### Abstract (150 words)
Multi-signal early warning system combining 6 theoretical frameworks. Key finding: hysteresis effect in tail risk.

### Introduction
- Motivation: Why crises are hard to predict
- Contribution: Integration + Hysteresis discovery + Economic significance
- Related literature: Systemic risk, critical transitions, catastrophe theory

### Methodology
- CARIA model: 6 signal components
- Factor analysis: Composite index extraction
- Cusp catastrophe: Bifurcation modeling
- Hysteresis framework: Path-dependent risk

### Results
- Factor loadings (Table 2)
- Hysteresis analysis (Table 3, Figure 3)
- OOS prediction (Table 4)
- Trading strategy (Table 5, Figure 5)

### Discussion
- Economic interpretation of hysteresis
- Implications for risk management
- Regulatory considerations
- Limitations

### Conclusion
- Summary of contributions
- Future research directions

---

## 🔬 Figures for Publication

### Figure 1: Fragility Index Time Series
- Panel A: Market price with crisis shading
- Panel B: Composite fragility index F_t
- Panel C: Structural components (absorption, entropy)

### Figure 2: Factor Loadings
- Horizontal bar chart showing contribution of each signal

### Figure 3: Hysteresis Effect (KEY FIGURE)
- Panel A: Tail probability by decile and path
- Panel B: Expected return by decile and path

### Figure 4: Cusp Catastrophe Surface
- 3D visualization of cusp potential with data points

### Figure 5: Strategy Performance
- Panel A: Cumulative returns (log scale)
- Panel B: Market exposure over time

### Figure 6: Model Comparison
- Box plot of OOS AUC across folds

---

## 📋 Tables for Publication

### Table 1: Summary Statistics
Descriptive stats for all fragility indicators

### Table 2: Factor Loadings
Loading of each signal on composite index

### Table 3: Hysteresis Analysis
Tail probability by fragility decile and path direction

### Table 4: OOS Prediction Results
AUC, PR-AUC, Brier score by model

### Table 5: Strategy Performance
CAGR, MaxDD, Sharpe, MAR for strategy vs. benchmark

---

## 🎯 Submission Checklist

- [ ] Run final analysis with complete data
- [ ] Generate all figures at 300 DPI
- [ ] Check table formatting for journal style
- [ ] Proofread abstract and introduction
- [ ] Verify all citations
- [ ] Prepare cover letter
- [ ] Check page limits
- [ ] Submit to SSRN for preprint

---

## 📚 Citation

```bibtex
@article{author2024caria,
  title={CARIA: Crisis Anticipation via Resonance, Integration, and Asymmetry},
  author={[Your Name]},
  journal={Working Paper},
  year={2024}
}
```

---

## 📧 Contact

For questions about the methodology or data, please contact:
- Email: [your.email@institution.edu]
- GitHub: [repository-url]











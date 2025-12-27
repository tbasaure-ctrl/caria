# QJE Submission Checklist

## Overview
- **Journal:** Quarterly Journal of Economics (QJE)
- **Manuscript Title:** When Stability Becomes Fragility: Phase Transitions in Financial Markets
- **Target Length:** ~30 pages (25 content + 5 references)
- **Current Status:** Ready for submission

---

## I. Manuscript Files

### Main Manuscript (`manuscript_qje_main.tex`)
- [x] Title page with author information and acknowledgments
- [x] Abstract (under 250 words)
- [x] Keywords and JEL codes
- [x] Double-spaced, 12pt font, 1.25" margins
- [x] Author-year citation style (AER/Chicago)
- [x] Sections properly numbered

### Content Sections (All Present)
1. [x] **Introduction** - Motivation, contributions, preview of results
2. [x] **Related Literature** - Four subsections covering relevant strands
3. [x] **Theoretical Framework** - Definitions, propositions with proofs, econometric specification
4. [x] **Data and Methodology** - Data sources, variable construction, estimation procedure
5. [x] **Empirical Results** - Main threshold regression, visual evidence
6. [x] **Robustness** - Surrogate tests, placebo, subsamples, alternatives, OOS forecasting
7. [x] **Structural Hysteresis** - Hysteresis loop analysis
8. [x] **Economic Significance** - Strategy validation
9. [x] **Discussion** - Policy implications, limitations
10. [x] **Conclusion** - Summary of findings and implications

### Online Appendix (`online_appendix_qje.tex`)
- [x] Appendix A: Full proofs of propositions
- [x] Appendix B: Model calibration exercise
- [x] Appendix C: Extended robustness tests (10+ tables)
- [x] Appendix D: Hysteresis analysis details
- [x] Appendix E: Strategy validation details
- [x] Appendix F: Replication code documentation

### Bibliography (`references_qje.bib`)
- [x] 51 references in AER/Chicago style
- [x] All citations properly formatted
- [x] Covers all major literature strands:
  - Systemic risk and financial networks
  - Volatility paradox and endogenous risk
  - Systemic risk measures
  - Spectral methods
  - Passive investing and ETFs
  - Econometric methods
  - Financial crises
  - Complex systems
  - Macroprudential policy

---

## II. Figures and Tables

### Main Manuscript Figures
1. [x] `Figure_Compression_Matrix.png` - Geometry of market compression
2. [x] `Figure_1_Historical_SE.png` - Historical ASF evolution
3. [x] `Figure_2_SE_vs_CVaR.png` - ASF vs forward tail risk scatter
4. [x] `Figure_SSE_Threshold.png` - Threshold selection (SSE curve)
5. [x] `Figure_Phase_Transition_Contour.png` - Phase transition surface
6. [x] `Figure_Marginal_Effect_C_Mean.png` - Marginal effect bow-tie
7. [x] `Figure_Hysteresis_Loop.png` - Macro-hysteresis cycle
8. [x] `Figure_Rolling_Beta3.png` - Rolling threshold estimates

### Main Manuscript Tables
1. [x] Table 1: Taxonomy of Risk Measures
2. [x] Table 2: Summary Statistics
3. [x] Table 3: Threshold Regression Results (Main)
4. [x] Table 4: Surrogate Data Falsification
5. [x] Table 5: Temporal Placebo Test
6. [x] Table 6: Subsample Analysis by Decade
7. [x] Table 7: Alternative Specifications
8. [x] Table 8: Out-of-Sample Forecast Comparison
9. [x] Table 9: Strategy Performance

---

## III. Formatting Requirements

### Document Formatting
- [x] 12-point font (Times or similar serif)
- [x] Double-spaced throughout
- [x] 1.25-inch margins on all sides
- [x] Page numbers included
- [x] Figures and tables embedded in text (not at end)

### Citation Style
- [x] Author-year format (Chicago/AER style)
- [x] `\citet{}` for textual citations
- [x] `\citep{}` for parenthetical citations
- [x] Bibliography sorted alphabetically

### Mathematical Notation
- [x] Equations numbered consecutively
- [x] Symbols defined on first use
- [x] Theorems/propositions formatted with `amsthm`

---

## IV. Cover Letter (`cover_letter_qje.tex`)

### Content Requirements
- [x] Addressed to QJE Editors
- [x] Manuscript title and author information
- [x] Summary of main contribution
- [x] Relevance to QJE scope
- [x] Declaration of no conflicts of interest
- [x] Statement that manuscript is not under review elsewhere
- [x] Contact information

---

## V. Supplementary Materials

### Replication Package
- [ ] `fetch_fmp_data.py` - Data download script
- [ ] `compute_spectral_entropy.py` - ASF calculation
- [ ] `global_phase_transition.py` - Main estimation
- [ ] `global_strategy_backtest.py` - Strategy backtest
- [ ] `robustness_tests.py` - All robustness checks
- [ ] `plot_figures.py` - Figure generation
- [ ] `README.md` - Replication instructions

### Data Files
- [ ] `prices_dataset.csv` - Raw price data (or API instructions)
- [ ] `derived_variables.csv` - Computed ASF, entropy, connectivity

---

## VI. Pre-Submission Checks

### Technical
- [ ] LaTeX compiles without errors
- [ ] All figures render correctly
- [ ] All cross-references work
- [ ] Bibliography generates properly
- [ ] No overfull/underfull boxes

### Content
- [x] Abstract under 250 words
- [x] All claims supported by evidence
- [x] All figures/tables referenced in text
- [x] Notation consistent throughout
- [x] Acknowledgments placeholder ready

### Ethical
- [x] No conflicts of interest to declare
- [x] Data sources properly cited
- [x] No IRB issues (public financial data)

---

## VII. Submission Process

### QJE-Specific Requirements
1. Submit via ScholarOne: https://mc.manuscriptcentral.com/qje
2. Upload files:
   - Main manuscript (PDF)
   - Online appendix (PDF)
   - Cover letter
   - Replication package (ZIP)
3. Enter metadata:
   - Title
   - Abstract
   - Keywords
   - JEL codes
   - Author information

### Recommended File Naming
- `manuscript_qje_main.pdf`
- `online_appendix_qje.pdf`
- `cover_letter_qje.pdf`
- `replication_package.zip`

---

## VIII. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2025 | Initial submission version |

---

## IX. Post-Submission

### If Desk Rejected
- Review editor comments
- Consider alternative journals (RFS, JF, JFE, AEJ:Macro)

### If R&R Received
- Address all referee comments systematically
- Prepare detailed response memo
- Track all changes in manuscript

---

## X. Summary Statistics

| Metric | Value |
|--------|-------|
| Main manuscript pages | ~30 |
| Online appendix pages | ~15 |
| Total figures | 8 (main) + 4 (appendix) |
| Total tables | 9 (main) + 8 (appendix) |
| References | 51 |
| Sample period | 1990-2024 |
| Primary observations | 939 (weekly) |
| Validation observations | 420 (monthly) |

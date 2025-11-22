# OpenBB Integration Overview

## SDK & Environment

- Added `openbb>=4.0.0` to `backend/requirements.txt`.
- Railway build installs the SDK; no additional system dependencies required.
- OpenBB is accessed exclusively via a new service module `api/services/openbb_client.py`, which provides:
  - `get_price_history(symbol, start_date)` – Yahoo-backed historical prices.
  - `get_multiples(symbol)` – EV/Sales, EV/EBITDA, ROIC, etc.
  - `get_financials(symbol)` – income statement, balance sheet, cash-flow results.
  - `get_ticker_data(symbol)` – aggregates the three datasets plus the latest price.
- The service includes a 6‑hour in-memory cache to smooth rate limits; backing Neon/Postgres caching can be layered on top via the same module.

## API Surface Rewired to OpenBB

- `/api/prices/*` now builds quotes from OpenBB price history instead of FMP.
- `SimpleValuationService` consumes OpenBB multiples & statements to power DCF, reverse DCF, and historical EV multiple fair values.
- Monte Carlo stock forecasts estimate `mu`/`sigma` solely from OpenBB price history.
- The scoring pipeline (quality, valuation, momentum, C‑Score) uses OpenBB datasets for all features—ROIC, margins, leverage, EV multiples, and price momentum.

## Expanded Coverage

- Because OpenBB taps Yahoo Finance under the hood, any ticker that Yahoo supports (US large/mid/small cap, TSX, STOXX, Nikkei, selected EM ADRs, etc.) can be fetched dynamically.
- The valuation and scoring endpoints no longer rely on pre-ingested universes; cache warming can be implemented by calling `openbb_client.get_ticker_data(symbol)` for desired regions.

## Models & C‑Score

- **Quality Model**: Combines ROIC/ROE, gross & operating margin stability, revenue/EBITDA CAGR, FCF volatility, leverage.
- **Valuation Model**: Blends margin of safety from EV/Sales & EV/EBITDA vs 3–5y medians plus DCF-based upside.
- **Momentum Model**: Uses 1/3/6/12‑month returns, realized volatility, drawdown profile, and RSI.
- **Qualitative Moat Proxy**: Derived from margin stability and growth persistence (placeholder until richer metadata is available).
- **C‑Score**: `0.35*Quality + 0.25*Valuation + 0.20*Momentum + 0.20*Moat`.
  - Returned via `/api/analysis/scoring/{ticker}` and the new alias `/api/score/{ticker}`.
  - Frontend highlights the C‑Score badge with color-coded classifications (Probable Outlier / High-Quality Compounder / Standard).

## Next Enhancements

- Persist OpenBB snapshots in Neon (e.g., `openbb_cache` table) for cold-start resilience and historical backtesting.
- Extend the qualitative moat component with board/insider data, patent density, or Glassdoor-style proxies delivered via OpenBB/Alt data.
- Expand the scoring service to expose factor attribution and SHAP-style explanations.
- Introduce Bayesian updating or ensemble models (XGBoost + rule-based) to smooth regime shifts.
- Wire factor analysis and screening endpoints (e.g., `/api/screener/cscore`) for portfolio-level views.


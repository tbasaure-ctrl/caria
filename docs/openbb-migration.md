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

## Enhancements (Nov 2025)

- ✅ **Persistent OpenBB cache**: Neon table `openbb_cache` now stores price history / multiples / profile snapshots so cold starts reuse previously fetched data.
- ✅ **Qualitative moat upgrade**: Moat score blends insider & institutional ownership, gross-margin stability, revenue persistence, R&D intensity, and scale (employees) using the OpenBB profile payload.
- ✅ **Factor attribution + explanations**: `/api/analysis/scoring` returns per-pillar driver weights plus natural-language rationales; the frontend renders attribution bars and commentary.
- ✅ **Bayesian-smoothed ensemble**: Quality/Valuation/Momentum scores combine rule-based metrics with trend proxies and a Bayesian prior to avoid noisy jumps.
- ✅ **C-Score screener**: `/api/screener/cscore` ranks user-supplied tickers by C-Score for quick portfolio filtering.
- ✅ **Deterministic OpenBB build**: `OPENBB_EXTENSION_LIST` (extensions above), `OPENBB_FORCE_EXTENSION_BUILD=true`, and `OPENBB_USER_DATA_PATH=/tmp/openbb` are now set so Railway can preload the SDK without writing outside `/tmp`.


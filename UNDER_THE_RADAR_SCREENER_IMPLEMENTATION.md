# Under-the-Radar Screener Implementation

## Overview

Implemented a lightweight "Under-the-Radar Screener" that detects small-cap stocks (50M-800M market cap) with:
1. Social momentum spikes (Reddit + Stocktwits)
2. Recent catalysts (news, earnings growth, insider buys)
3. Quality metrics (ROCE improvement, efficiency, FCF yield)
4. Size & liquidity filters (market cap, volume spikes)

Returns 0-3 high-conviction candidates per week.

## Architecture

### Backend

**Service**: `backend/api/services/under_the_radar_screener_service.py`
- `UnderTheRadarScreenerService`: Main service class
- Implements 4-step filtering pipeline:
  1. Social momentum detection
  2. Catalyst filter
  3. Quality filter
  4. Size & liquidity filter

**Endpoint**: `GET /api/screener/under-the-radar`
- Located in `backend/api/routes/screener.py`
- Requires authentication
- Returns `UnderTheRadarResponse` with candidates array

### Frontend

**Component**: `frontend/caria-app/components/widgets/UnderTheRadarScreener.tsx`
- React component with loading states
- Displays candidates with metrics, catalysts, and explanations
- Integrated into `ResearchSection.tsx` as the new Stock Screener

## Implementation Details

### Step 1: Social Momentum Detection

**Sources**:
- **Stocktwits**: Uses public API (`api.stocktwits.com/api/2/trending.json`)
- **Reddit**: Uses existing PRAW integration (r/wallstreetbets, r/stocks, r/investing, r/pennystocks)

**Logic**:
- Detects tickers mentioned in trending/active posts
- Only keeps tickers with spikes in **at least 2 independent sources**
- Reduces universe from thousands to <20 candidates

### Step 2: Catalyst Filter

**Checks** (last 30 days):
- **News**: Keywords like "acquisition", "FDA approval", "contract award", "phase 3", "merger"
- **Earnings**: Revenue growth > 50% YoY AND gross margins improving
- **Insider Activity**: Cluster of insider buys (≥2 in last 15 days)

**Data Sources**:
- FMP news endpoint
- FMP income statements (quarterly)
- FMP insider trading endpoint

### Step 3: Quality Filter

**Metrics Calculated**:
- `eficiencia = gross_margin / sga_ratio` (operational efficiency proxy)
- `roce_proxy = (operating_income * (1 - tax_rate)) / invested_capital`
- `delta_roce = roce_proxy_TTM - roce_proxy_1y_ago`
- `fcf_yield = fcf / market_cap * 100`
- `net_debt_ebitda = net_debt / ebitda`

**Hard Filters**:
- `eficiencia > 3.0` AND improving
- `delta_roce > +8 percentage points` (must improve significantly)
- `fcf_yield > 10%` (must be positive and substantial)
- `net_debt_ebitda < 1.5x` (low leverage)

**Data Sources**:
- FMP income statements (quarterly, last 2 quarters minimum)
- FMP balance sheets
- FMP cash flow statements

### Step 4: Size & Liquidity Filter

**Requirements**:
- Market cap: 50M - 800M USD
- Free float < 30M shares (estimated)
- Average daily volume > 300k
- Current volume spike > 10× average

**Data Sources**:
- FMP quote endpoint (market cap, shares outstanding)
- FMP price history (volume data)

## API Response Format

```json
{
  "candidates": [
    {
      "ticker": "ABC",
      "name": "ABC Company Inc.",
      "sector": "Technology",
      "social_spike": {
        "sources": ["stocktwits", "reddit"],
        "metrics": {
          "stocktwits_watchlist": 1500,
          "stocktwits_messages": 250,
          "reddit_mentions": 45,
          "reddit_sources": ["wallstreetbets", "stocks"]
        }
      },
      "catalysts": {
        "flags": [
          "Revenue growth: 65.3% YoY",
          "Gross margin improving: 42.1% vs 38.5%",
          "Insider buys: 3 in last 15 days"
        ],
        "details": {
          "revenue_growth_pct": 65.3,
          "gross_margin_improving": true,
          "insider_buys": 3
        }
      },
      "quality_metrics": {
        "eficiencia": 4.2,
        "roce_proxy": 18.5,
        "delta_roce": 12.3,
        "fcf_yield": 15.2,
        "net_debt_ebitda": 0.8
      },
      "liquidity": {
        "market_cap": 450000000,
        "avg_volume": 850000,
        "current_volume": 12000000,
        "volume_spike": 14.1,
        "free_float_est": 25000000
      },
      "explanation": "Social spike in stocktwits, reddit. Catalyst: Revenue growth: 65.3% YoY, Gross margin improving: 42.1% vs 38.5%. Quality: ROCE +12.3pp, FCF yield 15.2%"
    }
  ],
  "message": "Found 1 under-the-radar candidate(s)"
}
```

## Error Handling

- All API calls wrapped in try/except with graceful degradation
- Missing data returns `None` and candidate is filtered out
- Logging at appropriate levels (info, warning, debug)
- User-friendly error messages in frontend

## Performance Considerations

- **Lightweight**: Only fetches last 2 quarters of financials (not full history)
- **Caching**: Social data could be cached (not implemented yet)
- **Rate Limiting**: Respects API rate limits (FMP, Stocktwits, Reddit)
- **Early Exit**: Stops processing after finding 3 candidates

## Future Enhancements

1. **Historical Social Data**: Store 30-day averages for better spike detection
2. **Caching**: Cache social momentum data to reduce API calls
3. **More Sources**: Add Twitter/X, Discord, other social platforms
4. **Better Catalyst Detection**: Use NLP for news analysis
5. **Scheduling**: Run daily/weekly automatically
6. **Alerts**: Notify users when new candidates appear

## Testing

To test the screener:
1. Ensure FMP_API_KEY is set
2. Ensure Reddit credentials are set (optional, will skip if not available)
3. Call `GET /api/screener/under-the-radar` with auth token
4. Check logs for filtering steps

## Dependencies

- `requests`: For Stocktwits API
- `praw`: For Reddit API (optional, falls back gracefully)
- `FMPClient`: For financial data (already in codebase)
- FastAPI: For endpoint (already in codebase)

## Notes

- The screener is designed to be **lightweight** - no heavy historical database required
- Uses minimal API calls per candidate (only last 2 quarters)
- Can run on-demand or be scheduled daily
- Returns empty list if no candidates pass all filters (expected behavior)

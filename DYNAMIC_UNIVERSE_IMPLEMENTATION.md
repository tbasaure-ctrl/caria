# Dynamic Universe & Comprehensive Valuation - Implementation Plan

**Date**: 2025-11-25  
**Status**: READY TO IMPLEMENT  
**Priority**: P0 - Critical for Production

---

## 🎯 Objectives

### 1. **Self-Expanding Universe for Alpha Picker**
- Start with 128 stocks (current parquet files)
- Each time user valuates a new stock → fetch fundamentals from OpenBB → cache it
- Alpha Picker universe grows organically based on user exploration
- Result: Weeks/months later, could have 500+ stocks in screening universe

### 2. **Comprehensive Stock Valuation**
For ANY stock user requests, provide:
- ✅ **Reverse DCF** (already implemented in `simple_valuation.py`)
- ✅ **Multiples-based Valuation** (already implemented)
- ✅ **Monte Carlo Simulation** (already implemented in `monte_carlo_service.py`)
- ⚠️ **Integration**: Wire all three together in single endpoint response

---

## 📊 Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                    USER REQUESTS STOCK                      │
│                     (e.g., "NVDA")                          │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   Check: Is NVDA in cache?         │
        └────────┬───────────────────┬────────┘
                 │                   │
         YES (Fast ✅)          NO (Fetch 🔄)
                 │                   │
                 ▼                   ▼
     ┌─────────────────────┐  ┌──────────────────────┐
     │  Load from parquet  │  │ Fetch from OpenBB    │
     │  Quality + Value    │  │ get_key_metrics()    │
     │  (instant)          │  │ get_financials()     │
     └──────────┬──────────┘  │ get_multiples()      │
                │             │ get_growth()         │
                │             └──────────┬───────────┘
                │                        │
                │             ┌──────────▼────────────┐
                │             │ Save to cache         │
                │             │ (expandable storage)  │
                │             └──────────┬────────────┘
                │                        │
                ▼                        ▼
     ┌─────────────────────────────────────────────┐
     │      UNIFIED VALUATION PIPELINE             │
     │  1. Reverse DCF (implied growth)            │
     │  2. Multiples (PE, PB, PS historical avg)   │
     │  3. Monte Carlo (2yr forecast, 10k sims)    │
     └─────────────┬───────────────────────────────┘
                   │
                   ▼
     ┌─────────────────────────────────────────────┐
     │      RETURN COMPREHENSIVE RESULT            │
     │  {                                          │
     │    ticker, price, data_source,              │
     │    reverse_dcf: { implied_growth },         │
     │    multiples_val: { pe_fair, pb_fair },     │
     │    monte_carlo: { paths, percentiles }      │
     │  }                                          │
     └─────────────────────────────────────────────┘
```

---

## 🔧 Implementation Steps

### **Phase 1: Dynamic Cache System** (30 min)

#### 1.1 Create `FundamentalsCache` Service

**File**: `backend/api/services/fundamentals_cache.py`

```python
"""
Expandable fundamentals cache that grows as users explore stocks.
Combines static parquet files (128 stocks) with dynamic SQLite/postgres cache.
"""

class FundamentalsCacheService:
    def __init__(self):
        # Load initial 128 stocks from parquet
        self.static_cache = self._load_parquet_files()
        # Dynamic cache (database or file-based)
        self.dynamic_cache = self._init_dynamic_cache()
        
    def get_fundamentals(self, ticker: str) -> dict:
        """
        Get fundamentals for ticker. Checks cache first.
        If not found, fetches from OpenBB and caches it.
        """
        # Check static cache (parquet)
        if ticker in self.static_cache:
            return {"data": self.static_cache[ticker], "source": "static_cache"}
        
        # Check dynamic cache (DB)
        if ticker in self.dynamic_cache:
            return {"data": self.dynamic_cache[ticker], "source": "dynamic_cache"}
        
        # Fetch from OpenBB (real-time)
        data = self._fetch_from_openbb(ticker)
        
        # Save to dynamic cache
        self._save_to_cache(ticker, data)
        
        return {"data": data, "source": "realtime_fetched"}
    
    def _fetch_from_openbb(self, ticker: str) -> dict:
        """Fetch all fundamentals from OpenBB"""
        obb = OpenBBClient()
        return {
            "quality": obb.get_key_metrics(ticker),
            "value": obb.get_multiples(ticker),
            "growth": obb.get_growth(ticker),
            "financials": obb.get_financials(ticker)
        }
    
    def _save_to_cache(self, ticker: str, data: dict):
        """Save to dynamic cache (SQLite or append to parquet)"""
        # Option A: Database (recommended)
        # db.execute("INSERT INTO fundamentals_cache ...")
        
        # Option B: Append to parquet file
        # pd.concat([existing, new_row]).to_parquet(...)
        
        pass
    
    def get_all_cached_tickers(self) -> list[str]:
        """Return all tickers in static + dynamic cache (for Alpha Picker)"""
        return list(set(
            list(self.static_cache.keys()) + 
            list(self.dynamic_cache.keys())
        ))
```

#### 1.2 Update `FactorService` to use Dynamic Cache

**File**: `caria-lib/caria/services/factor_service.py`

Modify `_load_fundamentals()` to:
1. Load from parquet (128 stocks)
2. ALSO load from dynamic cache (growing set)
3. Combine both datasets

---

### **Phase 2: Comprehensive Valuation Endpoint** (45 min)

#### 2.1 Create Unified Valuation Service

**File**: `backend/api/services/comprehensive_valuation_service.py`

```python
"""
Unified valuation service that combines:
- Reverse DCF (implied growth rate)
- Multiples valuation (historical averages)
- Monte Carlo simulation (probabilistic forecast)
"""

class ComprehensiveValuationService:
    def __init__(self):
        self.simple_val = SimpleValuationService()
        self.monte_carlo = MonteCarloService()
        self.cache = FundamentalsCacheService()
    
    async def get_full_valuation(self, ticker: str) -> dict:
        """
        Get comprehensive valuation for any ticker.
        Fetches data if needed, runs all 3 valuation methods.
        """
        
        # 1. Get fundamentals (from cache or fetch)
        fund_result = self.cache.get_fundamentals(ticker)
        fundamentals = fund_result["data"]
        data_source = fund_result["source"]
        
        # 2. Get current price
        obb = OpenBBClient()
        price_data = obb.get_current_price(ticker)
        current_price = price_data["price"]
        
        # 3. Run all valuation methods in parallel
        reverse_dcf = self.simple_val._calculate_reverse_dcf(
            fundamentals, current_price, self.simple_val._default_assumptions()
        )
        
        multiples_val = self.simple_val._calculate_historical_multiples_valuation(
            ticker, fundamentals, current_price
        )
        
        monte_carlo_result = await self.monte_carlo.run_stock_forecast(
            ticker, horizon_years=2, simulations=10000
        )
        
        # 4. Combine results
        return {
            "ticker": ticker,
            "company_name": fundamentals.get("company_name", ticker),
            "current_price": current_price,
            "data_source": data_source,  # "static_cache", "dynamic_cache", or "realtime_fetched"
            "timestamp": datetime.now().isoformat(),
            
            "reverse_dcf": {
                "implied_growth_rate": reverse_dcf.get("implied_growth_rate"),
                "interpretation": self._interpret_reverse_dcf(reverse_dcf),
            },
            
            "multiples_valuation": {
                "pe_based_fair_value": multiples_val.get("pe_fair_value"),
                "pb_based_fair_value": multiples_val.get("pb_fair_value"),
                "ps_based_fair_value": multiples_val.get("ps_fair_value"),
                "average_fair_value": multiples_val.get("avg_fair_value"),
                "upside_downside": multiples_val.get("upside_percent"),
            },
            
            "monte_carlo": {
                "horizon_years": 2,
                "simulations": 10000,
                "percentiles": monte_carlo_result["percentiles"],
                "expected_value": monte_carlo_result["expected_value"],
                "probability_positive": monte_carlo_result["prob_positive"],
                "chart_data": monte_carlo_result["plotly_data"],
            },
            
            "summary": self._generate_summary(
                current_price, reverse_dcf, multiples_val, monte_carlo_result
            )
        }
    
    def _interpret_reverse_dcf(self, reverse_dcf: dict) -> str:
        """Human-readable interpretation of implied growth"""
        growth = reverse_dcf.get("implied_growth_rate", 0)
        
        if growth < 0.03:
            return "Market expects very low/negative growth"
        elif growth < 0.08:
            return "Conservative growth expectations priced in"
        elif growth < 0.15:
            return "Moderate growth expectations"
        elif growth < 0.25:
            return "High growth expectations"
        else:
            return "Very aggressive growth expectations - may be overvalued"
    
    def _generate_summary(self, price, reverse_dcf, multiples, mc) -> str:
        """Generate executive summary of valuation"""
        # Combine signals from all three methods
        # Return 2-3 sentence summary
        pass
```

#### 2.2 Create API Endpoint

**File**: `backend/api/routes/valuation.py` (update existing or create new)

```python
@router.post("/api/valuation/comprehensive")
async def get_comprehensive_valuation(
    request: ValuationRequest,
    current_user=Depends(get_current_user)
):
    """
    Get comprehensive valuation including:
    - Reverse DCF
    - Multiples-based valuation
    - Monte Carlo simulation
    
    If ticker not in cache, fetches from OpenBB and caches for future use.
    """
    service = ComprehensiveValuationService()
    result = await service.get_full_valuation(request.ticker.upper())
    return result
```

---

### **Phase 3: Update Alpha Picker** (20 min)

#### 3.1 Modify `AlphaService` to Use Growing Universe

**File**: `backend/api/services/alpha_service.py`

```python
def compute_alpha_picks(self, top_n_candidates: int = None) -> List[Dict[str, Any]]:
    """
    Computes top 3 stock picks based on CAS.
    Now uses ALL cached stocks (static 128 + dynamically added).
    """
    # Use all available stocks in cache
    cache = FundamentalsCacheService()
    all_tickers = cache.get_all_cached_tickers()
    
    # If top_n_candidates not specified, use ALL cached stocks
    if top_n_candidates is None:
        top_n_candidates = len(all_tickers)
    
    # Rest of logic remains same...
    candidates = self.factor_service.screen_companies(top_n=top_n_candidates)
    
    # ... existing code ...
```

#### 3.2 Update Frontend Alpha Picker

**File**: `frontend/caria-app/components/widgets/AlphaStockPicker.tsx`

Add universe size indicator:
```tsx
<div className="universe-info">
  Screening {universeSize} stocks (started with 128, growing based on user exploration)
</div>
```

---

## 📁 Files to Create/Modify

### New Files
- ✅ `backend/api/services/fundamentals_cache.py` (cache management)
- ✅ `backend/api/services/comprehensive_valuation_service.py` (unified valuation)

### Modified Files
- 📝 `backend/api/routes/valuation.py` (add comprehensive endpoint)
- 📝 `backend/api/services/alpha_service.py` (use dynamic universe)
- 📝 `caria-lib/caria/services/factor_service.py` (load from dynamic cache)
- 📝 `frontend/caria-app/components/widgets/AlphaStockPicker.tsx` (show universe size)
- 📝 `frontend/caria-app/components/widgets/ValuationTool.tsx` (show all 3 valuation methods)

---

## 🗄️ Dynamic Cache Storage Options

### Option A: **SQLite** (Recommended for MVP)
**Pros**:
- Simple, no external dependencies
- File-based, easy deployment
- Good for 10k-100k stocks

**Schema**:
```sql
CREATE TABLE fundamentals_cache (
    ticker TEXT PRIMARY KEY,
    fetched_at TIMESTAMP,
    quality_metrics JSON,
    value_metrics JSON,
    growth_metrics JSON,
    financials JSON
);
```

### Option B: **Append to Parquet** (Simpler)
**Pros**:
- No new dependencies
- Works with existing parquet infrastructure

**Cons**:
- Slower for lookups
- Need to reload entire file

### Option C: **PostgreSQL** (Production)
**Pros**:
- Already using Postgres for users/portfolios
- Better for large scale

**Cons**:
- More setup

**Recommendation**: Start with SQLite, migrate to Postgres later if needed.

---

## 🎨 Frontend UX Enhancements

### Valuation Tool UI
```tsx
<div className="valuation-container">
  {/* Header with data source badge */}
  <div className="header">
    <h2>{ticker} Valuation</h2>
    <span className={`badge ${dataSource}`}>
      {dataSource === 'static_cache' ? '📊 Cached' : 
       dataSource === 'dynamic_cache' ? '💾 Recently fetched' :
       '🔄 Real-time data'}
    </span>
  </div>
  
  {/* Three valuation methods side-by-side */}
  <div className="valuation-methods">
    <div className="method reverse-dcf">
      <h3>Reverse DCF</h3>
      <p>Implied Growth: {impliedGrowth}%</p>
      <p>{interpretation}</p>
    </div>
    
    <div className="method multiples">
      <h3>Multiples Valuation</h3>
      <p>Fair Value: ${avgFairValue}</p>
      <p>Upside: {upside}%</p>
    </div>
    
    <div className="method monte-carlo">
      <h3>Monte Carlo (2yr)</h3>
      <MonteCarloChart data={mcData} />
      <p>Expected: ${expectedValue}</p>
    </div>
  </div>
  
  {/* Executive Summary */}
  <div className="summary">
    {summary}
  </div>
</div>
```

---

## ⚡ Performance Considerations

### Cache Hit Rates
- **Static cache (128 stocks)**: Instant (< 10ms)
- **Dynamic cache**: Fast (< 50ms from SQLite)
- **Real-time fetch**: Moderate (2-5 seconds from OpenBB)

### Optimization Strategies
1. **Prefetch popular stocks**: Pre-cache S&P 500 in background
2. **Cache TTL**: Refresh data after 24 hours
3. **Batch updates**: Nightly job to refresh all cached stocks

---

## 🧪 Testing Plan

### Unit Tests
- Test cache hit/miss logic
- Test OpenBB fallback
- Test all 3 valuation methods

### Integration Tests
- Fetch new stock → cache → retrieve → verify
- Alpha Picker with growing universe

### Manual Testing Checklist
- [ ] Valuate AAPL (in static cache) → instant response
- [ ] Valuate OBSCURE-TICKER (not in cache) → fetches from OpenBB
- [ ] Valuate same OBSCURE-TICKER again → now cached
- [ ] Alpha Picker shows 129 stocks (128 + OBSCURE-TICKER)
- [ ] All 3 valuation methods return data
- [ ] Monte Carlo chart renders

---

## 📊 Expected Outcomes

### Week 1
- Universe: 128 → 150 stocks
- User explores 20-30 new stocks

### Month 1
- Universe: 128 → 300+ stocks
- Alpha Picker has richer dataset

### Month 6
- Universe: 500-1000 stocks
- Comprehensive coverage of user interests

---

## 🚀 Deployment Checklist

### Backend
- [ ] Add SQLite dependency (already included in Python)
- [ ] Create database schema migration
- [ ] Deploy new endpoints
- [ ] Test in staging

### Frontend
- [ ] Update ValuationTool component
- [ ] Add Monte Carlo visualization
- [ ] Test responsive layout

### Data
- [ ] Initial 128 stocks deployed (✅ DONE in commit d4bc352)
- [ ] Dynamic cache initialized
- [ ] Monitoring for cache hit rates

---

## 🔮 Future Enhancements

1. **Sector-aware caching**: Pre-cache entire sectors when user shows interest
2. **Collaborative filtering**: "Users who valued AAPL also valued..."
3. **Export universe**: Let users download current screening universe
4. **Custom universes**: Allow users to create personal watchlists for Alpha Picker

---

**Ready to Implement?** Let's start with Phase 1 (Dynamic Cache System).

*Created: 2025-11-25 02:17 ART*

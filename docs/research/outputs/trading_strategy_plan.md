# ASF Paper Trading Strategy - Implementation Plan

## Objective
Create a sophisticated forward paper trading strategy based on the Accumulated Spectral Fragility (ASF) framework to demonstrate real-world effectiveness of the research methodology.

## Strategy Overview

### Core Concept
The strategy exploits the **regime-dependent risk-return relationship** discovered in the research:
- **Contagion Regime** (low connectivity): Higher ASF → Higher risk → Reduce exposure
- **Coordination Regime** (high connectivity): Higher ASF → Lower volatility (stability illusion) → Maintain/increase exposure cautiously

### Universe
- **Core Assets**: SPY, QQQ, IWM, EFA (equities), TLT, IEF, AGG (bonds), GLD, SLV (commodities)
- **Alternative Signals**: VIX (volatility), HYG (credit risk)

## Strategy Components

### 1. Signal Generation
```
Daily:
1. Compute 63-day rolling correlation matrix
2. Extract eigenvalues → Spectral entropy
3. Update ASF (θ=0.995 EWM)
4. Compute connectivity (mean pairwise correlation)
5. Classify regime: Contagion (conn < τ) or Coordination (conn ≥ τ)
```

### 2. Position Sizing (Regime-Conditional)

#### Contagion Regime (Low Connectivity)
- ASF < 0.2: Full equity exposure (100% target)
- ASF 0.2-0.4: Reduce to 70% equity, 20% bonds, 10% gold
- ASF > 0.4: Defensive (40% equity, 40% bonds, 20% gold)

#### Coordination Regime (High Connectivity)
- Any ASF level: Defensive stance
- Maximum 50% equity exposure
- Increase cash/short-term bonds
- Alert: "Stability may be illusory"

### 3. Risk Management
- Maximum daily VaR: 2%
- Stop-loss: -5% portfolio drawdown → reduce all positions 50%
- Regime change confirmation: 5-day sustained crossing

### 4. Rebalancing
- Weekly rebalancing (every Friday close)
- Emergency rebalancing on regime change

## Files to Create

1. `asf_trading_engine.py` - Core signal computation
2. `portfolio_manager.py` - Position sizing and allocations
3. `paper_trader.py` - Execution simulation and logging
4. `performance_tracker.py` - Analytics and reporting
5. `run_daily_update.py` - Daily automation script

## Performance Metrics
- Sharpe Ratio vs benchmark (60/40)
- Maximum Drawdown
- Regime-conditional performance
- Signal accuracy (ASF predicting drawdowns)

## Next Steps
1. Implement core engine
2. Backtest on historical data (2007-2024)
3. Start forward paper trading (daily signals)
4. Weekly performance reports

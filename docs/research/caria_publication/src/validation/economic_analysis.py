"""
Economic Analysis for Crisis Detection
=======================================

This module implements economic analysis required for publication:
1. Crisis classification by type (liquidity, credit, macro)
2. Lead time calculation (anticipation horizon)
3. Portfolio de-risking simulation
4. Economic utility analysis

Required for Minor Concern 4.2 in reviewer feedback.

Author: Tomás Basaure
Date: December 2025
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import timedelta


# ==============================================================================
# CONFIGURATION
# ==============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Known crisis events with classification
CRISIS_CATALOG = {
    'flash_crash_2010': {
        'date': '2010-05-06',
        'type': 'liquidity',
        'description': 'Flash Crash - algorithmic trading cascade',
        'duration_days': 1,
        'max_drawdown': 0.09
    },
    'euro_crisis_2011': {
        'date': '2011-08-05',
        'type': 'credit',
        'description': 'European Debt Crisis - sovereign contagion',
        'duration_days': 60,
        'max_drawdown': 0.19
    },
    'china_crash_2015': {
        'date': '2015-08-24',
        'type': 'macro',
        'description': 'China Stock Market Crash - CNY devaluation',
        'duration_days': 5,
        'max_drawdown': 0.12
    },
    'covid_crash_2020': {
        'date': '2020-03-11',
        'type': 'liquidity',
        'description': 'COVID-19 Pandemic Crash - global liquidity crisis',
        'duration_days': 12,
        'max_drawdown': 0.34
    },
    'gfc_2008': {
        'date': '2008-09-15',
        'type': 'credit',
        'description': 'Global Financial Crisis - Lehman collapse',
        'duration_days': 180,
        'max_drawdown': 0.57
    },
    'fed_tightening_2022': {
        'date': '2022-01-03',
        'type': 'macro',
        'description': 'Fed Tightening Cycle - inflation response',
        'duration_days': 280,
        'max_drawdown': 0.25
    },
    'svb_collapse_2023': {
        'date': '2023-03-10',
        'type': 'credit',
        'description': 'SVB/Regional Bank Crisis',
        'duration_days': 10,
        'max_drawdown': 0.05
    }
}

CrisisType = str  # 'liquidity', 'credit', 'macro'


@dataclass
class CrisisEvent:
    """Container for a detected crisis event."""
    name: str
    start_date: pd.Timestamp
    signal_date: pd.Timestamp  # When signal first fired
    lead_time_days: int
    crisis_type: CrisisType
    max_drawdown: float
    detected: bool
    
    def __repr__(self):
        return f"CrisisEvent({self.name}, lead={self.lead_time_days}d, type={self.crisis_type})"


@dataclass
class EconomicUtility:
    """Container for economic utility analysis."""
    sharpe_ratio: float
    max_drawdown: float
    return_annualized: float
    vol_annualized: float
    crises_avoided: int
    crises_total: int
    false_alarm_rate: float
    
    def __repr__(self):
        return f"EconomicUtility(Sharpe={self.sharpe_ratio:.2f}, MaxDD={self.max_drawdown:.1%})"


# ==============================================================================
# CRISIS CLASSIFICATION
# ==============================================================================

def classify_crisis_type(
    signal_date: pd.Timestamp,
    known_events: Optional[Dict] = None,
    window_days: int = 30
) -> Tuple[CrisisType, str]:
    """
    Classify a detected crisis by type using known events catalog.
    
    Parameters:
    ===========
    signal_date : pd.Timestamp
        Date when crisis signal fired
    known_events : dict, optional
        Catalog of known events
    window_days : int
        Days around event to match
        
    Returns:
    ========
    Tuple[CrisisType, str] : (type, event_name)
    """
    if known_events is None:
        known_events = CRISIS_CATALOG
    
    for name, event in known_events.items():
        event_date = pd.Timestamp(event['date'])
        
        # Check if signal is within window of known event
        if abs((signal_date - event_date).days) <= window_days:
            return event['type'], name
    
    # Unknown crisis
    return 'unknown', 'unclassified'


def analyze_crisis_detection(
    predictions: pd.Series,
    prices: pd.Series,
    known_events: Optional[Dict] = None
) -> List[CrisisEvent]:
    """
    Analyze which crises were detected and calculate lead times.
    
    Parameters:
    ===========
    predictions : pd.Series
        Binary predictions (1 = crisis signal)
    prices : pd.Series
        Price series for drawdown calculation
    known_events : dict, optional
        Catalog of known events
        
    Returns:
    ========
    List[CrisisEvent] : List of detected/missed events
    """
    if known_events is None:
        known_events = CRISIS_CATALOG
    
    results = []
    
    for name, event in known_events.items():
        event_date = pd.Timestamp(event['date'])
        
        # Check if event is within prediction period
        if event_date < predictions.index.min() or event_date > predictions.index.max():
            continue
        
        # Find first signal before event
        pre_event_preds = predictions[
            (predictions.index < event_date) & 
            (predictions.index >= event_date - timedelta(days=60))
        ]
        
        signal_dates = pre_event_preds[pre_event_preds == 1].index
        
        if len(signal_dates) > 0:
            first_signal = signal_dates[0]
            lead_time = (event_date - first_signal).days
            detected = True
        else:
            first_signal = event_date
            lead_time = 0
            detected = False
        
        results.append(CrisisEvent(
            name=name,
            start_date=event_date,
            signal_date=first_signal,
            lead_time_days=lead_time,
            crisis_type=event['type'],
            max_drawdown=event['max_drawdown'],
            detected=detected
        ))
    
    return results


def calculate_lead_time_statistics(
    crisis_events: List[CrisisEvent]
) -> Dict:
    """
    Calculate lead time statistics for detected crises.
    
    Parameters:
    ===========
    crisis_events : list
        List of CrisisEvent objects
        
    Returns:
    ========
    Dict : Lead time statistics
    """
    detected = [e for e in crisis_events if e.detected]
    
    if len(detected) == 0:
        return {
            'mean_lead_time': np.nan,
            'median_lead_time': np.nan,
            'min_lead_time': np.nan,
            'max_lead_time': np.nan,
            'detection_rate': 0.0,
            'n_detected': 0,
            'n_total': len(crisis_events)
        }
    
    lead_times = [e.lead_time_days for e in detected]
    
    return {
        'mean_lead_time': np.mean(lead_times),
        'median_lead_time': np.median(lead_times),
        'min_lead_time': np.min(lead_times),
        'max_lead_time': np.max(lead_times),
        'detection_rate': len(detected) / len(crisis_events),
        'n_detected': len(detected),
        'n_total': len(crisis_events)
    }


def group_crises_by_type(
    crisis_events: List[CrisisEvent]
) -> Dict[CrisisType, List[CrisisEvent]]:
    """
    Group crisis events by type for analysis.
    
    Parameters:
    ===========
    crisis_events : list
        List of CrisisEvent objects
        
    Returns:
    ========
    Dict : Events grouped by type
    """
    groups = {'liquidity': [], 'credit': [], 'macro': [], 'unknown': []}
    
    for event in crisis_events:
        if event.crisis_type in groups:
            groups[event.crisis_type].append(event)
        else:
            groups['unknown'].append(event)
    
    return groups


# ==============================================================================
# PORTFOLIO DE-RISKING SIMULATION
# ==============================================================================

def dynamic_allocation(
    signal: str,
    base_allocation: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Determine portfolio allocation based on risk signal.
    
    Parameters:
    ===========
    signal : str
        Risk regime: 'CRITICAL', 'ALERT', 'FRAGILE', or 'NORMAL'
    base_allocation : dict, optional
        Default allocation when NORMAL
        
    Returns:
    ========
    Dict[str, float] : Asset allocation
    """
    if base_allocation is None:
        base_allocation = {'equity': 0.8, 'bonds': 0.2, 'cash': 0.0}
    
    allocations = {
        'CRITICAL': {'equity': 0.1, 'bonds': 0.3, 'cash': 0.6},
        'ALERT': {'equity': 0.3, 'bonds': 0.4, 'cash': 0.3},
        'FRAGILE': {'equity': 0.5, 'bonds': 0.3, 'cash': 0.2},
        'NORMAL': base_allocation
    }
    
    return allocations.get(signal, base_allocation)


def backtest_derisking_strategy(
    prices: pd.Series,
    signals: pd.Series,
    bond_prices: Optional[pd.Series] = None,
    transaction_cost: float = 0.001
) -> Tuple[pd.Series, EconomicUtility]:
    """
    Backtest a dynamic de-risking strategy.
    
    Parameters:
    ===========
    prices : pd.Series
        Equity prices
    signals : pd.Series
        Risk signals ('CRITICAL', 'ALERT', 'FRAGILE', 'NORMAL')
    bond_prices : pd.Series, optional
        Bond prices for allocation
    transaction_cost : float
        Transaction cost per trade (percentage)
        
    Returns:
    ========
    Tuple[pd.Series, EconomicUtility] : Portfolio values and utility metrics
    """
    # Align data
    common_idx = prices.index.intersection(signals.index)
    prices = prices.loc[common_idx]
    signals = signals.loc[common_idx]
    
    # Calculate returns
    equity_returns = prices.pct_change().fillna(0)
    
    if bond_prices is not None:
        bond_prices = bond_prices.loc[common_idx]
        bond_returns = bond_prices.pct_change().fillna(0)
    else:
        bond_returns = pd.Series(0.0001, index=common_idx)  # ~2.5% annual
    
    cash_return = 0.0001  # ~2.5% annual
    
    # Initialize portfolio
    portfolio_value = pd.Series(index=common_idx, dtype=float)
    portfolio_value.iloc[0] = 1.0
    
    prev_allocation = {'equity': 0.8, 'bonds': 0.2, 'cash': 0.0}
    
    for i in range(1, len(common_idx)):
        date = common_idx[i]
        signal = signals.iloc[i-1]
        
        # Get allocation
        if isinstance(signal, (int, float)):
            signal = 'ALERT' if signal == 1 else 'NORMAL'
        allocation = dynamic_allocation(signal)
        
        # Calculate returns
        port_return = (
            allocation['equity'] * equity_returns.iloc[i] +
            allocation['bonds'] * bond_returns.iloc[i] +
            allocation['cash'] * cash_return
        )
        
        # Transaction costs if allocation changed
        if allocation != prev_allocation:
            turnover = sum(abs(allocation.get(k, 0) - prev_allocation.get(k, 0)) 
                          for k in set(allocation) | set(prev_allocation))
            port_return -= transaction_cost * turnover / 2
        
        portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + port_return)
        prev_allocation = allocation
    
    # Calculate metrics
    returns = portfolio_value.pct_change().dropna()
    
    # Drawdowns
    rolling_max = portfolio_value.expanding().max()
    drawdowns = (portfolio_value - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Sharpe ratio
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Count crises avoided
    crisis_signals = (signals == 'CRITICAL') | (signals == 'ALERT') | (signals == 1)
    n_signals = crisis_signals.sum()
    
    # False alarm rate (signals that didn't precede a drop)
    false_alarms = 0
    for i in range(len(signals) - 5):
        if crisis_signals.iloc[i]:
            forward_return = prices.iloc[i+5] / prices.iloc[i] - 1
            if forward_return > 0:
                false_alarms += 1
    
    false_alarm_rate = false_alarms / max(n_signals, 1)
    
    utility = EconomicUtility(
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        return_annualized=ann_return,
        vol_annualized=ann_vol,
        crises_avoided=int(n_signals - false_alarms),
        crises_total=int(n_signals),
        false_alarm_rate=false_alarm_rate
    )
    
    return portfolio_value, utility


# ==============================================================================
# REPORT GENERATION
# ==============================================================================

def generate_economic_report(
    crisis_events: List[CrisisEvent],
    utility: EconomicUtility
) -> str:
    """
    Generate markdown report for economic analysis section.
    
    Parameters:
    ===========
    crisis_events : list
        Detected crisis events
    utility : EconomicUtility
        Portfolio utility metrics
        
    Returns:
    ========
    str : Markdown report
    """
    report = []
    report.append("# Economic Analysis Report")
    report.append("")
    
    # Lead time analysis
    lead_stats = calculate_lead_time_statistics(crisis_events)
    
    report.append("## 1. Crisis Detection Summary")
    report.append("")
    report.append(f"- **Detection Rate**: {lead_stats['detection_rate']:.1%} "
                  f"({lead_stats['n_detected']}/{lead_stats['n_total']} crises)")
    report.append(f"- **Mean Lead Time**: {lead_stats['mean_lead_time']:.1f} days")
    report.append(f"- **Median Lead Time**: {lead_stats['median_lead_time']:.1f} days")
    report.append("")
    
    # By crisis type
    report.append("## 2. Performance by Crisis Type")
    report.append("")
    report.append("| Type | Detected | Lead Time (mean) |")
    report.append("|------|----------|------------------|")
    
    grouped = group_crises_by_type(crisis_events)
    for crisis_type, events in grouped.items():
        if len(events) > 0:
            detected = [e for e in events if e.detected]
            mean_lead = np.mean([e.lead_time_days for e in detected]) if detected else 0
            report.append(f"| {crisis_type.capitalize()} | {len(detected)}/{len(events)} | {mean_lead:.1f} days |")
    
    report.append("")
    
    # Portfolio utility
    report.append("## 3. Portfolio De-Risking Utility")
    report.append("")
    report.append(f"- **Sharpe Ratio**: {utility.sharpe_ratio:.2f}")
    report.append(f"- **Max Drawdown**: {utility.max_drawdown:.1%}")
    report.append(f"- **Annualized Return**: {utility.return_annualized:.1%}")
    report.append(f"- **Annualized Volatility**: {utility.vol_annualized:.1%}")
    report.append(f"- **False Alarm Rate**: {utility.false_alarm_rate:.1%}")
    
    report.append("")
    report.append("## 4. Detected Events")
    report.append("")
    report.append("| Event | Date | Type | Lead Time | Detected |")
    report.append("|-------|------|------|-----------|----------|")
    
    for event in crisis_events:
        detected = "Yes" if event.detected else "No"
        report.append(f"| {event.name} | {event.start_date.date()} | "
                      f"{event.crisis_type} | {event.lead_time_days}d | {detected} |")
    
    return "\n".join(report)


# ==============================================================================
# TESTS
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ECONOMIC ANALYSIS MODULE TEST")
    print("=" * 60)
    
    np.random.seed(RANDOM_SEED)
    
    # Create synthetic data
    dates = pd.date_range('2007-01-01', '2024-12-31', freq='B')
    n = len(dates)
    
    # Synthetic prices with known crisis
    returns = np.random.randn(n) * 0.01
    
    # Inject crises at known dates
    gfc_idx = dates.get_loc(pd.Timestamp('2008-09-15'), method='nearest')
    covid_idx = dates.get_loc(pd.Timestamp('2020-03-11'), method='nearest')
    
    returns[gfc_idx:gfc_idx+30] = np.random.randn(30) * 0.03 - 0.02
    returns[covid_idx:covid_idx+15] = np.random.randn(15) * 0.05 - 0.03
    
    prices = pd.Series(100 * np.cumprod(1 + returns), index=dates)
    
    # Synthetic signals (with some lead time)
    signals = pd.Series('NORMAL', index=dates)
    signals.iloc[gfc_idx-10:gfc_idx+20] = 'CRITICAL'
    signals.iloc[covid_idx-5:covid_idx+10] = 'ALERT'
    
    print("\n1. Crisis Classification Test:")
    print("-" * 40)
    
    for date_str in ['2008-09-10', '2020-03-05', '2015-01-01']:
        date = pd.Timestamp(date_str)
        crisis_type, event = classify_crisis_type(date)
        print(f"   {date_str}: {crisis_type} ({event})")
    
    print("\n2. Crisis Detection Analysis:")
    print("-" * 40)
    
    # Convert signals to binary
    binary_signals = pd.Series(
        [1 if s in ['CRITICAL', 'ALERT'] else 0 for s in signals],
        index=signals.index
    )
    
    events = analyze_crisis_detection(binary_signals, prices)
    
    for event in events:
        status = "DETECTED" if event.detected else "MISSED"
        print(f"   {event.name}: {status}, lead={event.lead_time_days}d")
    
    print("\n3. Lead Time Statistics:")
    print("-" * 40)
    
    stats = calculate_lead_time_statistics(events)
    print(f"   Detection rate: {stats['detection_rate']:.1%}")
    print(f"   Mean lead time: {stats['mean_lead_time']:.1f} days")
    
    print("\n4. Portfolio Backtest:")
    print("-" * 40)
    
    portfolio, utility = backtest_derisking_strategy(prices, signals)
    print(f"   Final value: {portfolio.iloc[-1]:.2f}")
    print(f"   Sharpe ratio: {utility.sharpe_ratio:.2f}")
    print(f"   Max drawdown: {utility.max_drawdown:.1%}")
    
    print("\n" + "=" * 60)
    print("Economic analysis tests completed!")

/**
 * OpportunityRadar Widget
 * 
 * Event-driven scanner that detects:
 * - STEALTH signals: Volume divergence (high RVol) with flat price (accumulation)
 * - VELOCITY signals: Price breakout (>5%) with volume confirmation (momentum)
 * 
 * Uses the /api/screener/market-opportunities endpoint from MarketScannerService
 */

import React, { useState, useEffect } from 'react';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';

interface MarketSignal {
    ticker: string;
    price: number;
    change: number;
    rvol: number;
    market_cap?: number;
    signal_strength: string;
    tag: string;
    desc: string;
    social_spike?: {
        spike_ratio: number;
        mentions_today: number;
        has_spike: boolean;
    } | null;
}

interface ScannerResponse {
    momentum_signals: MarketSignal[];
    accumulation_signals: MarketSignal[];
}

const formatMarketCap = (mcap: number | undefined): string => {
    if (!mcap) return '—';
    if (mcap >= 1_000_000_000) return `${(mcap / 1_000_000_000).toFixed(1)}B`;
    if (mcap >= 1_000_000) return `${(mcap / 1_000_000).toFixed(0)}M`;
    return mcap.toLocaleString();
};

const SignalCard: React.FC<{ signal: MarketSignal; type: 'stealth' | 'velocity' }> = ({ signal, type }) => {
    const isVelocity = type === 'velocity';
    
    // Color schemes based on signal type
    const accentColor = isVelocity ? 'var(--color-warning)' : '#22D3EE'; // Amber vs Cyan
    const tagBgColor = isVelocity ? 'rgba(245, 158, 11, 0.15)' : 'rgba(34, 211, 238, 0.15)';
    
    return (
        <div 
            className="p-4 rounded-lg transition-all duration-200 hover:translate-y-[-2px]"
            style={{
                backgroundColor: 'var(--color-bg-tertiary)',
                border: `1px solid ${isVelocity ? 'rgba(245, 158, 11, 0.3)' : 'rgba(34, 211, 238, 0.3)'}`,
            }}
        >
            {/* Header: Ticker + Tag */}
            <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                    <span 
                        className="text-xl font-bold font-mono"
                        style={{ color: 'var(--color-text-primary)' }}
                    >
                        ${signal.ticker}
                    </span>
                    <span 
                        className="text-[10px] font-bold tracking-wider px-2 py-0.5 rounded"
                        style={{ 
                            backgroundColor: tagBgColor,
                            color: accentColor,
                        }}
                    >
                        {signal.tag}
                    </span>
                </div>
                <div className="text-right">
                    <div 
                        className="text-lg font-bold font-mono"
                        style={{ color: signal.change >= 0 ? 'var(--color-positive)' : 'var(--color-negative)' }}
                    >
                        {signal.change >= 0 ? '+' : ''}{signal.change.toFixed(2)}%
                    </div>
                </div>
            </div>

            {/* Metrics Row */}
            <div className="flex items-center gap-4 mb-3">
                <div>
                    <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>Price</span>
                    <div className="text-sm font-mono" style={{ color: 'var(--color-text-primary)' }}>
                        ${signal.price.toFixed(2)}
                    </div>
                </div>
                <div>
                    <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>RVol</span>
                    <div 
                        className="text-sm font-mono font-bold"
                        style={{ color: accentColor }}
                    >
                        {signal.rvol.toFixed(1)}x
                    </div>
                </div>
                <div>
                    <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>Mkt Cap</span>
                    <div className="text-sm font-mono" style={{ color: 'var(--color-text-secondary)' }}>
                        ${formatMarketCap(signal.market_cap)}
                    </div>
                </div>
                {signal.social_spike?.has_spike && (
                    <div>
                        <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>Social</span>
                        <div className="text-sm font-mono" style={{ color: 'var(--color-positive)' }}>
                            🔥 {signal.social_spike.spike_ratio}x
                        </div>
                    </div>
                )}
            </div>

            {/* Description */}
            <p 
                className="text-sm leading-relaxed"
                style={{ color: 'var(--color-text-secondary)' }}
            >
                {signal.desc}
            </p>

            {/* RVol Bar */}
            <div className="mt-3 h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--color-bg-surface)' }}>
                <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                        width: `${Math.min(signal.rvol * 30, 100)}%`,
                        backgroundColor: accentColor,
                    }}
                />
            </div>
        </div>
    );
};

export const OpportunityRadar: React.FC = () => {
    const [data, setData] = useState<ScannerResponse | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchOpportunities = async () => {
        setIsLoading(true);
        setError(null);
        
        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/screener/market-opportunities`);
            if (response.ok) {
                const result = await response.json();
                setData(result);
            } else {
                throw new Error('Failed to fetch opportunities');
            }
        } catch (err: any) {
            console.error('OpportunityRadar fetch error:', err);
            setError(err.message || 'Failed to load opportunities');
            // Fallback mock data for demo
            setData({
                momentum_signals: [
                    { ticker: 'SMCI', price: 42.50, change: 8.2, rvol: 2.1, market_cap: 24_000_000_000, signal_strength: 'High', tag: 'PRICE VELOCITY', desc: 'Price outlier (+8.2%) on expanded volume (2.1x avg).', social_spike: null },
                ],
                accumulation_signals: [
                    { ticker: 'PLTR', price: 68.30, change: 0.8, rvol: 2.4, market_cap: 45_000_000_000, signal_strength: 'Critical', tag: 'VOL DIVERGENCE', desc: 'Volume anomaly (2.4x avg) with flat price action (+0.8%).', social_spike: { spike_ratio: 1.8, mentions_today: 340, has_spike: true } },
                ],
            });
        }

        setIsLoading(false);
    };

    useEffect(() => {
        fetchOpportunities();
    }, []);

    const totalSignals = (data?.momentum_signals.length || 0) + (data?.accumulation_signals.length || 0);

    return (
        <div 
            className="rounded-xl p-6"
            style={{
                backgroundColor: 'var(--color-bg-secondary)',
                border: '1px solid var(--color-border-subtle)'
            }}
        >
            {/* Header */}
            <div className="flex items-center justify-between mb-2">
                <h2 
                    className="text-xl font-bold"
                    style={{
                        fontFamily: 'var(--font-display)',
                        color: 'var(--color-text-primary)'
                    }}
                >
                    Opportunity Radar
                </h2>
                <button
                    onClick={fetchOpportunities}
                    disabled={isLoading}
                    className="p-2 rounded-lg transition-all hover:bg-slate-700"
                    style={{ color: 'var(--color-text-muted)' }}
                    title="Refresh"
                >
                    {isLoading ? '⏳' : '🔄'}
                </button>
            </div>
            
            <p 
                className="text-sm mb-5"
                style={{ color: 'var(--color-text-muted)' }}
            >
                Real-time volume & price anomalies (Anti Mega-Cap filter active)
            </p>

            {/* Loading State */}
            {isLoading && (
                <div className="py-12 text-center">
                    <div 
                        className="w-8 h-8 mx-auto mb-3 border-2 border-t-transparent rounded-full animate-spin"
                        style={{ borderColor: 'var(--color-accent-primary)', borderTopColor: 'transparent' }}
                    />
                    <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
                        Scanning market anomalies...
                    </p>
                </div>
            )}

            {/* Error State */}
            {error && !isLoading && (
                <div 
                    className="mb-4 p-3 rounded-lg text-sm"
                    style={{ backgroundColor: 'rgba(245, 158, 11, 0.1)', color: 'var(--color-warning)' }}
                >
                    ⚠️ Using demo data (API not connected)
                </div>
            )}

            {/* Results */}
            {!isLoading && data && (
                <div className="space-y-6">
                    {/* Velocity Signals (Momentum) */}
                    {data.momentum_signals.length > 0 && (
                        <div>
                            <h3 
                                className="text-sm font-semibold uppercase tracking-wider mb-3 flex items-center gap-2"
                                style={{ color: 'var(--color-warning)' }}
                            >
                                <span>🚀</span> Velocity (Breakouts)
                            </h3>
                            <div className="space-y-3">
                                {data.momentum_signals.map((signal, idx) => (
                                    <SignalCard key={`mom-${idx}`} signal={signal} type="velocity" />
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Stealth Signals (Accumulation) */}
                    {data.accumulation_signals.length > 0 && (
                        <div>
                            <h3 
                                className="text-sm font-semibold uppercase tracking-wider mb-3 flex items-center gap-2"
                                style={{ color: '#22D3EE' }}
                            >
                                <span>🔍</span> Stealth (Accumulation)
                            </h3>
                            <div className="space-y-3">
                                {data.accumulation_signals.map((signal, idx) => (
                                    <SignalCard key={`acc-${idx}`} signal={signal} type="stealth" />
                                ))}
                            </div>
                        </div>
                    )}

                    {/* No Signals */}
                    {totalSignals === 0 && (
                        <div className="py-8 text-center">
                            <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
                                No unusual activity detected. Markets are quiet.
                            </p>
                        </div>
                    )}
                </div>
            )}

            {/* Disclaimer */}
            <div 
                className="mt-5 pt-4 border-t text-xs text-center"
                style={{ borderColor: 'var(--color-border-subtle)', color: 'var(--color-text-muted)' }}
            >
                ⚠️ Volume anomalies are not investment advice. Always DYOR.
            </div>
        </div>
    );
};


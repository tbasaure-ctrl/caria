import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { WidgetCard } from './WidgetCard';

interface MarketSignal {
    ticker: string;
    price: number;
    change: number;
    rvol: number;
    market_cap?: number | null;
    signal_strength: string;
    tag: string;
    desc: string;
    social_spike?: {
        spike_ratio?: number;
        mentions_today?: number;
        mentions_yesterday?: number;
        sentiment?: number;
        has_spike?: boolean;
    } | null;
}

interface MarketScannerResponse {
    momentum_signals: MarketSignal[];
    accumulation_signals: MarketSignal[];
}

export const MarketScanner: React.FC = () => {
    const { token } = useAuth();
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState<MarketScannerResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    const runScanner = async () => {
        setLoading(true);
        setError(null);
        setData(null);

        try {
            const response = await fetch('/api/screener/market-opportunities', {
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const result: MarketScannerResponse = await response.json();
            setData(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Error running scanner');
        } finally {
            setLoading(false);
        }
    };

    const formatNumber = (num: number): string => {
        if (num >= 1_000_000_000) {
            return `$${(num / 1_000_000_000).toFixed(2)}B`;
        } else if (num >= 1_000_000) {
            return `$${(num / 1_000_000).toFixed(2)}M`;
        } else if (num >= 1_000) {
            return `$${(num / 1_000).toFixed(2)}K`;
        }
        return `$${num.toFixed(2)}`;
    };

    const formatPrice = (price: number): string => {
        return `$${price.toFixed(2)}`;
    };

    const formatChange = (change: number): string => {
        const sign = change > 0 ? '+' : '';
        return `${sign}${change.toFixed(2)}%`;
    };

    return (
        <WidgetCard 
            title="MARKET SCANNER"
            tooltip="Event-driven social screener: Finds price anomalies first, validates with social noise. Avoids mega-cap bias."
        >
            <div className="mb-6">
                <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                    Professional scanner that inverts the traditional flow: searches for price anomalies first,
                    filters mega-caps, then validates with social spike ratio. Returns momentum and accumulation signals.
                </p>
            </div>

            <button
                onClick={runScanner}
                disabled={loading}
                className="px-6 py-3 font-semibold rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed mb-6"
                style={{
                    backgroundColor: 'var(--color-accent-primary)',
                    color: '#FFFFFF',
                }}
                onMouseEnter={(e) => {
                    if (!loading) {
                        e.currentTarget.style.transform = 'translateY(-2px)';
                        e.currentTarget.style.boxShadow = '0 4px 12px rgba(46, 124, 246, 0.3)';
                    }
                }}
                onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'none';
                }}
            >
                {loading ? 'Running Scanner...' : 'Run Scanner'}
            </button>

            {error && (
                <div className="mt-4 p-4 rounded-lg" style={{ backgroundColor: 'var(--color-negative-muted)', color: 'var(--color-negative)', border: '1px solid var(--color-negative)' }}>
                    <p>{error}</p>
                </div>
            )}

            {data && (
                <div className="mt-6">
                    <div style={{ 
                        display: 'grid', 
                        gridTemplateColumns: '1fr 1fr', 
                        gap: '20px',
                        maxWidth: '1200px',
                        margin: '0 auto'
                    }}>
                        {/* Volume Divergence Scanner */}
                        <div style={{
                            background: '#ffffff',
                            border: '1px solid #e0e0e0',
                            borderRadius: '4px',
                            overflow: 'hidden',
                            boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
                        }}>
                            <div style={{
                                padding: '12px 16px',
                                borderBottom: '1px solid #e0e0e0',
                                background: '#f9fafb',
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center'
                            }}>
                                <span style={{
                                    fontSize: '14px',
                                    fontWeight: 600,
                                    textTransform: 'uppercase',
                                    letterSpacing: '0.5px',
                                    color: '#666666'
                                }}>
                                    Volume Divergence Scanner
                                </span>
                                <span style={{ fontSize: '11px', color: '#888' }}>Live Feed</span>
                            </div>
                            <table style={{
                                width: '100%',
                                borderCollapse: 'collapse',
                                fontSize: '13px'
                            }}>
                                <thead>
                                    <tr>
                                        <th style={{
                                            textAlign: 'left',
                                            padding: '10px 16px',
                                            fontWeight: 500,
                                            color: '#666666',
                                            borderBottom: '1px solid #e0e0e0',
                                            fontSize: '11px',
                                            textTransform: 'uppercase'
                                        }}>Symbol</th>
                                        <th style={{
                                            textAlign: 'left',
                                            padding: '10px 16px',
                                            fontWeight: 500,
                                            color: '#666666',
                                            borderBottom: '1px solid #e0e0e0',
                                            fontSize: '11px',
                                            textTransform: 'uppercase'
                                        }}>Price / Chg</th>
                                        <th style={{
                                            textAlign: 'left',
                                            padding: '10px 16px',
                                            fontWeight: 500,
                                            color: '#666666',
                                            borderBottom: '1px solid #e0e0e0',
                                            fontSize: '11px',
                                            textTransform: 'uppercase'
                                        }}>Rel. Volume</th>
                                        <th style={{
                                            textAlign: 'left',
                                            padding: '10px 16px',
                                            fontWeight: 500,
                                            color: '#666666',
                                            borderBottom: '1px solid #e0e0e0',
                                            fontSize: '11px',
                                            textTransform: 'uppercase'
                                        }}>Signal Type</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {data.accumulation_signals.length === 0 ? (
                                        <tr>
                                            <td colSpan={4} style={{ padding: '20px', textAlign: 'center', color: '#888' }}>
                                                No accumulation signals found
                                            </td>
                                        </tr>
                                    ) : (
                                        data.accumulation_signals.map((signal, idx) => (
                                            <tr key={signal.ticker} style={{
                                                borderBottom: idx < data.accumulation_signals.length - 1 ? '1px solid #f0f0f0' : 'none'
                                            }}>
                                                <td style={{ padding: '12px 16px' }}>
                                                    <div style={{ fontWeight: 700, color: '#000' }}>{signal.ticker}</div>
                                                </td>
                                                <td style={{ padding: '12px 16px' }}>
                                                    <div style={{ fontFamily: "'Roboto Mono', 'Consolas', monospace", fontWeight: 500 }}>
                                                        {formatPrice(signal.price)}
                                                    </div>
                                                    <div style={{ 
                                                        fontFamily: "'Roboto Mono', 'Consolas', monospace",
                                                        color: signal.change > 0 ? '#006b3f' : signal.change < 0 ? '#d32f2f' : '#666666'
                                                    }}>
                                                        {formatChange(signal.change)}
                                                    </div>
                                                </td>
                                                <td style={{ padding: '12px 16px' }}>
                                                    <div style={{ 
                                                        fontFamily: "'Roboto Mono', 'Consolas', monospace",
                                                        color: '#0f4c81',
                                                        fontWeight: 500
                                                    }}>
                                                        {signal.rvol.toFixed(1)}x
                                                    </div>
                                                </td>
                                                <td style={{ padding: '12px 16px' }}>
                                                    <span style={{
                                                        display: 'inline-block',
                                                        padding: '2px 6px',
                                                        fontSize: '10px',
                                                        fontWeight: 600,
                                                        borderRadius: '2px',
                                                        textTransform: 'uppercase',
                                                        letterSpacing: '0.3px',
                                                        backgroundColor: '#e8f0fe',
                                                        color: '#0f4c81',
                                                        border: '1px solid #d2e3fc'
                                                    }}>
                                                        {signal.tag}
                                                    </span>
                                                </td>
                                            </tr>
                                        ))
                                    )}
                                </tbody>
                            </table>
                        </div>

                        {/* High Velocity Scanner */}
                        <div style={{
                            background: '#ffffff',
                            border: '1px solid #e0e0e0',
                            borderRadius: '4px',
                            overflow: 'hidden',
                            boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
                        }}>
                            <div style={{
                                padding: '12px 16px',
                                borderBottom: '1px solid #e0e0e0',
                                background: '#f9fafb',
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center'
                            }}>
                                <span style={{
                                    fontSize: '14px',
                                    fontWeight: 600,
                                    textTransform: 'uppercase',
                                    letterSpacing: '0.5px',
                                    color: '#666666'
                                }}>
                                    High Velocity Scanner
                                </span>
                                <span style={{ fontSize: '11px', color: '#888' }}>Top Gainers &gt; 5%</span>
                            </div>
                            <table style={{
                                width: '100%',
                                borderCollapse: 'collapse',
                                fontSize: '13px'
                            }}>
                                <thead>
                                    <tr>
                                        <th style={{
                                            textAlign: 'left',
                                            padding: '10px 16px',
                                            fontWeight: 500,
                                            color: '#666666',
                                            borderBottom: '1px solid #e0e0e0',
                                            fontSize: '11px',
                                            textTransform: 'uppercase'
                                        }}>Symbol</th>
                                        <th style={{
                                            textAlign: 'left',
                                            padding: '10px 16px',
                                            fontWeight: 500,
                                            color: '#666666',
                                            borderBottom: '1px solid #e0e0e0',
                                            fontSize: '11px',
                                            textTransform: 'uppercase'
                                        }}>Price / Chg</th>
                                        <th style={{
                                            textAlign: 'left',
                                            padding: '10px 16px',
                                            fontWeight: 500,
                                            color: '#666666',
                                            borderBottom: '1px solid #e0e0e0',
                                            fontSize: '11px',
                                            textTransform: 'uppercase'
                                        }}>Rel. Volume</th>
                                        <th style={{
                                            textAlign: 'left',
                                            padding: '10px 16px',
                                            fontWeight: 500,
                                            color: '#666666',
                                            borderBottom: '1px solid #e0e0e0',
                                            fontSize: '11px',
                                            textTransform: 'uppercase'
                                        }}>Signal Type</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {data.momentum_signals.length === 0 ? (
                                        <tr>
                                            <td colSpan={4} style={{ padding: '20px', textAlign: 'center', color: '#888' }}>
                                                No momentum signals found
                                            </td>
                                        </tr>
                                    ) : (
                                        data.momentum_signals.map((signal, idx) => (
                                            <tr key={signal.ticker} style={{
                                                borderBottom: idx < data.momentum_signals.length - 1 ? '1px solid #f0f0f0' : 'none'
                                            }}>
                                                <td style={{ padding: '12px 16px' }}>
                                                    <div style={{ fontWeight: 700, color: '#000' }}>{signal.ticker}</div>
                                                    {signal.social_spike?.has_spike && (
                                                        <div style={{ fontSize: '10px', color: '#888', marginTop: '2px' }}>
                                                            Social: {signal.social_spike.spike_ratio}x spike
                                                        </div>
                                                    )}
                                                </td>
                                                <td style={{ padding: '12px 16px' }}>
                                                    <div style={{ fontFamily: "'Roboto Mono', 'Consolas', monospace", fontWeight: 500 }}>
                                                        {formatPrice(signal.price)}
                                                    </div>
                                                    <div style={{ 
                                                        fontFamily: "'Roboto Mono', 'Consolas', monospace",
                                                        color: '#006b3f',
                                                        fontWeight: 500
                                                    }}>
                                                        {formatChange(signal.change)}
                                                    </div>
                                                </td>
                                                <td style={{ padding: '12px 16px' }}>
                                                    <div style={{ 
                                                        fontFamily: "'Roboto Mono', 'Consolas', monospace",
                                                        fontWeight: 500
                                                    }}>
                                                        {signal.rvol.toFixed(1)}x
                                                    </div>
                                                </td>
                                                <td style={{ padding: '12px 16px' }}>
                                                    <span style={{
                                                        display: 'inline-block',
                                                        padding: '2px 6px',
                                                        fontSize: '10px',
                                                        fontWeight: 600,
                                                        borderRadius: '2px',
                                                        textTransform: 'uppercase',
                                                        letterSpacing: '0.3px',
                                                        backgroundColor: '#e6f4ea',
                                                        color: '#006b3f',
                                                        border: '1px solid #ceead6'
                                                    }}>
                                                        {signal.tag}
                                                    </span>
                                                </td>
                                            </tr>
                                        ))
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}
        </WidgetCard>
    );
};

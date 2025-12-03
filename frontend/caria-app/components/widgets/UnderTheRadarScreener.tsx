import React, { useState } from 'react';
import { getToken } from '../../services/apiService';
import { WidgetCard } from './WidgetCard';

interface UnderTheRadarCandidate {
    ticker: string;
    name: string;
    sector: string;
    social_spike: {
        sources: string[];
        metrics: Record<string, any>;
    };
    catalysts: {
        flags: string[];
        details: Record<string, any>;
    };
    quality_metrics: {
        eficiencia: number;
        roce_proxy: number;
        delta_roce: number;
        fcf_yield: number;
        net_debt_ebitda: number;
    };
    liquidity: {
        market_cap: number;
        avg_volume: number;
        current_volume: number;
        volume_spike: number;
        free_float_est: number;
    };
    explanation: string;
}

interface UnderTheRadarResponse {
    candidates: UnderTheRadarCandidate[];
    message: string | null;
}

export const UnderTheRadarScreener: React.FC = () => {
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState<UnderTheRadarResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [expandedTicker, setExpandedTicker] = useState<string | null>(null);

    const runScreener = async () => {
        setLoading(true);
        setError(null);
        setData(null);

        try {
            const token = getToken();
            const headers: HeadersInit = {};
            if (token) {
                headers['Authorization'] = `Bearer ${token}`;
            }

            const response = await fetch('/api/screener/under-the-radar', {
                headers,
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const result: UnderTheRadarResponse = await response.json();
            setData(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Error running screener');
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

    return (
        <WidgetCard 
            title="UNDER-THE-RADAR SCREENER"
            tooltip="Bloomberg-style table mode: Discover undervalued mid-caps with strong quality, valuation, and momentum signals before Wall Street reacts."
        >
            <div className="mb-6">
                <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                    Event + social spark + quality filter for true emerging outliers.
                    Detects small-cap stocks (50M-800M market cap) with social momentum,
                    recent catalysts, and improving quality metrics.
                </p>
            </div>

            <button
                onClick={runScreener}
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
                {loading ? 'Running Screener...' : 'Run Screener'}
            </button>

            {error && (
                <div className="mt-4 p-4 rounded-lg" style={{ backgroundColor: 'var(--color-negative-muted)', color: 'var(--color-negative)', border: '1px solid var(--color-negative)' }}>
                    <p>{error}</p>
                </div>
            )}

            {data && (
                <div className="mt-6">
                    {data.candidates.length === 0 ? (
                        <div className="p-6 rounded-lg text-center" style={{ backgroundColor: 'var(--color-bg-tertiary)', color: 'var(--color-text-secondary)' }}>
                            <p>
                                {data.message || 'No stocks passed all filters this week.'}
                            </p>
                        </div>
                    ) : (
                        <div>
                            {/* Bloomberg-style Table */}
                            <div className="overflow-x-auto">
                                <table className="w-full" style={{ borderCollapse: 'separate', borderSpacing: 0 }}>
                                    <thead>
                                        <tr style={{ borderBottom: '2px solid var(--color-border-subtle)' }}>
                                            <th className="text-left py-3 px-4 text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>Ticker</th>
                                            <th className="text-left py-3 px-4 text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>Market Cap</th>
                                            <th className="text-right py-3 px-4 text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>ΔROIC</th>
                                            <th className="text-right py-3 px-4 text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>FCF Yield</th>
                                            <th className="text-right py-3 px-4 text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>Volume Spike</th>
                                            <th className="text-center py-3 px-4 text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>Catalyst?</th>
                                            <th className="text-center py-3 px-4 text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>Social</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {data.candidates.map((candidate, idx) => (
                                            <React.Fragment key={candidate.ticker}>
                                                <tr
                                                    className="transition-colors cursor-pointer"
                                                    style={{ 
                                                        backgroundColor: idx % 2 === 0 ? 'transparent' : 'var(--color-bg-tertiary)',
                                                        borderBottom: '1px solid var(--color-border-subtle)'
                                                    }}
                                                    onMouseEnter={(e) => {
                                                        e.currentTarget.style.backgroundColor = 'var(--color-bg-tertiary)';
                                                    }}
                                                    onMouseLeave={(e) => {
                                                        e.currentTarget.style.backgroundColor = idx % 2 === 0 ? 'transparent' : 'var(--color-bg-tertiary)';
                                                    }}
                                                    onClick={() => setExpandedTicker(expandedTicker === candidate.ticker ? null : candidate.ticker)}
                                                >
                                                    <td className="py-4 px-4">
                                                        <div>
                                                            <div className="font-mono font-bold" style={{ color: 'var(--color-text-primary)' }}>{candidate.ticker}</div>
                                                            <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>{candidate.name}</div>
                                                        </div>
                                                    </td>
                                                    <td className="py-4 px-4 font-mono text-sm" style={{ color: 'var(--color-text-primary)' }}>
                                                        {formatNumber(candidate.liquidity.market_cap)}
                                                    </td>
                                                    <td className="py-4 px-4 text-right font-mono text-sm" style={{ color: candidate.quality_metrics.delta_roce > 0 ? 'var(--color-positive)' : 'var(--color-text-secondary)' }}>
                                                        {candidate.quality_metrics.delta_roce > 0 ? '+' : ''}{candidate.quality_metrics.delta_roce.toFixed(1)}pp
                                                    </td>
                                                    <td className="py-4 px-4 text-right font-mono text-sm" style={{ color: 'var(--color-text-primary)' }}>
                                                        {candidate.quality_metrics.fcf_yield.toFixed(1)}%
                                                    </td>
                                                    <td className="py-4 px-4 text-right font-mono text-sm" style={{ color: candidate.liquidity.volume_spike > 2 ? 'var(--color-warning)' : 'var(--color-text-secondary)' }}>
                                                        {candidate.liquidity.volume_spike.toFixed(1)}x
                                                    </td>
                                                    <td className="py-4 px-4 text-center">
                                                        {candidate.catalysts.flags.length > 0 ? (
                                                            <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: 'var(--color-positive)' }} />
                                                        ) : (
                                                            <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>—</span>
                                                        )}
                                                    </td>
                                                    <td className="py-4 px-4 text-center">
                                                        {candidate.social_spike.sources.length > 0 ? (
                                                            <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: 'var(--color-accent-primary)' }} />
                                                        ) : (
                                                            <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>—</span>
                                                        )}
                                                    </td>
                                                </tr>
                                                {expandedTicker === candidate.ticker && (
                                                    <tr>
                                                        <td colSpan={7} className="p-6" style={{ backgroundColor: 'var(--color-bg-tertiary)' }}>
                                                            <div className="space-y-4">
                                                                <div>
                                                                    <p className="text-xs mb-1 uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>Why this stock surfaced?</p>
                                                                    <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                                                                        {candidate.explanation}
                                                                    </p>
                                                                </div>
                                                                {candidate.social_spike.sources.length > 0 && (
                                                                    <div>
                                                                        <p className="text-xs mb-2 uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>Social Momentum</p>
                                                                        <div className="flex flex-wrap gap-2">
                                                                            {candidate.social_spike.sources.map((source) => (
                                                                                <span
                                                                                    key={source}
                                                                                    className="px-2 py-1 text-xs rounded"
                                                                                    style={{ backgroundColor: 'var(--color-bg-surface)', color: 'var(--color-text-secondary)' }}
                                                                                >
                                                                                    {source}
                                                                                </span>
                                                                            ))}
                                                                        </div>
                                                                    </div>
                                                                )}
                                                                {candidate.catalysts.flags.length > 0 && (
                                                                    <div>
                                                                        <p className="text-xs mb-2 uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>Catalysts</p>
                                                                        <ul className="list-disc list-inside text-sm space-y-1" style={{ color: 'var(--color-text-secondary)' }}>
                                                                            {candidate.catalysts.flags.slice(0, 3).map((flag, i) => (
                                                                                <li key={i}>{flag}</li>
                                                                            ))}
                                                                        </ul>
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </td>
                                                    </tr>
                                                )}
                                            </React.Fragment>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </WidgetCard>
    );
};

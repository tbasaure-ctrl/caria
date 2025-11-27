import React, { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';

interface RadarResult {
    ticker: string;
    price?: number;
    mentions_today: number;
    spike_pct: number;
    sentiment_score: number;
    hype_density: number;
    market_cap_billions: number;
    tags: string[];
    insight: string;
    reddit_mentions: number;
    stocktwits_mentions: number;
    top_post?: string;
}

interface RadarResponse {
    results: RadarResult[];
    timeframe: string;
    total_detected: number;
    message: string;
}

const TagBadge: React.FC<{ tag: string }> = ({ tag }) => {
    const getTagConfig = (tag: string) => {
        if (tag.includes('🚀')) {
            return {
                bgColor: 'rgba(139, 92, 246, 0.15)', // Purple
                borderColor: 'rgba(139, 92, 246, 0.5)',
                textColor: '#a78bfa',
                icon: '🚀'
            };
        } else if (tag.includes('⚔️')) {
            return {
                bgColor: 'rgba(251, 146, 60, 0.15)', // Orange
                borderColor: 'rgba(251, 146, 60, 0.5)',
                textColor: '#fb923c',
                icon: '⚔️'
            };
        } else if (tag.includes('💎')) {
            return {
                bgColor: 'rgba(34, 197, 94, 0.15)', // Green
                borderColor: 'rgba(34, 197, 94, 0.5)',
                textColor: '#22c55e',
                icon: '💎'
            };
        }
        return {
            bgColor: 'var(--color-bg-surface)',
            borderColor: 'var(--color-border-subtle)',
            textColor: 'var(--color-text-secondary)',
            icon: '•'
        };
    };

    const config = getTagConfig(tag);

    return (
        <span
            className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium"
            style={{
                backgroundColor: config.bgColor,
                border: `1px solid ${config.borderColor}`,
                color: config.textColor
            }}
        >
            <span>{config.icon}</span>
            <span>{tag.replace(/[🚀⚔️💎]/g, '').trim()}</span>
        </span>
    );
};

const RadarCard: React.FC<{ result: RadarResult }> = ({ result }) => {
    return (
        <div
            className="rounded-xl p-5 transition-all duration-300"
            style={{
                backgroundColor: 'var(--color-bg-tertiary)',
                border: '1px solid var(--color-border-subtle)',
            }}
            onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'var(--color-border-emphasis)';
                e.currentTarget.style.transform = 'translateY(-2px)';
            }}
            onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'var(--color-border-subtle)';
                e.currentTarget.style.transform = 'translateY(0)';
            }}
        >
            {/* Header: Ticker + Price */}
            <div className="flex items-start justify-between mb-4">
                <div>
                    <div className="flex items-center gap-2 mb-2">
                        <span
                            className="text-xl font-bold font-mono"
                            style={{ color: 'var(--color-text-primary)' }}
                        >
                            {result.ticker}
                        </span>
                        {result.price && (
                            <span
                                className="text-sm font-mono"
                                style={{ color: 'var(--color-text-secondary)' }}
                            >
                                ${result.price.toFixed(2)}
                            </span>
                        )}
                    </div>
                    <div className="flex flex-wrap gap-2 mb-2">
                        {result.tags.map((tag, idx) => (
                            <TagBadge key={idx} tag={tag} />
                        ))}
                    </div>
                </div>
                
                {/* Mentions Badge */}
                <div className="text-center">
                    <div
                        className="text-2xl font-bold font-mono"
                        style={{
                            color: result.spike_pct > 200 ? 'var(--color-positive)' :
                                   result.spike_pct > 100 ? 'var(--color-accent-primary)' :
                                   'var(--color-text-secondary)'
                        }}
                    >
                        {result.mentions_today.toLocaleString()}
                    </div>
                    <div
                        className="text-[9px] font-medium tracking-widest uppercase"
                        style={{ color: 'var(--color-text-muted)' }}
                    >
                        Mentions
                    </div>
                </div>
            </div>

            {/* Insight */}
            <div
                className="pt-4 border-t mb-4"
                style={{ borderColor: 'var(--color-border-subtle)' }}
            >
                <p
                    className="text-xs font-semibold uppercase tracking-wider mb-2"
                    style={{ color: 'var(--color-text-muted)' }}
                >
                    Why it surfaced
                </p>
                <p
                    className="text-sm leading-relaxed mb-3"
                    style={{ color: 'var(--color-text-secondary)' }}
                >
                    {result.insight}
                </p>
                
                {/* Metrics strip */}
                <div className="flex items-center gap-4 text-xs font-mono flex-wrap">
                    {result.spike_pct > 0 && (
                        <div>
                            <span style={{ color: 'var(--color-text-muted)' }}>Spike: </span>
                            <span style={{ color: 'var(--color-positive)' }}>
                                +{result.spike_pct.toFixed(1)}%
                            </span>
                        </div>
                    )}
                    <div>
                        <span style={{ color: 'var(--color-text-muted)' }}>Sentiment: </span>
                        <span style={{ 
                            color: result.sentiment_score > 0.3 ? 'var(--color-positive)' :
                                   result.sentiment_score < -0.3 ? 'var(--color-negative)' :
                                   'var(--color-text-secondary)'
                        }}>
                            {result.sentiment_score > 0 ? '+' : ''}{result.sentiment_score.toFixed(2)}
                        </span>
                    </div>
                    {result.market_cap_billions > 0 && (
                        <div>
                            <span style={{ color: 'var(--color-text-muted)' }}>Market Cap: </span>
                            <span style={{ color: 'var(--color-text-primary)' }}>
                                ${result.market_cap_billions.toFixed(2)}B
                            </span>
                        </div>
                    )}
                    {result.hype_density > 0 && (
                        <div>
                            <span style={{ color: 'var(--color-text-muted)' }}>Hype Density: </span>
                            <span style={{ color: 'var(--color-text-primary)' }}>
                                {result.hype_density.toFixed(1)}
                            </span>
                        </div>
                    )}
                </div>
            </div>

            {/* Source breakdown */}
            <div className="flex items-center gap-4 text-xs">
                <div className="flex items-center gap-1">
                    <span style={{ color: '#FF4500' }}>📱</span>
                    <span style={{ color: 'var(--color-text-muted)' }}>
                        Reddit: {result.reddit_mentions}
                    </span>
                </div>
                <div className="flex items-center gap-1">
                    <span style={{ color: '#00D9FF' }}>💬</span>
                    <span style={{ color: 'var(--color-text-muted)' }}>
                        StockTwits: {result.stocktwits_mentions}
                    </span>
                </div>
            </div>

            {/* Top post preview */}
            {result.top_post && (
                <div
                    className="mt-3 pt-3 border-t text-xs italic"
                    style={{ 
                        borderColor: 'var(--color-border-subtle)',
                        color: 'var(--color-text-muted)'
                    }}
                >
                    "{result.top_post.substring(0, 100)}{result.top_post.length > 100 ? '...' : ''}"
                </div>
            )}
        </div>
    );
};

export const SocialRadar: React.FC = () => {
    const [data, setData] = useState<RadarResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [timeframe, setTimeframe] = useState<'hour' | 'day' | 'week'>('day');

    useEffect(() => {
        fetchRadarData();
    }, [timeframe]);

    const fetchRadarData = async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetchWithAuth(
                `${API_BASE_URL}/api/social/radar?timeframe=${timeframe}`
            );

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const radarData: RadarResponse = await response.json();
            setData(radarData);
        } catch (err) {
            console.error('Social Radar fetch error:', err);
            setError(err instanceof Error ? err.message : 'Failed to fetch social radar data');
        } finally {
            setLoading(false);
        }
    };

    return (
        <WidgetCard
            title="⚡ Movers & Shakers"
            tooltip="Social Radar: Detecta anomalías en social sentiment usando Velocity Spike, Rumble Score y Tiny Titan Ratio. Identifica stocks con aceleración de menciones, polémica (Bull/Bear War) o hype desproporcionado."
        >
            {/* Timeframe selector */}
            <div className="flex items-center justify-between mb-5">
                <div className="flex gap-2">
                    {(['hour', 'day', 'week'] as const).map(tf => (
                        <button
                            key={tf}
                            onClick={() => setTimeframe(tf)}
                            className="px-3 py-1 rounded text-xs transition-all"
                            style={{
                                backgroundColor: timeframe === tf ? 'var(--color-accent-primary)' : 'var(--color-bg-surface)',
                                color: timeframe === tf ? '#FFFFFF' : 'var(--color-text-secondary)',
                            }}
                        >
                            {tf.charAt(0).toUpperCase() + tf.slice(1)}
                        </button>
                    ))}
                </div>
                <button
                    onClick={fetchRadarData}
                    disabled={loading}
                    className="text-xs font-medium transition-colors disabled:opacity-50"
                    style={{ color: 'var(--color-text-muted)' }}
                    onMouseEnter={(e) => {
                        if (!loading) {
                            e.currentTarget.style.color = 'var(--color-accent-primary)';
                        }
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.color = 'var(--color-text-muted)';
                    }}
                >
                    {loading ? 'Loading...' : '🔄 Refresh'}
                </button>
            </div>

            {/* Error State */}
            {error && (
                <div
                    className="rounded-xl p-6 text-center mb-4"
                    style={{
                        backgroundColor: 'var(--color-negative-muted)',
                        border: '1px solid var(--color-negative)'
                    }}
                >
                    <p
                        className="text-sm mb-4"
                        style={{ color: 'var(--color-negative)' }}
                    >
                        {error}
                    </p>
                    <button
                        onClick={fetchRadarData}
                        className="px-4 py-2 rounded-lg text-sm font-medium"
                        style={{
                            backgroundColor: 'var(--color-bg-surface)',
                            color: 'var(--color-text-secondary)',
                        }}
                    >
                        Try Again
                    </button>
                </div>
            )}

            {/* Loading State */}
            {loading && !data && (
                <div
                    className="rounded-xl p-10 text-center"
                    style={{
                        backgroundColor: 'var(--color-bg-tertiary)',
                        border: '1px solid var(--color-border-subtle)'
                    }}
                >
                    <div
                        className="w-10 h-10 mx-auto mb-4 border-3 border-t-transparent rounded-full animate-spin"
                        style={{ borderColor: 'var(--color-accent-primary)', borderTopColor: 'transparent' }}
                    />
                    <p
                        className="text-sm animate-pulse"
                        style={{ color: 'var(--color-accent-primary)' }}
                    >
                        Analyzing social anomalies...
                    </p>
                </div>
            )}

            {/* Results */}
            {!loading && data && (
                <>
                    {data.results.length === 0 ? (
                        <div
                            className="rounded-xl p-10 text-center"
                            style={{
                                backgroundColor: 'var(--color-bg-tertiary)',
                                border: '1px solid var(--color-border-subtle)'
                            }}
                        >
                            <p
                                className="text-sm"
                                style={{ color: 'var(--color-text-secondary)' }}
                            >
                                No anomalies detected in this timeframe.
                            </p>
                            <p
                                className="text-xs mt-2"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                The radar looks for velocity spikes, controversy, and viral micro-caps.
                            </p>
                        </div>
                    ) : (
                        <>
                            <div className="mb-4">
                                <p
                                    className="text-xs"
                                    style={{ color: 'var(--color-text-muted)' }}
                                >
                                    {data.message} • {data.results.length} anomaly(ies) detected
                                </p>
                            </div>
                            <div className="grid md:grid-cols-2 gap-4 animate-fade-in">
                                {data.results.map((result) => (
                                    <RadarCard key={result.ticker} result={result} />
                                ))}
                            </div>
                        </>
                    )}
                </>
            )}

            {/* Disclaimer */}
            <div
                className="mt-6 pt-4 border-t text-center"
                style={{ borderColor: 'var(--color-border-subtle)' }}
            >
                <p
                    className="text-xs"
                    style={{ color: 'var(--color-text-muted)' }}
                >
                    ⚠️ Social anomalies are not investment advice. Always do your own research.
                </p>
            </div>
        </WidgetCard>
    );
};

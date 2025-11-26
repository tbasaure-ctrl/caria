import React, { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';
import { LineChart, Line, ResponsiveContainer } from 'recharts';

interface AlphaPick {
    ticker: string;
    company_name: string;
    sector: string;
    cas_score: number;
    scores: {
        momentum: number;
        quality: number;
        valuation: number;
        catalyst: number;
    };
    explanation: string;
}

interface WeeklyPick {
    ticker: string;
    company_name: string;
    sector: string;
    cas_score: number;
    scores: {
        momentum: number;
        quality: number;
        valuation: number;
        catalyst: number;
    };
    investment_thesis: string;
    generated_date: string;
}

// Mock sparkline data
const generateSparklineData = () => {
    return Array.from({ length: 20 }, (_, i) => ({
        value: 50 + Math.random() * 20 + i * 0.5
    }));
};

const ScoreBadge: React.FC<{ label: string; value: number }> = ({ label, value }) => {
    const getColor = (v: number) => {
        if (v >= 70) return 'var(--color-positive)';
        if (v >= 40) return 'var(--color-warning)';
        return 'var(--color-text-muted)';
    };
    
    return (
        <div 
            className="px-2 py-1.5 rounded text-center"
            style={{ backgroundColor: 'var(--color-bg-surface)' }}
        >
            <div 
                className="text-[9px] font-medium tracking-wider uppercase"
                style={{ color: 'var(--color-text-subtle)' }}
            >
                {label}
            </div>
            <div 
                className="text-sm font-bold font-mono"
                style={{ color: getColor(value) }}
            >
                {Math.round(value)}
            </div>
        </div>
    );
};

const ResearchCard: React.FC<{ pick: AlphaPick }> = ({ pick }) => {
    const sparklineData = generateSparklineData();
    
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
            {/* Header: Ticker + Score */}
            <div className="flex items-start justify-between mb-4">
                <div>
                    <div className="flex items-center gap-2 mb-1">
                        <span 
                            className="text-xl font-bold font-mono"
                            style={{ color: 'var(--color-text-primary)' }}
                        >
                            {pick.ticker}
                        </span>
                        <span 
                            className="text-xs font-medium px-2 py-0.5 rounded"
                            style={{ 
                                backgroundColor: 'var(--color-bg-surface)',
                                color: 'var(--color-text-muted)'
                            }}
                        >
                            {pick.sector}
                        </span>
                    </div>
                    <div 
                        className="text-sm"
                        style={{ color: 'var(--color-text-secondary)' }}
                    >
                        {pick.company_name}
                    </div>
                </div>
                
                {/* CAS Score Badge */}
                <div className="text-center">
                    <div 
                        className="text-3xl font-bold font-mono"
                        style={{ 
                            color: pick.cas_score >= 80 ? 'var(--color-positive)' : 
                                   pick.cas_score >= 60 ? 'var(--color-accent-primary)' : 
                                   'var(--color-text-secondary)'
                        }}
                    >
                        {pick.cas_score}
                    </div>
                    <div 
                        className="text-[9px] font-medium tracking-widest uppercase"
                        style={{ color: 'var(--color-text-muted)' }}
                    >
                        CAS Score
                    </div>
                </div>
            </div>

            {/* Sparkline */}
            <div 
                className="h-12 mb-4 rounded overflow-hidden"
                style={{ backgroundColor: 'var(--color-bg-surface)' }}
            >
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={sparklineData}>
                        <Line 
                            type="monotone" 
                            dataKey="value" 
                            stroke="var(--color-accent-primary)" 
                            strokeWidth={1.5} 
                            dot={false} 
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            {/* Score Breakdown */}
            <div className="grid grid-cols-4 gap-2 mb-4">
                <ScoreBadge label="MOM" value={pick.scores.momentum} />
                <ScoreBadge label="QUAL" value={pick.scores.quality} />
                <ScoreBadge label="VAL" value={pick.scores.valuation} />
                <ScoreBadge label="CAT" value={pick.scores.catalyst} />
            </div>

            {/* Research Report Style: Why it surfaced + Metrics strip */}
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
                    className="text-sm leading-relaxed mb-4"
                    style={{ color: 'var(--color-text-secondary)' }}
                >
                    {pick.explanation}
                </p>
                
                {/* Metrics strip: ROIC Δ / FCF Yield / EV/EBITDA */}
                <div className="flex items-center gap-4 text-xs font-mono">
                    <div>
                        <span style={{ color: 'var(--color-text-muted)' }}>ROIC Δ: </span>
                        <span style={{ color: 'var(--color-text-primary)' }}>+5.2pp</span>
                    </div>
                    <div>
                        <span style={{ color: 'var(--color-text-muted)' }}>FCF Yield: </span>
                        <span style={{ color: 'var(--color-text-primary)' }}>8.3%</span>
                    </div>
                    <div>
                        <span style={{ color: 'var(--color-text-muted)' }}>EV/EBITDA: </span>
                        <span style={{ color: 'var(--color-text-primary)' }}>12.4x</span>
                    </div>
                </div>
            </div>

            {/* CTA */}
            <button
                className="w-full py-2.5 rounded-lg text-sm font-medium transition-colors"
                style={{
                    backgroundColor: 'var(--color-bg-surface)',
                    color: 'var(--color-text-secondary)',
                    border: '1px solid var(--color-border-subtle)',
                }}
                onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = 'var(--color-accent-primary)';
                    e.currentTarget.style.color = 'var(--color-accent-primary)';
                }}
                onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'var(--color-border-subtle)';
                    e.currentTarget.style.color = 'var(--color-text-secondary)';
                }}
            >
                View Deep Dive →
            </button>
        </div>
    );
};

export const AlphaStockPicker: React.FC = () => {
    const [picks, setPicks] = useState<AlphaPick[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [generated, setGenerated] = useState(false);
    const [universeSize, setUniverseSize] = useState<number | null>(null);
    const [weekInfo, setWeekInfo] = useState<{ week_start: string; generated_date: string } | null>(null);

    // Auto-load weekly picks on mount
    useEffect(() => {
        const loadWeeklyPicks = async () => {
            try {
                const { fetchWithAuth, API_BASE_URL } = await import('../../services/apiService');
                const response = await fetchWithAuth(`${API_BASE_URL}/api/weekly-picks/current`);
                if (response.ok) {
                    const data = await response.json();
                    if (data.picks && data.picks.length > 0) {
                        const convertedPicks: AlphaPick[] = data.picks.map((wp: WeeklyPick) => ({
                            ticker: wp.ticker,
                            company_name: wp.company_name,
                            sector: wp.sector,
                            cas_score: wp.cas_score,
                            scores: wp.scores,
                            explanation: wp.investment_thesis
                        }));
                        setPicks(convertedPicks);
                        setGenerated(true);
                        setWeekInfo({
                            week_start: data.week_start,
                            generated_date: data.generated_date
                        });
                    }
                }
            } catch (e) {
                console.error('Failed to load weekly picks', e);
            }
        };
        loadWeeklyPicks();
    }, []);

    // Load universe stats
    useEffect(() => {
        const fetchStats = async () => {
            try {
                const { fetchWithAuth, API_BASE_URL } = await import('../../services/apiService');
                const res = await fetchWithAuth(`${API_BASE_URL}/api/valuation/cache/stats`);
                if (res.ok) {
                    const data = await res.json();
                    setUniverseSize(data.total_universe);
                }
            } catch (e) {
                console.error('Failed to load universe stats', e);
            }
        };
        fetchStats();
    }, []);

    const generatePicks = async () => {
        setLoading(true);
        setError(null);
        try {
            const { fetchWithAuth, API_BASE_URL } = await import('../../services/apiService');
            const response = await fetchWithAuth(`${API_BASE_URL}/api/alpha-picks/`);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch picks' }));
                throw new Error(errorData.detail || 'Failed to fetch picks');
            }
            const data = await response.json();
            if (!Array.isArray(data) || data.length === 0) {
                throw new Error('No picks generated. Please try again.');
            }
            setPicks(data);
            setGenerated(true);
        } catch (err: any) {
            console.error('Alpha Stock Picker error:', err);
            setError(err.message || 'Failed to generate picks. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <WidgetCard 
            title="ALPHA STOCK PICKER" 
            tooltip="Weekly 3-stock selection using Composite Alpha Score (CAS): Momentum, Quality, Valuation & Catalyst factors. Updated every Monday."
        >
            {/* Meta Info */}
            <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-4">
                    {universeSize !== null && (
                        <span 
                            className="text-xs font-mono"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            Screening {universeSize.toLocaleString()} stocks
                        </span>
                    )}
                    {weekInfo && (
                        <span 
                            className="text-xs"
                            style={{ color: 'var(--color-text-subtle)' }}
                        >
                            Week of {new Date(weekInfo.week_start).toLocaleDateString()}
                        </span>
                    )}
                </div>
                {generated && (
                    <button
                        onClick={generatePicks}
                        disabled={loading}
                        className="text-xs font-medium transition-colors"
                        style={{ color: 'var(--color-text-muted)' }}
                        onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-accent-primary)'}
                        onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-muted)'}
                    >
                        Refresh
                    </button>
                )}
            </div>

            {/* Initial State */}
            {!generated && !loading && !error && (
                <div 
                    className="rounded-xl p-10 text-center"
                    style={{ 
                        backgroundColor: 'var(--color-bg-tertiary)',
                        border: '1px solid var(--color-border-subtle)'
                    }}
                >
                    <div 
                        className="w-16 h-16 mx-auto mb-5 rounded-xl flex items-center justify-center"
                        style={{ backgroundColor: 'rgba(46, 124, 246, 0.12)' }}
                    >
                        <svg className="w-8 h-8" style={{ color: 'var(--color-accent-primary)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                        </svg>
                    </div>
                    <h3 
                        className="text-lg font-semibold mb-2"
                        style={{ color: 'var(--color-text-primary)' }}
                    >
                        Weekly Alpha Picks
                    </h3>
                    <p 
                        className="text-sm mb-6 max-w-sm mx-auto"
                        style={{ color: 'var(--color-text-secondary)' }}
                    >
                        Top 3 stock picks based on our Composite Alpha Score model—combining momentum, quality, valuation, and catalyst signals.
                    </p>
                    <button
                        onClick={generatePicks}
                        className="px-6 py-3 rounded-lg font-semibold text-sm transition-all duration-200"
                        style={{
                            backgroundColor: 'var(--color-accent-primary)',
                            color: '#FFFFFF',
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.transform = 'translateY(-2px)';
                            e.currentTarget.style.boxShadow = '0 4px 12px rgba(46, 124, 246, 0.3)';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.transform = 'translateY(0)';
                            e.currentTarget.style.boxShadow = 'none';
                        }}
                    >
                        Load Weekly Picks
                    </button>
                </div>
            )}

            {/* Loading State */}
            {loading && (
                <div 
                    className="rounded-xl p-10 text-center"
                    style={{ 
                        backgroundColor: 'var(--color-bg-tertiary)',
                        border: '1px solid var(--color-border-subtle)'
                    }}
                >
                    <div className="w-10 h-10 mx-auto mb-4 border-3 border-t-transparent rounded-full animate-spin"
                        style={{ borderColor: 'var(--color-accent-primary)', borderTopColor: 'transparent' }}
                    />
                    <p 
                        className="text-sm animate-pulse"
                        style={{ color: 'var(--color-accent-primary)' }}
                    >
                        Running alpha models...
                    </p>
                </div>
            )}

            {/* Error State */}
            {error && (
                <div 
                    className="rounded-xl p-6 text-center"
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
                        onClick={generatePicks}
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

            {/* Results */}
            {generated && !loading && !error && (
                <div className="grid md:grid-cols-3 gap-4 animate-fade-in">
                    {picks.map(pick => (
                        <ResearchCard key={pick.ticker} pick={pick} />
                    ))}
                </div>
            )}
        </WidgetCard>
    );
};

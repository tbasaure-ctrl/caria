import React, { useState } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';

interface HiddenGem {
    ticker: string;
    cScore: number;
    hiddenGemScore: number;
    classification: string | null;
    qualityScore: number;
    valuationScore: number;
    momentumScore: number;
    current_price: number | null;
    details: any;
    explanations: any;
}

const getClassificationStyle = (classification: string | null) => {
    if (!classification) return { color: 'var(--color-text-muted)', bg: 'var(--color-bg-surface)' };
    if (classification === 'Hidden Gem') return { color: 'var(--color-positive)', bg: 'var(--color-positive-muted)' };
    if (classification === 'Investable') return { color: 'var(--color-accent-primary)', bg: 'rgba(46, 124, 246, 0.15)' };
    if (classification === 'Watchlist') return { color: 'var(--color-warning)', bg: 'var(--color-warning-muted)' };
    return { color: 'var(--color-text-muted)', bg: 'var(--color-bg-surface)' };
};

const ScoreBar: React.FC<{ value: number; color: string }> = ({ value, color }) => (
    <div className="flex items-center gap-2">
        <div 
            className="flex-1 h-1.5 rounded-full overflow-hidden"
            style={{ backgroundColor: 'var(--color-bg-surface)' }}
        >
            <div 
                className="h-full rounded-full transition-all duration-500"
                style={{ 
                    width: `${Math.min(value, 100)}%`,
                    backgroundColor: color
                }}
            />
        </div>
        <span 
            className="text-xs font-mono font-medium w-8 text-right"
            style={{ color }}
        >
            {Math.round(value)}
        </span>
    </div>
);

export const HiddenGemsScreener: React.FC = () => {
    const [gems, setGems] = useState<HiddenGem[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [generated, setGenerated] = useState(false);

    const generateGems = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/screener/hidden-gems?limit=10`);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch hidden gems' }));
                throw new Error(errorData.detail || 'Failed to fetch hidden gems');
            }
            const data = await response.json();
            if (!data.results || data.results.length === 0) {
                throw new Error('No hidden gems found. Try again later.');
            }
            setGems(data.results);
            setGenerated(true);
        } catch (err: any) {
            console.error('Hidden Gems Screener error:', err);
            setError(err.message || 'Failed to generate hidden gems. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const scoreExplanation = (
        <div className="space-y-2 text-sm">
            <p className="font-medium" style={{ color: 'var(--color-text-primary)' }}>
                Hidden Gem Score (0-100)
            </p>
            <ul className="space-y-1" style={{ color: 'var(--color-text-secondary)' }}>
                <li>• <strong>Quality (30%):</strong> Financial health & profitability</li>
                <li>• <strong>Valuation (30%):</strong> Attractive price vs fundamentals</li>
                <li>• <strong>Momentum (25%):</strong> Recent price performance</li>
                <li>• <strong>Bonus/Penalty (15%):</strong> Exceptional metrics or red flags</li>
            </ul>
            <p className="pt-2 text-xs" style={{ color: 'var(--color-text-muted)' }}>
                Stocks scoring 80+ are classified as "Hidden Gems"—undervalued mid-caps with strong fundamentals.
            </p>
            <p className="pt-2 text-xs" style={{ color: 'var(--color-negative)' }}>
                ⚠️ Not financial advice. Always conduct your own research.
            </p>
        </div>
    );

    return (
        <WidgetCard 
            title="HIDDEN GEMS SCREENER" 
            tooltip={scoreExplanation}
        >
            {/* Header Controls */}
            <div className="flex items-center justify-between mb-5">
                <span 
                    className="text-xs"
                    style={{ color: 'var(--color-text-muted)' }}
                >
                    Mid-cap stocks ($50M - $10B market cap)
                </span>
                {generated && (
                    <button
                        onClick={generateGems}
                        disabled={loading}
                        className="text-xs font-medium transition-colors"
                        style={{ color: 'var(--color-text-muted)' }}
                        onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-positive)'}
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
                        style={{ backgroundColor: 'var(--color-positive-muted)' }}
                    >
                        <svg className="w-8 h-8" style={{ color: 'var(--color-positive)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                        </svg>
                    </div>
                    <h3 
                        className="text-lg font-semibold mb-2"
                        style={{ color: 'var(--color-text-primary)' }}
                    >
                        Discover Hidden Gems
                    </h3>
                    <p 
                        className="text-sm mb-6 max-w-sm mx-auto"
                        style={{ color: 'var(--color-text-secondary)' }}
                    >
                        Screen for undervalued mid-cap stocks with strong quality, valuation, and momentum scores—before Wall Street notices.
                    </p>
                    <button
                        onClick={generateGems}
                        className="px-6 py-3 rounded-lg font-semibold text-sm transition-all duration-200"
                        style={{
                            backgroundColor: 'var(--color-positive)',
                            color: '#FFFFFF',
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.transform = 'translateY(-2px)';
                            e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 200, 83, 0.3)';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.transform = 'translateY(0)';
                            e.currentTarget.style.boxShadow = 'none';
                        }}
                    >
                        Start Screening
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
                        style={{ borderColor: 'var(--color-positive)', borderTopColor: 'transparent' }}
                    />
                    <p 
                        className="text-sm animate-pulse"
                        style={{ color: 'var(--color-positive)' }}
                    >
                        Screening mid-cap universe...
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
                        onClick={generateGems}
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

            {/* Results - Bloomberg Table Style */}
            {generated && !loading && !error && (
                <div className="space-y-3 animate-fade-in">
                    {/* Table Header */}
                    <div 
                        className="grid grid-cols-12 gap-3 px-4 py-2 text-[10px] font-medium tracking-wider uppercase"
                        style={{ color: 'var(--color-text-muted)' }}
                    >
                        <div className="col-span-3">Ticker</div>
                        <div className="col-span-2 text-right">Score</div>
                        <div className="col-span-2">Quality</div>
                        <div className="col-span-2">Value</div>
                        <div className="col-span-2">Momentum</div>
                        <div className="col-span-1 text-right">Price</div>
                    </div>

                    {/* Table Rows */}
                    {gems.map((gem, idx) => {
                        const classStyle = getClassificationStyle(gem.classification);
                        
                        return (
                            <div 
                                key={gem.ticker}
                                className="grid grid-cols-12 gap-3 items-center px-4 py-3 rounded-lg transition-all duration-200"
                                style={{
                                    backgroundColor: 'var(--color-bg-tertiary)',
                                    border: '1px solid var(--color-border-subtle)',
                                    animationDelay: `${idx * 50}ms`
                                }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.borderColor = 'var(--color-border-emphasis)';
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.borderColor = 'var(--color-border-subtle)';
                                }}
                            >
                                {/* Ticker + Classification */}
                                <div className="col-span-3">
                                    <div className="flex items-center gap-2">
                                        <span 
                                            className="text-base font-bold font-mono"
                                            style={{ color: 'var(--color-text-primary)' }}
                                        >
                                            {gem.ticker}
                                        </span>
                                    </div>
                                    <span 
                                        className="text-[10px] font-medium px-1.5 py-0.5 rounded"
                                        style={{ 
                                            backgroundColor: classStyle.bg,
                                            color: classStyle.color
                                        }}
                                    >
                                        {gem.classification || 'N/A'}
                                    </span>
                                </div>

                                {/* Hidden Gem Score */}
                                <div className="col-span-2 text-right">
                                    <span 
                                        className="text-xl font-bold font-mono"
                                        style={{ 
                                            color: gem.hiddenGemScore >= 80 ? 'var(--color-positive)' : 
                                                   gem.hiddenGemScore >= 60 ? 'var(--color-accent-primary)' : 
                                                   'var(--color-text-secondary)'
                                        }}
                                    >
                                        {gem.hiddenGemScore.toFixed(0)}
                                    </span>
                                </div>

                                {/* Quality Score */}
                                <div className="col-span-2">
                                    <ScoreBar value={gem.qualityScore} color="var(--color-accent-primary)" />
                                </div>

                                {/* Valuation Score */}
                                <div className="col-span-2">
                                    <ScoreBar value={gem.valuationScore} color="#8B5CF6" />
                                </div>

                                {/* Momentum Score */}
                                <div className="col-span-2">
                                    <ScoreBar value={gem.momentumScore} color="var(--color-warning)" />
                                </div>

                                {/* Price */}
                                <div className="col-span-1 text-right">
                                    <span 
                                        className="text-sm font-mono"
                                        style={{ color: 'var(--color-text-secondary)' }}
                                    >
                                        {gem.current_price 
                                            ? `$${gem.current_price.toFixed(2)}` 
                                            : '—'
                                        }
                                    </span>
                                </div>
                            </div>
                        );
                    })}

                    {/* Footer Legend */}
                    <div 
                        className="flex items-center justify-center gap-6 pt-4 text-[10px]"
                        style={{ color: 'var(--color-text-muted)' }}
                    >
                        <div className="flex items-center gap-1.5">
                            <span 
                                className="w-2 h-2 rounded-full"
                                style={{ backgroundColor: 'var(--color-positive)' }}
                            />
                            Hidden Gem (80+)
                        </div>
                        <div className="flex items-center gap-1.5">
                            <span 
                                className="w-2 h-2 rounded-full"
                                style={{ backgroundColor: 'var(--color-accent-primary)' }}
                            />
                            Investable (60-79)
                        </div>
                        <div className="flex items-center gap-1.5">
                            <span 
                                className="w-2 h-2 rounded-full"
                                style={{ backgroundColor: 'var(--color-warning)' }}
                            />
                            Watchlist (40-59)
                        </div>
                    </div>
                </div>
            )}
        </WidgetCard>
    );
};

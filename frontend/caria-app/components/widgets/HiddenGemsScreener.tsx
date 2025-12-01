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

const MOCK_GEMS: HiddenGem[] = [
    { ticker: "CROX", cScore: 85, hiddenGemScore: 88, classification: "Hidden Gem", qualityScore: 90, valuationScore: 85, momentumScore: 89, current_price: 124.50, details: {}, explanations: {} },
    { ticker: "ELF", cScore: 82, hiddenGemScore: 84, classification: "Hidden Gem", qualityScore: 88, valuationScore: 70, momentumScore: 95, current_price: 168.20, details: {}, explanations: {} },
    { ticker: "HIMS", cScore: 78, hiddenGemScore: 79, classification: "Investable", qualityScore: 75, valuationScore: 80, momentumScore: 82, current_price: 14.30, details: {}, explanations: {} },
    { ticker: "CELH", cScore: 75, hiddenGemScore: 76, classification: "Investable", qualityScore: 80, valuationScore: 65, momentumScore: 78, current_price: 52.10, details: {}, explanations: {} },
    { ticker: "SOFI", cScore: 70, hiddenGemScore: 72, classification: "Investable", qualityScore: 70, valuationScore: 75, momentumScore: 70, current_price: 7.80, details: {}, explanations: {} },
];

const getClassificationStyle = (classification: string | null) => {
    if (!classification) return { color: 'var(--color-text-muted)', bg: 'var(--color-bg-surface)' };
    if (classification === 'Hidden Gem') return { color: 'var(--color-positive)', bg: 'var(--color-positive-muted)' };
    if (classification === 'Investable') return { color: 'var(--color-accent-primary)', bg: 'rgba(46, 124, 246, 0.15)' };
    if (classification === 'Watchlist') return { color: 'var(--color-warning)', bg: 'var(--color-warning-muted)' };
    return { color: 'var(--color-text-muted)', bg: 'var(--color-bg-surface)' };
};

const ScoreBar: React.FC<{ value: number; color: string; label?: string }> = ({ value, color, label }) => (
    <div className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-2 w-full">
        {label && <span className="text-[10px] text-text-muted sm:hidden">{label}</span>}
        <div className="flex items-center gap-2 w-full">
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
    </div>
);

export const HiddenGemsScreener: React.FC = () => {
    const [gems, setGems] = useState<HiddenGem[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [generated, setGenerated] = useState(false);
    const [isMock, setIsMock] = useState(false);

    const generateGems = async () => {
        setLoading(true);
        setError(null);
        setIsMock(false);
        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/screener/hidden-gems?limit=10`);
            if (!response.ok) {
                // Fallback to mock
                console.warn("API failed, using mock data");
                setIsMock(true);
                setGems(MOCK_GEMS);
                setGenerated(true);
                return; 
            }
            const data = await response.json();
            if (!data.results || data.results.length === 0) {
                 // Fallback if empty
                setIsMock(true);
                setGems(MOCK_GEMS);
                setGenerated(true);
                return;
            }
            setGems(data.results);
            setGenerated(true);
        } catch (err: any) {
            console.error('Hidden Gems Screener error:', err);
            // Use mock on error
             setIsMock(true);
             setGems(MOCK_GEMS);
             setGenerated(true);
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
                    <div className="flex items-center gap-4">
                        {isMock && <span className="text-[10px] text-warning italic hidden sm:inline">Demo Mode</span>}
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
                    </div>
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

            {/* Results - Responsive Table/Card Layout */}
            {generated && !loading && (
                <div className="space-y-3 animate-fade-in">
                    {/* Table Header - Hidden on Mobile */}
                    <div 
                        className="hidden sm:grid grid-cols-12 gap-3 px-4 py-2 text-[10px] font-medium tracking-wider uppercase"
                        style={{ color: 'var(--color-text-muted)' }}
                    >
                        <div className="col-span-3">Ticker</div>
                        <div className="col-span-2 text-right">Score</div>
                        <div className="col-span-2">Quality</div>
                        <div className="col-span-2">Value</div>
                        <div className="col-span-2">Momentum</div>
                        <div className="col-span-1 text-right">Price</div>
                    </div>

                    {/* Rows / Cards */}
                    {gems.map((gem, idx) => {
                        const classStyle = getClassificationStyle(gem.classification);
                        
                        return (
                            <div 
                                key={gem.ticker}
                                className="flex flex-col sm:grid sm:grid-cols-12 gap-3 sm:items-center px-4 py-3 rounded-lg transition-all duration-200"
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
                                <div className="col-span-3 flex justify-between sm:block items-center">
                                    <div className="flex items-center gap-2">
                                        <span 
                                            className="text-base font-bold font-mono"
                                            style={{ color: 'var(--color-text-primary)' }}
                                        >
                                            {gem.ticker}
                                        </span>
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
                                    {/* Mobile Price Display */}
                                    <span className="sm:hidden text-sm font-mono text-white">
                                        {gem.current_price ? `$${gem.current_price.toFixed(2)}` : '—'}
                                    </span>
                                </div>

                                {/* Hidden Gem Score */}
                                <div className="col-span-2 flex justify-between sm:justify-end items-center border-b sm:border-b-0 border-white/5 pb-2 sm:pb-0 mb-2 sm:mb-0">
                                    <span className="text-[10px] text-text-muted sm:hidden uppercase">Gem Score</span>
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
                                    <ScoreBar value={gem.qualityScore} color="var(--color-accent-primary)" label="Quality" />
                                </div>

                                {/* Valuation Score */}
                                <div className="col-span-2">
                                    <ScoreBar value={gem.valuationScore} color="#8B5CF6" label="Value" />
                                </div>

                                {/* Momentum Score */}
                                <div className="col-span-2">
                                    <ScoreBar value={gem.momentumScore} color="var(--color-warning)" label="Momentum" />
                                </div>

                                {/* Price (Desktop Only) */}
                                <div className="col-span-1 text-right hidden sm:block">
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
                        className="flex flex-wrap items-center justify-center gap-4 sm:gap-6 pt-4 text-[10px]"
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

import React, { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';
import { BASE_INDUSTRIES } from '../../data/industries';

interface IndustryCard {
    name: string;
    status: string;
    growth_signal: string;
    valuation_context: string;
    flows_activity: string;
    representative_tickers: string[];
}

interface IndustryDetail {
    name: string;
    thesis_summary: string[];
    aggregate_kpis: {
        revenue_growth: number;
        margins: number;
        ev_ebitda_median: number;
        market_cap_total: number;
    };
    stage: string;
    leaders_challengers: Array<{
        ticker: string;
        name: string;
        market_share: string;
    }>;
    key_risks: string[];
    caria_signals: {
        alpha_picker_appearances: number;
        screener_appearances: number;
        crisis_sensitivity: string;
    };
    recent_headlines: string[];
    learning_resources: {
        lectures: string[];
        videos: string[];
    };
}

const StatusPill: React.FC<{ status: string }> = ({ status }) => {
    const colorMap: Record<string, { bg: string; text: string }> = {
        'Emerging': { bg: 'rgba(46, 124, 246, 0.15)', text: 'var(--color-accent-primary)' },
        'Mature': { bg: 'rgba(139, 92, 246, 0.15)', text: '#8B5CF6' },
        'Overheated': { bg: 'rgba(255, 152, 0, 0.15)', text: 'var(--color-warning)' },
        'Under Pressure': { bg: 'rgba(239, 68, 68, 0.15)', text: 'var(--color-negative)' },
    };
    const colors = colorMap[status] || { bg: 'var(--color-bg-surface)', text: 'var(--color-text-muted)' };
    
    return (
        <span
            className="px-3 py-1 rounded-full text-xs font-semibold"
            style={{ backgroundColor: colors.bg, color: colors.text }}
        >
            {status}
        </span>
    );
};

const IndustryCardComponent: React.FC<{ 
    industry: IndustryCard; 
    onSelect: (name: string) => void;
}> = ({ industry, onSelect }) => {
    return (
        <div
            className="rounded-xl p-6 cursor-pointer transition-all duration-300"
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
            onClick={() => onSelect(industry.name)}
        >
            <div className="flex items-start justify-between mb-4">
                <h3 
                    className="text-lg font-bold"
                    style={{ color: 'var(--color-text-primary)' }}
                >
                    {industry.name}
                </h3>
                <StatusPill status={industry.status} />
            </div>
            
            <div className="space-y-3 mb-4">
                <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                    {industry.growth_signal}
                </p>
                <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                    {industry.valuation_context}
                </p>
                <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
                    {industry.flows_activity}
                </p>
            </div>
            
            <div className="flex items-center gap-2 pt-4 border-t" style={{ borderColor: 'var(--color-border-subtle)' }}>
                <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>Examples:</span>
                {industry.representative_tickers.map(ticker => (
                    <span
                        key={ticker}
                        className="px-2 py-1 rounded text-xs font-mono font-semibold"
                        style={{
                            backgroundColor: 'var(--color-bg-surface)',
                            color: 'var(--color-text-primary)'
                        }}
                    >
                        {ticker}
                    </span>
                ))}
            </div>
            
            <button
                className="w-full mt-4 py-2 rounded-lg text-sm font-medium transition-colors"
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
                Open Industry Page →
            </button>
        </div>
    );
};

export const IndustryResearch: React.FC = () => {
    const [industries, setIndustries] = useState<IndustryCard[]>([]);
    const [selectedIndustry, setSelectedIndustry] = useState<IndustryDetail | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [featuredIndustry, setFeaturedIndustry] = useState<IndustryCard | null>(null);

    useEffect(() => {
        const loadIndustries = async () => {
            setLoading(true);
            try {
                const response = await fetchWithAuth(`${API_BASE_URL}/api/industry-research/industries`);
                if (response.ok) {
                    const data = await response.json();
                    setIndustries(data);
                    // Set featured industry (first one or random)
                    if (data.length > 0) {
                        setFeaturedIndustry(data[0]);
                    }
                } else {
                    // Fallback to using base industries config if API fails
                    const fallbackIndustries: IndustryCard[] = BASE_INDUSTRIES.map(industry => ({
                        name: industry.name,
                        status: industry.status,
                        growth_signal: `Growth drivers and market dynamics for ${industry.name}`,
                        valuation_context: `Valuation metrics vs historical averages`,
                        flows_activity: `ETF flows and M&A activity trends`,
                        representative_tickers: industry.representative_tickers
                    }));
                    setIndustries(fallbackIndustries);
                    if (fallbackIndustries.length > 0) {
                        setFeaturedIndustry(fallbackIndustries[0]);
                    }
                }
            } catch (e) {
                console.error('Failed to load industries', e);
                // Fallback to using base industries config
                const fallbackIndustries: IndustryCard[] = BASE_INDUSTRIES.map(industry => ({
                    name: industry.name,
                    status: industry.status,
                    growth_signal: `Growth drivers and market dynamics for ${industry.name}`,
                    valuation_context: `Valuation metrics vs historical averages`,
                    flows_activity: `ETF flows and M&A activity trends`,
                    representative_tickers: industry.representative_tickers
                }));
                setIndustries(fallbackIndustries);
                if (fallbackIndustries.length > 0) {
                    setFeaturedIndustry(fallbackIndustries[0]);
                }
            } finally {
                setLoading(false);
            }
        };
        loadIndustries();
    }, []);

    const handleSelectIndustry = async (name: string) => {
        setLoading(true);
        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/industry-research/industries/${encodeURIComponent(name)}`);
            if (response.ok) {
                const data = await response.json();
                setSelectedIndustry(data);
            }
        } catch (e) {
            console.error('Failed to load industry detail', e);
            setError('Failed to load industry details');
        } finally {
            setLoading(false);
        }
    };

    if (selectedIndustry) {
        return (
            <WidgetCard 
                title="INDUSTRY RESEARCH"
                tooltip="Deep dive into sector analysis, trends, and investment opportunities"
            >
                <div className="space-y-6">
                    <button
                        onClick={() => setSelectedIndustry(null)}
                        className="text-sm font-medium mb-4"
                        style={{ color: 'var(--color-accent-primary)' }}
                    >
                        ← Back to Industries
                    </button>
                    
                    {/* Sector Sheet - Top Section */}
                    <div 
                        className="mb-6 p-6 rounded-xl"
                        style={{
                            backgroundColor: 'var(--color-bg-secondary)',
                            border: '1px solid var(--color-border-subtle)',
                        }}
                    >
                        <div className="flex items-start justify-between mb-4">
                            <div>
                                <h2 
                                    className="text-2xl font-bold mb-2"
                                    style={{ 
                                        fontFamily: 'var(--font-display)',
                                        color: 'var(--color-text-primary)' 
                                    }}
                                >
                                    {selectedIndustry.name}
                                </h2>
                                <div className="flex items-center gap-2">
                                    <StatusPill status={selectedIndustry.stage} />
                                    <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                                        Stage: {selectedIndustry.stage}
                                    </span>
                                </div>
                            </div>
                        </div>
                        
                        {/* Thesis Summary - Bullets */}
                        <div className="mb-4">
                            <h3 className="text-xs font-semibold mb-2 uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>
                                Thesis Summary
                            </h3>
                            <ul className="space-y-1.5">
                                {selectedIndustry.thesis_summary.map((point, idx) => (
                                    <li key={idx} className="text-sm flex items-start gap-2" style={{ color: 'var(--color-text-secondary)' }}>
                                        <span style={{ color: 'var(--color-accent-primary)' }}>•</span>
                                        <span>{point}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                        
                        {/* Aggregate KPIs */}
                        <div>
                            <h3 className="text-xs font-semibold mb-3 uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>
                                Aggregate KPIs
                            </h3>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                {Object.entries(selectedIndustry.aggregate_kpis).map(([key, value]) => (
                                    <div key={key} className="p-3 rounded-lg" style={{ backgroundColor: 'var(--color-bg-tertiary)' }}>
                                        <div className="text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--color-text-muted)' }}>
                                            {key.replace(/_/g, ' ')}
                                        </div>
                                        <div className="text-lg font-bold font-mono" style={{ color: 'var(--color-text-primary)' }}>
                                            {typeof value === 'number' && value > 1000000000 
                                                ? `$${(value / 1000000000).toFixed(1)}B`
                                                : typeof value === 'number' && value > 1000000
                                                ? `$${(value / 1000000).toFixed(1)}M`
                                                : typeof value === 'number'
                                                ? value.toFixed(1) + (key.includes('growth') || key.includes('margin') ? '%' : 'x')
                                                : value}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                    
                    <div>
                        <h3 className="text-sm font-semibold mb-3 uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>
                            Leaders & Challengers
                        </h3>
                        <div className="space-y-2">
                            {selectedIndustry.leaders_challengers.map((company, idx) => (
                                <div key={idx} className="p-3 rounded-lg flex items-center justify-between" style={{ backgroundColor: 'var(--color-bg-tertiary)' }}>
                                    <div>
                                        <span className="font-mono font-semibold mr-2" style={{ color: 'var(--color-text-primary)' }}>
                                            {company.ticker}
                                        </span>
                                        <span className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                                            {company.name}
                                        </span>
                                    </div>
                                    <span className="text-sm font-mono" style={{ color: 'var(--color-text-muted)' }}>
                                        {company.market_share}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                    
                    <div>
                        <h3 className="text-sm font-semibold mb-3 uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>
                            Key Risks
                        </h3>
                        <ul className="space-y-2">
                            {selectedIndustry.key_risks.map((risk, idx) => (
                                <li key={idx} className="text-sm flex items-start gap-2" style={{ color: 'var(--color-text-secondary)' }}>
                                    <span style={{ color: 'var(--color-negative)' }}>⚠</span>
                                    <span>{risk}</span>
                                </li>
                            ))}
                        </ul>
                    </div>
                    
                    <div>
                        <h3 className="text-sm font-semibold mb-3 uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>
                            Caria Signals
                        </h3>
                        <p className="text-xs mb-3" style={{ color: 'var(--color-text-muted)' }}>
                            % of tickers in this industry that appear in:
                        </p>
                        <div className="grid grid-cols-3 gap-4">
                            <div className="p-3 rounded-lg text-center" style={{ backgroundColor: 'var(--color-bg-tertiary)' }}>
                                <div className="text-xs mb-1" style={{ color: 'var(--color-text-muted)' }}>Alpha Stock Picker</div>
                                <div className="text-lg font-bold font-mono" style={{ color: 'var(--color-text-primary)' }}>
                                    {typeof selectedIndustry.caria_signals.alpha_picker_appearances === 'number' 
                                        ? `${selectedIndustry.caria_signals.alpha_picker_appearances}%`
                                        : selectedIndustry.caria_signals.alpha_picker_appearances}
                                </div>
                            </div>
                            <div className="p-3 rounded-lg text-center" style={{ backgroundColor: 'var(--color-bg-tertiary)' }}>
                                <div className="text-xs mb-1" style={{ color: 'var(--color-text-muted)' }}>Under-the-Radar Screener</div>
                                <div className="text-lg font-bold font-mono" style={{ color: 'var(--color-text-primary)' }}>
                                    {typeof selectedIndustry.caria_signals.screener_appearances === 'number' 
                                        ? `${selectedIndustry.caria_signals.screener_appearances}%`
                                        : selectedIndustry.caria_signals.screener_appearances}
                                </div>
                            </div>
                            <div className="p-3 rounded-lg text-center" style={{ backgroundColor: 'var(--color-bg-tertiary)' }}>
                                <div className="text-xs mb-1" style={{ color: 'var(--color-text-muted)' }}>Crisis / Macro Sensitivity</div>
                                <div className="text-lg font-bold" style={{ color: 'var(--color-text-primary)' }}>
                                    {selectedIndustry.caria_signals.crisis_sensitivity}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Recent Headlines */}
                    {selectedIndustry.recent_headlines && selectedIndustry.recent_headlines.length > 0 && (
                        <div>
                            <h3 className="text-sm font-semibold mb-3 uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>
                                Recent Headlines
                            </h3>
                            <div className="space-y-2">
                                {selectedIndustry.recent_headlines.map((headline, idx) => (
                                    <div 
                                        key={idx} 
                                        className="p-3 rounded-lg text-sm flex items-start gap-2" 
                                        style={{ backgroundColor: 'var(--color-bg-tertiary)' }}
                                    >
                                        <span style={{ color: 'var(--color-accent-primary)' }}>📰</span>
                                        <span style={{ color: 'var(--color-text-secondary)' }}>{headline}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Learning Resources */}
                    {selectedIndustry.learning_resources && (
                        <div>
                            <h3 className="text-sm font-semibold mb-3 uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>
                                Learning
                            </h3>
                            <div className="space-y-4">
                                {/* Lectures */}
                                {selectedIndustry.learning_resources.lectures && selectedIndustry.learning_resources.lectures.length > 0 && (
                                    <div>
                                        <div className="text-xs font-medium mb-2 uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>
                                            Recommended Lectures
                                        </div>
                                        <ul className="space-y-2">
                                            {selectedIndustry.learning_resources.lectures.slice(0, 2).map((lecture, idx) => (
                                                <li key={idx} className="text-sm flex items-start gap-2" style={{ color: 'var(--color-text-secondary)' }}>
                                                    <span style={{ color: 'var(--color-accent-primary)' }}>📚</span>
                                                    <span>{lecture}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                                
                                {/* Videos/Podcasts */}
                                {selectedIndustry.learning_resources.videos && selectedIndustry.learning_resources.videos.length > 0 && (
                                    <div>
                                        <div className="text-xs font-medium mb-2 uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>
                                            Video / Podcast
                                        </div>
                                        <ul className="space-y-2">
                                            {selectedIndustry.learning_resources.videos.slice(0, 1).map((video, idx) => (
                                                <li key={idx} className="text-sm flex items-start gap-2" style={{ color: 'var(--color-text-secondary)' }}>
                                                    <span style={{ color: 'var(--color-accent-primary)' }}>🎥</span>
                                                    <span>{video}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            </WidgetCard>
        );
    }

    return (
        <WidgetCard 
            title="INDUSTRY RESEARCH"
            tooltip="Deep dive into sector analysis, trends, and investment opportunities"
        >
            {loading && !industries.length && (
                <div className="text-center py-12">
                    <div className="w-10 h-10 mx-auto mb-4 border-3 border-t-transparent rounded-full animate-spin"
                        style={{ borderColor: 'var(--color-accent-primary)', borderTopColor: 'transparent' }}
                    />
                    <p style={{ color: 'var(--color-text-muted)' }}>Loading industries...</p>
                </div>
            )}
            
            {error && (
                <div className="p-4 rounded-lg mb-4" style={{ backgroundColor: 'var(--color-negative-muted)', color: 'var(--color-negative)' }}>
                    {error}
                </div>
            )}
            
            {industries.length > 0 && (
                <div>
                    {/* Industry of the Month - Large Card */}
                    {featuredIndustry && (
                        <div className="mb-6">
                            <div className="flex items-center gap-2 mb-3">
                                <span 
                                    className="text-xs font-semibold uppercase tracking-wider"
                                    style={{ color: 'var(--color-text-muted)' }}
                                >
                                    Industry of the Month
                                </span>
                            </div>
                            <div 
                                className="p-6 rounded-xl cursor-pointer transition-all duration-300"
                                style={{
                                    backgroundColor: 'var(--color-bg-secondary)',
                                    border: '2px solid var(--color-accent-primary)',
                                }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.transform = 'translateY(-2px)';
                                    e.currentTarget.style.boxShadow = '0 8px 24px rgba(46, 124, 246, 0.2)';
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.transform = 'translateY(0)';
                                    e.currentTarget.style.boxShadow = 'none';
                                }}
                                onClick={() => handleSelectIndustry(featuredIndustry.name)}
                            >
                                <div className="flex items-start justify-between mb-4">
                                    <div className="flex-1">
                                        <h2 
                                            className="text-2xl font-bold mb-3"
                                            style={{ 
                                                fontFamily: 'var(--font-display)',
                                                color: 'var(--color-text-primary)' 
                                            }}
                                        >
                                            {featuredIndustry.name}
                                        </h2>
                                        <StatusPill status={featuredIndustry.status} />
                                    </div>
                                </div>
                                
                                <p className="text-sm mb-4" style={{ color: 'var(--color-text-secondary)' }}>
                                    {featuredIndustry.growth_signal}
                                </p>
                                
                                <div className="flex items-center gap-2 pt-4 border-t" style={{ borderColor: 'var(--color-border-subtle)' }}>
                                    <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>Examples:</span>
                                    {featuredIndustry.representative_tickers.slice(0, 4).map(ticker => (
                                        <span
                                            key={ticker}
                                            className="px-2 py-1 rounded text-xs font-mono font-semibold"
                                            style={{
                                                backgroundColor: 'var(--color-bg-surface)',
                                                color: 'var(--color-text-primary)'
                                            }}
                                        >
                                            {ticker}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}
                    
                    {/* Trending Industries - Smaller Cards */}
                    <div className="mb-4">
                        <h3 
                            className="text-lg font-semibold mb-4"
                            style={{ color: 'var(--color-text-primary)' }}
                        >
                            Trending Industries
                        </h3>
                    </div>
                    
                    <div className="grid md:grid-cols-3 gap-4">
                        {industries.filter(ind => ind.name !== featuredIndustry?.name).slice(0, 3).map(industry => (
                            <div
                                key={industry.name}
                                className="rounded-xl p-4 cursor-pointer transition-all duration-300"
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
                                onClick={() => handleSelectIndustry(industry.name)}
                            >
                                <div className="flex items-start justify-between mb-3">
                                    <h3 
                                        className="text-base font-bold flex-1"
                                        style={{ color: 'var(--color-text-primary)' }}
                                    >
                                        {industry.name}
                                    </h3>
                                    <StatusPill status={industry.status} />
                                </div>
                                
                                <p className="text-xs mb-3 line-clamp-2" style={{ color: 'var(--color-text-secondary)' }}>
                                    {industry.growth_signal}
                                </p>
                                
                                <div className="flex items-center gap-1.5 flex-wrap">
                                    {industry.representative_tickers.slice(0, 3).map(ticker => (
                                        <span
                                            key={ticker}
                                            className="px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold"
                                            style={{
                                                backgroundColor: 'var(--color-bg-surface)',
                                                color: 'var(--color-text-primary)'
                                            }}
                                        >
                                            {ticker}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </WidgetCard>
    );
};

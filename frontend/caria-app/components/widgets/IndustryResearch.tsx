import React, { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';

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

    useEffect(() => {
        const loadIndustries = async () => {
            setLoading(true);
            try {
                const response = await fetchWithAuth(`${API_BASE_URL}/api/industry-research/industries`);
                if (response.ok) {
                    const data = await response.json();
                    setIndustries(data);
                }
            } catch (e) {
                console.error('Failed to load industries', e);
                setError('Failed to load industries');
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
                    
                    <div>
                        <h2 
                            className="text-2xl font-bold mb-2"
                            style={{ color: 'var(--color-text-primary)' }}
                        >
                            {selectedIndustry.name}
                        </h2>
                        <StatusPill status={selectedIndustry.stage} />
                    </div>
                    
                    <div>
                        <h3 className="text-sm font-semibold mb-3 uppercase tracking-wider" style={{ color: 'var(--color-text-muted)' }}>
                            Thesis Summary
                        </h3>
                        <ul className="space-y-2">
                            {selectedIndustry.thesis_summary.map((point, idx) => (
                                <li key={idx} className="text-sm flex items-start gap-2" style={{ color: 'var(--color-text-secondary)' }}>
                                    <span style={{ color: 'var(--color-accent-primary)' }}>•</span>
                                    <span>{point}</span>
                                </li>
                            ))}
                        </ul>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {Object.entries(selectedIndustry.aggregate_kpis).map(([key, value]) => (
                            <div key={key} className="p-4 rounded-lg" style={{ backgroundColor: 'var(--color-bg-tertiary)' }}>
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
                        <div className="grid grid-cols-3 gap-4">
                            <div className="p-3 rounded-lg text-center" style={{ backgroundColor: 'var(--color-bg-tertiary)' }}>
                                <div className="text-xs mb-1" style={{ color: 'var(--color-text-muted)' }}>Alpha Picker</div>
                                <div className="text-lg font-bold font-mono" style={{ color: 'var(--color-text-primary)' }}>
                                    {selectedIndustry.caria_signals.alpha_picker_appearances}
                                </div>
                            </div>
                            <div className="p-3 rounded-lg text-center" style={{ backgroundColor: 'var(--color-bg-tertiary)' }}>
                                <div className="text-xs mb-1" style={{ color: 'var(--color-text-muted)' }}>Screener</div>
                                <div className="text-lg font-bold font-mono" style={{ color: 'var(--color-text-primary)' }}>
                                    {selectedIndustry.caria_signals.screener_appearances}
                                </div>
                            </div>
                            <div className="p-3 rounded-lg text-center" style={{ backgroundColor: 'var(--color-bg-tertiary)' }}>
                                <div className="text-xs mb-1" style={{ color: 'var(--color-text-muted)' }}>Crisis Sensitivity</div>
                                <div className="text-lg font-bold" style={{ color: 'var(--color-text-primary)' }}>
                                    {selectedIndustry.caria_signals.crisis_sensitivity}
                                </div>
                            </div>
                        </div>
                    </div>
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
                    <div className="mb-6">
                        <h3 
                            className="text-lg font-semibold mb-2"
                            style={{ color: 'var(--color-text-primary)' }}
                        >
                            This Week's Deep Dive
                        </h3>
                        <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                            Explore sector-level insights, trends, and investment opportunities
                        </p>
                    </div>
                    
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {industries.map(industry => (
                            <IndustryCardComponent
                                key={industry.name}
                                industry={industry}
                                onSelect={handleSelectIndustry}
                            />
                        ))}
                    </div>
                </div>
            )}
        </WidgetCard>
    );
};

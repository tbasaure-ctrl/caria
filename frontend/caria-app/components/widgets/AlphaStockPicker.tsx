import React, { useState } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';

interface FundamentalPick {
    symbol: string;
    name?: string; // Optional as backend might not always fill it
    sector?: string;
    quality_score: number;
    valuation_score: number;
    momentum_score: number;
    catalyst_score: number;
    risk_penalty: number;
    c_score: number;
}

interface ScreenResponse {
    picks: FundamentalPick[];
    timestamp: string;
}

const ScoreBadge: React.FC<{ label: string; value: number; max?: number }> = ({ label, value, max = 100 }) => {
    // Normalized for display if needed, but raw scores are fine if context is clear
    // Assuming scores are roughly 0-100 or similar magnitude after normalization
    // But service returns raw weighted components in some cases? 
    // Wait, service returns "normalized" 0-100 relative scores for Q/V/M/C.
    // Let's display them directly.
    
    const getColor = (v: number) => {
        if (v >= 70) return 'var(--color-positive)';
        if (v >= 40) return 'var(--color-warning)';
        return 'var(--color-text-muted)';
    };
    
    return (
        <div className="px-2 py-1.5 rounded bg-bg-surface/50 border border-white/5 text-center flex-1">
            <div className="text-[9px] font-bold tracking-wider uppercase text-text-muted mb-1">{label}</div>
            <div className="text-sm font-mono font-bold" style={{ color: getColor(value) }}>
                {Math.round(value)}
            </div>
        </div>
    );
};

const AlphaCard: React.FC<{ pick: FundamentalPick; rank: number }> = ({ pick, rank }) => {
    return (
        <div className="rounded-lg p-5 bg-bg-tertiary border border-white/5 hover:border-accent-cyan/30 transition-all duration-300 group">
            <div className="flex justify-between items-start mb-4">
                <div>
                    <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-bold bg-accent-cyan/10 text-accent-cyan px-1.5 py-0.5 rounded">
                            #{rank}
                        </span>
                        <span className="text-xl font-display font-bold text-white tracking-wide group-hover:text-accent-cyan transition-colors">
                            {pick.symbol}
                        </span>
                    </div>
                    <div className="text-xs text-text-secondary truncate max-w-[150px]">
                        {pick.name || pick.symbol}
                    </div>
                    <div className="text-[10px] text-text-muted uppercase tracking-wider mt-0.5">
                        {pick.sector || 'Unknown Sector'}
                    </div>
                </div>
                
                <div className="text-right">
                    <div className="text-2xl font-mono font-bold text-accent-primary">
                        {Math.round(pick.c_score)}
                    </div>
                    <div className="text-[9px] text-text-muted uppercase tracking-widest">C-Score</div>
                </div>
            </div>

            {/* Scores Grid */}
            <div className="grid grid-cols-4 gap-2">
                <ScoreBadge label="QUAL" value={pick.quality_score} />
                <ScoreBadge label="VAL" value={pick.valuation_score} />
                <ScoreBadge label="MOM" value={pick.momentum_score} />
                <ScoreBadge label="CAT" value={pick.catalyst_score} />
            </div>
            
            {/* Risk Warning if penalty exists */}
            {pick.risk_penalty < 0 && (
                <div className="mt-3 text-[10px] text-negative flex items-center gap-1">
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                    Risk Penalty Applied: {Math.round(pick.risk_penalty)}
                </div>
            )}
        </div>
    );
};

export const AlphaStockPicker: React.FC = () => {
    const [data, setData] = useState<ScreenResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const runScreener = async () => {
        setLoading(true);
        setError(null);
        try {
            // Using POST to trigger run, or could use GET if cached results available
            // For demo/user request "run screener", we trigger it
            const response = await fetchWithAuth(`${API_BASE_URL}/api/screener/run-fundamental`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error('Screener run failed');
            }
            
            const result = await response.json();
            setData(result);
        } catch (err: any) {
            console.error('Screener Error:', err);
            setError('Failed to run stock screener. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <WidgetCard 
            title="Alpha Stock Picker" 
            tooltip="Algorithmic screener ranking stocks by Quality, Value, Momentum, and Catalysts (C-Score)."
            action={{ label: 'Run Screen', onClick: runScreener }}
        >
            <div className="space-y-4">
                {loading ? (
                    <div className="py-12 text-center">
                        <div className="w-8 h-8 border-2 border-accent-cyan border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
                        <p className="text-xs text-text-muted">Crunching fundamental data...</p>
                    </div>
                ) : error ? (
                    <div className="p-4 rounded bg-negative-muted/20 border border-negative/30 text-center">
                        <p className="text-xs text-negative">{error}</p>
                        <button onClick={runScreener} className="mt-2 text-xs underline text-text-secondary">Retry</button>
                    </div>
                ) : data ? (
                    <div className="space-y-3">
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-xs text-text-muted">Top Ranked Opportunities</span>
                            <span className="text-[10px] text-text-subtle">Live Data</span>
                        </div>
                        {data.picks.map((pick, idx) => (
                            <AlphaCard key={pick.symbol} pick={pick} rank={idx + 1} />
                        ))}
                    </div>
                ) : (
                    <div className="py-12 text-center border border-dashed border-white/5 rounded-lg">
                        <p className="text-sm text-text-muted mb-2">Discover high-potential stocks</p>
                        <button 
                            onClick={runScreener}
                            className="px-4 py-2 rounded bg-accent-primary/10 text-accent-primary hover:bg-accent-primary/20 text-xs font-bold uppercase tracking-wider transition-colors"
                        >
                            Start Screening
                        </button>
                    </div>
                )}
            </div>
        </WidgetCard>
    );
};

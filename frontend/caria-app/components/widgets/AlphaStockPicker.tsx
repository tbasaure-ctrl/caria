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

// Mock sparkline data generator (since we don't have real history in this payload yet)
const generateSparklineData = () => {
    return Array.from({ length: 20 }, (_, i) => ({
        value: 50 + Math.random() * 20 + i * 0.5 // Slight upward trend
    }));
};

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
                        // Convert WeeklyPick format to AlphaPick format
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
                // Fallback to manual generation
            }
        };
        loadWeeklyPicks();
    }, []);

    // Load universe stats on mount
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
        <WidgetCard title="ALPHA STOCK PICKER" tooltip="Weekly 3-stock selection using Momentum, Quality, Valuation & Catalysts">
            {universeSize !== null && (
                <div className="text-xs text-slate-400 mt-1">
                    Screening {universeSize} stocks (growing with user exploration)
                </div>
            )}
            <div className="min-h-[400px] w-full bg-slate-900/50 rounded border border-slate-800 p-6 flex flex-col items-center justify-center relative overflow-hidden">
                {/* Week info */}
                {weekInfo && (
                    <div className="text-xs text-slate-500 mb-2 self-start">
                        Week of {new Date(weekInfo.week_start).toLocaleDateString()} • Auto-updated weekly
                    </div>
                )}
                {/* Initial state */}
                {!generated && !loading && (
                    <div className="text-center z-10">
                        <div className="mb-6 text-slate-400 max-w-md mx-auto">
                            This week's top 3 stock picks based on our Composite Alpha Score (CAS) model.
                        </div>
                        <button
                            onClick={generatePicks}
                            className="bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 px-8 rounded-full shadow-lg hover:shadow-blue-500/20 transition-all transform hover:scale-105"
                        >
                            Load Weekly Picks
                        </button>
                    </div>
                )}
                {/* Loading */}
                {loading && (
                    <div className="flex flex-col items-center z-10">
                        <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                        <div className="text-blue-400 animate-pulse">Running Alpha Models...</div>
                    </div>
                )}
                {/* Error */}
                {error && (
                    <div className="text-center z-10">
                        <div className="text-red-400 mb-4">{error}</div>
                        <button
                            onClick={generatePicks}
                            className="bg-slate-700 hover:bg-slate-600 text-white py-2 px-6 rounded-full text-sm"
                        >
                            Try Again
                        </button>
                    </div>
                )}
                {/* Results */}
                {generated && !loading && !error && (
                    <div className="w-full grid grid-cols-1 md:grid-cols-3 gap-4 z-10 animate-fade-in">
                        {picks.map(pick => (
                            <div key={pick.ticker} className="bg-slate-800/80 border border-slate-700 rounded-xl p-4 flex flex-col hover:border-blue-500/50 transition-colors">
                                <div className="flex justify-between items-start mb-2">
                                    <div>
                                        <h3 className="text-xl font-bold text-white">{pick.ticker}</h3>
                                        <div className="text-xs text-slate-400">{pick.company_name}</div>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-2xl font-bold text-blue-400">{pick.cas_score}</div>
                                        <div className="text-[10px] text-slate-500 uppercase tracking-wider">CAS Score</div>
                                    </div>
                                </div>
                                <div className="h-16 w-full mb-4 opacity-50">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={generateSparklineData()}>
                                            <Line type="monotone" dataKey="value" stroke="#60a5fa" strokeWidth={2} dot={false} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                                <div className="grid grid-cols-4 gap-1 mb-4 text-center">
                                    <ScoreBadge label="MOM" value={pick.scores.momentum} />
                                    <ScoreBadge label="QUAL" value={pick.scores.quality} />
                                    <ScoreBadge label="VAL" value={pick.scores.valuation} />
                                    <ScoreBadge label="CAT" value={pick.scores.catalyst} />
                                </div>
                                <div className="mt-auto pt-3 border-t border-slate-700/50">
                                    <p className="text-xs text-slate-300 leading-relaxed">{pick.explanation}</p>
                                </div>
                            </div>
                        ))}
                        <div className="col-span-1 md:col-span-3 flex justify-center mt-6">
                            <button onClick={generatePicks} className="text-xs text-slate-500 hover:text-slate-300 underline">
                                Refresh Picks
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </WidgetCard>
    );
};

const ScoreBadge: React.FC<{ label: string; value: number }> = ({ label, value }) => {
    const getColor = (v: number) => {
        if (v >= 70) return 'text-green-400';
        if (v >= 40) return 'text-yellow-400';
        return 'text-red-400';
    };
    return (
        <div className="bg-slate-900/50 rounded p-1">
            <div className="text-[9px] text-slate-500 uppercase">{label}</div>
            <div className={`text-sm font-bold ${getColor(value)}`}>{Math.round(value)}</div>
        </div>
    );
};

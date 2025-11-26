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

    const getClassificationColor = (classification: string | null) => {
        if (!classification) return 'text-slate-400';
        if (classification === 'Hidden Gem') return 'text-emerald-400';
        if (classification === 'Investable') return 'text-blue-400';
        if (classification === 'Watchlist') return 'text-yellow-400';
        return 'text-slate-400';
    };

    const scoreExplanation = (
        <div>
            <p className="mb-2"><strong>How the Hidden Gem Score Works:</strong></p>
            <p className="mb-2 text-sm">The Hidden Gem Score (0-100) combines three key factors:</p>
            <ul className="list-disc list-inside mb-2 text-sm space-y-1">
                <li><strong>Quality Score (30%):</strong> Measures financial health, profitability, and operational efficiency</li>
                <li><strong>Valuation Score (30%):</strong> Assesses whether the stock is trading at an attractive price relative to fundamentals</li>
                <li><strong>Momentum Score (25%):</strong> Evaluates recent price performance and market sentiment</li>
                <li><strong>Bonus/Penalty (15%):</strong> Adjustments for exceptional metrics or red flags</li>
            </ul>
            <p className="mb-2 text-sm">Stocks scoring 80+ are classified as "Hidden Gems" - undervalued mid-caps with strong fundamentals.</p>
            <p className="mt-3 pt-2 border-t border-slate-600 text-xs text-red-300">
                <strong>⚠️ Disclaimer:</strong> This is not financial advice. All investments carry risk. Please conduct your own research and consult with a financial advisor before making investment decisions.
            </p>
        </div>
    );

    return (
        <WidgetCard 
            title="HIDDEN GEMS SCREENER" 
            tooltip={scoreExplanation}
        >
            <div className="min-h-[400px] w-full bg-slate-900/50 rounded border border-slate-800 p-6 flex flex-col items-center justify-center relative overflow-hidden">
                {/* Initial state */}
                {!generated && !loading && (
                    <div className="text-center z-10">
                        <div className="mb-6 text-slate-400 max-w-md mx-auto">
                            Screen for undervalued mid-cap stocks (50M-10B market cap) with strong quality, valuation, and momentum scores.
                        </div>
                        <button
                            onClick={generateGems}
                            className="bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-3 px-8 rounded-full shadow-lg hover:shadow-emerald-500/20 transition-all transform hover:scale-105"
                        >
                            Discover Hidden Gems
                        </button>
                    </div>
                )}
                {/* Loading */}
                {loading && (
                    <div className="flex flex-col items-center z-10">
                        <div className="w-12 h-12 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                        <div className="text-emerald-400 animate-pulse">Screening mid-cap universe...</div>
                    </div>
                )}
                {/* Error */}
                {error && (
                    <div className="text-center z-10">
                        <div className="text-red-400 mb-4">{error}</div>
                        <button
                            onClick={generateGems}
                            className="bg-slate-700 hover:bg-slate-600 text-white py-2 px-6 rounded-full text-sm"
                        >
                            Try Again
                        </button>
                    </div>
                )}
                {/* Results */}
                {generated && !loading && !error && (
                    <div className="w-full space-y-3 z-10 animate-fade-in">
                        {gems.map((gem, idx) => (
                            <div 
                                key={gem.ticker} 
                                className="bg-slate-800/80 border border-slate-700 rounded-xl p-4 hover:border-emerald-500/50 transition-colors"
                            >
                                <div className="flex justify-between items-start mb-3">
                                    <div>
                                        <h3 className="text-xl font-bold text-white">{gem.ticker}</h3>
                                        <div className={`text-sm font-semibold ${getClassificationColor(gem.classification)}`}>
                                            {gem.classification || 'N/A'}
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-2xl font-bold text-emerald-400">{gem.hiddenGemScore.toFixed(0)}</div>
                                        <div className="text-[10px] text-slate-500 uppercase tracking-wider">Hidden Gem Score</div>
                                    </div>
                                </div>
                                
                                <div className="grid grid-cols-3 gap-2 mb-3 text-center">
                                    <div className="bg-slate-900/50 rounded p-2">
                                        <div className="text-[9px] text-slate-500 uppercase">Quality</div>
                                        <div className="text-sm font-bold text-blue-400">{gem.qualityScore.toFixed(0)}</div>
                                    </div>
                                    <div className="bg-slate-900/50 rounded p-2">
                                        <div className="text-[9px] text-slate-500 uppercase">Valuation</div>
                                        <div className="text-sm font-bold text-purple-400">{gem.valuationScore.toFixed(0)}</div>
                                    </div>
                                    <div className="bg-slate-900/50 rounded p-2">
                                        <div className="text-[9px] text-slate-500 uppercase">Momentum</div>
                                        <div className="text-sm font-bold text-orange-400">{gem.momentumScore.toFixed(0)}</div>
                                    </div>
                                </div>

                                {gem.current_price && (
                                    <div className="text-xs text-slate-400 text-right">
                                        ${gem.current_price.toFixed(2)}
                                    </div>
                                )}
                            </div>
                        ))}
                        <div className="flex justify-center mt-4">
                            <button 
                                onClick={generateGems} 
                                className="text-xs text-slate-500 hover:text-slate-300 underline"
                            >
                                Refresh Gems
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </WidgetCard>
    );
};

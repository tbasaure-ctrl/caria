import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';

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
    const { token } = useAuth();
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState<UnderTheRadarResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    const runScreener = async () => {
        setLoading(true);
        setError(null);
        setData(null);

        try {
            const response = await fetch('/api/screener/under-the-radar', {
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
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

    const formatVolume = (num: number): string => {
        if (num >= 1_000_000) {
            return `${(num / 1_000_000).toFixed(2)}M`;
        } else if (num >= 1_000) {
            return `${(num / 1_000).toFixed(2)}K`;
        }
        return num.toFixed(0);
    };

    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
            <div className="mb-6">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                    Under-the-Radar Screener
                </h2>
                <p className="text-gray-600 dark:text-gray-400 text-sm">
                    Event + social spark + quality filter for true emerging outliers.
                    Detects small-cap stocks (50M-800M market cap) with social momentum,
                    recent catalysts, and improving quality metrics.
                </p>
            </div>

            <button
                onClick={runScreener}
                disabled={loading}
                className="w-full md:w-auto px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-semibold rounded-lg transition-colors duration-200 disabled:cursor-not-allowed"
            >
                {loading ? 'Running Screener...' : 'Run Screener'}
            </button>

            {error && (
                <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                    <p className="text-red-800 dark:text-red-200">{error}</p>
                </div>
            )}

            {data && (
                <div className="mt-6">
                    {data.candidates.length === 0 ? (
                        <div className="p-6 bg-gray-50 dark:bg-gray-700 rounded-lg">
                            <p className="text-gray-700 dark:text-gray-300 text-center">
                                {data.message || 'No stocks passed all filters this week.'}
                            </p>
                        </div>
                    ) : (
                        <div className="space-y-6">
                            {data.message && (
                                <p className="text-sm text-gray-600 dark:text-gray-400 text-center">
                                    {data.message}
                                </p>
                            )}
                            {data.candidates.map((candidate, idx) => (
                                <div
                                    key={candidate.ticker}
                                    className="border border-gray-200 dark:border-gray-700 rounded-lg p-6 hover:shadow-lg transition-shadow"
                                >
                                    <div className="flex items-start justify-between mb-4">
                                        <div>
                                            <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                                                {candidate.ticker}
                                            </h3>
                                            <p className="text-sm text-gray-600 dark:text-gray-400">
                                                {candidate.name} • {candidate.sector}
                                            </p>
                                        </div>
                                        <div className="text-right">
                                            <p className="text-sm text-gray-500 dark:text-gray-400">Market Cap</p>
                                            <p className="text-lg font-semibold text-gray-900 dark:text-white">
                                                {formatNumber(candidate.liquidity.market_cap)}
                                            </p>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                                        <div>
                                            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">FCF Yield</p>
                                            <p className="text-sm font-semibold text-gray-900 dark:text-white">
                                                {candidate.quality_metrics.fcf_yield.toFixed(1)}%
                                            </p>
                                        </div>
                                        <div>
                                            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Δ ROCE</p>
                                            <p className="text-sm font-semibold text-green-600 dark:text-green-400">
                                                +{candidate.quality_metrics.delta_roce.toFixed(1)}pp
                                            </p>
                                        </div>
                                        <div>
                                            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Volume Spike</p>
                                            <p className="text-sm font-semibold text-orange-600 dark:text-orange-400">
                                                {candidate.liquidity.volume_spike.toFixed(1)}x
                                            </p>
                                        </div>
                                        <div>
                                            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Avg Volume</p>
                                            <p className="text-sm font-semibold text-gray-900 dark:text-white">
                                                {formatVolume(candidate.liquidity.avg_volume)}
                                            </p>
                                        </div>
                                    </div>

                                    <div className="mb-4">
                                        <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">Social Momentum</p>
                                        <div className="flex flex-wrap gap-2">
                                            {candidate.social_spike.sources.map((source) => (
                                                <span
                                                    key={source}
                                                    className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 text-xs rounded"
                                                >
                                                    {source}
                                                </span>
                                            ))}
                                        </div>
                                    </div>

                                    <div className="mb-4">
                                        <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">Catalysts</p>
                                        <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
                                            {candidate.catalysts.flags.slice(0, 3).map((flag, i) => (
                                                <li key={i}>{flag}</li>
                                            ))}
                                        </ul>
                                    </div>

                                    <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                                        <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Why this stock surfaced?</p>
                                        <p className="text-sm text-gray-700 dark:text-gray-300">
                                            {candidate.explanation}
                                        </p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

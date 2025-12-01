import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';
import { WidgetCard } from './WidgetCard';

interface Scenario {
    price: number;
    return_pct: number;
}

interface RiskRewardData {
    ticker: string;
    horizon_months: number;
    current_price: number;
    scenarios: {
        bear: Scenario;
        base: Scenario;
        bull: Scenario;
    };
    metrics: {
        upside: number;
        downside: number;
        rrr: number;
        expected_value: number;
    };
    explanations: {
        summary: string;
        ev_breakdown: string;
        analogy: string;
        position_sizing: string;
    };
    volatility_metrics: {
        volatility: number;
        max_drawdown: number;
    };
}

export const RiskRewardWidget: React.FC = () => {
    const [ticker, setTicker] = useState('');
    const [horizonMonths, setHorizonMonths] = useState(24);
    const [probabilities, setProbabilities] = useState({
        bear: 0.20,
        base: 0.50,
        bull: 0.30,
    });
    const [data, setData] = useState<RiskRewardData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [showEducation, setShowEducation] = useState(false);

    useEffect(() => {
        if (ticker && ticker.length >= 1) {
            const timeoutId = setTimeout(() => {
                analyzeRiskReward();
            }, 800); // Debounce
            return () => clearTimeout(timeoutId);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [ticker, horizonMonths, probabilities]);

    const analyzeRiskReward = async () => {
        if (!ticker || ticker.length < 1) return;

        setLoading(true);
        setError(null);

        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/risk-reward/analyze`, {
                method: 'POST',
                body: JSON.stringify({
                    ticker: ticker.toUpperCase(),
                    horizon_months: horizonMonths,
                    probabilities: probabilities,
                }),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }

            const result = await response.json();
            setData(result);
        } catch (err: any) {
            console.error('Risk-reward analysis error:', err);
            setError(err?.message || 'Failed to analyze risk-reward');
            setData(null);
        } finally {
            setLoading(false);
        }
    };

    const handleProbabilityChange = (scenario: 'bear' | 'base' | 'bull', value: number) => {
        const newProbs = { ...probabilities };
        newProbs[scenario] = Math.max(0, Math.min(1, value / 100));

        // Normalize to sum to 100%
        const total = newProbs.bear + newProbs.base + newProbs.bull;
        if (total > 0) {
            newProbs.bear = newProbs.bear / total;
            newProbs.base = newProbs.base / total;
            newProbs.bull = newProbs.bull / total;
        }

        setProbabilities(newProbs);
    };

    // Prepare chart data
    const chartData = data ? [
        {
            name: 'Bear',
            return_pct: data.scenarios.bear.return_pct * 100,
            price: data.scenarios.bear.price,
            probability: probabilities.bear * 100,
        },
        {
            name: 'Base',
            return_pct: data.scenarios.base.return_pct * 100,
            price: data.scenarios.base.price,
            probability: probabilities.base * 100,
        },
        {
            name: 'Bull',
            return_pct: data.scenarios.bull.return_pct * 100,
            price: data.scenarios.bull.price,
            probability: probabilities.bull * 100,
        },
    ] : [];

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const d = payload[0].payload;
            return (
                <div className="bg-[#0B1221] border border-white/10 rounded-lg p-3 shadow-xl">
                    <p className="font-bold text-white mb-1">{d.name} Case</p>
                    <p className="text-sm text-text-secondary">
                        Return: <span className={d.return_pct >= 0 ? 'text-green-400' : 'text-red-400'}>
                            {d.return_pct > 0 ? '+' : ''}{d.return_pct.toFixed(1)}%
                        </span>
                    </p>
                    <p className="text-sm text-text-secondary">Price: ${d.price.toFixed(2)}</p>
                    <p className="text-sm text-text-secondary">Probability: {d.probability.toFixed(0)}%</p>
                </div>
            );
        }
        return null;
    };

    const tooltipContent = (
        <div className="space-y-2">
            <p><strong>Risk-Reward Ratio:</strong> Measures potential upside vs downside. A 2:1 ratio means you could gain $2 for every $1 at risk.</p>
            <p><strong>Expected Value:</strong> The probability-weighted average return across all scenarios.</p>
            <p>Adjust probabilities to see how different assumptions affect the outcome.</p>
        </div>
    );

    return (
        <WidgetCard title="RISK-REWARD ENGINE" tooltip={tooltipContent}>
            <div className="flex flex-col h-full overflow-y-auto space-y-4">
                {/* Input Section */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {/* Ticker Input */}
                    <div>
                        <label className="block text-xs font-medium text-text-muted mb-2 uppercase tracking-wider">
                            Ticker Symbol
                        </label>
                        <input
                            type="text"
                            value={ticker}
                            onChange={(e) => setTicker(e.target.value.toUpperCase())}
                            placeholder="e.g. AAPL"
                            className="w-full px-4 py-2.5 rounded-lg text-sm bg-[#0B1221] border border-white/10 text-white placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-accent-cyan/50 focus:border-accent-cyan/50 transition-all"
                        />
                    </div>

                    {/* Time Horizon */}
                    <div>
                        <label className="block text-xs font-medium text-text-muted mb-2 uppercase tracking-wider">
                            Time Horizon
                        </label>
                        <div className="flex gap-2">
                            {[12, 24, 36].map((months) => (
                                <button
                                    key={months}
                                    onClick={() => setHorizonMonths(months)}
                                    className={`flex-1 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                                        horizonMonths === months
                                            ? 'bg-accent-cyan text-white shadow-[0_0_12px_rgba(34,211,238,0.3)]'
                                            : 'bg-[#0B1221] border border-white/10 text-text-secondary hover:text-white hover:border-white/20'
                                    }`}
                                >
                                    {months}M
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Loading State */}
                {loading && (
                    <div className="text-center py-8">
                        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-accent-cyan"></div>
                        <p className="text-sm text-text-muted mt-2">Analyzing scenarios...</p>
                    </div>
                )}

                {/* Error State */}
                {error && (
                    <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/30">
                        <p className="text-sm text-red-400">{error}</p>
                    </div>
                )}

                {/* Empty State */}
                {!data && !loading && !error && (
                    <div className="text-center py-8 sm:py-12">
                        <div className="w-16 h-16 mx-auto mb-4 rounded-xl bg-accent-cyan/10 flex items-center justify-center">
                            <svg className="w-8 h-8 text-accent-cyan" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                        </div>
                        <h3 className="text-base font-semibold text-white mb-2">Quantify Risk vs Reward</h3>
                        <p className="text-sm text-text-secondary max-w-md mx-auto mb-4">
                            Enter a ticker to see Bear/Base/Bull scenarios, calculate Expected Value, and learn to think like a rational investor.
                        </p>
                        <div className="flex flex-wrap justify-center gap-2">
                            {['AAPL', 'NVDA', 'MSFT', 'TSLA'].map((t) => (
                                <button
                                    key={t}
                                    onClick={() => setTicker(t)}
                                    className="px-4 py-1.5 rounded-full text-xs font-medium bg-white/5 border border-white/10 text-text-secondary hover:text-white hover:border-accent-cyan/50 transition-all"
                                >
                                    {t}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {/* Results */}
                {data && !loading && (
                    <>
                        {/* Current Price & Ticker */}
                        <div className="flex items-center justify-between bg-[#0B1221] rounded-lg p-4 border border-white/5">
                            <div>
                                <span className="text-xs text-text-muted uppercase tracking-wider">Analyzing</span>
                                <h3 className="text-xl font-bold text-white">{data.ticker}</h3>
                            </div>
                            <div className="text-right">
                                <span className="text-xs text-text-muted uppercase tracking-wider">Current Price</span>
                                <p className="text-xl font-bold text-accent-cyan">${data.current_price.toFixed(2)}</p>
                            </div>
                        </div>

                        {/* Scenario Chart */}
                        <div>
                            <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-accent-cyan"></span>
                                Scenario Outcomes ({horizonMonths} Months)
                            </h3>
                            <div className="w-full bg-[#0B1221] rounded-lg p-4 border border-white/5" style={{ minHeight: '200px', height: '200px' }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                        <XAxis
                                            dataKey="name"
                                            stroke="rgba(255,255,255,0.3)"
                                            tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                                        />
                                        <YAxis
                                            stroke="rgba(255,255,255,0.3)"
                                            tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                                            tickFormatter={(value) => `${value > 0 ? '+' : ''}${value.toFixed(0)}%`}
                                        />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Bar dataKey="return_pct" radius={[6, 6, 0, 0]}>
                                            {chartData.map((entry, index) => (
                                                <Cell
                                                    key={`cell-${index}`}
                                                    fill={
                                                        entry.name === 'Bear' ? '#ef4444' :
                                                        entry.name === 'Base' ? '#fbbf24' :
                                                        '#22c55e'
                                                    }
                                                />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Probability Sliders */}
                        <div className="bg-[#0B1221] rounded-lg p-4 border border-white/5">
                            <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-accent-cyan"></span>
                                Adjust Probabilities
                            </h3>
                            <div className="space-y-3">
                                {(['bear', 'base', 'bull'] as const).map((scenario) => (
                                    <div key={scenario} className="flex items-center gap-4">
                                        <span className={`text-xs font-medium w-12 ${
                                            scenario === 'bear' ? 'text-red-400' :
                                            scenario === 'base' ? 'text-yellow-400' :
                                            'text-green-400'
                                        }`}>
                                            {scenario.charAt(0).toUpperCase() + scenario.slice(1)}
                                        </span>
                                        <input
                                            type="range"
                                            min="5"
                                            max="80"
                                            value={probabilities[scenario] * 100}
                                            onChange={(e) => handleProbabilityChange(scenario, parseFloat(e.target.value))}
                                            className="flex-1 h-1.5 rounded-lg appearance-none cursor-pointer bg-white/10"
                                            style={{ accentColor: scenario === 'bear' ? '#ef4444' : scenario === 'base' ? '#fbbf24' : '#22c55e' }}
                                        />
                                        <span className="text-sm font-mono text-white w-12 text-right">
                                            {(probabilities[scenario] * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Key Metrics */}
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                            <div className="bg-[#0B1221] rounded-lg p-4 border border-white/5">
                                <p className="text-xs text-text-muted mb-1 uppercase tracking-wider">Upside</p>
                                <p className="text-xl font-bold text-green-400">
                                    +{(data.metrics.upside * 100).toFixed(0)}%
                                </p>
                            </div>
                            <div className="bg-[#0B1221] rounded-lg p-4 border border-white/5">
                                <p className="text-xs text-text-muted mb-1 uppercase tracking-wider">Downside</p>
                                <p className="text-xl font-bold text-red-400">
                                    -{(data.metrics.downside * 100).toFixed(0)}%
                                </p>
                            </div>
                            <div className="bg-[#0B1221] rounded-lg p-4 border border-white/5">
                                <p className="text-xs text-text-muted mb-1 uppercase tracking-wider">Risk-Reward</p>
                                <p className="text-xl font-bold text-white">
                                    {data.metrics.rrr.toFixed(1)}:1
                                </p>
                            </div>
                            <div className="bg-[#0B1221] rounded-lg p-4 border border-white/5">
                                <p className="text-xs text-text-muted mb-1 uppercase tracking-wider">Expected Value</p>
                                <p className={`text-xl font-bold ${data.metrics.expected_value >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                    {data.metrics.expected_value >= 0 ? '+' : ''}{(data.metrics.expected_value * 100).toFixed(0)}%
                                </p>
                            </div>
                        </div>

                        {/* Educational Toggle */}
                        <button
                            onClick={() => setShowEducation(!showEducation)}
                            className="w-full py-3 rounded-lg bg-accent-cyan/10 border border-accent-cyan/30 text-accent-cyan text-sm font-medium hover:bg-accent-cyan/20 transition-all flex items-center justify-center gap-2"
                        >
                            {showEducation ? 'Hide' : 'Show'} Educational Breakdown
                            <svg
                                className={`w-4 h-4 transition-transform ${showEducation ? 'rotate-180' : ''}`}
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                            </svg>
                        </button>

                        {/* Educational Content */}
                        {showEducation && (
                            <div className="space-y-4 bg-[#0B1221] rounded-lg p-4 border border-accent-cyan/20 animate-fade-in">
                                <div>
                                    <h4 className="text-sm font-semibold text-accent-cyan mb-2">📊 Summary</h4>
                                    <p className="text-sm text-text-secondary leading-relaxed">
                                        {data.explanations.summary}
                                    </p>
                                </div>

                                <div>
                                    <h4 className="text-sm font-semibold text-accent-cyan mb-2">🧮 Expected Value Math</h4>
                                    <p className="text-sm text-text-secondary font-mono bg-black/30 p-3 rounded">
                                        {data.explanations.ev_breakdown}
                                    </p>
                                </div>

                                <div>
                                    <h4 className="text-sm font-semibold text-accent-cyan mb-2">🎰 Poker Analogy</h4>
                                    <p className="text-sm text-text-secondary leading-relaxed">
                                        {data.explanations.analogy}
                                    </p>
                                </div>

                                <div>
                                    <h4 className="text-sm font-semibold text-accent-cyan mb-2">📐 Position Sizing Guidance</h4>
                                    <p className="text-sm text-text-secondary leading-relaxed">
                                        {data.explanations.position_sizing}
                                    </p>
                                </div>

                                <div className="pt-2 border-t border-white/10">
                                    <p className="text-xs text-text-muted italic">
                                        ⚠️ This is educational content, not financial advice. Always do your own research and consider consulting a financial advisor.
                                    </p>
                                </div>
                            </div>
                        )}
                    </>
                )}
            </div>
        </WidgetCard>
    );
};

export default RiskRewardWidget;


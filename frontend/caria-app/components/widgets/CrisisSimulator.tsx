import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';

const CRISES = [
    { id: '1929_depression', name: 'Great Depression (1929)' },
    { id: '1939_wwii', name: 'WWII Start (1939)' },
    { id: '1962_cuban_missile', name: 'Cuban Missile Crisis (1962)' },
    { id: '1963_jfk', name: 'Kennedy Assassination (1963)' },
    { id: '1987_black_monday', name: 'Black Monday (1987)' },
    { id: '2000_dot_com', name: 'Dot Com Bubble (2000)' },
    { id: '2001_911', name: '9/11 Attacks (2001)' },
    { id: '2008_gfc', name: 'Global Financial Crisis (2008)' },
    { id: '2011_euro_debt', name: 'European Debt Crisis (2011)' },
    { id: '2018_trade_war', name: '2018 Trade War / Fed Tightening' },
    { id: '2020_covid', name: 'COVID-19 Crash (2020)' },
    { id: '2022_inflation', name: '2022 Inflation Bear Market' },
];

interface SimulationResult {
    crisis_name: string;
    dates: string[];
    portfolio_values: number[];
    benchmark_values: number[];
    metrics: {
        max_drawdown: number;
        total_return: number;
        benchmark_return: number;
    };
}

type Timeframe = '1d' | '1m' | '1y';

export const CrisisSimulator: React.FC = () => {
    const [selectedCrisis, setSelectedCrisis] = useState(CRISES[7].id); // Default 2008
    const [timeframe, setTimeframe] = useState<Timeframe>('1m');
    const [result, setResult] = useState<SimulationResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    
    const timeframeLabels = {
        '1d': '1 Day',
        '1m': '1 Month', 
        '1y': '1 Year'
    };
    
    const timeframeMessages = {
        '1d': 'Even the worst days eventually end.',
        '1m': 'Markets often recover faster than fear suggests.',
        '1y': 'History shows: this too shall pass.'
    };

    const filterDataByTimeframe = (data: SimulationResult, tf: Timeframe): SimulationResult => {
        if (!data.dates || data.dates.length === 0) return data;
        
        const dates = data.dates.map(d => new Date(d));
        const portfolioValues = data.portfolio_values;
        const benchmarkValues = data.benchmark_values;
        
        let startIndex = 0;
        let endIndex = dates.length;
        
        if (tf === '1d') {
            // Show first day only
            endIndex = Math.min(1, dates.length);
        } else if (tf === '1m') {
            // Show first month (approximately 20-22 trading days)
            endIndex = Math.min(22, dates.length);
        } else if (tf === '1y') {
            // Show all data (1 year or full crisis period)
            endIndex = dates.length;
        }
        
        const filteredDates = dates.slice(startIndex, endIndex).map(d => d.toISOString().split('T')[0]);
        const filteredPortfolio = portfolioValues.slice(startIndex, endIndex);
        const filteredBenchmark = benchmarkValues.slice(startIndex, endIndex);
        
        // Recalculate metrics for filtered data
        const initialPortfolio = filteredPortfolio[0] || 100;
        const finalPortfolio = filteredPortfolio[filteredPortfolio.length - 1] || 100;
        const portfolioReturn = (finalPortfolio / initialPortfolio) - 1;
        
        const initialBenchmark = filteredBenchmark[0] || 100;
        const finalBenchmark = filteredBenchmark[filteredBenchmark.length - 1] || 100;
        const benchmarkReturn = (finalBenchmark / initialBenchmark) - 1;
        
        // Calculate max drawdown for filtered period
        let maxDrawdown = 0;
        let peak = initialPortfolio;
        for (const value of filteredPortfolio) {
            if (value > peak) peak = value;
            const drawdown = (value - peak) / peak;
            if (drawdown < maxDrawdown) maxDrawdown = drawdown;
        }
        
        return {
            ...data,
            dates: filteredDates,
            portfolio_values: filteredPortfolio,
            benchmark_values: filteredBenchmark,
            metrics: {
                max_drawdown: maxDrawdown,
                total_return: portfolioReturn,
                benchmark_return: benchmarkReturn,
            }
        };
    };

    const handleSimulate = async () => {
        setLoading(true);
        setError(null);
        try {
            // Mock portfolio for now - in real app, fetch from state/context
            const portfolio = [
                { ticker: 'AAPL', quantity: 10, weight: 0.5 },
                { ticker: 'MSFT', quantity: 5, weight: 0.5 },
            ];

            const response = await fetchWithAuth(`${API_BASE_URL}/api/simulation/crisis`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    portfolio,
                    crisis_id: selectedCrisis,
                }),
            });

            if (!response.ok) {
                throw new Error('Simulation failed');
            }

            const data = await response.json();
            
            // Filter data based on selected timeframe
            const filteredData = filterDataByTimeframe(data, timeframe);
            setResult(filteredData);
        } catch (err: any) {
            setError(err.message || 'Failed to run simulation');
        } finally {
            setLoading(false);
        }
    };

    return (
        <WidgetCard title="HISTORICAL CRISIS SIMULATOR" tooltip="Stress test your portfolio against major historical market crashes. See how markets recover over different timeframes.">
            <div className="space-y-4">
                <div className="flex gap-2">
                    <select
                        value={selectedCrisis}
                        onChange={(e) => setSelectedCrisis(e.target.value)}
                        className="flex-1 bg-gray-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200"
                    >
                        {CRISES.map((c) => (
                            <option key={c.id} value={c.id}>
                                {c.name}
                            </option>
                        ))}
                    </select>
                    <button
                        onClick={handleSimulate}
                        disabled={loading}
                        className="bg-red-900/50 hover:bg-red-900/70 text-red-100 px-4 py-2 rounded text-sm font-semibold border border-red-800/50 transition-colors"
                    >
                        {loading ? 'Simulating...' : 'Run Simulation'}
                    </button>
                </div>
                
                {/* Timeframe Selector */}
                <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-500">View recovery:</span>
                    <div className="flex gap-1 bg-gray-900/50 rounded p-1">
                        {(['1d', '1m', '1y'] as Timeframe[]).map((tf) => (
                            <button
                                key={tf}
                                onClick={() => {
                                    setTimeframe(tf);
                                    // Re-filter existing result if available
                                    if (result) {
                                        const filtered = filterDataByTimeframe(result, tf);
                                        setResult(filtered);
                                    }
                                }}
                                className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                                    timeframe === tf 
                                        ? 'bg-blue-600 text-white' 
                                        : 'text-slate-400 hover:text-white hover:bg-slate-700'
                                }`}
                            >
                                {timeframeLabels[tf]}
                            </button>
                        ))}
                    </div>
                </div>

                {error && <div className="text-red-400 text-xs">{error}</div>}

                {result && (
                    <div className="space-y-4 animate-fade-in">
                        <div className="grid grid-cols-3 gap-2">
                            <div className="bg-gray-900/50 p-2 rounded border border-slate-800">
                                <div className="text-xs text-slate-500">Max Drawdown</div>
                                <div className="text-lg font-mono text-red-400">
                                    {(result.metrics.max_drawdown * 100).toFixed(1)}%
                                </div>
                            </div>
                            <div className="bg-gray-900/50 p-2 rounded border border-slate-800">
                                <div className="text-xs text-slate-500">Portfolio Return</div>
                                <div className={`text-lg font-mono ${result.metrics.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                    {(result.metrics.total_return * 100).toFixed(1)}%
                                </div>
                            </div>
                            <div className="bg-gray-900/50 p-2 rounded border border-slate-800">
                                <div className="text-xs text-slate-500">S&P 500 Return</div>
                                <div className={`text-lg font-mono ${result.metrics.benchmark_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                    {(result.metrics.benchmark_return * 100).toFixed(1)}%
                                </div>
                            </div>
                        </div>

                        <div className="h-64 w-full bg-gray-900/30 rounded border border-slate-800/50 p-1">
                            <Plot
                                data={[
                                    {
                                        x: result.dates,
                                        y: result.portfolio_values,
                                        type: 'scatter',
                                        mode: 'lines',
                                        name: 'Portfolio',
                                        line: { color: '#3b82f6', width: 2 },
                                    },
                                    {
                                        x: result.dates,
                                        y: result.benchmark_values,
                                        type: 'scatter',
                                        mode: 'lines',
                                        name: 'S&P 500',
                                        line: { color: '#64748b', width: 2, dash: 'dot' },
                                    },
                                ]}
                                layout={{
                                    autosize: true,
                                    margin: { l: 40, r: 20, t: 20, b: 40 },
                                    paper_bgcolor: 'rgba(0,0,0,0)',
                                    plot_bgcolor: 'rgba(0,0,0,0)',
                                    xaxis: {
                                        gridcolor: '#1e293b',
                                        tickfont: { color: '#94a3b8', size: 10 },
                                    },
                                    yaxis: {
                                        gridcolor: '#1e293b',
                                        tickfont: { color: '#94a3b8', size: 10 },
                                        title: 'Value (Rebased to 100)',
                                        titlefont: { color: '#64748b', size: 10 },
                                    },
                                    legend: {
                                        orientation: 'h',
                                        y: 1.1,
                                        font: { color: '#cbd5e1' },
                                    },
                                }}
                                useResizeHandler
                                style={{ width: '100%', height: '100%' }}
                                config={{ displayModeBar: false }}
                            />
                        </div>
                        
                        {/* Inspirational message */}
                        <div className="text-center py-2 px-4 bg-emerald-900/20 rounded border border-emerald-800/30">
                            <p className="text-emerald-300 text-sm italic">
                                "{timeframeMessages[timeframe]}"
                            </p>
                        </div>
                    </div>
                )}
            </div>
        </WidgetCard>
    );
};

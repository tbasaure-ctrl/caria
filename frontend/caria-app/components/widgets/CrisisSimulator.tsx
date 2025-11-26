import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';

const CRISES = [
    { id: '1929_depression', name: 'Great Depression', year: '1929' },
    { id: '1939_wwii', name: 'WWII Start', year: '1939' },
    { id: '1962_cuban_missile', name: 'Cuban Missile Crisis', year: '1962' },
    { id: '1963_jfk', name: 'Kennedy Assassination', year: '1963' },
    { id: '1987_black_monday', name: 'Black Monday', year: '1987' },
    { id: '2000_dot_com', name: 'Dot Com Bubble', year: '2000' },
    { id: '2001_911', name: '9/11 Attacks', year: '2001' },
    { id: '2008_gfc', name: 'Global Financial Crisis', year: '2008' },
    { id: '2011_euro_debt', name: 'European Debt Crisis', year: '2011' },
    { id: '2018_trade_war', name: 'Trade War', year: '2018' },
    { id: '2020_covid', name: 'COVID-19 Crash', year: '2020' },
    { id: '2022_inflation', name: 'Inflation Bear Market', year: '2022' },
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
    const [selectedCrisis, setSelectedCrisis] = useState(CRISES[7].id);
    const [timeframe, setTimeframe] = useState<Timeframe>('1m');
    const [result, setResult] = useState<SimulationResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    
    const timeframeLabels = { '1d': '1 Day', '1m': '1 Month', '1y': '1 Year' };
    const timeframeMessages = {
        '1d': 'Even the worst days eventually end.',
        '1m': 'Markets often recover faster than fear suggests.',
        '1y': 'History shows: this too shall pass.'
    };

    const filterDataByTimeframe = (data: SimulationResult, tf: Timeframe): SimulationResult => {
        if (!data.dates || data.dates.length === 0) return data;
        
        const dates = data.dates.map(d => new Date(d));
        let endIndex = dates.length;
        
        if (tf === '1d') endIndex = Math.min(1, dates.length);
        else if (tf === '1m') endIndex = Math.min(22, dates.length);
        
        const filteredDates = dates.slice(0, endIndex).map(d => d.toISOString().split('T')[0]);
        const filteredPortfolio = data.portfolio_values.slice(0, endIndex);
        const filteredBenchmark = data.benchmark_values.slice(0, endIndex);
        
        const initialPortfolio = filteredPortfolio[0] || 100;
        const finalPortfolio = filteredPortfolio[filteredPortfolio.length - 1] || 100;
        const portfolioReturn = (finalPortfolio / initialPortfolio) - 1;
        
        const initialBenchmark = filteredBenchmark[0] || 100;
        const finalBenchmark = filteredBenchmark[filteredBenchmark.length - 1] || 100;
        const benchmarkReturn = (finalBenchmark / initialBenchmark) - 1;
        
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
            metrics: { max_drawdown: maxDrawdown, total_return: portfolioReturn, benchmark_return: benchmarkReturn }
        };
    };

    const handleSimulate = async () => {
        setLoading(true);
        setError(null);
        try {
            const portfolio = [
                { ticker: 'AAPL', quantity: 10, weight: 0.5 },
                { ticker: 'MSFT', quantity: 5, weight: 0.5 },
            ];

            const response = await fetchWithAuth(`${API_BASE_URL}/api/simulation/crisis`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ portfolio, crisis_id: selectedCrisis }),
            });

            if (!response.ok) throw new Error('Simulation failed');

            const data = await response.json();
            setResult(filterDataByTimeframe(data, timeframe));
        } catch (err: any) {
            setError(err.message || 'Failed to run simulation');
        } finally {
            setLoading(false);
        }
    };

    const selectedCrisisData = CRISES.find(c => c.id === selectedCrisis);

    return (
        <WidgetCard 
            title="CRISIS SIMULATOR" 
            tooltip="Stress test your portfolio against major historical market crashes. Visualize how your holdings would perform across different recovery timeframes."
        >
            <div className="grid lg:grid-cols-2 gap-6">
                {/* Left Column - Controls */}
                <div className="space-y-4">
                    {/* Crisis Selector */}
                    <div>
                        <label 
                            className="block text-xs font-medium tracking-wider uppercase mb-2"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            Historical Crisis
                        </label>
                        <select
                            value={selectedCrisis}
                            onChange={(e) => setSelectedCrisis(e.target.value)}
                            className="w-full px-4 py-3 rounded-lg text-sm"
                            style={{
                                backgroundColor: 'var(--color-bg-tertiary)',
                                border: '1px solid var(--color-border-subtle)',
                                color: 'var(--color-text-primary)',
                            }}
                        >
                            {CRISES.map((c) => (
                                <option key={c.id} value={c.id}>
                                    {c.name} ({c.year})
                                </option>
                            ))}
                        </select>
                    </div>

                    {/* Timeframe Selector */}
                    <div>
                        <label 
                            className="block text-xs font-medium tracking-wider uppercase mb-2"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            Recovery View
                        </label>
                        <div 
                            className="flex gap-2"
                        >
                            {(['1d', '1m', '1y'] as Timeframe[]).map((tf) => (
                                <button
                                    key={tf}
                                    onClick={() => {
                                        setTimeframe(tf);
                                        if (result) setResult(filterDataByTimeframe(result, tf));
                                    }}
                                    className="flex-1 px-4 py-2.5 rounded-lg text-sm font-medium transition-all"
                                    style={{
                                        backgroundColor: timeframe === tf ? 'var(--color-accent-primary)' : 'var(--color-bg-tertiary)',
                                        color: timeframe === tf ? '#FFFFFF' : 'var(--color-text-secondary)',
                                        border: timeframe === tf ? '1px solid var(--color-accent-primary)' : '1px solid var(--color-border-subtle)',
                                    }}
                                >
                                    {timeframeLabels[tf]}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Run Button */}
                    <button
                        onClick={handleSimulate}
                        disabled={loading}
                        className="w-full px-6 py-3 rounded-lg font-semibold text-sm transition-all duration-200 disabled:opacity-50"
                        style={{
                            backgroundColor: 'var(--color-negative)',
                            color: '#FFFFFF',
                        }}
                        onMouseEnter={(e) => {
                            if (!loading) {
                                e.currentTarget.style.backgroundColor = '#E53935';
                                e.currentTarget.style.transform = 'translateY(-1px)';
                            }
                        }}
                        onMouseLeave={(e) => {
                            if (!loading) {
                                e.currentTarget.style.backgroundColor = 'var(--color-negative)';
                                e.currentTarget.style.transform = 'translateY(0)';
                            }
                        }}
                    >
                        {loading ? 'Simulating...' : 'Run Stress Test'}
                    </button>

                    {/* Crisis Info */}
                    {selectedCrisisData && (
                        <div 
                            className="p-4 rounded-lg"
                            style={{
                                backgroundColor: 'var(--color-bg-tertiary)',
                                border: '1px solid var(--color-border-subtle)',
                            }}
                        >
                            <div 
                                className="text-xs font-medium tracking-wider uppercase mb-2"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Selected Crisis
                            </div>
                            <div 
                                className="text-lg font-semibold"
                                style={{ color: 'var(--color-text-primary)' }}
                            >
                                {selectedCrisisData.name}
                            </div>
                            <div 
                                className="text-sm mt-1"
                                style={{ color: 'var(--color-text-secondary)' }}
                            >
                                Year: {selectedCrisisData.year}
                            </div>
                        </div>
                    )}
                </div>

                {/* Right Column - Results */}
                <div className="space-y-4">

                    {error && (
                        <div 
                            className="px-4 py-3 rounded-lg text-sm"
                            style={{
                                backgroundColor: 'var(--color-negative-muted)',
                                color: 'var(--color-negative)',
                                border: '1px solid var(--color-negative)',
                            }}
                        >
                            {error}
                        </div>
                    )}

                    {/* Results */}
                    {result && (
                        <div className="space-y-4 animate-fade-in">
                            {/* Metrics Grid */}
                            <div className="grid grid-cols-3 gap-3">
                            <div 
                                className="p-4 rounded-lg text-center"
                                style={{
                                    backgroundColor: 'var(--color-bg-tertiary)',
                                    border: '1px solid var(--color-border-subtle)',
                                }}
                            >
                                <div 
                                    className="text-[10px] font-medium tracking-wider uppercase mb-1"
                                    style={{ color: 'var(--color-text-muted)' }}
                                >
                                    Max Drawdown
                                </div>
                                <div 
                                    className="text-2xl font-bold font-mono"
                                    style={{ color: 'var(--color-negative)' }}
                                >
                                    {(result.metrics.max_drawdown * 100).toFixed(1)}%
                                </div>
                            </div>
                            <div 
                                className="p-4 rounded-lg text-center"
                                style={{
                                    backgroundColor: 'var(--color-bg-tertiary)',
                                    border: '1px solid var(--color-border-subtle)',
                                }}
                            >
                                <div 
                                    className="text-[10px] font-medium tracking-wider uppercase mb-1"
                                    style={{ color: 'var(--color-text-muted)' }}
                                >
                                    Portfolio Return
                                </div>
                                <div 
                                    className="text-2xl font-bold font-mono"
                                    style={{ 
                                        color: result.metrics.total_return >= 0 
                                            ? 'var(--color-positive)' 
                                            : 'var(--color-negative)' 
                                    }}
                                >
                                    {(result.metrics.total_return * 100).toFixed(1)}%
                                </div>
                            </div>
                            <div 
                                className="p-4 rounded-lg text-center"
                                style={{
                                    backgroundColor: 'var(--color-bg-tertiary)',
                                    border: '1px solid var(--color-border-subtle)',
                                }}
                            >
                                <div 
                                    className="text-[10px] font-medium tracking-wider uppercase mb-1"
                                    style={{ color: 'var(--color-text-muted)' }}
                                >
                                    S&P 500 Return
                                </div>
                                <div 
                                    className="text-2xl font-bold font-mono"
                                    style={{ 
                                        color: result.metrics.benchmark_return >= 0 
                                            ? 'var(--color-positive)' 
                                            : 'var(--color-negative)' 
                                    }}
                                >
                                    {(result.metrics.benchmark_return * 100).toFixed(1)}%
                                </div>
                            </div>
                        </div>

                            {/* Chart */}
                            <div 
                                className="rounded-lg overflow-hidden"
                                style={{
                                    backgroundColor: 'var(--color-bg-tertiary)',
                                    border: '1px solid var(--color-border-subtle)',
                                }}
                            >
                                <Plot
                                    data={[
                                        {
                                            x: result.dates,
                                            y: result.portfolio_values,
                                            type: 'scatter',
                                            mode: 'lines',
                                            name: 'Portfolio',
                                            line: { color: '#2E7CF6', width: 2.5 },
                                        },
                                        {
                                            x: result.dates,
                                            y: result.benchmark_values,
                                            type: 'scatter',
                                            mode: 'lines',
                                            name: 'S&P 500',
                                            line: { color: '#6B7A8F', width: 2, dash: 'dot' },
                                        },
                                    ]}
                                    layout={{
                                        autosize: true,
                                        margin: { l: 60, r: 30, t: 30, b: 50 },
                                        paper_bgcolor: '#0F1419',
                                        plot_bgcolor: '#0F1419',
                                        xaxis: {
                                            gridcolor: '#1E2733',
                                            tickfont: { color: '#6B7A8F', size: 11 },
                                        },
                                        yaxis: {
                                            gridcolor: '#1E2733',
                                            tickfont: { color: '#6B7A8F', size: 11 },
                                            title: 'Value (Rebased)',
                                            titlefont: { color: '#B4BCC8', size: 12 },
                                        },
                                        legend: {
                                            orientation: 'h',
                                            y: 1.15,
                                            x: 0.5,
                                            xanchor: 'center',
                                            font: { color: '#B4BCC8', size: 12 },
                                        },
                                    }}
                                    useResizeHandler
                                    style={{ width: '100%', height: '300px' }}
                                    config={{ displayModeBar: false }}
                                />
                            </div>
                            
                            {/* Inspirational Message */}
                            <div 
                                className="text-center py-3 px-4 rounded-lg"
                                style={{
                                    backgroundColor: 'var(--color-positive-muted)',
                                    border: '1px solid rgba(0, 200, 83, 0.25)',
                                }}
                            >
                                <p 
                                    className="text-sm italic"
                                    style={{ color: 'var(--color-positive)' }}
                                >
                                    "{timeframeMessages[timeframe]}"
                                </p>
                            </div>
                        </div>
                    )}

                    {/* Empty State */}
                    {!result && !loading && !error && (
                        <div 
                            className="text-center py-16 rounded-lg"
                            style={{
                                backgroundColor: 'var(--color-bg-tertiary)',
                                border: '1px solid var(--color-border-subtle)',
                            }}
                        >
                            <div 
                                className="w-16 h-16 mx-auto mb-4 rounded-xl flex items-center justify-center"
                                style={{ backgroundColor: 'var(--color-negative-muted)' }}
                            >
                                <svg className="w-8 h-8" style={{ color: 'var(--color-negative)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
                                </svg>
                            </div>
                            <p 
                                className="text-sm"
                                style={{ color: 'var(--color-text-secondary)' }}
                            >
                                Select a crisis and run the stress test to see how your portfolio would perform
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </WidgetCard>
    );
};

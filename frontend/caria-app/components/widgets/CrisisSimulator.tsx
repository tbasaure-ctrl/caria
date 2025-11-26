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

// Crisis event dates for markers
const CRISIS_EVENTS: Record<string, { date: string; label: string; time?: string }> = {
    '2008_gfc': { date: '2008-09-15', label: 'Lehman Brothers Bankruptcy', time: '08:00' },
    '2001_911': { date: '2001-09-11', label: '9/11 Attacks', time: '08:46' },
    '1987_black_monday': { date: '1987-10-19', label: 'Black Monday', time: '09:30' },
    '1963_jfk': { date: '1963-11-22', label: 'JFK Assassination', time: '12:30' },
    '1962_cuban_missile': { date: '1962-10-16', label: 'Cuban Missile Crisis Start' },
    '2020_covid': { date: '2020-03-16', label: 'COVID-19 Market Crash' },
    '2000_dot_com': { date: '2000-03-10', label: 'Dot Com Peak' },
    '1929_depression': { date: '1929-10-24', label: 'Black Thursday' },
    '1939_wwii': { date: '1939-09-01', label: 'WWII Start' },
    '2011_euro_debt': { date: '2011-05-02', label: 'Greek Bailout' },
    '2018_trade_war': { date: '2018-09-24', label: 'Trade War Escalation' },
    '2022_inflation': { date: '2022-01-26', label: 'Fed Rate Hike Start' },
};

export const CrisisSimulator: React.FC = () => {
    const [selectedCrisis, setSelectedCrisis] = useState(CRISES[7].id);
    const [result, setResult] = useState<SimulationResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

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
            setResult(data);
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
                                        // Event marker - find closest date index
                                        ...(CRISIS_EVENTS[selectedCrisis] && result.dates.length > 0 ? (() => {
                                            const eventDate = CRISIS_EVENTS[selectedCrisis].date;
                                            const dates = result.dates.map(d => new Date(d).toISOString().split('T')[0]);
                                            const eventIndex = dates.findIndex(d => d >= eventDate);
                                            const markerIndex = eventIndex >= 0 ? eventIndex : Math.floor(dates.length / 2);
                                            const markerY = result.portfolio_values[markerIndex] || result.portfolio_values[0] || 100;
                                            return [{
                                                x: [result.dates[markerIndex] || eventDate],
                                                y: [markerY],
                                                type: 'scatter',
                                                mode: 'markers',
                                                name: CRISIS_EVENTS[selectedCrisis].label,
                                                marker: { 
                                                    size: 15, 
                                                    color: '#ef4444',
                                                    symbol: 'diamond',
                                                    line: { width: 2, color: '#FFFFFF' }
                                                },
                                                showlegend: false,
                                                hovertemplate: `${CRISIS_EVENTS[selectedCrisis].label}${CRISIS_EVENTS[selectedCrisis].time ? ` (${CRISIS_EVENTS[selectedCrisis].time})` : ''}<extra></extra>`,
                                            }];
                                        })() : []),
                                    ]}
                                    layout={{
                                        autosize: true,
                                        margin: { l: 60, r: 30, t: 50, b: 50 },
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
                                        annotations: CRISIS_EVENTS[selectedCrisis] && result.dates.length > 0 ? (() => {
                                            const eventDate = CRISIS_EVENTS[selectedCrisis].date;
                                            const dates = result.dates.map(d => new Date(d).toISOString().split('T')[0]);
                                            const eventIndex = dates.findIndex(d => d >= eventDate);
                                            const markerIndex = eventIndex >= 0 ? eventIndex : Math.floor(dates.length / 2);
                                            const markerY = result.portfolio_values[markerIndex] || result.portfolio_values[0] || 100;
                                            return [{
                                                x: result.dates[markerIndex] || eventDate,
                                                y: markerY,
                                                text: CRISIS_EVENTS[selectedCrisis].label + (CRISIS_EVENTS[selectedCrisis].time ? ` (${CRISIS_EVENTS[selectedCrisis].time})` : ''),
                                                showarrow: true,
                                                arrowhead: 2,
                                                arrowsize: 1,
                                                arrowwidth: 2,
                                                arrowcolor: '#ef4444',
                                                ax: 0,
                                                ay: -40,
                                                font: { color: '#ef4444', size: 11 },
                                                bgcolor: 'rgba(239, 68, 68, 0.1)',
                                                bordercolor: '#ef4444',
                                                borderwidth: 1,
                                                borderpad: 4,
                                            }];
                                        })() : [],
                                    }}
                                    useResizeHandler
                                    style={{ width: '100%', height: '300px' }}
                                    config={{ displayModeBar: false }}
                                />
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

import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';
import { AlertTriangle, Activity, Calendar, TrendingDown } from 'lucide-react';

const CRISES = [
    { id: '1929_depression', name: 'Great Depression', year: '1929', type: 'Recession' },
    { id: '1939_wwii', name: 'WWII Start', year: '1939', type: 'Geopolitical' },
    { id: '1962_cuban_missile', name: 'Cuban Missile Crisis', year: '1962', type: 'Geopolitical' },
    { id: '1963_jfk', name: 'Kennedy Assassination', year: '1963', type: 'Shock' },
    { id: '1987_black_monday', name: 'Black Monday', year: '1987', type: 'Crash' },
    { id: '2000_dot_com', name: 'Dot Com Bubble', year: '2000', type: 'Bubble' },
    { id: '2001_911', name: '9/11 Attacks', year: '2001', type: 'Shock' },
    { id: '2008_gfc', name: 'Global Financial Crisis', year: '2008', type: 'Recession' },
    { id: '2011_euro_debt', name: 'European Debt Crisis', year: '2011', type: 'Debt' },
    { id: '2018_trade_war', name: 'Trade War', year: '2018', type: 'Geopolitical' },
    { id: '2020_covid', name: 'COVID-19 Crash', year: '2020', type: 'Pandemic' },
    { id: '2022_inflation', name: 'Inflation Bear Market', year: '2022', type: 'Inflation' },
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

export const CrisisSimulator: React.FC = () => {
    const [selectedCrisis, setSelectedCrisis] = useState(CRISES[7].id); // Default to GFC
    const [result, setResult] = useState<SimulationResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleSimulate = async () => {
        setLoading(true);
        setError(null);
        try {
            // Mock portfolio for simulation if real one is empty/unavailable
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
            className="h-full"
        >
            <div className="flex flex-col h-full gap-6">
                {/* Header / Controls */}
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                    <div className="w-full sm:w-64">
                        <div className="relative">
                            <select
                                value={selectedCrisis}
                                onChange={(e) => setSelectedCrisis(e.target.value)}
                                className="w-full pl-4 pr-10 py-2.5 bg-[#0B1221] border border-cyan-900/30 rounded-lg text-cyan-100 text-sm focus:outline-none focus:border-cyan-500/50 appearance-none shadow-[0_0_10px_rgba(6,182,212,0.1)]"
                            >
                                {CRISES.map((c) => (
                                    <option key={c.id} value={c.id}>
                                        {c.year} - {c.name}
                                    </option>
                                ))}
                            </select>
                            <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-cyan-500">
                                <Calendar className="w-4 h-4" />
                            </div>
                        </div>
                    </div>

                    <button
                        onClick={handleSimulate}
                        disabled={loading}
                        className="w-full sm:w-auto px-6 py-2.5 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white text-sm font-bold rounded-lg shadow-[0_0_15px_rgba(6,182,212,0.3)] transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                        {loading ? (
                            <>
                                <Activity className="w-4 h-4 animate-spin" />
                                SIMULATING...
                            </>
                        ) : (
                            <>
                                <AlertTriangle className="w-4 h-4" />
                                RUN STRESS TEST
                            </>
                        )}
                    </button>
                </div>

                {/* Main Content Area */}
                <div className="flex-1 min-h-[300px] relative rounded-xl overflow-hidden border border-white/5 bg-[#050912]">
                    {/* Background Grid */}
                    <div className="absolute inset-0 bg-[linear-gradient(rgba(6,182,212,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.03)_1px,transparent_1px)] bg-[size:20px_20px]" />

                    {error && (
                        <div className="absolute inset-0 flex items-center justify-center z-20 bg-black/50 backdrop-blur-sm">
                            <div className="bg-red-900/20 border border-red-500/50 text-red-400 px-6 py-4 rounded-lg flex items-center gap-3">
                                <AlertTriangle className="w-5 h-5" />
                                {error}
                            </div>
                        </div>
                    )}

                    {!result && !loading && !error && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center text-cyan-900/40 z-10">
                            <Activity className="w-24 h-24 mb-4 opacity-20" />
                            <p className="text-sm font-mono uppercase tracking-widest">Select a crisis to begin simulation</p>
                        </div>
                    )}

                    {result && (
                        <div className="absolute inset-0 p-4 flex flex-col">
                            {/* Chart */}
                            <div className="flex-1 w-full min-h-0">
                                <Plot
                                    data={[
                                        {
                                            x: result.dates,
                                            y: result.portfolio_values,
                                            type: 'scatter',
                                            mode: 'lines',
                                            name: 'Your Portfolio',
                                            line: { color: '#06b6d4', width: 3, shape: 'spline' }, // Cyan-500
                                            fill: 'tozeroy',
                                            fillcolor: 'rgba(6, 182, 212, 0.1)'
                                        },
                                        {
                                            x: result.dates,
                                            y: result.benchmark_values,
                                            type: 'scatter',
                                            mode: 'lines',
                                            name: 'S&P 500',
                                            line: { color: '#64748b', width: 2, dash: 'dot' }, // Slate-500
                                        }
                                    ]}
                                    layout={{
                                        autosize: true,
                                        margin: { l: 40, r: 20, t: 20, b: 40 },
                                        paper_bgcolor: 'transparent',
                                        plot_bgcolor: 'transparent',
                                        xaxis: {
                                            gridcolor: 'rgba(255,255,255,0.05)',
                                            tickfont: { color: '#94a3b8', size: 10 },
                                            showgrid: true,
                                            zeroline: false,
                                        },
                                        yaxis: {
                                            gridcolor: 'rgba(255,255,255,0.05)',
                                            tickfont: { color: '#94a3b8', size: 10 },
                                            showgrid: true,
                                            zeroline: false,
                                        },
                                        legend: {
                                            orientation: 'h',
                                            y: 1,
                                            x: 0,
                                            font: { color: '#e2e8f0' },
                                            bgcolor: 'rgba(0,0,0,0.5)'
                                        },
                                        hovermode: 'x unified',
                                        dragmode: false,
                                    }}
                                    useResizeHandler
                                    style={{ width: '100%', height: '100%' }}
                                    config={{ displayModeBar: false }}
                                />
                            </div>

                            {/* Metrics Footer */}
                            <div className="mt-4 grid grid-cols-3 gap-4 border-t border-white/10 pt-4">
                                <div className="text-center">
                                    <div className="text-[10px] text-cyan-500/70 uppercase tracking-wider mb-1">Max Drawdown</div>
                                    <div className="text-2xl font-bold text-red-500 font-mono">
                                        {(result.metrics.max_drawdown * 100).toFixed(1)}%
                                    </div>
                                </div>
                                <div className="text-center border-l border-white/10">
                                    <div className="text-[10px] text-cyan-500/70 uppercase tracking-wider mb-1">Recovery Time</div>
                                    <div className="text-2xl font-bold text-white font-mono">
                                        -- M
                                    </div>
                                </div>
                                <div className="text-center border-l border-white/10">
                                    <div className="text-[10px] text-cyan-500/70 uppercase tracking-wider mb-1">Volatility</div>
                                    <div className="text-2xl font-bold text-yellow-500 font-mono">
                                        HIGH
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </WidgetCard>
    );
};

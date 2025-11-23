/**
 * Monte Carlo Simulation Widget with optimized Plotly visualization.
 * Per audit document (3.1): Uses Scattergl (WebGL) for rendering thousands of lines.
 */

import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';

interface MonteCarloResult {
    paths: number[][];
    final_values: number[];
    percentiles: {
        p5: number;
        p10: number;
        p25: number;
        p50: number;
        p75: number;
        p90: number;
        p95: number;
    };
    metrics: {
        mean: number;
        median: number;
        std: number;
        var_5pct: number;
        cvar_5pct: number;
        prob_final_less_invested: number;
        moic_median: number;
    };
    plotly_data: {
        x: (number | null)[];
        y: (number | null)[];
        type: string;
        mode: string;
        line: { width: number; color: string };
        name: string;
    };
    histogram: {
        x: number[];
        type: string;
        nbinsx: number;
        marker: { color: string; line: { color: string; width: number } };
        name: string;
    };
    simulation_params: {
        initial_value: number;
        mu: number;
        sigma: number;
        years: number;
        simulations: number;
    };
}

export const MonteCarloSimulation: React.FC = () => {
    const [initialValue, setInitialValue] = useState(100000);
    const [mu, setMu] = useState(0.10);
    const [sigma, setSigma] = useState(0.25);
    const [years, setYears] = useState(5);
    const [simulations, setSimulations] = useState(10000);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<MonteCarloResult | null>(null);

    const handleSimulate = async () => {
        setIsLoading(true);
        setError(null);
        setResult(null);

        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/montecarlo/simulate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    initial_value: initialValue,
                    mu: mu,
                    sigma: sigma,
                    years: years,
                    simulations: simulations,
                    contributions_per_year: 0.0,
                    annual_fee: 0.0,
                }),
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({ detail: 'Simulation failed' }));
                throw new Error(errData.detail || 'Simulation failed');
            }

            const data: MonteCarloResult = await response.json();
            setResult(data);
        } catch (err: any) {
            setError('Coming soon... Monte Carlo simulations are being enhanced for more accurate predictions.');
            console.error('Monte Carlo error:', err);
        } finally {
            setIsLoading(false);
        }
    };

    // Prepare Plotly layout
    const plotLayout = {
        title: {
            text: `Monte Carlo Simulation (${result?.simulation_params.simulations.toLocaleString()} paths)`,
            font: { color: '#E0E1DD', size: 14 },
        },
        xaxis: {
            title: 'Years',
            gridcolor: '#334155',
            color: '#94a3b8',
        },
        yaxis: {
            title: 'Portfolio Value ($)',
            gridcolor: '#334155',
            color: '#94a3b8',
        },
        plot_bgcolor: '#0f172a',
        paper_bgcolor: '#0f172a',
        font: { color: '#94a3b8' },
        showlegend: false,
        margin: { l: 60, r: 20, t: 50, b: 50 },
    };

    const histogramLayout = {
        title: {
            text: 'Distribution of Final Values',
            font: { color: '#E0E1DD', size: 14 },
        },
        xaxis: {
            title: 'Final Value ($)',
            gridcolor: '#334155',
            color: '#94a3b8',
        },
        yaxis: {
            title: 'Frequency',
            gridcolor: '#334155',
            color: '#94a3b8',
        },
        plot_bgcolor: '#0f172a',
        paper_bgcolor: '#0f172a',
        font: { color: '#94a3b8' },
        margin: { l: 60, r: 20, t: 50, b: 50 },
    };

    return (
        <WidgetCard
            title="MONTE CARLO SIMULATION"
            tooltip="Simulación de miles de escenarios futuros para tu cartera. Visualiza probabilidades, riesgo (VaR/CVaR) y distribución de resultados."
        >
            <div className="space-y-4">
                {/* Input Form */}
                <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                        <label className="text-xs text-slate-400">Initial Value ($)</label>
                        <input
                            type="number"
                            value={initialValue}
                            onChange={(e) => setInitialValue(parseFloat(e.target.value) || 0)}
                            className="w-full mt-1 bg-gray-800 border border-slate-700 rounded-md py-1 px-2 text-slate-100"
                            disabled={isLoading}
                        />
                    </div>
                    <div>
                        <label className="text-xs text-slate-400">Expected Return (μ)</label>
                        <input
                            type="number"
                            step="0.01"
                            value={mu}
                            onChange={(e) => setMu(parseFloat(e.target.value) || 0)}
                            className="w-full mt-1 bg-gray-800 border border-slate-700 rounded-md py-1 px-2 text-slate-100"
                            disabled={isLoading}
                        />
                    </div>
                    <div>
                        <label className="text-xs text-slate-400">Volatility (σ)</label>
                        <input
                            type="number"
                            step="0.01"
                            value={sigma}
                            onChange={(e) => setSigma(parseFloat(e.target.value) || 0)}
                            className="w-full mt-1 bg-gray-800 border border-slate-700 rounded-md py-1 px-2 text-slate-100"
                            disabled={isLoading}
                        />
                    </div>
                    <div>
                        <label className="text-xs text-slate-400">Years</label>
                        <input
                            type="number"
                            value={years}
                            onChange={(e) => setYears(parseInt(e.target.value) || 1)}
                            className="w-full mt-1 bg-gray-800 border border-slate-700 rounded-md py-1 px-2 text-slate-100"
                            disabled={isLoading}
                        />
                    </div>
                </div>

                <button
                    onClick={handleSimulate}
                    disabled={isLoading || initialValue <= 0 || sigma <= 0}
                    className="w-full bg-slate-700 text-white font-bold py-2 px-4 rounded-md hover:bg-slate-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                >
                    {isLoading ? 'Running Simulation...' : 'Run Simulation'}
                </button>

                {error && (
                    <div className="text-sm text-red-400 bg-red-900/30 p-2 rounded-md">{error}</div>
                )}

                {/* Results */}
                {result && (
                    <div className="space-y-4 fade-in">
                        {/* Percentiles */}
                        <div className="grid grid-cols-3 gap-2 text-xs">
                            <div className="bg-gray-900/70 p-2 rounded-md">
                                <div className="text-slate-400">P10</div>
                                <div className="text-slate-100 font-mono">${result.percentiles.p10.toLocaleString()}</div>
                            </div>
                            <div className="bg-gray-900/70 p-2 rounded-md">
                                <div className="text-slate-400">P50 (Median)</div>
                                <div className="text-slate-100 font-mono">${result.percentiles.p50.toLocaleString()}</div>
                            </div>
                            <div className="bg-gray-900/70 p-2 rounded-md">
                                <div className="text-slate-400">P90</div>
                                <div className="text-slate-100 font-mono">${result.percentiles.p90.toLocaleString()}</div>
                            </div>
                        </div>

                        {/* Metrics */}
                        <div className="space-y-1 text-xs">
                            <div className="flex justify-between">
                                <span className="text-slate-400">Expected Value:</span>
                                <span className="text-slate-100 font-mono">${result.metrics.mean.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-400">VaR (5%):</span>
                                <span className="text-red-400 font-mono">${result.metrics.var_5pct.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-400">CVaR (5%):</span>
                                <span className="text-red-400 font-mono">${result.metrics.cvar_5pct.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-400">Prob. Loss:</span>
                                <span className="text-slate-300 font-mono">{(result.metrics.prob_final_less_invested * 100).toFixed(1)}%</span>
                            </div>
                        </div>

                        {/* Plotly Visualization - Scattergl for performance */}
                        <div className="bg-gray-900/50 rounded-md p-2">
                            <Plot
                                data={[result.plotly_data] as any}
                                layout={plotLayout}
                                config={{ displayModeBar: false, responsive: true }}
                                style={{ width: '100%', height: '300px' }}
                                useResizeHandler={true}
                            />
                        </div>

                        {/* Histogram */}
                        {result.histogram && (
                            <div className="bg-gray-900/50 rounded-md p-2">
                                <Plot
                                    data={[result.histogram] as any}
                                    layout={histogramLayout}
                                    config={{ displayModeBar: false, responsive: true }}
                                    style={{ width: '100%', height: '250px' }}
                                    useResizeHandler={true}
                                />
                            </div>
                        )}
                    </div>
                )}

                {!result && !isLoading && !error && (
                    <div className="text-center text-xs text-slate-500 py-4">
                        <p>Run a Monte Carlo simulation to see projected portfolio paths.</p>
                    </div>
                )}
            </div>
        </WidgetCard>
    );
};


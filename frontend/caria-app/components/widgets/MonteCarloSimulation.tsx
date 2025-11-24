/**
 * Monte Carlo Simulation – Stock Price Forecast
 * Explore a range of possible future prices based on historical volatility and your assumptions.
 */

import React, { useState } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';

export const MonteCarloSimulation: React.FC = () => {
    const [years, setYears] = useState(10);
    const [simulationPaths, setSimulationPaths] = useState(10000);
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [expectedGrowth, setExpectedGrowth] = useState(0.10);
    const [volatility, setVolatility] = useState(0.25);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<any>(null);

    const handleReset = () => {
        setYears(10);
        setSimulationPaths(10000);
        setExpectedGrowth(0.10);
        setVolatility(0.25);
        setShowAdvanced(false);
    };

    const handleSimulate = async () => {
        setIsLoading(true);
        setError(null);
        setResult(null);

        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/montecarlo/simulate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    years,
                    n_paths: simulationPaths,
                    growth_rate: expectedGrowth,
                    annual_volatility: volatility,
                }),
            });

            if (!response.ok) {
                throw new Error('Simulation failed');
            }

            const data = await response.json();
            setResult(data);
        } catch (err: any) {
            setError('Simulation failed. Please try again.');
            console.error('Monte Carlo error:', err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <WidgetCard title="MONTE CARLO – STOCK PRICE FORECAST">
            <div className="space-y-4">
                {/* Header with info icon */}
                <div className="flex items-center gap-2">
                    <p className="text-sm text-slate-400">
                        Explore a range of possible future prices based on historical volatility and your assumptions.
                    </p>
                    <button
                        className="text-slate-400 hover:text-slate-200 transition-colors"
                        title="We simulate thousands of possible future prices using your assumptions for growth and volatility, based on historical data. This is not a prediction, but a way to visualize risk and the range of potential outcomes."
                    >
                        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                        </svg>
                    </button>
                </div>

                {/* Learn more link */}
                <a
                    href="https://en.wikipedia.org/wiki/Monte_Carlo_method"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-blue-400 hover:text-blue-300 underline"
                >
                    Learn more about how Monte Carlo simulations work.
                </a>

                {/* Controls */}
                <div className="space-y-3">
                    {/* Years */}
                    <div>
                        <label className="block text-sm font-semibold text-slate-300 mb-1">Years</label>
                        <p className="text-xs text-slate-500 mb-2">Number of years to project into the future.</p>
                        <input
                            type="number"
                            value={years}
                            onChange={(e) => setYears(parseInt(e.target.value) || 1)}
                            className="w-full bg-gray-800 border border-slate-700 rounded-md py-2 px-3 text-slate-100 focus:outline-none focus:ring-1 focus:ring-blue-500"
                            placeholder="10"
                            min="1"
                            max="50"
                            disabled={isLoading}
                        />
                    </div>

                    {/* Simulation Paths */}
                    <div>
                        <label className="block text-sm font-semibold text-slate-300 mb-1">Simulation Paths</label>
                        <p className="text-xs text-slate-500 mb-2">More paths create a smoother distribution but may take longer to compute.</p>
                        <select
                            value={simulationPaths}
                            onChange={(e) => setSimulationPaths(parseInt(e.target.value))}
                            className="w-full bg-gray-800 border border-slate-700 rounded-md py-2 px-3 text-slate-100 focus:outline-none focus:ring-1 focus:ring-blue-500"
                            disabled={isLoading}
                        >
                            <option value={1000}>1,000</option>
                            <option value={5000}>5,000</option>
                            <option value={10000}>10,000 (default)</option>
                            <option value={20000}>20,000</option>
                        </select>
                    </div>

                    {/* Advanced Settings Toggle */}
                    <button
                        onClick={() => setShowAdvanced(!showAdvanced)}
                        className="flex items-center gap-2 text-sm text-slate-300 hover:text-white transition-colors"
                    >
                        <span>Advanced settings</span>
                        <svg
                            className={`w-4 h-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`}
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                    </button>

                    {/* Advanced Settings Panel */}
                    {showAdvanced && (
                        <div className="space-y-3 bg-gray-900/50 p-4 rounded-md border border-slate-800">
                            {/* Expected Annual Growth */}
                            <div>
                                <label className="block text-sm font-semibold text-slate-300 mb-1">Expected annual growth</label>
                                <p className="text-xs text-slate-500 mb-2">Baseline annual return assumption. Defaults are based on historical data.</p>
                                <input
                                    type="number"
                                    step="0.01"
                                    value={expectedGrowth}
                                    onChange={(e) => setExpectedGrowth(parseFloat(e.target.value) || 0)}
                                    className="w-full bg-gray-800 border border-slate-700 rounded-md py-2 px-3 text-slate-100 focus:outline-none focus:ring-1 focus:ring-blue-500"
                                    placeholder="0.10"
                                    disabled={isLoading}
                                />
                            </div>

                            {/* Volatility */}
                            <div>
                                <label className="block text-sm font-semibold text-slate-300 mb-1">Annual volatility</label>
                                <p className="text-xs text-slate-500 mb-2">How much the price tends to move up or down in a typical year.</p>
                                <input
                                    type="number"
                                    step="0.01"
                                    value={volatility}
                                    onChange={(e) => setVolatility(parseFloat(e.target.value) || 0)}
                                    className="w-full bg-gray-800 border border-slate-700 rounded-md py-2 px-3 text-slate-100 focus:outline-none focus:ring-1 focus:ring-blue-500"
                                    placeholder="0.25"
                                    disabled={isLoading}
                                />
                            </div>
                        </div>
                    )}
                </div>

                {/* Action Buttons */}
                <div className="flex gap-2">
                    <button
                        onClick={handleSimulate}
                        disabled={isLoading}
                        className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-md transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isLoading ? 'Running...' : 'Run Simulation'}
                    </button>
                    <button
                        onClick={handleReset}
                        disabled={isLoading}
                        className="px-4 py-2 text-sm text-slate-400 hover:text-slate-200 underline transition-colors disabled:opacity-50"
                    >
                        Reset to defaults
                    </button>
                </div>

                {/* Error Message */}
                {error && (
                    <div className="text-sm text-red-400 bg-red-900/30 p-3 rounded-md border border-red-800">
                        {error}
                    </div>
                )}

                {/* Results */}
                {result && (
                    <div className="space-y-4 fade-in">
                        {/* Percentiles */}
                        <div className="grid grid-cols-3 gap-2">
                            <div className="bg-gray-900/70 p-3 rounded-md text-center">
                                <div className="text-xs text-slate-400">P10</div>
                                <div className="text-lg font-semibold text-slate-100 font-mono">
                                    ${result.percentiles?.p10?.toLocaleString() || 'N/A'}
                                </div>
                            </div>
                            <div className="bg-gray-900/70 p-3 rounded-md text-center">
                                <div className="text-xs text-slate-400">P50 (Median)</div>
                                <div className="text-lg font-semibold text-slate-100 font-mono">
                                    ${result.percentiles?.p50?.toLocaleString() || 'N/A'}
                                </div>
                            </div>
                            <div className="bg-gray-900/70 p-3 rounded-md text-center">
                                <div className="text-xs text-slate-400">P90</div>
                                <div className="text-lg font-semibold text-slate-100 font-mono">
                                    ${result.percentiles?.p90?.toLocaleString() || 'N/A'}
                                </div>
                            </div>
                        </div>

                        {/* Charts placeholder - would use Recharts or similar */}
                        <div className="bg-gray-900/50 p-4 rounded-md border border-slate-800">
                            <p className="text-sm text-slate-400 text-center">
                                Charts will display here (Price over time and histogram)
                            </p>
                        </div>

                        {/* Educational Disclaimer */}
                        <p className="text-xs text-slate-500 text-center italic">
                            These simulations are for educational purposes only and are not a guarantee of future performance.
                        </p>
                    </div>
                )}

                {/* Initial State */}
                {!result && !isLoading && !error && (
                    <div className="text-center text-sm text-slate-500 py-6">
                        <p>Configure your parameters and run the simulation to visualize potential outcomes.</p>
                    </div>
                )}
            </div>
        </WidgetCard>
    );
};

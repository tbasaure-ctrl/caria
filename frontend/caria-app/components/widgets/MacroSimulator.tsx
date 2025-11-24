import React, { useState } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';

interface MacroResult {
    portfolio_impact_pct: number;
    market_impact_pct: number;
    details: {
        ticker: string;
        impact_pct: number;
        contribution: number;
    }[];
}

export const MacroSimulator: React.FC = () => {
    const [inflation, setInflation] = useState(0);
    const [rates, setRates] = useState(0);
    const [gdp, setGdp] = useState(0);
    const [result, setResult] = useState<MacroResult | null>(null);
    const [loading, setLoading] = useState(false);

    const handleSimulate = async () => {
        setLoading(true);
        try {
            // Mock portfolio
            const portfolio = [
                { ticker: 'AAPL', quantity: 10, weight: 0.5 },
                { ticker: 'MSFT', quantity: 5, weight: 0.5 },
            ];

            const response = await fetchWithAuth(`${API_BASE_URL}/api/simulation/macro`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    portfolio,
                    params: { inflation, rates, gdp },
                }),
            });

            if (response.ok) {
                const data = await response.json();
                setResult(data);
            }
        } catch (error) {
            console.error('Macro sim error', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <WidgetCard title="MACRO MULTIVERSE" tooltip="Simulate portfolio performance under different macroeconomic conditions.">
            <div className="space-y-6">
                <div className="space-y-4">
                    <div>
                        <div className="flex justify-between text-xs text-slate-400 mb-1">
                            <span>Inflation Shock</span>
                            <span>{inflation > 0 ? '+' : ''}{inflation}%</span>
                        </div>
                        <input
                            type="range"
                            min="-5"
                            max="5"
                            step="0.5"
                            value={inflation}
                            onChange={(e) => setInflation(parseFloat(e.target.value))}
                            className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                        />
                    </div>
                    <div>
                        <div className="flex justify-between text-xs text-slate-400 mb-1">
                            <span>Interest Rates Shock</span>
                            <span>{rates > 0 ? '+' : ''}{rates}%</span>
                        </div>
                        <input
                            type="range"
                            min="-3"
                            max="3"
                            step="0.25"
                            value={rates}
                            onChange={(e) => setRates(parseFloat(e.target.value))}
                            className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                        />
                    </div>
                    <div>
                        <div className="flex justify-between text-xs text-slate-400 mb-1">
                            <span>GDP Growth Shock</span>
                            <span>{gdp > 0 ? '+' : ''}{gdp}%</span>
                        </div>
                        <input
                            type="range"
                            min="-5"
                            max="5"
                            step="0.5"
                            value={gdp}
                            onChange={(e) => setGdp(parseFloat(e.target.value))}
                            className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
                        />
                    </div>
                </div>

                <button
                    onClick={handleSimulate}
                    disabled={loading}
                    className="w-full bg-slate-700 hover:bg-slate-600 text-slate-200 py-2 rounded text-sm font-semibold transition-colors"
                >
                    {loading ? 'Calculating...' : 'Simulate Scenario'}
                </button>

                {result && (
                    <div className="bg-gray-900/50 p-4 rounded border border-slate-800 space-y-3 animate-fade-in">
                        <div className="flex justify-between items-center">
                            <span className="text-sm text-slate-400">Portfolio Impact</span>
                            <span className={`text-lg font-bold ${result.portfolio_impact_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                {result.portfolio_impact_pct > 0 ? '+' : ''}{result.portfolio_impact_pct.toFixed(2)}%
                            </span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-xs text-slate-500">Market Impact</span>
                            <span className={`text-sm ${result.market_impact_pct >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                                {result.market_impact_pct > 0 ? '+' : ''}{result.market_impact_pct.toFixed(2)}%
                            </span>
                        </div>

                        <div className="pt-2 border-t border-slate-800">
                            <div className="text-xs text-slate-500 mb-2">Asset Breakdown</div>
                            <div className="space-y-1">
                                {result.details.map((d) => (
                                    <div key={d.ticker} className="flex justify-between text-xs">
                                        <span className="text-slate-300">{d.ticker}</span>
                                        <span className={d.impact_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                                            {d.impact_pct.toFixed(2)}%
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </WidgetCard>
    );
};

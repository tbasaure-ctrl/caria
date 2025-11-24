/**
 * Professional Holdings Graph with Recharts
 * Displays portfolio holdings with bar chart visualization and sortable table
 */

import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { fetchHoldingsWithPrices, HoldingWithPrice } from '../../services/apiService';
import { WidgetCard } from './WidgetCard';

type SortOption = 'ticker' | 'return' | 'value';

export const HoldingsGraph: React.FC = () => {
    const [holdings, setHoldings] = useState<HoldingWithPrice[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [sortOption, setSortOption] = useState<SortOption>('value');

    useEffect(() => {
        loadHoldings();
    }, []);

    const loadHoldings = async () => {
        try {
            setError(null);
            const data = await fetchHoldingsWithPrices();
            setHoldings(data.holdings || []);
        } catch (err) {
            console.error('Error loading holdings:', err);
            setError('Could not load holdings.');
        } finally {
            setLoading(false);
        }
    };

    const getSortedHoldings = () => {
        const sorted = [...holdings];
        switch (sortOption) {
            case 'ticker':
                return sorted.sort((a, b) => a.ticker.localeCompare(b.ticker));
            case 'return':
                return sorted.sort((a, b) => (b.gain_loss_pct || 0) - (a.gain_loss_pct || 0));
            case 'value':
                return sorted.sort((a, b) => (b.current_value || 0) - (a.current_value || 0));
            default:
                return sorted;
        }
    };

    const sortedData = getSortedHoldings();

    // Calculate total portfolio value
    const totalValue = holdings.reduce((sum, h) => sum + (h.current_value || 0), 0);

    // Prepare chart data
    const chartData = sortedData.map(h => ({
        ticker: h.ticker,
        value: h.current_value || 0,
        returnPct: h.gain_loss_pct || 0,
        cost: (h.quantity * h.average_cost),
    }));

    if (loading) {
        return (
            <WidgetCard title="HOLDINGS GRAPH" tooltip="Visual breakdown of your portfolio holdings">
                <div className="text-sm text-slate-400">Loading...</div>
            </WidgetCard>
        );
    }

    if (error) {
        return (
            <WidgetCard title="HOLDINGS GRAPH" tooltip="Visual breakdown of your portfolio holdings">
                <div className="text-sm text-red-400">{error}</div>
            </WidgetCard>
        );
    }

    return (
        <WidgetCard title="HOLDINGS GRAPH" tooltip="Visual breakdown of your portfolio holdings by market value and return">
            <div className="space-y-4">
                {/* Sort Controls */}
                <div className="flex justify-between items-center">
                    <div className="text-sm text-slate-400">
                        Total Value: <span className="text-slate-200 font-semibold">${totalValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                    </div>
                    <div className="flex bg-slate-800 rounded p-0.5 gap-1">
                        <button
                            onClick={() => setSortOption('value')}
                            className={`px-3 py-1 text-xs rounded transition-colors ${sortOption === 'value'
                                    ? 'bg-blue-600 text-white'
                                    : 'text-slate-400 hover:text-slate-200'
                                }`}
                        >
                            By Value
                        </button>
                        <button
                            onClick={() => setSortOption('return')}
                            className={`px-3 py-1 text-xs rounded transition-colors ${sortOption === 'return'
                                    ? 'bg-blue-600 text-white'
                                    : 'text-slate-400 hover:text-slate-200'
                                }`}
                        >
                            By Return %
                        </button>
                        <button
                            onClick={() => setSortOption('ticker')}
                            className={`px-3 py-1 text-xs rounded transition-colors ${sortOption === 'ticker'
                                    ? 'bg-blue-600 text-white'
                                    : 'text-slate-400 hover:text-slate-200'
                                }`}
                        >
                            By Ticker
                        </button>
                    </div>
                </div>

                {holdings.length === 0 ? (
                    <div className="text-center text-sm text-slate-500 py-8">
                        No holdings to display. Add holdings to see the graph.
                    </div>
                ) : (
                    <>
                        {/* Bar Chart */}
                        <div className="bg-gray-900/50 p-4 rounded-md border border-slate-800">
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                    <XAxis
                                        dataKey="ticker"
                                        stroke="#94a3b8"
                                        tick={{ fill: '#94a3b8', fontSize: 12 }}
                                    />
                                    <YAxis
                                        stroke="#94a3b8"
                                        tick={{ fill: '#94a3b8', fontSize: 12 }}
                                        tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: '#1e293b',
                                            border: '1px solid #475569',
                                            borderRadius: '0.375rem',
                                            color: '#e2e8f0'
                                        }}
                                        formatter={(value: number) => [`$${value.toLocaleString(undefined, { minimumFractionDigits: 2 })}`, 'Value']}
                                    />
                                    <Legend wrapperStyle={{ color: '#94a3b8' }} />
                                    <Bar dataKey="value" name="Market Value">
                                        {chartData.map((entry, index) => (
                                            <Cell
                                                key={`cell-${index}`}
                                                fill={entry.returnPct >= 0 ? '#10b981' : '#ef4444'}
                                            />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Table */}
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="border-b border-slate-800">
                                        <th className="text-left py-2 px-3 text-slate-400 font-semibold">Ticker</th>
                                        <th className="text-right py-2 px-3 text-slate-400 font-semibold">Quantity</th>
                                        <th className="text-right py-2 px-3 text-slate-400 font-semibold">Avg Cost</th>
                                        <th className="text-right py-2 px-3 text-slate-400 font-semibold">Market Value</th>
                                        <th className="text-right py-2 px-3 text-slate-400 font-semibold">Return %</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {sortedData.map((holding) => (
                                        <tr
                                            key={holding.id}
                                            className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors"
                                        >
                                            <td className="py-2 px-3 text-slate-200 font-semibold">{holding.ticker}</td>
                                            <td className="py-2 px-3 text-right text-slate-300">{holding.quantity}</td>
                                            <td className="py-2 px-3 text-right text-slate-300">
                                                ${holding.average_cost.toFixed(2)}
                                            </td>
                                            <td className="py-2 px-3 text-right text-slate-200 font-mono">
                                                ${(holding.current_value || 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                            </td>
                                            <td className={`py-2 px-3 text-right font-semibold ${(holding.gain_loss_pct || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                                                }`}>
                                                {(holding.gain_loss_pct || 0) >= 0 ? '+' : ''}
                                                {(holding.gain_loss_pct || 0).toFixed(2)}%
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </>
                )}
            </div>
        </WidgetCard>
    );
};

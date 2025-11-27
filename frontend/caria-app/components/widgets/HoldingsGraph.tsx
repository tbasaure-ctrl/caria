/**
 * Professional Holdings Graph with Recharts
 * Displays portfolio holdings with artistic area chart visualization
 */

import React, { useState, useEffect } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
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
                <div className="text-sm text-text-muted">Loading...</div>
            </WidgetCard>
        );
    }

    if (error) {
        return (
            <WidgetCard title="HOLDINGS GRAPH" tooltip="Visual breakdown of your portfolio holdings">
                <div className="text-sm text-negative">{error}</div>
            </WidgetCard>
        );
    }

    return (
        <WidgetCard title="HOLDINGS GRAPH" tooltip="Visual breakdown of your portfolio holdings by market value and return">
            <div className="space-y-6">
                {/* Sort Controls */}
                <div className="flex justify-between items-center">
                    <div className="text-sm text-text-muted font-mono">
                        Total Value: <span className="text-text-primary font-semibold text-lg ml-2">${totalValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                    </div>
                    <div className="flex bg-bg-tertiary border border-white/5 rounded p-0.5 gap-1">
                        <button
                            onClick={() => setSortOption('value')}
                            className={`px-3 py-1 text-xs font-medium rounded transition-colors ${sortOption === 'value'
                                    ? 'bg-accent-primary/20 text-accent-primary'
                                    : 'text-text-muted hover:text-text-primary'
                                }`}
                        >
                            Value
                        </button>
                        <button
                            onClick={() => setSortOption('return')}
                            className={`px-3 py-1 text-xs font-medium rounded transition-colors ${sortOption === 'return'
                                    ? 'bg-accent-primary/20 text-accent-primary'
                                    : 'text-text-muted hover:text-text-primary'
                                }`}
                        >
                            Return %
                        </button>
                        <button
                            onClick={() => setSortOption('ticker')}
                            className={`px-3 py-1 text-xs font-medium rounded transition-colors ${sortOption === 'ticker'
                                    ? 'bg-accent-primary/20 text-accent-primary'
                                    : 'text-text-muted hover:text-text-primary'
                                }`}
                        >
                            Ticker
                        </button>
                    </div>
                </div>

                {holdings.length === 0 ? (
                    <div className="text-center text-sm text-text-muted py-12">
                        No holdings to display. Add holdings to see the visualization.
                    </div>
                ) : (
                    <>
                        {/* Area Chart - Artistic Style */}
                        <div className="h-[300px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                                    <defs>
                                        <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#D4AF37" stopOpacity={0.3}/>
                                            <stop offset="95%" stopColor="#D4AF37" stopOpacity={0}/>
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                                    <XAxis
                                        dataKey="ticker"
                                        stroke="#64748B"
                                        tick={{ fill: '#64748B', fontSize: 11 }}
                                        axisLine={false}
                                        tickLine={false}
                                    />
                                    <YAxis
                                        stroke="#64748B"
                                        tick={{ fill: '#64748B', fontSize: 11 }}
                                        tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                                        axisLine={false}
                                        tickLine={false}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: '#0B101B',
                                            border: '1px solid rgba(255,255,255,0.1)',
                                            borderRadius: '8px',
                                            color: '#F1F5F9',
                                            fontFamily: 'var(--font-mono)'
                                        }}
                                        formatter={(value: number) => [`$${value.toLocaleString(undefined, { minimumFractionDigits: 2 })}`, 'Value']}
                                        cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 1 }}
                                    />
                                    <Area 
                                        type="monotone" 
                                        dataKey="value" 
                                        stroke="#D4AF37" 
                                        strokeWidth={2}
                                        fillOpacity={1} 
                                        fill="url(#colorValue)" 
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Modern Table */}
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="border-b border-white/5 text-xs uppercase tracking-wider">
                                        <th className="text-left py-3 px-4 text-text-muted font-medium">Ticker</th>
                                        <th className="text-right py-3 px-4 text-text-muted font-medium">Quantity</th>
                                        <th className="text-right py-3 px-4 text-text-muted font-medium">Avg Cost</th>
                                        <th className="text-right py-3 px-4 text-text-muted font-medium">Market Value</th>
                                        <th className="text-right py-3 px-4 text-text-muted font-medium">Return</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-white/5">
                                    {sortedData.map((holding) => (
                                        <tr
                                            key={holding.id}
                                            className="hover:bg-white/5 transition-colors group"
                                        >
                                            <td className="py-3 px-4 text-text-primary font-medium group-hover:text-accent-cyan transition-colors">{holding.ticker}</td>
                                            <td className="py-3 px-4 text-right text-text-secondary font-mono">{holding.quantity}</td>
                                            <td className="py-3 px-4 text-right text-text-secondary font-mono">
                                                ${holding.average_cost.toFixed(2)}
                                            </td>
                                            <td className="py-3 px-4 text-right text-text-primary font-mono font-medium">
                                                ${(holding.current_value || 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                            </td>
                                            <td className={`py-3 px-4 text-right font-mono font-medium ${(holding.gain_loss_pct || 0) >= 0 ? 'text-positive' : 'text-negative'
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

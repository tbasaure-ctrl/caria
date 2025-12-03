import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { CurrencyRate, fetchCurrencyHistory, CurrencyHistory } from '../../../services/apiService';

interface CurrencyMonitorProps {
    data: CurrencyRate[];
    loading: boolean;
}

export const CurrencyMonitor: React.FC<CurrencyMonitorProps> = ({
    data,
    loading,
}) => {
    const [selectedPair, setSelectedPair] = useState<string | null>(null);
    const [historyData, setHistoryData] = useState<CurrencyHistory | null>(null);
    const [historyLoading, setHistoryLoading] = useState(false);
    const [timeRange, setTimeRange] = useState<30 | 90 | 180 | 365>(365);

    useEffect(() => {
        if (selectedPair) {
            loadHistory(selectedPair, timeRange);
        }
    }, [selectedPair, timeRange]);

    const loadHistory = async (pair: string, days: number) => {
        setHistoryLoading(true);
        try {
            const response = await fetchCurrencyHistory(pair, days);
            setHistoryData(response.history);
        } catch (error) {
            console.error('Error loading currency history:', error);
        } finally {
            setHistoryLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-[400px]">
                <div className="text-text-muted text-sm">Loading currency data...</div>
            </div>
        );
    }

    if (data.length === 0) {
        return (
            <div className="flex items-center justify-center h-[400px]">
                <div className="text-text-muted text-sm">No currency data available</div>
            </div>
        );
    }

    const formatRate = (rate: number) => {
        return rate.toFixed(4);
    };

    const formatChange = (change: number | undefined, changePct: number | undefined) => {
        if (change === undefined || changePct === undefined) return 'N/A';
        const sign = change >= 0 ? '+' : '';
        return `${sign}${change.toFixed(4)} (${sign}${changePct.toFixed(2)}%)`;
    };

    const getChangeColor = (change: number | undefined) => {
        if (change === undefined) return 'text-text-muted';
        return change >= 0 ? 'text-positive' : 'text-negative';
    };

    // Prepare chart data
    const chartData = historyData
        ? historyData.dates.map((date, index) => ({
              date: new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
              rate: historyData.rates[index],
          }))
        : [];

    return (
        <div className="space-y-6">
            {/* Currency List */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {data.map(rate => (
                    <div
                        key={rate.currency_pair}
                        onClick={() => setSelectedPair(rate.currency_pair)}
                        className={`p-3 rounded border cursor-pointer transition-all ${
                            selectedPair === rate.currency_pair
                                ? 'border-accent-cyan/50 bg-accent-cyan/10'
                                : 'border-white/10 hover:border-white/20 bg-white/5'
                        }`}
                    >
                        <div className="flex items-center justify-between mb-2">
                            <div className="font-mono font-bold text-sm text-white">
                                {rate.base_currency}/{rate.quote_currency}
                            </div>
                            <div className="text-xs text-text-muted">{rate.currency_pair}</div>
                        </div>
                        <div className="text-lg font-mono text-white mb-2">
                            {formatRate(rate.rate)}
                        </div>
                        <div className="space-y-1 text-xs">
                            <div className="flex justify-between">
                                <span className="text-text-muted">1D:</span>
                                <span className={getChangeColor(rate.change_1d)}>
                                    {formatChange(rate.change_1d, rate.change_pct_1d)}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-text-muted">1W:</span>
                                <span className={getChangeColor(rate.change_1w)}>
                                    {formatChange(rate.change_1w, rate.change_pct_1w)}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-text-muted">1M:</span>
                                <span className={getChangeColor(rate.change_1m)}>
                                    {formatChange(rate.change_1m, rate.change_pct_1m)}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-text-muted">1Y:</span>
                                <span className={getChangeColor(rate.change_1y)}>
                                    {formatChange(rate.change_1y, rate.change_pct_1y)}
                                </span>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Historical Chart */}
            {selectedPair && (
                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <h4 className="text-sm font-medium text-white">
                            Historical Trend: {selectedPair}
                        </h4>
                        <div className="flex gap-2">
                            {([30, 90, 180, 365] as const).map(days => (
                                <button
                                    key={days}
                                    onClick={() => setTimeRange(days)}
                                    className={`px-2 py-1 text-xs rounded transition-colors ${
                                        timeRange === days
                                            ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/30'
                                            : 'text-text-muted hover:text-text-secondary hover:bg-white/5'
                                    }`}
                                >
                                    {days === 30 ? '1M' : days === 90 ? '3M' : days === 180 ? '6M' : '1Y'}
                                </button>
                            ))}
                        </div>
                    </div>

                    {historyLoading ? (
                        <div className="flex items-center justify-center h-[300px]">
                            <div className="text-text-muted text-sm">Loading history...</div>
                        </div>
                    ) : chartData.length > 0 ? (
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                <XAxis
                                    dataKey="date"
                                    stroke="rgba(255,255,255,0.5)"
                                    tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 10 }}
                                    angle={-45}
                                    textAnchor="end"
                                    height={60}
                                />
                                <YAxis
                                    stroke="rgba(255,255,255,0.5)"
                                    tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 10 }}
                                    domain={['auto', 'auto']}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: 'rgba(0,0,0,0.9)',
                                        border: '1px solid rgba(255,255,255,0.2)',
                                        borderRadius: '4px',
                                    }}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="rate"
                                    stroke="#22d3ee"
                                    strokeWidth={2}
                                    dot={false}
                                    name="Exchange Rate"
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="flex items-center justify-center h-[300px]">
                            <div className="text-text-muted text-sm">No historical data available</div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};


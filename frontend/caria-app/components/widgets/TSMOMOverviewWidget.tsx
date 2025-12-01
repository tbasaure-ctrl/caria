import React, { useEffect, useState } from 'react';
import { TrendingUp } from 'lucide-react';
import { API_BASE_URL, getToken } from '../../services/apiService';

interface MarketTrend {
    symbol: string;
    name: string;
    trend: 'Bullish' | 'Bearish' | 'Neutral' | 'High Risk Bullish';
    strength: number;
    volatility: number;
}

const MARKETS = [
    { symbol: 'SPY', name: 'S&P 500' },
    { symbol: 'QQQ', name: 'Nasdaq 100' },
    { symbol: 'IWM', name: 'Russell 2000' },
    { symbol: 'GLD', name: 'Gold' },
    { symbol: 'USO', name: 'Oil' },
    { symbol: 'UUP', name: 'USD Index' },
    { symbol: 'BTCUSD', name: 'Bitcoin' },
];

export const TSMOMOverviewWidget: React.FC = () => {
    const [trends, setTrends] = useState<MarketTrend[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchTrends = async () => {
            try {
                const token = getToken();
                const headers: HeadersInit = { 'Content-Type': 'application/json' };
                if (token) headers['Authorization'] = `Bearer ${token}`;

                const promises = MARKETS.map(async (market) => {
                    try {
                        const res = await fetch(`${API_BASE_URL}/api/analysis/tsmom/${market.symbol}`, { headers });
                        if (res.ok) {
                            const data = await res.json();
                            return {
                                symbol: market.symbol,
                                name: market.name,
                                trend: data.trend_direction,
                                strength: data.trend_strength_12m,
                                volatility: data.annualized_volatility
                            };
                        }
                    } catch (e) {
                        return null;
                    }
                    return null;
                });

                const results = await Promise.all(promises);
                setTrends(results.filter((t): t is MarketTrend => t !== null));
            } finally {
                setLoading(false);
            }
        };
        fetchTrends();
    }, []);

    if (loading) return <div className="h-64 bg-bg-secondary border border-white/10 rounded-lg flex items-center justify-center text-text-muted animate-pulse">Scanning Global Markets...</div>;

    return (
        <div className="bg-bg-secondary border border-white/10 rounded-lg overflow-hidden">
            <div className="p-4 border-b border-white/10 flex items-center justify-between">
                <h3 className="text-sm font-display font-bold text-white">Global TSMOM Matrix</h3>
                <TrendingUp className="w-4 h-4 text-accent-primary" />
            </div>
            <div className="p-4 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
                {trends.map((market) => (
                    <div key={market.symbol} className="p-3 bg-white/5 rounded border border-white/5 hover:border-white/20 transition-colors">
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-xs font-bold text-white">{market.name}</span>
                            <div className={`w-2 h-2 rounded-full ${
                                market.trend === 'Bullish' ? 'bg-positive shadow-[0_0_6px_rgba(16,185,129,0.4)]' : 
                                market.trend === 'Bearish' ? 'bg-negative' : 'bg-warning'
                            }`} />
                        </div>
                        <div className="text-xs text-text-secondary space-y-1">
                            <div className="flex justify-between">
                                <span>Trend</span>
                                <span className={market.trend === 'Bullish' ? 'text-positive' : 'text-negative'}>
                                    {(market.strength * 100).toFixed(1)}%
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span>Vol</span>
                                <span className="text-text-muted">{(market.volatility * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
            <div className="px-4 py-2 bg-bg-tertiary text-[10px] text-text-muted border-t border-white/5">
                * 12-month Time Series Momentum vs. Annualized Volatility
            </div>
        </div>
    );
};


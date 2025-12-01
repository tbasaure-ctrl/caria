import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { fetchHoldingsWithPrices, HoldingsWithPrices, HoldingWithPrice, API_BASE_URL, getToken } from '../../services/apiService';
import { getGuestHoldings } from '../../services/guestStorageService';
import { MoreHorizontal, ArrowUpRight, ArrowDownRight, TrendingUp, AlertCircle } from 'lucide-react';

// TSMOM Status Dot Component
const TrendDot: React.FC<{ ticker: string }> = ({ ticker }) => {
    const [trend, setTrend] = useState<'Bullish' | 'Bearish' | 'Neutral' | 'High Risk Bullish' | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchTrend = async () => {
            try {
                // Avoid fetching if not a valid ticker
                if (!ticker || ticker.length > 10) return;
                
                const token = getToken();
                const headers: HeadersInit = { 'Content-Type': 'application/json' };
                if (token) headers['Authorization'] = `Bearer ${token}`;

                const response = await fetch(`${API_BASE_URL}/api/analysis/tsmom/${ticker}`, { headers });
                if (response.ok) {
                    const data = await response.json();
                    setTrend(data.trend_direction);
                }
            } catch (error) {
                // Silent fail for list view
            } finally {
                setLoading(false);
            }
        };
        fetchTrend();
    }, [ticker]);

    if (loading) return <div className="w-2 h-2 rounded-full bg-white/10 animate-pulse" />;

    let color = 'bg-gray-500';
    let tooltip = 'Neutral';

    if (trend === 'Bullish') { color = 'bg-positive shadow-[0_0_8px_rgba(16,185,129,0.4)]'; tooltip = 'Positive Trend'; }
    else if (trend === 'Bearish') { color = 'bg-negative'; tooltip = 'Negative Trend'; }
    else if (trend === 'High Risk Bullish') { color = 'bg-warning'; tooltip = 'High Risk Trend'; }

    return (
        <div className="group relative flex items-center justify-center w-full h-full">
            <div className={`w-2.5 h-2.5 rounded-full ${color} cursor-help`} />
            <div className="absolute bottom-full mb-2 hidden group-hover:block bg-black border border-white/10 px-2 py-1 text-[10px] rounded whitespace-nowrap z-10">
                {tooltip}
            </div>
        </div>
    );
};

// Helper to convert guest holdings (simplified for list view)
const convertGuestHoldings = (guestHoldings: any[]): HoldingsWithPrices => {
    return {
        holdings: guestHoldings.map(h => ({
            ...h,
            current_price: h.average_cost, // Mock
            current_value: h.quantity * h.average_cost,
            gain_loss: 0,
            gain_loss_pct: 0,
            cost_basis: h.quantity * h.average_cost,
            price_change: 0,
            price_change_pct: 0
        })),
        total_value: 0,
        total_cost: 0,
        total_gain_loss: 0,
        total_gain_loss_pct: 0
    };
};

export const PortfolioPage: React.FC = () => {
    const navigate = useNavigate();
    const [portfolioData, setPortfolioData] = useState<HoldingsWithPrices | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const loadData = async () => {
            setLoading(true);
            try {
                if (getToken()) {
                    const data = await fetchHoldingsWithPrices();
                    setPortfolioData(data);
                } else {
                    const guest = getGuestHoldings();
                    setPortfolioData(convertGuestHoldings(guest));
                }
            } catch (e) {
                console.error(e);
            } finally {
                setLoading(false);
            }
        };
        loadData();
    }, []);

    if (loading) return <div className="text-sm text-text-muted animate-pulse">Loading portfolio...</div>;

    const holdings = portfolioData?.holdings || [];

    return (
        <div className="animate-fade-in">
            {/* Header Stats */}
            <div className="mb-8 grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="p-4 rounded-lg border border-white/5 bg-bg-secondary">
                    <div className="text-[10px] text-text-muted uppercase tracking-widest mb-1">Total Equity</div>
                    <div className="text-2xl font-display text-white">
                        ${portfolioData?.total_value.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                    </div>
                </div>
                <div className="p-4 rounded-lg border border-white/5 bg-bg-secondary">
                    <div className="text-[10px] text-text-muted uppercase tracking-widest mb-1">Total Return</div>
                    <div className={`text-2xl font-mono ${portfolioData?.total_gain_loss_pct && portfolioData.total_gain_loss_pct >= 0 ? 'text-positive' : 'text-negative'}`}>
                        {portfolioData?.total_gain_loss_pct >= 0 ? '+' : ''}{portfolioData?.total_gain_loss_pct.toFixed(2)}%
                    </div>
                </div>
                <div className="p-4 rounded-lg border border-white/5 bg-bg-secondary flex items-center justify-between">
                    <div>
                        <div className="text-[10px] text-text-muted uppercase tracking-widest mb-1">Health Check</div>
                        <div className="text-sm text-text-secondary">TSMOM Signals Active</div>
                    </div>
                    <TrendingUp className="w-5 h-5 text-accent-primary opacity-50" />
                </div>
            </div>

            {/* GitHub-style Table */}
            <div className="border border-white/10 rounded-lg overflow-hidden bg-bg-secondary/50">
                {/* Table Header */}
                <div className="grid grid-cols-12 gap-4 px-4 py-3 bg-bg-tertiary border-b border-white/10 text-[10px] text-text-muted uppercase tracking-wider font-medium">
                    <div className="col-span-4 sm:col-span-3">Asset</div>
                    <div className="col-span-3 sm:col-span-2 text-right">Price</div>
                    <div className="col-span-3 sm:col-span-2 text-right">Value</div>
                    <div className="hidden sm:block sm:col-span-2 text-right">Return</div>
                    <div className="col-span-2 sm:col-span-1 text-center">12m Trend</div>
                    <div className="hidden sm:block sm:col-span-2 text-right">Action</div>
                </div>

                {/* Table Rows */}
                <div className="divide-y divide-white/5">
                    {holdings.map((holding) => (
                        <div 
                            key={holding.ticker}
                            onClick={() => navigate(`/analysis?ticker=${holding.ticker}`)}
                            className="grid grid-cols-12 gap-4 px-4 py-3 hover:bg-white/5 transition-colors cursor-pointer items-center group"
                        >
                            {/* Asset */}
                            <div className="col-span-4 sm:col-span-3 flex items-center gap-3">
                                <div className="w-8 h-8 rounded bg-white/5 flex items-center justify-center text-[10px] font-bold text-accent-cyan">
                                    {holding.ticker.substring(0, 2)}
                                </div>
                                <div>
                                    <div className="text-sm font-bold text-white font-mono">{holding.ticker}</div>
                                    <div className="text-[10px] text-text-muted">{holding.quantity} units</div>
                                </div>
                            </div>

                            {/* Price */}
                            <div className="col-span-3 sm:col-span-2 text-right text-sm font-mono text-text-secondary">
                                ${holding.current_price?.toFixed(2)}
                            </div>

                            {/* Value */}
                            <div className="col-span-3 sm:col-span-2 text-right">
                                <div className="text-sm font-mono text-white">
                                    ${holding.current_value?.toLocaleString()}
                                </div>
                            </div>

                            {/* Return (Hidden on Mobile) */}
                            <div className={`hidden sm:block sm:col-span-2 text-right text-sm font-mono ${holding.gain_loss_pct >= 0 ? 'text-positive' : 'text-negative'}`}>
                                {holding.gain_loss_pct >= 0 ? '+' : ''}{holding.gain_loss_pct.toFixed(2)}%
                            </div>

                            {/* 12m Trend (TSMOM) */}
                            <div className="col-span-2 sm:col-span-1 flex justify-center">
                                <TrendDot ticker={holding.ticker} />
                            </div>

                            {/* Action (Hidden on Mobile) */}
                            <div className="hidden sm:flex sm:col-span-2 justify-end">
                                <button className="p-1.5 rounded hover:bg-white/10 text-text-muted hover:text-white transition-colors opacity-0 group-hover:opacity-100">
                                    <ArrowUpRight className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                    ))}

                    {holdings.length === 0 && (
                        <div className="p-8 text-center text-text-muted text-sm">
                            No holdings found. Add assets to start tracking.
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};


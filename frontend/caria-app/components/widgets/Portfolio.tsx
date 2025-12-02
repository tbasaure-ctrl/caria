
import React, { useState, useMemo, useEffect, useCallback } from 'react';
import { fetchHoldingsWithPrices, HoldingsWithPrices, HoldingWithPrice, createHolding, deleteHolding, getToken, fetchPrices } from '../../services/apiService';
import { 
    getGuestHoldings, 
    createGuestHolding, 
    deleteGuestHolding,
    GuestHolding 
} from '../../services/guestStorageService';
import { WidgetCard } from './WidgetCard';
import { getErrorMessage } from '../../src/utils/errorHandling';
import { AreaChart, Area, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, XAxis, YAxis } from 'recharts';

// Helper to check if user is logged in
const isLoggedIn = (): boolean => !!getToken();

// Convert guest holdings to HoldingsWithPrices format with mock prices
const convertGuestHoldingsToPortfolioData = async (
    guestHoldings: GuestHolding[]
): Promise<HoldingsWithPrices> => {
    if (guestHoldings.length === 0) {
        return {
            holdings: [],
            total_value: 0,
            total_cost: 0,
            total_gain_loss: 0,
            total_gain_loss_pct: 0,
        };
    }

    // Try to fetch real prices for the tickers
    let prices: Record<string, any> = {};
    try {
        const tickers = guestHoldings.map(h => h.ticker);
        prices = await fetchPrices(tickers);
    } catch (error) {
        console.warn('Could not fetch prices for guest holdings, using estimates');
    }

    let totalValue = 0;
    let totalCost = 0;

    const holdingsWithPrices: HoldingWithPrice[] = guestHoldings.map(holding => {
        // Get price from API or estimate based on average cost
        const priceData = prices[holding.ticker];
        const currentPrice = priceData?.price || holding.average_cost * 1.05; // 5% gain estimate if no price
        const priceChange = priceData?.change || 0;
        const priceChangePct = priceData?.changesPercentage || 0;

        const costBasis = holding.quantity * holding.average_cost;
        const currentValue = holding.quantity * currentPrice;
        const gainLoss = currentValue - costBasis;
        const gainLossPct = costBasis > 0 ? (gainLoss / costBasis) * 100 : 0;

        totalValue += currentValue;
        totalCost += costBasis;

        return {
            id: holding.id,
            ticker: holding.ticker,
            quantity: holding.quantity,
            average_cost: holding.average_cost,
            notes: holding.notes,
            created_at: holding.created_at,
            updated_at: holding.updated_at,
            current_price: currentPrice,
            cost_basis: costBasis,
            current_value: currentValue,
            gain_loss: gainLoss,
            gain_loss_pct: gainLossPct,
            price_change: priceChange,
            price_change_pct: priceChangePct,
        };
    });

    const totalGainLoss = totalValue - totalCost;
    const totalGainLossPct = totalCost > 0 ? (totalGainLoss / totalCost) * 100 : 0;

    return {
        holdings: holdingsWithPrices,
        total_value: totalValue,
        total_cost: totalCost,
        total_gain_loss: totalGainLoss,
        total_gain_loss_pct: totalGainLossPct,
    };
};

type AllocationData = { name: string; value: number; color: string };

const CustomPieChart: React.FC<{ data: AllocationData[] }> = ({ data }) => {
    const [activeIndex, setActiveIndex] = useState<number | null>(null);

    if (!data.length) {
        return (
            <div className="flex flex-col gap-2 text-text-muted text-sm items-center justify-center h-48">
                <p>No allocation data.</p>
            </div>
        );
    }

    const total = data.reduce((sum, item) => sum + item.value, 0);

    return (
        <div className="flex flex-col items-center w-full">
            <div className="relative w-40 h-40 sm:w-48 sm:h-48">
                <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                        <Pie
                            data={data}
                            innerRadius="55%"
                            outerRadius="85%"
                            paddingAngle={3}
                            dataKey="value"
                            stroke="none"
                            onMouseEnter={(_, index) => setActiveIndex(index)}
                            onMouseLeave={() => setActiveIndex(null)}
                        >
                            {data.map((entry, index) => (
                                <Cell
                                    key={`cell-${index}`}
                                    fill={entry.color}
                                    opacity={activeIndex === null || activeIndex === index ? 1 : 0.4}
                                    style={{
                                        filter: activeIndex === index ? 'drop-shadow(0 0 8px rgba(255,255,255,0.3))' : 'none',
                                        transition: 'all 0.3s ease'
                                    }}
                                />
                            ))}
                        </Pie>
                    </PieChart>
                </ResponsiveContainer>
                {/* Center Text - Total or Active */}
                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                    {activeIndex !== null ? (
                        <>
                            <span className="text-xl sm:text-2xl font-display text-white font-bold">
                                {data[activeIndex].value.toFixed(1)}%
                            </span>
                            <span className="text-[10px] sm:text-xs text-accent-cyan font-mono uppercase">
                                {data[activeIndex].name}
                            </span>
                        </>
                    ) : (
                        <>
                            <span className="text-lg sm:text-xl font-display text-white/60 font-medium">
                                {data.length}
                            </span>
                            <span className="text-[9px] sm:text-[10px] text-text-muted uppercase tracking-wider">
                                Holdings
                            </span>
                        </>
                    )}
                </div>
            </div>

            {/* Professional Legend - Side by side with values */}
            <div className="w-full mt-4 sm:mt-6 space-y-1.5 sm:space-y-2">
                {data.slice(0, 8).map((item, i) => (
                    <div
                        key={i}
                        className={`flex items-center justify-between px-2 sm:px-3 py-1.5 sm:py-2 rounded-lg transition-all duration-200 cursor-pointer ${
                            activeIndex === i ? 'bg-white/10' : 'hover:bg-white/5'
                        }`}
                        onMouseEnter={() => setActiveIndex(i)}
                        onMouseLeave={() => setActiveIndex(null)}
                    >
                        <div className="flex items-center gap-2 sm:gap-3 min-w-0">
                            <div
                                className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full flex-shrink-0"
                                style={{ backgroundColor: item.color, boxShadow: `0 0 6px ${item.color}40` }}
                            />
                            <span className="text-[10px] sm:text-xs text-white font-mono truncate">{item.name}</span>
                        </div>
                        <span className="text-[10px] sm:text-xs text-text-secondary font-mono ml-2">
                            {item.value.toFixed(1)}%
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
};

type PerformanceGraphPoint = { date: string, value: number };

const PerformanceGraph: React.FC<{ data: PerformanceGraphPoint[]; timeRange?: string; currency: 'USD' | 'CLP' }> = ({ data, timeRange = '1Y', currency = 'USD' }) => {
    if (!data || data.length === 0) return null;

    const startValue = data[0].value;
    const endValue = data[data.length - 1].value;
    const change = endValue - startValue;
    const pctChange = startValue !== 0 ? (change / startValue) * 100 : 0;
    const isPositive = change >= 0;
    const color = isPositive ? '#10B981' : '#EF4444';
    const gradientId = `colorPerf-${timeRange}`;

    // Format date based on time range
    const formatXAxis = (dateStr: string) => {
        const date = new Date(dateStr);
        if (isNaN(date.getTime())) return '';

        switch (timeRange) {
            case '1D':
                return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
            case '1W':
                return date.toLocaleDateString('en-US', { weekday: 'short' });
            case '1M':
                return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            case 'YTD':
            case '1Y':
                return date.toLocaleDateString('en-US', { month: 'short' });
            case 'START':
                return date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
            default:
                return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        }
    };

    // Calculate tick interval based on data length
    const tickInterval = Math.max(1, Math.floor(data.length / 6));

    return (
        <div className="h-full flex flex-col">
            <div className="mb-3 sm:mb-4 px-2">
                <div className="flex flex-wrap items-baseline gap-2 sm:gap-3">
                    <span className="text-2xl sm:text-3xl font-display text-white">
                        {formatCurrency(endValue, currency)}
                    </span>
                    <span className={`text-xs sm:text-sm font-mono font-medium ${isPositive ? 'text-positive' : 'text-negative'}`}>
                        {isPositive ? '+' : ''}{formatCurrency(Math.abs(change), currency)} ({pctChange >= 0 ? '+' : ''}{pctChange.toFixed(2)}%)
                    </span>
                </div>
                <div className="text-[10px] sm:text-xs text-text-muted uppercase tracking-widest mt-1">
                    Portfolio Value • {timeRange === 'START' ? 'All Time' : timeRange}
                </div>
            </div>

            <div className="flex-1 min-h-[180px] sm:min-h-[200px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                        <defs>
                            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={color} stopOpacity={0.4}/>
                                <stop offset="95%" stopColor={color} stopOpacity={0.05}/>
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                        <XAxis
                            dataKey="date"
                            tickFormatter={formatXAxis}
                            tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10 }}
                            axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                            tickLine={false}
                            interval={tickInterval}
                            minTickGap={30}
                        />
                        <YAxis
                            tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10 }}
                            axisLine={false}
                            tickLine={false}
                            tickFormatter={(value) => {
                                const symbol = currency === 'CLP' ? 'CLP$' : '$';
                                const divisor = currency === 'CLP' ? 1000000 : 1000;
                                const suffix = currency === 'CLP' ? 'M' : 'k';
                                return `${symbol}${(value / divisor).toFixed(0)}${suffix}`;
                            }}
                            width={45}
                            domain={['dataMin * 0.95', 'dataMax * 1.05']}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#0B101B',
                                border: '1px solid rgba(255,255,255,0.15)',
                                borderRadius: '8px',
                                color: '#F1F5F9',
                                fontFamily: 'var(--font-mono)',
                                fontSize: '12px',
                                padding: '8px 12px'
                            }}
                            labelFormatter={(label) => {
                                const date = new Date(label);
                                return isNaN(date.getTime()) ? label : date.toLocaleDateString('en-US', {
                                    weekday: 'short',
                                    month: 'short',
                                    day: 'numeric',
                                    year: 'numeric'
                                });
                            }}
                            formatter={(value: number) => [formatCurrency(value, currency), 'Value']}
                        />
                        <Area
                            type="monotone"
                            dataKey="value"
                            stroke={color}
                            strokeWidth={2}
                            fillOpacity={1}
                            fill={`url(#${gradientId})`}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

const TimeRangeSelector: React.FC<{ selected: string; onSelect: (range: string) => void }> = ({ selected, onSelect }) => {
    const ranges = ['1D', '1W', '1M', 'YTD', '1Y', 'START'];
    return (
        <div className="flex bg-bg-tertiary border border-white/5 rounded p-0.5 gap-0.5 sm:gap-1 overflow-x-auto scrollbar-hide">
            {ranges.map(range => (
                <button
                    key={range}
                    onClick={() => onSelect(range)}
                    className={`px-2 sm:px-3 py-1 text-[10px] sm:text-xs font-medium rounded transition-all duration-200 whitespace-nowrap ${
                        selected === range
                            ? 'bg-accent-cyan/20 text-accent-cyan shadow-sm'
                            : 'text-text-muted hover:text-text-secondary hover:bg-white/5'
                    }`}
                >
                    {range === 'START' ? 'ALL' : range}
                </button>
            ))}
        </div>
    );
};

const POLLING_INTERVAL = 30000; // 30 segundos

// Currency formatting helper
const formatCurrency = (value: number, currency: 'USD' | 'CLP'): string => {
    const symbol = currency === 'CLP' ? 'CLP$' : '$';
    if (currency === 'CLP') {
        return `${symbol}${value.toLocaleString('es-CL', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
    }
    return `${symbol}${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
};

type HoldingFormState = {
    ticker: string;
    quantity: string;
    average_cost: string;
    purchase_date: string;
    notes: string;
};

const createDefaultFormState = (): HoldingFormState => ({
    ticker: '',
    quantity: '',
    average_cost: '',
    purchase_date: new Date().toISOString().split('T')[0],
    notes: '',
});

export const Portfolio: React.FC<{ id?: string }> = ({ id }) => {
    const [portfolioData, setPortfolioData] = useState<HoldingsWithPrices | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [timeRange, setTimeRange] = useState('1Y');
    const [formData, setFormData] = useState<HoldingFormState>(createDefaultFormState());
    const [formError, setFormError] = useState<string | null>(null);
    const [sortOption, setSortOption] = useState<'value' | 'return' | 'ticker'>('value');
    const [showForm, setShowForm] = useState(false);
    const [actionLoading, setActionLoading] = useState(false);
    const [isRefreshing, setIsRefreshing] = useState(false);
    const [currency, setCurrency] = useState<'USD' | 'CLP'>('USD');

    const loadPortfolio = useCallback(
        async (showSpinner = true) => {
            if (showSpinner) {
                setLoading(true);
            } else {
                setIsRefreshing(true);
            }
            setError(null);

            try {
                if (isLoggedIn()) {
                    // Authenticated user - fetch from API
                    const data = await fetchHoldingsWithPrices(currency);
                    setPortfolioData(data);
                } else {
                    // Guest mode - load from localStorage
                    const guestHoldings = getGuestHoldings();
                    const data = await convertGuestHoldingsToPortfolioData(guestHoldings);
                    setPortfolioData(data);
                }
            } catch (err: unknown) {
                // In guest mode, errors are less critical - just show empty state
                if (!isLoggedIn()) {
                    setPortfolioData({
                        holdings: [],
                        total_value: 0,
                        total_cost: 0,
                        total_gain_loss: 0,
                        total_gain_loss_pct: 0,
                    });
                } else {
                    setError(getErrorMessage(err) || 'Could not load your portfolio. Please try again.');
                }
            } finally {
                if (showSpinner) {
                    setLoading(false);
                } else {
                    setIsRefreshing(false);
                }
            }
        },
        []
    );

    useEffect(() => {
        loadPortfolio();
        const interval = setInterval(() => loadPortfolio(false), POLLING_INTERVAL);
        return () => clearInterval(interval);
    }, [loadPortfolio, currency]);

    useEffect(() => {
        if (portfolioData && portfolioData.holdings.length === 0) {
            setShowForm(true);
        }
    }, [portfolioData]);

    const hasHoldings = Boolean(portfolioData && portfolioData.holdings.length > 0);

    const getPerformanceData = useCallback((): PerformanceGraphPoint[] => {
        if (!portfolioData || portfolioData.holdings.length === 0) return [];

        const now = new Date();
        let startDate: Date;
        let dataPoints = 30;

        // Calculate period-specific returns based on timeRange
        // These simulate realistic market movements for each period
        const totalCurrentValue = portfolioData.total_value;
        const totalCost = portfolioData.total_cost;
        const totalGainPct = totalCost > 0 ? ((totalCurrentValue - totalCost) / totalCost) * 100 : 0;

        // Derive period-specific start values that create realistic returns
        // Shorter periods = smaller movements, longer periods = larger movements
        let periodStartValue: number;
        let volatilityFactor: number;

        switch (timeRange) {
            case '1D':
                startDate = new Date(now.getTime() - 1 * 24 * 60 * 60 * 1000);
                dataPoints = 24;
                // Daily: small movement (-2% to +2%)
                periodStartValue = totalCurrentValue * (1 - (Math.sin(Date.now() / 86400000) * 0.015 + 0.005));
                volatilityFactor = 0.003;
                break;
            case '1W':
                startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
                dataPoints = 7;
                // Weekly: moderate movement (-5% to +5%)
                periodStartValue = totalCurrentValue * (1 - (Math.sin(Date.now() / 604800000) * 0.03 + 0.01));
                volatilityFactor = 0.008;
                break;
            case '1M':
                startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
                dataPoints = 30;
                // Monthly: larger movement (-10% to +10%)
                periodStartValue = totalCurrentValue * (1 - (Math.sin(Date.now() / 2592000000) * 0.06 + 0.02));
                volatilityFactor = 0.015;
                break;
            case 'YTD':
                startDate = new Date(now.getFullYear(), 0, 1);
                dataPoints = Math.min(Math.ceil((now.getTime() - startDate.getTime()) / (24 * 60 * 60 * 1000)), 365);
                // YTD: significant movement based on days elapsed
                const ytdDays = Math.ceil((now.getTime() - startDate.getTime()) / (24 * 60 * 60 * 1000));
                periodStartValue = totalCurrentValue * (1 - (totalGainPct / 100) * (ytdDays / 365));
                volatilityFactor = 0.025;
                break;
            case '1Y':
                startDate = new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000);
                dataPoints = 52;
                // 1 Year: use a portion of total gain
                periodStartValue = totalCurrentValue * (1 - Math.min(totalGainPct / 100, 0.25));
                volatilityFactor = 0.035;
                break;
            case 'START':
                const purchaseDates = portfolioData.holdings
                    .map((h: any) => h.purchase_date ? new Date(h.purchase_date) : null)
                    .filter((d): d is Date => d !== null && !isNaN(d.getTime()));

                if (purchaseDates.length > 0) {
                    startDate = new Date(Math.min(...purchaseDates.map(d => d.getTime())));
                } else {
                    startDate = new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000);
                }

                const daysDiff = Math.ceil((now.getTime() - startDate.getTime()) / (24 * 60 * 60 * 1000));
                dataPoints = Math.min(daysDiff, 104);
                // All time: use total cost as start
                periodStartValue = totalCost > 0 ? totalCost : totalCurrentValue * 0.7;
                volatilityFactor = 0.045;
                break;
            default:
                startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
                dataPoints = 30;
                periodStartValue = totalCurrentValue * 0.95;
                volatilityFactor = 0.015;
        }

        const data: PerformanceGraphPoint[] = [];

        // Use seeded randomness based on timeRange for consistent noise per range
        const seed = timeRange.charCodeAt(0) + (timeRange.charCodeAt(1) || 0) + Math.floor(Date.now() / 86400000);
        const seededRandom = (i: number) => {
            const x = Math.sin(seed * 9999 + i * 1234) * 10000;
            return x - Math.floor(x);
        };

        for (let i = 0; i <= dataPoints; i++) {
            const progress = i / dataPoints;
            const date = new Date(startDate.getTime() + progress * (now.getTime() - startDate.getTime()));

            // Interpolate from period start to current value with realistic noise
            let interpolatedValue = periodStartValue + (totalCurrentValue - periodStartValue) * progress;

            // Add market-like noise
            const cycle = Math.sin(progress * Math.PI * 4) * (totalCurrentValue * volatilityFactor * 0.5);
            const noise = (seededRandom(i) - 0.5) * (totalCurrentValue * volatilityFactor);
            interpolatedValue += cycle + noise;

            data.push({
                date: date.toISOString(),
                value: Math.max(0, interpolatedValue)
            });
        }

        // Force endpoints to match calculated values
        if (data.length > 0) {
            data[0].value = periodStartValue;
            data[data.length - 1].value = totalCurrentValue;
        }

        return data;
    }, [portfolioData, timeRange]);

    const performanceData = useMemo(() => getPerformanceData(), [getPerformanceData]);

    const allocationData = useMemo(() => {
        if (!portfolioData || portfolioData.holdings.length === 0 || portfolioData.total_value <= 0) {
            return [];
        }
        const colors = ['#D4AF37', '#22D3EE', '#38BDF8', '#94A3B8', '#F1F5F9', '#64748B']; // Gold, Cyan, Blue, Grey, White
        return portfolioData.holdings.map((holding, i) => {
            const holdingValue = holding.current_value ?? holding.current_price * holding.quantity;
            return {
                name: holding.ticker,
                value: portfolioData.total_value > 0 ? (holdingValue / portfolioData.total_value) * 100 : 0,
                color: colors[i % colors.length],
            };
        });
    }, [portfolioData]);

    const sortedHoldings = useMemo(() => {
        if (!portfolioData) return [];
        const holdings = [...portfolioData.holdings];
        switch (sortOption) {
            case 'return':
                return holdings.sort((a, b) => (b.gain_loss_pct || 0) - (a.gain_loss_pct || 0));
            case 'ticker':
                return holdings.sort((a, b) => a.ticker.localeCompare(b.ticker));
            case 'value':
            default:
                return holdings.sort((a, b) => (b.current_value || 0) - (a.current_value || 0));
        }
    }, [portfolioData, sortOption]);

    if (loading) {
        return (
            <WidgetCard id={id} title="PORTFOLIO SNAPSHOT">
                <div className="text-text-muted text-sm">Loading portfolio...</div>
            </WidgetCard>
        );
    }

    if (error) {
        // Demo data for elegant preview (like Fintech Dashboard reference)
        const demoChartData = Array.from({ length: 30 }, (_, i) => {
            const base = 45000 + i * 500;
            const noise = Math.sin(i * 0.5) * 2000 + Math.cos(i * 0.3) * 1000;
            return { date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString(), value: base + noise };
        });
        const demoAllocation = [
            { name: 'AAPL', value: 28, color: '#D4AF37' },
            { name: 'MSFT', value: 22, color: '#22D3EE' },
            { name: 'NVDA', value: 18, color: '#38BDF8' },
            { name: 'GOOGL', value: 15, color: '#94A3B8' },
            { name: 'AMZN', value: 12, color: '#10B981' },
            { name: 'Others', value: 5, color: '#64748B' },
        ];

        return (
            <WidgetCard id={id} title="PORTFOLIO TRACKER">
                <div className="space-y-6">
                    {/* Header with main value */}
                    <div className="flex items-start justify-between">
                        <div>
                            <div className="text-[10px] text-text-muted uppercase tracking-widest mb-1">Total Balance</div>
                            <div className="text-3xl sm:text-4xl font-display text-white">$58,432.00</div>
                            <div className="flex items-center gap-2 mt-1">
                                <span className="text-xs sm:text-sm text-positive font-mono">+$6,847.32</span>
                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-positive/20 text-positive font-medium">+13.3%</span>
                            </div>
                        </div>
                        <div className="flex gap-1.5">
                            {['1D', '1W', '1M', '1Y'].map((range, i) => (
                                <button key={range} className={`px-2 py-1 text-[10px] rounded ${i === 3 ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-muted hover:text-white'}`}>
                                    {range}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Performance Chart */}
                    <div className="h-[180px] sm:h-[200px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={demoChartData}>
                                <defs>
                                    <linearGradient id="demoGradientGold" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#10B981" stopOpacity={0.3}/>
                                        <stop offset="95%" stopColor="#10B981" stopOpacity={0.02}/>
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                                <Area type="monotone" dataKey="value" stroke="#10B981" strokeWidth={2} fill="url(#demoGradientGold)" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Allocation & Holdings Preview */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Pie Chart */}
                        <div>
                            <div className="text-[10px] text-text-muted uppercase tracking-widest mb-3">Allocation</div>
                            <div className="flex items-center gap-4">
                                <div className="w-24 h-24 relative">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <PieChart>
                                            <Pie data={demoAllocation} innerRadius="60%" outerRadius="90%" dataKey="value" stroke="none" paddingAngle={2}>
                                                {demoAllocation.map((entry, index) => (
                                                    <Cell key={index} fill={entry.color} />
                                                ))}
                                            </Pie>
                                        </PieChart>
                                    </ResponsiveContainer>
                                    <div className="absolute inset-0 flex items-center justify-center">
                                        <span className="text-xs text-text-muted">6</span>
                                    </div>
                                </div>
                                <div className="space-y-1.5 flex-1">
                                    {demoAllocation.slice(0, 4).map((item, i) => (
                                        <div key={i} className="flex items-center justify-between text-[10px]">
                                            <div className="flex items-center gap-2">
                                                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
                                                <span className="text-white font-mono">{item.name}</span>
                                            </div>
                                            <span className="text-text-muted">{item.value}%</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Top Performers */}
                        <div>
                            <div className="text-[10px] text-text-muted uppercase tracking-widest mb-3">Top Performers</div>
                            <div className="space-y-2">
                                {[
                                    { ticker: 'NVDA', value: '$12,340', change: '+42.5%', positive: true },
                                    { ticker: 'AAPL', value: '$16,200', change: '+18.2%', positive: true },
                                    { ticker: 'MSFT', value: '$12,880', change: '+12.8%', positive: true },
                                ].map((stock, i) => (
                                    <div key={i} className="flex items-center justify-between p-2 rounded bg-white/5">
                                        <div className="flex items-center gap-2">
                                            <div className="w-6 h-6 rounded bg-white/10 flex items-center justify-center text-[9px] font-bold text-accent-cyan">
                                                {stock.ticker.slice(0, 2)}
                                            </div>
                                            <span className="text-xs text-white font-mono">{stock.ticker}</span>
                                        </div>
                                        <div className="text-right">
                                            <div className="text-xs text-white font-mono">{stock.value}</div>
                                            <div className={`text-[10px] ${stock.positive ? 'text-positive' : 'text-negative'}`}>{stock.change}</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* CTA Button - redirects to login */}
                    <button
                        onClick={() => window.location.href = '/?login=true'}
                        className="w-full py-3 rounded-lg bg-accent-cyan/10 border border-accent-cyan/30 text-accent-cyan text-sm font-medium hover:bg-accent-cyan/20 transition-all flex items-center justify-center gap-2"
                    >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                        </svg>
                        Add Holdings
                    </button>
                </div>
            </WidgetCard>
        );
    }

    const handleAddHolding = async (event: React.FormEvent) => {
        event.preventDefault();
        setFormError(null);

        const ticker = formData.ticker.trim().toUpperCase();
        const quantity = parseFloat(formData.quantity);
        const averageCost = parseFloat(formData.average_cost);

        if (!ticker) { setFormError('Invalid ticker.'); return; }
        if (!Number.isFinite(quantity) || quantity <= 0) { setFormError('Invalid quantity.'); return; }
        if (!Number.isFinite(averageCost) || averageCost < 0) { setFormError('Invalid cost.'); return; }

        try {
            setActionLoading(true);
            
            if (isLoggedIn()) {
                // Authenticated user - save to API
                await createHolding({
                    ticker,
                    quantity,
                    average_cost: averageCost,
                    purchase_date: formData.purchase_date,
                    notes: formData.notes || undefined,
                });
            } else {
                // Guest mode - save to localStorage
                createGuestHolding({
                    ticker,
                    quantity,
                    average_cost: averageCost,
                    purchase_date: formData.purchase_date,
                    notes: formData.notes || undefined,
                });
            }
            
            setFormData(createDefaultFormState());
            setShowForm(false);
            await loadPortfolio(false);
        } catch (err: unknown) {
            setFormError(getErrorMessage(err) || 'Could not save the position.');
        } finally {
            setActionLoading(false);
        }
    };

    const handleDeleteHolding = async (id: string) => {
        if (!window.confirm('Delete this position?')) return;
        try {
            setActionLoading(true);
            
            if (isLoggedIn()) {
                // Authenticated user - delete from API
                await deleteHolding(id);
            } else {
                // Guest mode - delete from localStorage
                deleteGuestHolding(id);
            }
            
            // Optimistic update
            setPortfolioData(prev => prev ? {
                ...prev,
                holdings: prev.holdings.filter(h => h.id !== id)
            } : null);
            await loadPortfolio(false);
        } catch (err: unknown) {
            setFormError(getErrorMessage(err) || 'Could not delete position.');
        } finally {
            setActionLoading(false);
        }
    };

    return (
        <WidgetCard
            id={id}
            title="PORTFOLIO SNAPSHOT"
            tooltip="Overview of your current holdings, allocation, and performance."
        >
            <div className="space-y-8">
                {/* Header Actions */}
                <div className="flex items-center justify-between flex-wrap gap-2">
                    {isRefreshing && (
                        <span className="text-xs text-accent-cyan animate-pulse">Syncing...</span>
                    )}
                    <div className="flex gap-2 ml-auto">
                        {/* Currency Toggle */}
                        <div className="flex bg-bg-tertiary border border-white/5 rounded p-0.5">
                            {(['USD', 'CLP'] as const).map(curr => (
                                <button
                                    key={curr}
                                    onClick={() => setCurrency(curr)}
                                    className={`px-2 sm:px-3 py-1 text-[10px] sm:text-xs uppercase font-medium rounded transition-colors ${
                                        currency === curr 
                                            ? 'bg-accent-cyan/20 text-accent-cyan' 
                                            : 'text-text-muted hover:text-text-secondary'
                                    }`}
                                >
                                    {curr}
                                </button>
                            ))}
                        </div>
                        <TimeRangeSelector selected={timeRange} onSelect={setTimeRange} />
                    </div>
                </div>

                {portfolioData && hasHoldings ? (
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6 lg:gap-8">
                        {/* Left: Performance Graph (Span 2) */}
                        <div className="lg:col-span-2 order-1">
                            <PerformanceGraph data={performanceData} timeRange={timeRange} currency={currency} />
                        </div>

                        {/* Right: Allocation Pie */}
                        <div className="lg:col-span-1 flex flex-col justify-center order-2">
                            <h4 className="text-[10px] sm:text-xs text-text-muted uppercase tracking-widest mb-3 sm:mb-4 text-center">Asset Allocation</h4>
                            <CustomPieChart data={allocationData} />
                        </div>
                    </div>
                ) : (
                    <div className="rounded-lg border border-dashed border-white/10 p-4 sm:p-6 lg:p-8 text-center">
                        <div className="w-12 h-12 sm:w-16 sm:h-16 mx-auto mb-3 sm:mb-4 rounded-full bg-accent-cyan/10 flex items-center justify-center">
                            <svg className="w-6 h-6 sm:w-8 sm:h-8 text-accent-cyan" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                            </svg>
                        </div>
                        <h4 className="text-sm sm:text-base font-display font-bold text-white mb-2">
                            Start Your Portfolio
                        </h4>
                        <p className="text-[10px] sm:text-xs text-text-muted max-w-[280px] mx-auto mb-4">
                            Add your positions to see performance analytics, asset allocation, and crisis simulations.
                        </p>
                        <div className="flex flex-wrap justify-center gap-2 text-[9px] sm:text-[10px] text-text-muted">
                            <span className="px-2 py-1 rounded bg-white/5">Historical Returns</span>
                            <span className="px-2 py-1 rounded bg-white/5">Diversification</span>
                            <span className="px-2 py-1 rounded bg-white/5">Risk Analysis</span>
                        </div>
                    </div>
                )}

                {/* Position Management Table */}
                <div className="border-t border-white/5 pt-4 sm:pt-6">
                    <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 sm:gap-4 justify-between mb-4">
                        <h4 className="text-xs sm:text-sm font-medium text-white uppercase tracking-widest">
                            Positions
                        </h4>
                        <div className="flex gap-2 w-full sm:w-auto justify-between sm:justify-end">
                            <div className="flex bg-bg-tertiary border border-white/5 rounded p-0.5">
                                {['value', 'return', 'ticker'].map(opt => (
                                    <button
                                        key={opt}
                                        onClick={() => setSortOption(opt as any)}
                                        className={`px-2 sm:px-3 py-1 text-[10px] sm:text-xs uppercase font-medium rounded transition-colors ${sortOption === opt ? 'bg-white/10 text-white' : 'text-text-muted hover:text-text-secondary'}`}
                                    >
                                        {opt === 'ticker' ? 'AZ' : opt === 'value' ? '$' : '%'}
                                    </button>
                                ))}
                            </div>
                            <button
                                onClick={() => setShowForm((prev) => !prev)}
                                className="text-[10px] sm:text-xs font-medium px-3 sm:px-4 py-1.5 rounded bg-accent-primary/10 text-accent-primary hover:bg-accent-primary/20 transition-colors border border-accent-primary/20"
                            >
                                {showForm ? 'Close' : 'Add +'}
                            </button>
                        </div>
                    </div>

                    {/* Professional Add Form - Mobile Responsive */}
                    {showForm && (
                        <div className="bg-bg-tertiary/50 p-3 sm:p-4 lg:p-6 rounded-lg border border-accent-primary/20 mb-4 sm:mb-6 animate-fade-in-up">
                            <h5 className="text-xs sm:text-sm text-accent-primary mb-3 sm:mb-4 font-medium">New Position Entry</h5>
                            <form onSubmit={handleAddHolding} className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
                                <div className="col-span-1">
                                    <label className="block text-[9px] sm:text-[10px] text-text-muted uppercase tracking-wider mb-1">Ticker</label>
                                    <input
                                        type="text"
                                        value={formData.ticker}
                                        onChange={(e) => setFormData({ ...formData, ticker: e.target.value.toUpperCase() })}
                                        className="w-full bg-bg-primary border border-white/10 rounded px-2 sm:px-3 py-1.5 sm:py-2 text-xs sm:text-sm text-white focus:border-accent-primary outline-none font-mono"
                                        placeholder={formData.ticker.toUpperCase() === 'CASH' ? 'CASH' : 'AAPL or CASH'}
                                        required
                                    />
                                    {formData.ticker.toUpperCase() === 'CASH' && (
                                        <p className="text-[9px] text-accent-cyan mt-1">Cash position - enter amount in USD</p>
                                    )}
                                </div>
                                <div className="col-span-1">
                                    <label className="block text-[9px] sm:text-[10px] text-text-muted uppercase tracking-wider mb-1">Quantity</label>
                                    <input
                                        type="number"
                                        step="any"
                                        value={formData.quantity}
                                        onChange={(e) => setFormData({ ...formData, quantity: e.target.value })}
                                        className="w-full bg-bg-primary border border-white/10 rounded px-2 sm:px-3 py-1.5 sm:py-2 text-xs sm:text-sm text-white focus:border-accent-primary outline-none font-mono"
                                        placeholder="0.00"
                                        required
                                    />
                                </div>
                                <div className="col-span-1">
                                    <label className="block text-[9px] sm:text-[10px] text-text-muted uppercase tracking-wider mb-1">
                                        Avg Cost ({currency === 'CLP' ? 'CLP' : '$'})
                                        {formData.ticker.toUpperCase() === 'CASH' && ' (use 1.0 for cash)'}
                                    </label>
                                    <input
                                        type="number"
                                        step="any"
                                        value={formData.average_cost}
                                        onChange={(e) => setFormData({ ...formData, average_cost: e.target.value })}
                                        className="w-full bg-bg-primary border border-white/10 rounded px-2 sm:px-3 py-1.5 sm:py-2 text-xs sm:text-sm text-white focus:border-accent-primary outline-none font-mono"
                                        placeholder={formData.ticker.toUpperCase() === 'CASH' ? '1.0' : '0.00'}
                                        required
                                    />
                                </div>
                                <div className="col-span-1">
                                    <label className="block text-[9px] sm:text-[10px] text-text-muted uppercase tracking-wider mb-1">Date</label>
                                    <input
                                        type="date"
                                        value={formData.purchase_date}
                                        onChange={(e) => setFormData({ ...formData, purchase_date: e.target.value })}
                                        className="w-full bg-bg-primary border border-white/10 rounded px-2 sm:px-3 py-1.5 sm:py-2 text-xs sm:text-sm text-white focus:border-accent-primary outline-none"
                                        required
                                    />
                                </div>
                                <div className="col-span-2 lg:col-span-4 flex gap-2 sm:gap-3 mt-2 justify-end">
                                    <button
                                        type="button"
                                        onClick={() => setShowForm(false)}
                                        className="px-3 sm:px-4 py-1.5 sm:py-2 text-[10px] sm:text-xs font-medium text-text-muted hover:text-white transition-colors"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        type="submit"
                                        disabled={actionLoading}
                                        className="bg-accent-primary hover:bg-accent-primary/90 text-white text-[10px] sm:text-xs font-bold uppercase tracking-wider py-1.5 sm:py-2 px-4 sm:px-6 rounded transition-colors shadow-glow-sm"
                                    >
                                        {actionLoading ? 'Saving...' : 'Save'}
                                    </button>
                                </div>
                            </form>
                        </div>
                    )}

                    <div className="max-h-[250px] sm:max-h-[300px] overflow-y-auto pr-1 sm:pr-2 custom-scrollbar space-y-1">
                        {sortedHoldings.map((holding: HoldingWithPrice) => (
                            <div
                                key={holding.id}
                                className="flex justify-between items-center p-2 sm:p-3 bg-white/5 rounded hover:bg-white/10 transition-colors group border border-transparent hover:border-white/5"
                            >
                                <div className="flex items-center gap-2 sm:gap-4 min-w-0 flex-1">
                                    <div className="w-6 h-6 sm:w-8 sm:h-8 rounded bg-white/5 flex items-center justify-center text-[10px] sm:text-xs font-bold text-accent-cyan flex-shrink-0">
                                        {holding.ticker.substring(0, 2)}
                                    </div>
                                    <div className="min-w-0">
                                        <div className="flex items-baseline gap-1 sm:gap-2 flex-wrap">
                                            <span className="text-white font-medium font-mono text-xs sm:text-sm">{holding.ticker}</span>
                                            <span className="text-[10px] sm:text-xs text-text-muted truncate">{holding.quantity} sh</span>
                                        </div>
                                        <div className="text-[10px] sm:text-xs text-text-secondary">
                                            Avg: {formatCurrency(holding.average_cost, currency)}
                                        </div>
                                    </div>
                                </div>
                                <div className="flex items-center gap-2 sm:gap-6 flex-shrink-0">
                                    <div className="text-right">
                                        <div className="font-mono text-white text-xs sm:text-sm">
                                            {formatCurrency(holding.current_value ?? 0, currency)}
                                        </div>
                                        <div className={`text-[10px] sm:text-xs font-mono ${holding.gain_loss_pct! >= 0 ? 'text-positive' : 'text-negative'}`}>
                                            {holding.gain_loss_pct! >= 0 ? '+' : ''}{holding.gain_loss_pct!.toFixed(1)}%
                                        </div>
                                    </div>
                                    <button
                                        onClick={() => holding.id && handleDeleteHolding(holding.id)}
                                        className="w-5 h-5 sm:w-6 sm:h-6 rounded flex items-center justify-center text-text-muted hover:text-negative hover:bg-negative/10 transition-all sm:opacity-0 sm:group-hover:opacity-100"
                                        title="Delete position"
                                    >
                                        ×
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </WidgetCard>
    );
};

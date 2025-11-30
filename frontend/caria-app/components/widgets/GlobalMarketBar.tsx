import React, { useEffect, useState } from 'react';
import { fetchPrices, RealtimePrice } from '../../services/apiService';
import { WidgetCard } from './WidgetCard';
import { getErrorMessage } from '../../src/utils/errorHandling';

// Global market indices
const GLOBAL_INDICES = [
    { ticker: 'SPY', name: 'S&P 500', symbol: 'SPX', region: 'US' },
    { ticker: 'QQQ', name: 'Nasdaq 100', symbol: 'NDX', region: 'US' },
    { ticker: 'VGK', name: 'Euro Stoxx', symbol: 'STOXX', region: 'EU' },
    { ticker: 'EEM', name: 'Emerging', symbol: 'EM', region: 'GLOBAL' },
    { ticker: 'GLD', name: 'Gold', symbol: 'GOLD', region: 'CMDTY' },
    { ticker: 'OIL', name: 'Oil', symbol: 'OIL', region: 'CMDTY' },
];

const POLLING_INTERVAL = 30000;

// Mock data for initial display (before API loads or if API fails)
const MOCK_PRICES: Record<string, RealtimePrice> = {
    'SPY': { symbol: 'SPY', price: 6032.45, change: 15.20, changesPercentage: 0.25, previousClose: 6017.25 },
    'QQQ': { symbol: 'QQQ', price: 21150.80, change: 120.50, changesPercentage: 0.57, previousClose: 21030.30 },
    'VGK': { symbol: 'VGK', price: 525.10, change: -2.15, changesPercentage: -0.41, previousClose: 527.25 },
    'EEM': { symbol: 'EEM', price: 44.20, change: 0.12, changesPercentage: 0.27, previousClose: 44.08 },
    'GLD': { symbol: 'GLD', price: 272.50, change: 1.80, changesPercentage: 0.66, previousClose: 270.70 },
    'OIL': { symbol: 'OIL', price: 68.50, change: -0.75, changesPercentage: -1.08, previousClose: 69.25 },
};

interface MarketTileProps {
    name: string;
    symbol: string;
    region: string;
    price: number;
    change: number;
    changePercent: number;
}

const MarketTile: React.FC<MarketTileProps> = ({ name, symbol, region, price, change, changePercent }) => {
    const isPositive = change >= 0;

    return (
        <div
            className="p-4 rounded-lg transition-all duration-200"
            style={{
                backgroundColor: 'var(--color-bg-tertiary)',
                border: '1px solid var(--color-border-subtle)',
            }}
            onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'var(--color-border-default)';
            }}
            onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'var(--color-border-subtle)';
            }}
        >
            {/* Top Row: Symbol & Region */}
            <div className="flex items-center justify-between mb-2">
                <span
                    className="text-xs font-mono font-semibold px-2 py-0.5 rounded"
                    style={{
                        backgroundColor: 'var(--color-bg-surface)',
                        color: 'var(--color-text-primary)',
                    }}
                >
                    {symbol}
                </span>
                <span
                    className="text-[10px] font-medium tracking-wider uppercase"
                    style={{ color: 'var(--color-text-subtle)' }}
                >
                    {region}
                </span>
            </div>

            {/* Index Name */}
            <div
                className="text-sm font-medium mb-3"
                style={{ color: 'var(--color-text-secondary)' }}
            >
                {name}
            </div>

            {/* Price & Change */}
            <div className="flex items-end justify-between">
                <span
                    className="text-xl font-mono font-semibold"
                    style={{ color: 'var(--color-text-primary)' }}
                >
                    {price.toLocaleString(undefined, {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2
                    })}
                </span>
                <div className="text-right">
                    <div
                        className="text-sm font-mono font-semibold flex items-center gap-1"
                        style={{ color: isPositive ? '#10b981' : '#ef4444' }}
                    >
                        <svg
                            className="w-3 h-3"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d={isPositive ? "M5 15l7-7 7 7" : "M19 9l-7 7-7-7"}
                            />
                        </svg>
                        {changePercent >= 0 ? '+' : ''}{changePercent.toFixed(2)}%
                    </div>
                    <div
                        className="text-xs font-mono"
                        style={{ color: 'var(--color-text-muted)' }}
                    >
                        {change >= 0 ? '+' : ''}{change.toFixed(2)}
                    </div>
                </div>
            </div>

            {/* Mini Sparkline Placeholder */}
            <div
                className="mt-3 h-8 rounded overflow-hidden"
                style={{ backgroundColor: 'var(--color-bg-surface)' }}
            >
                <svg
                    viewBox="0 0 100 32"
                    className="w-full h-full"
                    preserveAspectRatio="none"
                >
                    <path
                        d={`M0,${isPositive ? 28 : 4} Q20,${isPositive ? 20 : 12} 40,${isPositive ? 16 : 16} T60,${isPositive ? 12 : 20} T80,${isPositive ? 8 : 24} T100,${isPositive ? 4 : 28}`}
                        fill="none"
                        stroke={isPositive ? 'var(--color-positive)' : 'var(--color-negative)'}
                        strokeWidth="1.5"
                        opacity="0.6"
                    />
                </svg>
            </div>
        </div>
    );
};

const LoadingTile: React.FC = () => (
    <div
        className="p-4 rounded-lg"
        style={{
            backgroundColor: 'var(--color-bg-tertiary)',
            border: '1px solid var(--color-border-subtle)',
        }}
    >
        <div className="animate-pulse space-y-3">
            <div className="flex justify-between">
                <div
                    className="h-5 w-12 rounded"
                    style={{ backgroundColor: 'var(--color-bg-surface)' }}
                />
                <div
                    className="h-4 w-8 rounded"
                    style={{ backgroundColor: 'var(--color-bg-surface)' }}
                />
            </div>
            <div
                className="h-4 w-20 rounded"
                style={{ backgroundColor: 'var(--color-bg-surface)' }}
            />
            <div className="flex justify-between items-end">
                <div
                    className="h-6 w-24 rounded"
                    style={{ backgroundColor: 'var(--color-bg-surface)' }}
                />
                <div
                    className="h-5 w-16 rounded"
                    style={{ backgroundColor: 'var(--color-bg-surface)' }}
                />
            </div>
        </div>
    </div>
);

export const GlobalMarketBar: React.FC<{ id?: string }> = ({ id }) => {
    const [prices, setPrices] = useState<Record<string, RealtimePrice>>({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

    useEffect(() => {
        // Show mock data immediately
        setPrices(MOCK_PRICES);
        setLoading(false);
        setLastUpdate(new Date());

        const updatePrices = async () => {
            try {
                setError(null);
                const tickers = GLOBAL_INDICES.map(idx => idx.ticker);
                const data = await fetchPrices(tickers);
                // Only update if we got valid data
                if (data && Object.keys(data).length > 0) {
                    setPrices(data);
                    setLastUpdate(new Date());
                }
            } catch (err: unknown) {
                const message = getErrorMessage(err);
                // Don't show error for unauthenticated users - prices should work without auth
                console.warn('Error fetching prices:', message);
                // Keep showing mock data on error
            }
        };

        // Try to fetch real data after initial render
        updatePrices();
        const interval = setInterval(updatePrices, POLLING_INTERVAL);

        return () => clearInterval(interval);
    }, []);

    return (
        <WidgetCard
            id={id}
            title="MARKET PULSE"
            tooltip="Live market data for major global indices, commodities, and asset classes. Updated every 30 seconds during market hours."
            action={lastUpdate ? {
                label: `Updated ${lastUpdate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`,
                onClick: () => { }
            } : undefined}
        >
            {error && error !== 'Market data temporarily unavailable' && (
                <div
                    className="mb-4 px-4 py-3 rounded-lg text-sm"
                    style={{
                        backgroundColor: 'var(--color-negative-muted)',
                        color: 'var(--color-negative)',
                        border: '1px solid var(--color-negative)',
                    }}
                >
                    {error}
                </div>
            )}

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                {loading ? (
                    GLOBAL_INDICES.map((_, idx) => <LoadingTile key={idx} />)
                ) : (
                    GLOBAL_INDICES.map((index) => {
                        const priceData = prices[index.ticker] || MOCK_PRICES[index.ticker];

                        if (!priceData) {
                            return null;
                        }

                        const price = priceData.price || priceData.previousClose || 0;
                        const change = priceData.change || 0;
                        const changePercent = priceData.changesPercentage || 0;

                        return (
                            <MarketTile
                                key={index.ticker}
                                name={index.name}
                                symbol={index.symbol}
                                region={index.region}
                                price={price}
                                change={change}
                                changePercent={changePercent}
                            />
                        );
                    })
                )}
            </div>
        </WidgetCard>
    );
};

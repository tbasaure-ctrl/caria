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
    { ticker: 'TLT', name: 'Treasuries', symbol: 'TLT', region: 'BOND' },
];

const POLLING_INTERVAL = 30000;

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
                        style={{ color: isPositive ? 'var(--color-positive)' : 'var(--color-negative)' }}
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
        const updatePrices = async () => {
            try {
                setError(null);
                const tickers = GLOBAL_INDICES.map(idx => idx.ticker);
                const data = await fetchPrices(tickers);
                setPrices(data);
                setLastUpdate(new Date());
                setLoading(false);
            } catch (err: unknown) {
                const message = getErrorMessage(err);
                if (message.includes('401') || message.includes('403')) {
                    setError('Please log in to view market data');
                } else {
                    setError('Market data temporarily unavailable');
                }
                setLoading(false);
            }
        };

        updatePrices();
        const interval = setInterval(updatePrices, POLLING_INTERVAL);

        return () => clearInterval(interval);
    }, []);

    return (
        <WidgetCard
            id={id}
            title="GLOBAL MARKETS"
            tooltip="Live market data for major global indices, commodities, and asset classes. Updated every 30 seconds during market hours."
            action={lastUpdate ? {
                label: `Updated ${lastUpdate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`,
                onClick: () => {}
            } : undefined}
        >
            {error && (
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
                        const priceData = prices[index.ticker];
                        
                        if (!priceData) {
                            return (
                                <div 
                                    key={index.ticker}
                                    className="p-4 rounded-lg"
                                    style={{
                                        backgroundColor: 'var(--color-bg-tertiary)',
                                        border: '1px solid var(--color-border-subtle)',
                                    }}
                                >
                                    <div className="flex items-center justify-between mb-2">
                                        <span 
                                            className="text-xs font-mono font-semibold"
                                            style={{ color: 'var(--color-text-muted)' }}
                                        >
                                            {index.symbol}
                                        </span>
                                    </div>
                                    <div 
                                        className="text-sm"
                                        style={{ color: 'var(--color-text-muted)' }}
                                    >
                                        No data
                                    </div>
                                </div>
                            );
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

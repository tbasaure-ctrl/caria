
import React, { useEffect, useState } from 'react';
import { fetchPrices, RealtimePrice } from '../../services/apiService';
import { ArrowUpIcon, ArrowDownIcon } from '../Icons';
import { WidgetCard } from './WidgetCard';
import { getErrorMessage } from '../../src/utils/errorHandling';

// Principales índices globales (ETFs que representan índices)
// Usando ETFs reales que se pueden obtener de FMP API
const GLOBAL_INDICES = [
    { ticker: 'SPY', name: 'S&P 500', region: 'USA' },
    { ticker: 'VGK', name: 'STOXX 600', region: 'Europe' }, // VGK es un ETF europeo amplio
    { ticker: 'EEM', name: 'Emerging Markets', region: 'Chile' }, // EEM para mercados emergentes
];

const POLLING_INTERVAL = 30000; // 30 segundos

export const GlobalMarketBar: React.FC<{id?: string}> = ({id}) => {
    const [prices, setPrices] = useState<Record<string, RealtimePrice>>({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const updatePrices = async () => {
            try {
                setError(null);
                const tickers = GLOBAL_INDICES.map(idx => idx.ticker);
                const data = await fetchPrices(tickers);
                setPrices(data);
                setLoading(false);
            } catch (err: unknown) {
                const message = getErrorMessage(err);
                // Check if it's an authentication error
                if (message.includes('401') || message.includes('403')) {
                    setError('Please log in to view market data');
                } else if (message.includes('Failed to connect') || message.includes('network')) {
                    setError('Unable to connect to market data service');
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

    if (loading) {
        return (
            <WidgetCard
                id={id}
                title="GLOBAL MARKETS"
                tooltip="Principales índices de mercado globales en tiempo real: S&P 500, STOXX 600, y mercados emergentes."
            >
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {GLOBAL_INDICES.map(index => (
                        <div key={index.name} className="border rounded-lg p-4" style={{ backgroundColor: 'var(--color-bg-tertiary)', borderColor: 'var(--color-bg-tertiary)' }}>
                            <div style={{ color: 'var(--color-text-secondary)' }} className="text-sm">Cargando...</div>
                        </div>
                    ))}
                </div>
            </WidgetCard>
        );
    }

    return (
        <WidgetCard
            id={id}
            title="GLOBAL MARKETS"
            tooltip="Principales índices de mercado globales en tiempo real: S&P 500, STOXX 600, y mercados emergentes."
        >
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {GLOBAL_INDICES.map(index => {
                const priceData = prices[index.ticker];
                if (!priceData) {
                    return (
                        <div key={index.name} style={{ backgroundColor: 'var(--color-bg-tertiary)', borderColor: 'var(--color-bg-tertiary)' }} className="border rounded-lg p-4">
                            <div className="flex justify-between items-baseline">
                                <h3 className="text-sm font-bold" style={{ color: 'var(--color-text-primary)' }}>{index.name}</h3>
                                <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>{index.region}</p>
                            </div>
                            <div className="mt-2 text-sm" style={{ color: 'var(--color-text-muted)' }}>No disponible</div>
                        </div>
                    );
                }

                const value = priceData.price || priceData.previousClose || 0;
                const change = priceData.change || 0;
                const changePercent = priceData.changesPercentage || 0;
                const isPositive = change >= 0;

                return (
                    <div
                        key={index.name}
                        className="border rounded-lg p-4 transition-all hover:bg-opacity-80"
                        style={{
                            backgroundColor: 'var(--color-bg-tertiary)',
                            borderColor: 'var(--color-bg-tertiary)'
                        }}
                    >
                        <div className="flex justify-between items-baseline">
                            <h3 className="text-sm font-bold" style={{ color: 'var(--color-text-primary)' }}>{index.name}</h3>
                            <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>{index.region}</p>
                        </div>
                        <div className="mt-2 flex justify-between items-end">
                            <p className="text-2xl" style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-text-primary)' }}>
                                {value.toFixed(2)}
                            </p>
                            <span className={`text-md font-semibold flex items-center ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                                {isPositive ? <ArrowUpIcon className="w-4 h-4 mr-1"/> : <ArrowDownIcon className="w-4 h-4 mr-1"/>}
                                {changePercent >= 0 ? '+' : ''}{changePercent.toFixed(2)}%
                            </span>
                        </div>
                    </div>
                );
            })}
            </div>
        </WidgetCard>
    );
};

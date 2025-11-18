
import React, { useEffect, useState } from 'react';
import { fetchPrices, RealtimePrice } from '../../services/apiService';
import { ArrowUpIcon, ArrowDownIcon } from '../Icons';
import { WidgetCard } from './WidgetCard';

// Principales índices globales (ETFs que representan índices)
const GLOBAL_INDICES = [
    { ticker: 'SPY', name: 'S&P 500', region: 'USA' },
    { ticker: 'STOXX', name: 'STOXX 600', region: 'Europe' },
    { ticker: 'EEM', name: 'S&P IPSA', region: 'Chile' }, // Usando EEM como proxy, puedes cambiarlo
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
            } catch (err) {
                console.error('Error fetching global market indices:', err);
                setError(err instanceof Error ? err.message : 'Error cargando índices');
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

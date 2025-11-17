
import React, { useEffect, useState } from 'react';
import { fetchPrices, RealtimePrice } from '../../services/apiService';
import { ArrowUpIcon, ArrowDownIcon } from '../Icons';

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
            <div id={id} className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {GLOBAL_INDICES.map(index => (
                    <div key={index.name} className="bg-gray-950/50 border border-slate-800/50 rounded-lg p-4">
                        <div className="text-slate-400 text-sm">Cargando...</div>
                    </div>
                ))}
            </div>
        );
    }

    return (
        <div id={id} className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {GLOBAL_INDICES.map(index => {
                const priceData = prices[index.ticker];
                if (!priceData) {
                    return (
                        <div key={index.name} className="bg-gray-950/50 border border-slate-800/50 rounded-lg p-4">
                            <div className="flex justify-between items-baseline">
                                <h3 className="text-sm font-bold text-slate-300">{index.name}</h3>
                                <p className="text-xs text-slate-500">{index.region}</p>
                            </div>
                            <div className="mt-2 text-slate-500 text-sm">No disponible</div>
                        </div>
                    );
                }

                const value = priceData.price || priceData.previousClose || 0;
                const change = priceData.change || 0;
                const changePercent = priceData.changesPercentage || 0;
                const isPositive = change >= 0;

                return (
                    <div key={index.name} className="bg-gray-950/50 border border-slate-800/50 rounded-lg p-4 transition-all hover:border-slate-700 hover:bg-gray-900">
                        <div className="flex justify-between items-baseline">
                            <h3 className="text-sm font-bold text-slate-300">{index.name}</h3>
                            <p className="text-xs text-slate-500">{index.region}</p>
                        </div>
                        <div className="mt-2 flex justify-between items-end">
                            <p className="text-2xl font-mono text-slate-100">{value.toFixed(2)}</p>
                            <span className={`text-md font-semibold flex items-center ${isPositive ? 'text-blue-400' : 'text-red-400'}`}>
                                {isPositive ? <ArrowUpIcon className="w-4 h-4 mr-1"/> : <ArrowDownIcon className="w-4 h-4 mr-1"/>}
                                {changePercent >= 0 ? '+' : ''}{changePercent.toFixed(2)}%
                            </span>
                        </div>
                    </div>
                );
            })}
        </div>
    );
};


import React, { useEffect, useState } from 'react';
import { fetchPrices, RealtimePrice } from '../../services/apiService';
import { ArrowUpIcon, ArrowDownIcon } from '../Icons';
import { WidgetCard } from './WidgetCard';

// Principales índices del mercado (ETFs que representan índices)
const MARKET_INDICES = ['SPY', 'QQQ', 'DIA', 'IWM']; // S&P 500, NASDAQ, Dow, Russell 2000

const POLLING_INTERVAL = 30000; // 30 segundos

export const MarketIndices: React.FC = () => {
    const [prices, setPrices] = useState<Record<string, RealtimePrice>>({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const updatePrices = async () => {
            try {
                setError(null);
                const data = await fetchPrices(MARKET_INDICES);
                setPrices(data);
                setLoading(false);
            } catch (err) {
                console.error('Error fetching market indices:', err);
                setError(err instanceof Error ? err.message : 'Error cargando precios');
                setLoading(false);
            }
        };

        // Actualizar inmediatamente
        updatePrices();

        // Actualizar cada 30 segundos
        const interval = setInterval(updatePrices, POLLING_INTERVAL);

        return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <WidgetCard title="MARKET OVERVIEW">
                <div className="text-slate-400 text-sm">Cargando precios...</div>
            </WidgetCard>
        );
    }

    if (error) {
        return (
            <WidgetCard title="MARKET OVERVIEW">
                <div className="text-red-400 text-sm">{error}</div>
                <div className="text-slate-500 text-xs mt-1">Usando datos mock</div>
            </WidgetCard>
        );
    }

    return (
        <WidgetCard title="MARKET OVERVIEW">
            <div className="space-y-2">
                {MARKET_INDICES.map(ticker => {
                    const priceData = prices[ticker];
                    if (!priceData) {
                        return (
                            <div key={ticker} className="flex justify-between items-center text-sm">
                                <span className="text-slate-200">{ticker}</span>
                                <span className="text-slate-500 text-xs">No disponible</span>
                            </div>
                        );
                    }

                    const value = priceData.price || priceData.previousClose || 0;
                    const change = priceData.change || 0;
                    const changePercent = priceData.changesPercentage || 0;
                    const isPositive = change >= 0;

                    return (
                        <div key={ticker} className="flex justify-between items-center text-sm">
                            <span className="text-slate-200">{ticker}</span>
                            <div className="text-right">
                                <span className="font-mono">${value.toFixed(2)}</span>
                                <span className={`ml-2 font-semibold flex items-center justify-end ${isPositive ? 'text-blue-400' : 'text-red-400'}`}>
                                    {isPositive ? <ArrowUpIcon className="w-3 h-3 mr-1"/> : <ArrowDownIcon className="w-3 h-3 mr-1"/>}
                                    {changePercent.toFixed(2)}%
                                </span>
                            </div>
                        </div>
                    );
                })}
            </div>
        </WidgetCard>
    );
};

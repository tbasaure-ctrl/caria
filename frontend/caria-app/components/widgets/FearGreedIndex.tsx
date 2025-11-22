/**
 * FearGreedIndex Widget - CNN Fear and Greed Index in real-time.
 * Displays market sentiment gauge similar to CNN's original design but adapted to app style.
 */

import React, { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';
import { getErrorMessage } from '../../src/utils/errorHandling';

interface FearGreedData {
    value: number; // 0-100
    classification: string;
    timestamp: string;
    previous_close: number | null;
    change: number | null;
}

const CLASSIFICATION_CONFIG: Record<string, { label: string; color: string; zone: string }> = {
    'Extreme Fear': { label: 'EXTREME FEAR', color: '#ef4444', zone: '0-25' },
    'Fear': { label: 'FEAR', color: '#f59e0b', zone: '25-45' },
    'Neutral': { label: 'NEUTRAL', color: '#6b7280', zone: '45-55' },
    'Greed': { label: 'GREED', color: '#3b82f6', zone: '55-75' },
    'Extreme Greed': { label: 'EXTREME GREED', color: '#10b981', zone: '75-100' },
};

export const FearGreedIndex: React.FC = () => {
    const [data, setData] = useState<FearGreedData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadFearGreedIndex();
        // Refresh every 5 minutes
        const interval = setInterval(loadFearGreedIndex, 5 * 60 * 1000);
        return () => clearInterval(interval);
    }, []);

    const loadFearGreedIndex = async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await fetchWithAuth(`${API_BASE_URL}/api/market/fear-greed`);

            if (!response.ok) {
                throw new Error('Failed to load Fear and Greed Index');
            }

            const fearGreedData: FearGreedData = await response.json();
            setData(fearGreedData);
        } catch (err: unknown) {
            const errorMsg = getErrorMessage(err);
            setError('Coming soon... Fear & Greed Index is being enhanced with real-time updates.');
        } finally {
            setLoading(false);
        }
    };

    const renderGauge = (value: number, color: string) => {
        const angle = -90 + (value / 100) * 180;

        return (
            <div className="relative w-48 h-24 mx-auto mb-4">
                <svg viewBox="0 0 100 50" className="w-full h-full">
                    {/* Background arc */}
                    <path
                        d="M 10 50 A 40 40 0 0 1 90 50"
                        fill="none"
                        stroke="var(--color-bg-tertiary)"
                        strokeWidth="6"
                        strokeLinecap="round"
                    />
                    {/* Colored progress arc */}
                    <path
                        d="M 10 50 A 40 40 0 0 1 90 50"
                        fill="none"
                        stroke={color}
                        strokeWidth="6"
                        strokeDasharray="125.6"
                        strokeDashoffset={125.6 - (value / 100) * 125.6}
                        strokeLinecap="round"
                        style={{
                            transition: 'stroke-dashoffset 0.8s ease-out',
                            filter: 'drop-shadow(0 0 4px currentColor)'
                        }}
                    />
                </svg>
                {/* Needle */}
                <div
                    className="absolute bottom-0 left-1/2 origin-bottom transition-transform duration-700 ease-out"
                    style={{
                        transform: `translateX(-50%) rotate(${angle}deg)`,
                        width: '2px',
                        height: '20px',
                        backgroundColor: 'var(--color-cream)'
                    }}
                ></div>
                {/* Center dot */}
                <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-3 h-3 rounded-full"
                    style={{ backgroundColor: 'var(--color-cream)', boxShadow: '0 0 8px var(--color-cream)' }}></div>
            </div>
        );
    };

    if (loading) {
        return (
            <WidgetCard
                title="FEAR & GREED"
                id="fear-greed-widget"
                tooltip="Índice CNN Fear & Greed en tiempo real. Mide el sentimiento del mercado de 0 (Miedo Extremo) a 100 (Avaricia Extrema)."
            >
                <div className="text-center h-[124px] flex items-center justify-center">
                    <p className="text-slate-500">Loading index...</p>
                </div>
            </WidgetCard>
        );
    }

    if (error) {
        return (
            <WidgetCard
                title="FEAR & GREED"
                id="fear-greed-widget"
                tooltip="Índice CNN Fear & Greed en tiempo real. Mide el sentimiento del mercado de 0 (Miedo Extremo) a 100 (Avaricia Extrema)."
            >
                <div className="text-center h-[124px] flex flex-col items-center justify-center">
                    <div className="text-sm mb-2 text-red-400">{error}</div>
                    <button
                        onClick={loadFearGreedIndex}
                        className="text-xs underline text-blue-400"
                    >
                        Retry
                    </button>
                </div>
            </WidgetCard>
        );
    }

    if (!data) {
        return (
            <WidgetCard
                title="FEAR & GREED"
                id="fear-greed-widget"
                tooltip="Índice CNN Fear & Greed en tiempo real. Mide el sentimiento del mercado de 0 (Miedo Extremo) a 100 (Avaricia Extrema)."
            >
                <div className="text-center h-[124px] flex items-center justify-center">
                    <p className="text-slate-500">No data available</p>
                </div>
            </WidgetCard>
        );
    }

    const config = CLASSIFICATION_CONFIG[data.classification] || {
        label: data.classification,
        color: '#6b7280',
        zone: '',
    };

    return (
        <WidgetCard
            title="FEAR & GREED"
            id="fear-greed-widget"
            tooltip="Índice CNN Fear & Greed en tiempo real. Mide el sentimiento del mercado de 0 (Miedo Extremo) a 100 (Avaricia Extrema)."
        >
            <div className="text-center">
                {renderGauge(data.value, config.color)}

                <div className="flex flex-col items-center">
                    <p className="text-2xl font-bold mt-3 mb-1"
                        style={{ fontFamily: 'var(--font-display)', color: config.color }}>
                        {config.label}
                    </p>

                    {/* Change indicator */}
                    {data.change !== null && data.previous_close !== null && (
                        <div className="flex items-center gap-2 text-xs mb-2">
                            <span className="text-slate-500">Prev: {data.previous_close}</span>
                            <span
                                style={{
                                    color: data.change >= 0 ? '#10b981' : '#ef4444',
                                    fontWeight: 'bold',
                                }}
                            >
                                {data.change >= 0 ? '+' : ''}{data.change}
                            </span>
                        </div>
                    )}

                    <p className="text-xs mt-1 px-4 italic text-slate-500 leading-relaxed">
                        Emotions driving the US stock market on a given day
                    </p>
                </div>
            </div>
        </WidgetCard>
    );
};


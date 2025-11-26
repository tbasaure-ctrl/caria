/**
 * FearGreedIndex Widget - CNN Fear and Greed Index
 * Displays market sentiment gauge with Bloomberg-style design
 */

import React, { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';
import { getErrorMessage } from '../../src/utils/errorHandling';

interface FearGreedData {
    value: number;
    classification: string;
    timestamp: string;
    previous_close: number | null;
    change: number | null;
}

const CLASSIFICATION_CONFIG: Record<string, { label: string; color: string; zone: string }> = {
    'Extreme Fear': { label: 'EXTREME FEAR', color: 'var(--color-negative)', zone: '0-25' },
    'Fear': { label: 'FEAR', color: 'var(--color-warning)', zone: '25-45' },
    'Neutral': { label: 'NEUTRAL', color: 'var(--color-text-muted)', zone: '45-55' },
    'Greed': { label: 'GREED', color: 'var(--color-accent-primary)', zone: '55-75' },
    'Extreme Greed': { label: 'EXTREME GREED', color: 'var(--color-positive)', zone: '75-100' },
};

export const FearGreedIndex: React.FC = () => {
    const [data, setData] = useState<FearGreedData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadFearGreedIndex();
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
            setError('Fear & Greed Index temporarily unavailable');
        } finally {
            setLoading(false);
        }
    };

    const renderGauge = (value: number, color: string) => {
        const angle = -90 + (value / 100) * 180;

        return (
            <div className="relative w-48 h-24 mx-auto mb-4">
                <svg viewBox="0 0 100 50" className="w-full h-full">
                    {/* Background arc - more visible */}
                    <path
                        d="M 10 50 A 40 40 0 0 1 90 50"
                        fill="none"
                        stroke="rgba(107, 122, 143, 0.2)"
                        strokeWidth="8"
                        strokeLinecap="round"
                    />
                    {/* Colored progress arc */}
                    <path
                        d="M 10 50 A 40 40 0 0 1 90 50"
                        fill="none"
                        stroke={color}
                        strokeWidth="8"
                        strokeDasharray="125.6"
                        strokeDashoffset={125.6 - (value / 100) * 125.6}
                        strokeLinecap="round"
                        style={{
                            transition: 'stroke-dashoffset 0.8s ease-out',
                            filter: 'drop-shadow(0 0 6px currentColor)'
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
                        backgroundColor: 'var(--color-text-primary)'
                    }}
                />
                {/* Center dot */}
                <div 
                    className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-3 h-3 rounded-full"
                    style={{ 
                        backgroundColor: 'var(--color-text-primary)', 
                        boxShadow: '0 0 8px var(--color-text-primary)' 
                    }}
                />
            </div>
        );
    };

    if (loading) {
        return (
            <WidgetCard
                title="FEAR & GREED INDEX"
                tooltip="CNN Fear & Greed Index measures market sentiment from 0 (Extreme Fear) to 100 (Extreme Greed)."
            >
                <div className="text-center py-8 flex flex-col items-center justify-center">
                    <div 
                        className="w-8 h-8 border-2 border-t-transparent rounded-full animate-spin mb-3"
                        style={{ borderColor: 'var(--color-accent-primary)', borderTopColor: 'transparent' }}
                    />
                    <p style={{ color: 'var(--color-text-muted)' }}>Loading sentiment data...</p>
                </div>
            </WidgetCard>
        );
    }

    if (error) {
        return (
            <WidgetCard
                title="FEAR & GREED INDEX"
                tooltip="CNN Fear & Greed Index measures market sentiment from 0 (Extreme Fear) to 100 (Extreme Greed)."
            >
                <div className="text-center py-8 flex flex-col items-center justify-center">
                    <div 
                        className="text-sm mb-4"
                        style={{ color: 'var(--color-text-secondary)' }}
                    >
                        {error}
                    </div>
                    <button
                        onClick={loadFearGreedIndex}
                        className="text-xs font-medium px-4 py-2 rounded-lg transition-colors"
                        style={{
                            backgroundColor: 'var(--color-bg-surface)',
                            color: 'var(--color-text-secondary)',
                            border: '1px solid var(--color-border-subtle)',
                        }}
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
                title="FEAR & GREED INDEX"
                tooltip="CNN Fear & Greed Index measures market sentiment from 0 (Extreme Fear) to 100 (Extreme Greed)."
            >
                <div 
                    className="text-center py-8"
                    style={{ color: 'var(--color-text-muted)' }}
                >
                    No data available
                </div>
            </WidgetCard>
        );
    }

    const config = CLASSIFICATION_CONFIG[data.classification] || {
        label: data.classification,
        color: 'var(--color-text-muted)',
        zone: '',
    };

    return (
        <WidgetCard
            title="FEAR & GREED INDEX"
            tooltip="CNN Fear & Greed Index measures market sentiment from 0 (Extreme Fear) to 100 (Extreme Greed). Based on 7 market indicators."
        >
            <div className="flex flex-col items-center">
                {renderGauge(data.value, config.color)}
                
                {/* Value Display */}
                <div className="flex items-center gap-3 mb-2">
                    <span 
                        className="text-4xl font-bold font-mono"
                        style={{ color: config.color }}
                    >
                        {data.value}
                    </span>
                    {data.change !== null && (
                        <span 
                            className="text-sm font-mono font-medium"
                            style={{ 
                                color: data.change >= 0 ? 'var(--color-positive)' : 'var(--color-negative)'
                            }}
                        >
                            {data.change >= 0 ? '+' : ''}{data.change}
                        </span>
                    )}
                </div>

                {/* Classification Label */}
                <div 
                    className="text-lg font-semibold mb-3"
                    style={{ 
                        fontFamily: 'var(--font-display)', 
                        color: config.color 
                    }}
                >
                    {config.label}
                </div>

                {/* Previous Close */}
                {data.previous_close !== null && (
                    <div 
                        className="text-xs mb-3 px-3 py-1 rounded-full"
                        style={{ 
                            backgroundColor: 'var(--color-bg-surface)',
                            color: 'var(--color-text-muted)'
                        }}
                    >
                        Previous: {data.previous_close}
                    </div>
                )}

                {/* Description */}
                <p 
                    className="text-xs text-center italic"
                    style={{ color: 'var(--color-text-muted)' }}
                >
                    Emotions driving the US stock market
                </p>
            </div>
        </WidgetCard>
    );
};

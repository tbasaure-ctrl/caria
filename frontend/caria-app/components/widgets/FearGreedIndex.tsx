/**
 * FearGreedIndex Widget - CNN Fear and Greed Index in real-time.
 * Displays market sentiment gauge similar to CNN's original design but adapted to app style.
 */

import React, { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';

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
        } catch (err: any) {
            console.error('Error loading Fear and Greed Index:', err);
            setError(err.message || 'Error loading index');
        } finally {
            setLoading(false);
        }
    };

    const renderGauge = (value: number, color: string) => {
        // Calculate angle for needle (0-100 maps to -90deg to +90deg)
        const angle = -90 + (value / 100) * 180;

        return (
            <div className="relative w-full max-w-xs mx-auto mb-4" style={{ height: '140px' }}>
                <svg viewBox="0 0 200 100" className="w-full h-full">
                    {/* Background arc (full semicircle) */}
                    <path
                        d="M 20 80 A 60 60 0 0 1 180 80"
                        fill="none"
                        stroke="var(--color-bg-tertiary)"
                        strokeWidth="8"
                        strokeLinecap="round"
                    />

                    {/* Colored zones */}
                    {/* Extreme Fear (0-25) */}
                    <path
                        d="M 20 80 A 60 60 0 0 1 65 80"
                        fill="none"
                        stroke="#ef4444"
                        strokeWidth="8"
                        strokeLinecap="round"
                        opacity="0.3"
                    />
                    {/* Fear (25-45) */}
                    <path
                        d="M 65 80 A 60 60 0 0 1 101 80"
                        fill="none"
                        stroke="#f59e0b"
                        strokeWidth="8"
                        strokeLinecap="round"
                        opacity="0.3"
                    />
                    {/* Neutral (45-55) */}
                    <path
                        d="M 101 80 A 60 60 0 0 1 119 80"
                        fill="none"
                        stroke="#6b7280"
                        strokeWidth="8"
                        strokeLinecap="round"
                        opacity="0.3"
                    />
                    {/* Greed (55-75) */}
                    <path
                        d="M 119 80 A 60 60 0 0 1 155 80"
                        fill="none"
                        stroke="#3b82f6"
                        strokeWidth="8"
                        strokeLinecap="round"
                        opacity="0.3"
                    />
                    {/* Extreme Greed (75-100) */}
                    <path
                        d="M 155 80 A 60 60 0 0 1 180 80"
                        fill="none"
                        stroke="#10b981"
                        strokeWidth="8"
                        strokeLinecap="round"
                        opacity="0.3"
                    />

                    {/* Progress arc (current value) */}
                    <path
                        d="M 20 80 A 60 60 0 0 1 180 80"
                        fill="none"
                        stroke={color}
                        strokeWidth="10"
                        strokeDasharray="188.5"
                        strokeDashoffset={188.5 - (value / 100) * 188.5}
                        strokeLinecap="round"
                        style={{
                            transition: 'stroke-dashoffset 0.8s ease-out, stroke 0.5s ease-out',
                            filter: 'drop-shadow(0 0 6px currentColor)',
                        }}
                    />

                    {/* Zone labels */}
                    <text x="20" y="95" fontSize="8" fill="var(--color-text-muted)" textAnchor="start">
                        0
                    </text>
                    <text x="50" y="95" fontSize="8" fill="var(--color-text-muted)" textAnchor="middle">
                        25
                    </text>
                    <text x="100" y="95" fontSize="8" fill="var(--color-text-muted)" textAnchor="middle">
                        50
                    </text>
                    <text x="150" y="95" fontSize="8" fill="var(--color-text-muted)" textAnchor="middle">
                        75
                    </text>
                    <text x="180" y="95" fontSize="8" fill="var(--color-text-muted)" textAnchor="end">
                        100
                    </text>

                    {/* Zone labels on arc */}
                    <text x="42" y="70" fontSize="7" fill="var(--color-text-muted)" textAnchor="middle" transform="rotate(-45 42 70)">
                        EXTREME FEAR
                    </text>
                    <text x="83" y="60" fontSize="7" fill="var(--color-text-muted)" textAnchor="middle" transform="rotate(-22.5 83 60)">
                        FEAR
                    </text>
                    <text x="100" y="50" fontSize="7" fill="var(--color-text-muted)" textAnchor="middle">
                        NEUTRAL
                    </text>
                    <text x="117" y="60" fontSize="7" fill="var(--color-text-muted)" textAnchor="middle" transform="rotate(22.5 117 60)">
                        GREED
                    </text>
                    <text x="158" y="70" fontSize="7" fill="var(--color-text-muted)" textAnchor="middle" transform="rotate(45 158 70)">
                        EXTREME GREED
                    </text>
                </svg>

                {/* Needle */}
                <div
                    className="absolute bottom-0 left-1/2 origin-bottom transition-transform duration-700 ease-out"
                    style={{
                        transform: `translateX(-50%) rotate(${angle}deg)`,
                        width: '3px',
                        height: '50px',
                        backgroundColor: 'var(--color-cream)',
                        borderRadius: '2px',
                        boxShadow: '0 0 8px rgba(255, 255, 255, 0.5)',
                        zIndex: 10,
                    }}
                />

                {/* Center dot */}
                <div
                    className="absolute bottom-0 left-1/2 -translate-x-1/2 w-4 h-4 rounded-full"
                    style={{
                        backgroundColor: 'var(--color-cream)',
                        boxShadow: '0 0 12px rgba(255, 255, 255, 0.8)',
                        zIndex: 11,
                    }}
                />

                {/* Value display */}
                <div
                    className="absolute bottom-8 left-1/2 -translate-x-1/2 text-center"
                    style={{ zIndex: 12 }}
                >
                    <div
                        className="text-4xl font-bold"
                        style={{
                            fontFamily: 'var(--font-display)',
                            color: color,
                            textShadow: `0 0 10px ${color}40`,
                        }}
                    >
                        {value}
                    </div>
                </div>
            </div>
        );
    };

    if (loading) {
        return (
            <WidgetCard
                title="Fear & Greed Index"
                id="fear-greed-widget"
                tooltip="Índice CNN Fear & Greed en tiempo real. Mide el sentimiento del mercado de 0 (Miedo Extremo) a 100 (Avaricia Extrema)."
            >
                <div className="text-center py-8">
                    <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                        Loading Fear & Greed Index...
                    </div>
                </div>
            </WidgetCard>
        );
    }

    if (error) {
        return (
            <WidgetCard
                title="Fear & Greed Index"
                id="fear-greed-widget"
                tooltip="Índice CNN Fear & Greed en tiempo real. Mide el sentimiento del mercado de 0 (Miedo Extremo) a 100 (Avaricia Extrema)."
            >
                <div className="text-center py-8">
                    <div className="text-sm mb-2" style={{ color: '#ef4444' }}>{error}</div>
                    <button
                        onClick={loadFearGreedIndex}
                        className="text-xs underline"
                        style={{ color: 'var(--color-primary)' }}
                    >
                        Retry
                    </button>
                </div>
            </WidgetCard>
        );
    }

    if (!data) {
        return null;
    }

    const config = CLASSIFICATION_CONFIG[data.classification] || {
        label: data.classification,
        color: '#6b7280',
        zone: '',
    };

    return (
        <WidgetCard
            title="Fear & Greed Index"
            id="fear-greed-widget"
            tooltip="Índice CNN Fear & Greed en tiempo real. Mide el sentimiento del mercado de 0 (Miedo Extremo) a 100 (Avaricia Extrema)."
        >
            <div className="text-center">
                {/* Gauge */}
                {renderGauge(data.value, config.color)}

                {/* Classification */}
                <div
                    className="text-xl font-bold mb-2"
                    style={{
                        fontFamily: 'var(--font-display)',
                        color: config.color,
                    }}
                >
                    {config.label}
                </div>

                {/* Change indicator */}
                {data.change !== null && data.previous_close !== null && (
                    <div className="flex items-center justify-center gap-2 text-sm mb-2">
                        <span style={{ color: 'var(--color-text-secondary)' }}>Previous:</span>
                        <span style={{ color: 'var(--color-text-secondary)' }}>{data.previous_close}</span>
                        <span
                            style={{
                                color: data.change >= 0 ? '#10b981' : '#ef4444',
                                fontWeight: 'bold',
                            }}
                        >
                            {data.change >= 0 ? '+' : ''}
                            {data.change}
                        </span>
                    </div>
                )}

                {/* Source attribution */}
                <div className="text-xs mt-4" style={{ color: 'var(--color-text-muted)' }}>
                    Source: CNN.com
                </div>

                {/* Description */}
                <div
                    className="text-xs mt-2 px-4 italic"
                    style={{
                        fontFamily: 'var(--font-body)',
                        color: 'var(--color-text-muted)',
                        lineHeight: '1.5',
                    }}
                >
                    Emotions driving the US stock market on a given day
                </div>
            </div>
        </WidgetCard>
    );
};


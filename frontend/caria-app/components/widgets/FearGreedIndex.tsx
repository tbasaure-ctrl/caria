/**
 * FearGreedIndex Widget - CNN Fear and Greed Index
 * Displays market sentiment gauge with Bloomberg-style design
 */

import React, { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';
import { API_BASE_URL } from '../../services/apiService';

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

// Mock data for initial display
const MOCK_FEAR_GREED: FearGreedData = {
    value: 65,
    classification: 'Greed',
    timestamp: new Date().toISOString(),
    previous_close: 62,
    change: 3,
};

export const FearGreedIndex: React.FC = () => {
    const [data, setData] = useState<FearGreedData | null>(MOCK_FEAR_GREED);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        // Show mock data immediately
        setData(MOCK_FEAR_GREED);
        setLoading(false);

        // Try to load real data
        loadFearGreedIndex();
        const interval = setInterval(loadFearGreedIndex, 5 * 60 * 1000);
        return () => clearInterval(interval);
    }, []);

    const loadFearGreedIndex = async () => {
        try {
            setError(null);
            // Try without auth first, fallback to with auth
            let response = await fetch(`${API_BASE_URL}/api/market/fear-greed`);
            
            if (!response.ok && response.status === 401) {
                // If unauthorized, try with auth
                const token = localStorage.getItem('caria-auth-token');
                if (token) {
                    response = await fetch(`${API_BASE_URL}/api/market/fear-greed`, {
                        headers: {
                            'Authorization': `Bearer ${token}`,
                        },
                    });
                }
            }

            if (response.ok) {
                const fearGreedData: FearGreedData = await response.json();
                setData(fearGreedData);
            } else {
                // Keep showing mock data on error
                console.warn('Failed to load Fear & Greed Index, using mock data');
            }
        } catch (err: unknown) {
            // Keep showing mock data on error
            console.warn('Error loading Fear & Greed Index, using mock data');
        }
    };

    const renderGauge = (value: number, color: string) => {
        const angle = -90 + (value / 100) * 180;
        // Ensure value is between 0 and 100
        const clampedValue = Math.max(0, Math.min(100, value));
        // Calculate dash offset for semicircle fill
        const circumference = Math.PI * 40; // radius is 40
        const dashOffset = circumference - (clampedValue / 100) * circumference;
        
        // Resolve CSS variable to actual color value
        const getComputedColor = (cssVar: string): string => {
            if (cssVar.startsWith('var(')) {
                // Extract variable name
                const varName = cssVar.match(/var\(--([^)]+)\)/)?.[1];
                if (varName) {
                    // Map CSS variables to actual colors
                    const colorMap: Record<string, string> = {
                        'color-positive': '#27ae60',
                        'color-negative': '#c0392b',
                        'color-warning': '#f39c12',
                        'color-accent-primary': '#3498db',
                        'color-text-muted': '#7f8c8d',
                    };
                    return colorMap[varName] || '#3498db';
                }
            }
            return color;
        };
        const resolvedColor = getComputedColor(color);

        return (
            <div className="relative w-48 h-24 mx-auto mb-4">
                <svg viewBox="0 0 100 50" className="w-full h-full">
                    {/* Background arc - darker base */}
                    <path
                        d="M 10 50 A 40 40 0 0 1 90 50"
                        fill="none"
                        stroke="rgba(107, 122, 143, 0.3)"
                        strokeWidth="10"
                        strokeLinecap="round"
                    />
                    {/* Colored progress arc - bright but dark fill */}
                    <path
                        d="M 10 50 A 40 40 0 0 1 90 50"
                        fill="none"
                        stroke={resolvedColor}
                        strokeWidth="10"
                        strokeDasharray={circumference}
                        strokeDashoffset={dashOffset}
                        strokeLinecap="round"
                        opacity={1}
                        style={{
                            transition: 'stroke-dashoffset 0.8s ease-out',
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

    // Always show data (mock or real)
    if (!data) {
        setData(MOCK_FEAR_GREED);
        return null;
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

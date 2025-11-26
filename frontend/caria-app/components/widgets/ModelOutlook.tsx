import React from 'react';
import { WidgetCard } from './WidgetCard';

interface RegimeData {
    regime: string;
    confidence: number;
}

const regimeConfig: { [key: string]: { label: string; value: number; color: string; description: string } } = {
    expansion: { 
        label: 'Expansion', 
        value: 85, 
        color: 'var(--color-positive)',
        description: 'Market conditions are favorable for growth assets'
    },
    slowdown: { 
        label: 'Slowdown', 
        value: 60, 
        color: 'var(--color-warning)',
        description: 'Economic momentum is decelerating'
    },
    recession: { 
        label: 'Recession', 
        value: 35, 
        color: 'var(--color-negative)',
        description: 'Defensive positioning recommended'
    },
    stress: { 
        label: 'Market Stress', 
        value: 15, 
        color: 'var(--color-negative)',
        description: 'High volatility environment'
    },
    default: { 
        label: 'Awaiting Data...', 
        value: 0, 
        color: 'var(--color-text-muted)',
        description: 'Regime detection in progress'
    },
};

const Gauge: React.FC<{ value: number; color: string }> = ({ value, color }) => {
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

export const ModelOutlook: React.FC<{ regimeData: RegimeData | null; isLoading: boolean }> = ({ regimeData, isLoading }) => {
    
    const renderContent = () => {
        if (isLoading) {
            return (
                <div 
                    className="text-center py-8 flex flex-col items-center justify-center"
                >
                    <div 
                        className="w-8 h-8 border-2 border-t-transparent rounded-full animate-spin mb-3"
                        style={{ borderColor: 'var(--color-accent-primary)', borderTopColor: 'transparent' }}
                    />
                    <p style={{ color: 'var(--color-text-muted)' }}>Loading regime data...</p>
                </div>
            );
        }
        
        const currentRegimeKey = regimeData?.regime && regimeConfig[regimeData.regime] ? regimeData.regime : 'default';
        const { label, value, color, description } = regimeConfig[currentRegimeKey];

        return (
            <div className="flex flex-col items-center">
                <Gauge value={value} color={color} />
                
                <div 
                    className="text-2xl font-bold mt-2 mb-2"
                    style={{ 
                        fontFamily: 'var(--font-display)', 
                        color: color 
                    }}
                >
                    {label}
                </div>
                
                {regimeData?.confidence && (
                    <div 
                        className="text-xs font-mono mb-3 px-3 py-1 rounded-full"
                        style={{ 
                            backgroundColor: 'var(--color-bg-surface)',
                            color: 'var(--color-text-secondary)'
                        }}
                    >
                        Confidence: {(regimeData.confidence * 100).toFixed(0)}%
                    </div>
                )}
                
                <p 
                    className="text-sm text-center max-w-xs"
                    style={{ color: 'var(--color-text-secondary)' }}
                >
                    {description}
                </p>
                
                <p 
                    className="text-xs mt-4 text-center italic"
                    style={{ color: 'var(--color-text-muted)' }}
                >
                    HMM-based regime detection with 78% validation accuracy
                </p>
            </div>
        );
    }

    return (
        <WidgetCard
            title="MODEL OUTLOOK"
            tooltip="Current macroeconomic regime detected by our Hidden Markov Model. Indicates whether the market is in expansion, slowdown, recession, or stress conditions."
        >
            {renderContent()}
        </WidgetCard>
    );
};

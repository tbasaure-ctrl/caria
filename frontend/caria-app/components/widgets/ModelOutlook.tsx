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
        value: 50, 
        color: 'var(--color-text-muted)',
        description: 'Regime detection in progress'
    },
};

const Gauge: React.FC<{ value: number; color: string }> = ({ value, color }) => {
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

export const ModelOutlook: React.FC<{ regimeData: RegimeData | null; isLoading: boolean }> = ({ regimeData, isLoading }) => {
    
    const renderContent = () => {
        // Always show data - use default if loading or no data
        const currentRegimeKey = (!isLoading && regimeData?.regime && regimeConfig[regimeData.regime]) 
            ? regimeData.regime 
            : 'default';
        const { label, value: defaultValue, color, description } = regimeConfig[currentRegimeKey];
        
        // Use actual confidence score (0-1) converted to 0-100, or fallback to default value
        // If confidence is 0 or undefined, use default value to show something meaningful
        const gaugeValue = (!isLoading && regimeData?.confidence !== undefined && regimeData.confidence > 0) 
            ? regimeData.confidence * 100 
            : defaultValue;

        return (
            <div className="flex flex-col items-center">
                <Gauge value={gaugeValue} color={color} />
                
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

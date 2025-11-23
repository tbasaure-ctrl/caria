
import React from 'react';
import { WidgetCard } from './WidgetCard';

interface RegimeData {
    regime: string;
    confidence: number;
}

const regimeConfig: { [key: string]: { label: string; value: number; color: string } } = {
    expansion: { label: 'Expansion', value: 85, color: 'var(--color-accent)' },
    slowdown: { label: 'Slowdown', value: 60, color: 'var(--color-secondary)' },
    recession: { label: 'Recession', value: 35, color: 'var(--color-text-muted)' },
    stress: { label: 'Market Stress', value: 15, color: 'var(--color-primary)' },
    default: { label: 'Awaiting Data...', value: 0, color: 'var(--color-text-muted)' },
};

const Gauge: React.FC<{ value: number; color: string }> = ({ value, color }) => {
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
                 style={{backgroundColor: 'var(--color-cream)', boxShadow: '0 0 8px var(--color-cream)'}}></div>
        </div>
    );
};

export const ModelOutlook: React.FC<{ regimeData: RegimeData | null; isLoading: boolean }> = ({ regimeData, isLoading }) => {
    
    const renderContent = () => {
        if (isLoading) {
            return (
                <div className="text-center h-[124px] flex items-center justify-center">
                    <p className="text-slate-500">Loading outlook...</p>
                </div>
            );
        }
        
        const currentRegimeKey = regimeData?.regime && regimeConfig[regimeData.regime] ? regimeData.regime : 'default';
        const { label, value, color } = regimeConfig[currentRegimeKey];

        return (
            <div className="text-center">
                <Gauge value={value} color={color} />
                <p className="text-2xl font-bold mt-3 mb-2"
                   style={{fontFamily: 'var(--font-display)', color: 'var(--color-cream)'}}>
                    {label}
                </p>
                <p className="text-xs mt-3 px-4 italic"
                   style={{
                     fontFamily: 'var(--font-body)',
                     color: 'var(--color-text-muted)',
                     lineHeight: '1.5'
                   }}>
                    As of November 2025, this model achieves 78% accuracy in post-training validation.
                </p>
            </div>
        );
    }

    return (
        <WidgetCard
            title="MODEL OUTLOOK"
            tooltip="Régimen macroeconómico actual detectado por el modelo. Indica si el mercado está en expansión, desaceleración, recesión o estrés."
        >
            {renderContent()}
        </WidgetCard>
    );
};

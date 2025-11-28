import React, { useState } from 'react';

interface WidgetCardProps {
    title: string;
    children: React.ReactNode;
    id?: string;
    className?: string;
    tooltip?: string | React.ReactNode;
    action?: {
        label: string;
        onClick: () => void;
    };
    compact?: boolean;
}

export const WidgetCard: React.FC<WidgetCardProps> = ({ 
    title, 
    children, 
    id, 
    className = '', 
    tooltip,
    action,
    compact = false 
}) => {
    const [showTooltip, setShowTooltip] = useState(false);

    return (
        <div
            id={id}
            className={`rounded-lg transition-all duration-300 group border border-white/5 hover:border-accent-cyan/50 hover:shadow-[0_0_20px_rgba(34,211,238,0.15)] ${className}`}
            style={{
                backgroundColor: 'transparent', 
            }}
        >
            {/* Widget Header - "Blue Line - Box - Blue Line" Style */}
            <div className="flex items-center justify-center mb-4 relative">
                {/* Left Line */}
                <div className="h-px flex-1 bg-gradient-to-r from-transparent via-accent-cyan/30 to-accent-cyan/60" />
                
                {/* Title Box */}
                <div className="mx-4 relative">
                    <div 
                        className="px-6 py-1.5 rounded border border-accent-cyan/30 bg-accent-cyan/5 backdrop-blur-sm shadow-[0_0_15px_rgba(34,211,238,0.1)] flex items-center gap-2 group-hover:border-accent-cyan/50 group-hover:shadow-[0_0_20px_rgba(34,211,238,0.2)] transition-all duration-300"
                    >
                        <h3
                            className="text-xs font-bold tracking-[0.15em] uppercase text-accent-cyan whitespace-nowrap group-hover:text-white transition-colors duration-300"
                            style={{ fontFamily: 'var(--font-mono)', textShadow: '0 0 10px rgba(34,211,238,0.4)' }}
                        >
                            {title}
                        </h3>
                        
                        {/* Tooltip Icon */}
                        {tooltip && (
                            <button
                                className="text-accent-cyan/50 hover:text-accent-cyan transition-colors ml-1"
                                onMouseEnter={() => setShowTooltip(true)}
                                onMouseLeave={() => setShowTooltip(false)}
                                onClick={() => setShowTooltip(!showTooltip)}
                            >
                                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                            </button>
                        )}
                    </div>

                    {/* Tooltip Popup */}
                    {showTooltip && (
                        <div className="absolute z-50 w-64 p-3 rounded-lg shadow-xl animate-fade-in left-1/2 -translate-x-1/2 top-full mt-3 bg-[#0B1221] border border-accent-cyan/20">
                            <div className="text-xs text-text-secondary leading-relaxed text-center">
                                {tooltip}
                            </div>
                        </div>
                    )}
                </div>

                {/* Right Line */}
                <div className="h-px flex-1 bg-gradient-to-l from-transparent via-accent-cyan/30 to-accent-cyan/60" />

                {/* Action Button */}
                {action && (
                    <button
                        onClick={action.onClick}
                        className="absolute right-0 text-[10px] font-medium px-2 py-1 rounded text-accent-cyan/70 hover:text-accent-cyan hover:bg-accent-cyan/10 transition-colors uppercase tracking-wider"
                    >
                        {action.label}
                    </button>
                )}
            </div>

            {/* Widget Content */}
            <div className={`bg-[#0B1221]/60 border border-white/5 rounded-xl backdrop-blur-sm group-hover:border-accent-cyan/20 transition-colors duration-300 ${compact ? 'p-4' : 'p-5'}`}>
                {children}
            </div>
        </div>
    );
};

// Helper Components (Updated to match)
interface DataCardProps {
    label: string;
    value: string | number;
    change?: string;
    isPositive?: boolean;
    sublabel?: string;
}

export const DataCard: React.FC<DataCardProps> = ({ label, value, change, isPositive, sublabel }) => {
    return (
        <div className="p-4 rounded-lg bg-[#0F1623] border border-white/5 hover:border-accent-cyan/20 transition-colors duration-300 hover:shadow-glow-sm">
            <div className="text-[10px] font-bold tracking-widest uppercase mb-2 text-text-muted">
                {label}
            </div>
            <div className="flex items-baseline gap-2">
                <span className="text-xl font-mono text-text-primary text-shadow-sm">
                    {value}
                </span>
                {change && (
                    <span className={`text-xs font-mono font-medium ${isPositive ? 'text-positive' : 'text-negative'}`}>
                        {change}
                    </span>
                )}
            </div>
            {sublabel && (
                <div className="text-[10px] mt-1 text-text-subtle uppercase tracking-wide">
                    {sublabel}
                </div>
            )}
        </div>
    );
};

interface MetricRowProps {
    items: { label: string; value: string | number; color?: string }[];
}

export const MetricRow: React.FC<MetricRowProps> = ({ items }) => {
    return (
        <div className="flex divide-x divide-white/5 rounded-lg overflow-hidden bg-[#0F1623] border border-white/5 hover:border-accent-cyan/20 transition-colors duration-300">
            {items.map((item, idx) => (
                <div key={idx} className="flex-1 px-3 py-2 text-center">
                    <div className="text-[9px] font-bold tracking-widest uppercase mb-1 text-text-muted">
                        {item.label}
                    </div>
                    <div 
                        className="text-sm font-mono"
                        style={{ color: item.color || 'var(--color-text-primary)' }}
                    >
                        {item.value}
                    </div>
                </div>
            ))}
        </div>
    );
};

interface SectionDividerProps {
    label?: string;
}

export const SectionDivider: React.FC<SectionDividerProps> = ({ label }) => {
    return (
        <div className="relative my-8">
            <div className="absolute inset-0 flex items-center" aria-hidden="true">
                <div className="w-full h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
            </div>
            {label && (
                <div className="relative flex justify-center">
                    <span className="px-4 py-1 text-[10px] font-bold tracking-[0.2em] uppercase text-text-muted bg-bg-primary border border-white/5 rounded-full shadow-sm">
                        {label}
                    </span>
                </div>
            )}
        </div>
    );
};

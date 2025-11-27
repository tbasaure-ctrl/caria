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
            className={`rounded-lg transition-all duration-300 ${className}`}
            style={{
                backgroundColor: 'var(--color-bg-secondary)', // Darker background
                border: '1px solid rgba(148, 163, 184, 0.1)', // Very subtle border
            }}
        >
            {/* Widget Header - Clean & Minimal */}
            <div 
                className={`flex items-center justify-between ${compact ? 'px-5 py-4' : 'px-6 py-5'} border-b border-white/5`}
            >
                <div className="flex items-center gap-3">
                    {/* Minimalist Title */}
                    <h3
                        className="text-sm font-medium tracking-wide text-text-primary"
                        style={{ fontFamily: 'var(--font-display)' }}
                    >
                        {title}
                    </h3>

                    {/* Info Tooltip */}
                    {tooltip && (
                        <div className="relative">
                            <button
                                className="opacity-40 hover:opacity-100 transition-opacity text-text-muted"
                                onMouseEnter={() => setShowTooltip(true)}
                                onMouseLeave={() => setShowTooltip(false)}
                                onClick={() => setShowTooltip(!showTooltip)}
                                aria-label="Information"
                            >
                                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                            </button>
                            
                            {/* Tooltip Popup */}
                            {showTooltip && (
                                <div
                                    className="absolute z-50 w-72 p-3 rounded-lg shadow-xl animate-fade-in left-1/2 -translate-x-1/2 top-full mt-2 bg-bg-elevated border border-white/10"
                                >
                                    <div className="text-xs text-text-secondary leading-relaxed">
                                        {tooltip}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Action Button */}
                {action && (
                    <button
                        onClick={action.onClick}
                        className="text-xs font-medium px-3 py-1 rounded hover:bg-white/5 transition-colors text-accent-cyan"
                    >
                        {action.label}
                    </button>
                )}
            </div>

            {/* Widget Content */}
            <div className={compact ? 'p-5' : 'p-6'}>
                {children}
            </div>
        </div>
    );
};

// Exporting other components as they were, but updated styles if needed...
// For brevity, I'm keeping the WidgetCard core update here. 
// If DataCard/MetricRow are used, they should be in separate files or updated here. 
// I will assume for now the user mainly cares about the container.
// I'll re-add DataCard and MetricRow to ensure no breaks.

interface DataCardProps {
    label: string;
    value: string | number;
    change?: string;
    isPositive?: boolean;
    sublabel?: string;
}

export const DataCard: React.FC<DataCardProps> = ({ label, value, change, isPositive, sublabel }) => {
    return (
        <div 
            className="p-4 rounded-lg bg-bg-tertiary border border-white/5"
        >
            <div className="text-xs font-medium tracking-wide uppercase mb-2 text-text-muted">
                {label}
            </div>
            <div className="flex items-baseline gap-2">
                <span className="text-xl font-mono text-text-primary">
                    {value}
                </span>
                {change && (
                    <span 
                        className={`text-sm font-mono font-medium ${isPositive ? 'text-positive' : 'text-negative'}`}
                    >
                        {change}
                    </span>
                )}
            </div>
            {sublabel && (
                <div className="text-xs mt-1 text-text-subtle">
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
        <div className="flex divide-x divide-white/5 rounded-lg overflow-hidden bg-bg-tertiary border border-white/5">
            {items.map((item, idx) => (
                <div key={idx} className="flex-1 px-4 py-3 text-center">
                    <div className="text-[10px] font-medium tracking-wider uppercase mb-1 text-text-muted">
                        {item.label}
                    </div>
                    <div 
                        className="text-lg font-mono"
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
        <div className="relative my-6">
            <div className="absolute inset-0 flex items-center" aria-hidden="true">
                <div className="w-full h-px bg-white/5" />
            </div>
            {label && (
                <div className="relative flex justify-center">
                    <span className="px-3 text-xs font-medium tracking-wide uppercase bg-bg-secondary text-text-muted">
                        {label}
                    </span>
                </div>
            )}
        </div>
    );
};

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
            className={`rounded-xl transition-all duration-300 ${className}`}
            style={{
                backgroundColor: 'var(--color-bg-secondary)',
                border: '1px solid var(--color-border-subtle)',
            }}
        >
            {/* Widget Header - Bloomberg Editorial Style */}
            <div 
                className={`flex items-center justify-between ${compact ? 'px-5 py-4' : 'px-6 py-5'} border-b`}
                style={{ borderColor: 'var(--color-border-subtle)' }}
            >
                <div className="flex items-center gap-3">
                    {/* Accent Bar */}
                    <div 
                        className="w-1 h-5 rounded-full"
                        style={{ backgroundColor: 'var(--color-accent-primary)' }}
                    />
                    
                    {/* Title */}
                    <h3
                        className="text-xs font-semibold tracking-[0.08em] uppercase"
                        style={{
                            fontFamily: 'var(--font-body)',
                            color: 'var(--color-text-secondary)',
                        }}
                    >
                        {title}
                    </h3>

                    {/* Info Tooltip */}
                    {tooltip && (
                        <div className="relative">
                            <button
                                className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold transition-all duration-200"
                                style={{
                                    backgroundColor: 'var(--color-bg-surface)',
                                    color: 'var(--color-text-muted)',
                                    border: '1px solid var(--color-border-subtle)',
                                }}
                                onMouseEnter={() => setShowTooltip(true)}
                                onMouseLeave={() => setShowTooltip(false)}
                                onClick={() => setShowTooltip(!showTooltip)}
                                aria-label="Information"
                            >
                                ?
                            </button>
                            
                            {/* Tooltip Popup */}
                            {showTooltip && (
                                <div
                                    className="absolute z-50 w-80 p-4 rounded-lg shadow-xl animate-fade-in"
                                    style={{
                                        left: '50%',
                                        top: 'calc(100% + 8px)',
                                        transform: 'translateX(-50%)',
                                        backgroundColor: 'var(--color-bg-elevated)',
                                        border: '1px solid var(--color-border-default)',
                                    }}
                                >
                                    {/* Tooltip Arrow */}
                                    <div 
                                        className="absolute w-3 h-3 rotate-45"
                                        style={{
                                            top: '-6px',
                                            left: '50%',
                                            transform: 'translateX(-50%)',
                                            backgroundColor: 'var(--color-bg-elevated)',
                                            borderLeft: '1px solid var(--color-border-default)',
                                            borderTop: '1px solid var(--color-border-default)',
                                        }}
                                    />
                                    
                                    <div className="relative">
                                        <div className="flex justify-between items-start mb-2">
                                            <span
                                                className="text-[10px] font-semibold tracking-wide uppercase"
                                                style={{ color: 'var(--color-accent-primary)' }}
                                            >
                                                Info
                                            </span>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    setShowTooltip(false);
                                                }}
                                                className="text-sm leading-none"
                                                style={{ color: 'var(--color-text-muted)' }}
                                            >
                                                ×
                                            </button>
                                        </div>
                                        <div 
                                            className="text-sm leading-relaxed"
                                            style={{ color: 'var(--color-text-secondary)' }}
                                        >
                                            {tooltip}
                                        </div>
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
                        className="text-xs font-medium px-3 py-1.5 rounded transition-colors"
                        style={{
                            backgroundColor: 'var(--color-bg-surface)',
                            color: 'var(--color-text-secondary)',
                            border: '1px solid var(--color-border-subtle)',
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.borderColor = 'var(--color-accent-primary)';
                            e.currentTarget.style.color = 'var(--color-accent-primary)';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.borderColor = 'var(--color-border-subtle)';
                            e.currentTarget.style.color = 'var(--color-text-secondary)';
                        }}
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

// Specialized Widget variants for different use cases

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
            className="p-4 rounded-lg"
            style={{ 
                backgroundColor: 'var(--color-bg-tertiary)',
                border: '1px solid var(--color-border-subtle)'
            }}
        >
            <div 
                className="text-xs font-medium tracking-wide uppercase mb-2"
                style={{ color: 'var(--color-text-muted)' }}
            >
                {label}
            </div>
            <div className="flex items-baseline gap-2">
                <span 
                    className="text-2xl font-semibold font-mono"
                    style={{ color: 'var(--color-text-primary)' }}
                >
                    {value}
                </span>
                {change && (
                    <span 
                        className="text-sm font-mono font-medium"
                        style={{ color: isPositive ? 'var(--color-positive)' : 'var(--color-negative)' }}
                    >
                        {change}
                    </span>
                )}
            </div>
            {sublabel && (
                <div 
                    className="text-xs mt-1"
                    style={{ color: 'var(--color-text-subtle)' }}
                >
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
        <div 
            className="flex divide-x rounded-lg overflow-hidden"
            style={{ 
                backgroundColor: 'var(--color-bg-tertiary)',
                borderColor: 'var(--color-border-subtle)'
            }}
        >
            {items.map((item, idx) => (
                <div key={idx} className="flex-1 px-4 py-3 text-center">
                    <div 
                        className="text-[10px] font-medium tracking-wider uppercase mb-1"
                        style={{ color: 'var(--color-text-muted)' }}
                    >
                        {item.label}
                    </div>
                    <div 
                        className="text-lg font-semibold font-mono"
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
            <div 
                className="absolute inset-0 flex items-center"
                aria-hidden="true"
            >
                <div 
                    className="w-full h-px"
                    style={{ backgroundColor: 'var(--color-border-subtle)' }}
                />
            </div>
            {label && (
                <div className="relative flex justify-center">
                    <span 
                        className="px-3 text-xs font-medium tracking-wide uppercase"
                        style={{ 
                            backgroundColor: 'var(--color-bg-secondary)',
                            color: 'var(--color-text-muted)'
                        }}
                    >
                        {label}
                    </span>
                </div>
            )}
        </div>
    );
};

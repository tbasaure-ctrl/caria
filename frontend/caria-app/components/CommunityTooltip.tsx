import React from 'react';

interface CommunityTooltipProps {
    description: string;
    community: string;
}

export const CommunityTooltip: React.FC<CommunityTooltipProps> = ({
    description,
    community,
}) => {
    return (
        <div
            className="absolute z-10 p-3 rounded-lg shadow-lg max-w-xs"
            style={{
                backgroundColor: 'var(--color-bg-primary)',
                border: '1px solid var(--color-bg-tertiary)',
                top: '100%',
                left: '50%',
                transform: 'translateX(-50%)',
                marginTop: '8px',
            }}
        >
            <div 
                className="text-sm"
                style={{ 
                    color: 'var(--color-text-primary)',
                    fontFamily: 'var(--font-body)',
                }}
            >
                {description}
            </div>
            {/* Arrow */}
            <div
                className="absolute -top-2 left-1/2 transform -translate-x-1/2 w-0 h-0"
                style={{
                    borderLeft: '8px solid transparent',
                    borderRight: '8px solid transparent',
                    borderBottom: '8px solid var(--color-bg-primary)',
                }}
            />
        </div>
    );
};


import React from 'react';

interface CommunityInfo {
    name: string;
    icon: string;
    description: string;
    color: string;
}

interface CommunityCardProps {
    community: CommunityInfo;
    response: string;
    impactScore: number;
}

export const CommunityCard: React.FC<CommunityCardProps> = ({
    community,
    response,
    impactScore,
}) => {
    const getImpactColor = () => {
        if (impactScore > 0.1) return '#10b981'; // green
        if (impactScore < -0.1) return '#ef4444'; // red
        return '#6b7280'; // gray
    };

    const getImpactLabel = () => {
        if (impactScore > 0.1) return 'Positivo';
        if (impactScore < -0.1) return 'Negativo';
        return 'Neutral';
    };

    return (
        <div 
            className="rounded-lg p-4 h-full transition-all"
            style={{
                backgroundColor: 'var(--color-bg-secondary)',
                border: `2px solid ${community.color}40`,
                boxShadow: `0 2px 8px ${community.color}20`,
            }}
        >
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <span className="text-2xl">{community.icon}</span>
                    <h3 
                        className="font-semibold"
                        style={{ 
                            color: community.color,
                            fontFamily: 'var(--font-display)',
                        }}
                    >
                        {community.name}
                    </h3>
                </div>
                <div 
                    className="px-2 py-1 rounded text-xs font-medium"
                    style={{
                        backgroundColor: `${getImpactColor()}20`,
                        color: getImpactColor(),
                    }}
                >
                    {getImpactLabel()}
                </div>
            </div>

            {/* Response */}
            <div 
                className="text-sm leading-relaxed"
                style={{ 
                    color: 'var(--color-text-primary)',
                    fontFamily: 'var(--font-body)',
                }}
            >
                {response}
            </div>

            {/* Impact Score */}
            <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--color-bg-tertiary)' }}>
                <div className="flex items-center justify-between text-xs">
                    <span style={{ color: 'var(--color-text-secondary)' }}>Impacto:</span>
                    <span 
                        className="font-medium"
                        style={{ color: getImpactColor() }}
                    >
                        {impactScore >= 0 ? '+' : ''}{impactScore.toFixed(2)}
                    </span>
                </div>
            </div>
        </div>
    );
};


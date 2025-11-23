import React from 'react';

interface ProtectionVisualizationProps {
    protectionLevel: 'high' | 'medium' | 'low';
    exposureScore: number; // 0-100
}

export const ProtectionVisualization: React.FC<ProtectionVisualizationProps> = ({
    protectionLevel,
    exposureScore,
}) => {
    const getProtectionColor = () => {
        switch (protectionLevel) {
            case 'high':
                return '#10b981'; // green
            case 'medium':
                return '#f59e0b'; // amber
            case 'low':
                return '#ef4444'; // red
            default:
                return '#6b7280'; // gray
        }
    };

    const getProtectionLabel = () => {
        switch (protectionLevel) {
            case 'high':
                return 'Alta Protección';
            case 'medium':
                return 'Protección Moderada';
            case 'low':
                return 'Baja Protección';
            default:
                return 'Desconocido';
        }
    };

    // Calculate gauge angle (0-180 degrees, where 0 = left, 180 = right)
    // Score 0-100 maps to 0-180 degrees
    const gaugeAngle = (exposureScore / 100) * 180;

    return (
        <div className="space-y-4">
            <div className="text-center">
                <h3 
                    className="text-lg font-semibold mb-2"
                    style={{ 
                        fontFamily: 'var(--font-display)',
                        color: 'var(--color-cream)',
                    }}
                >
                    Nivel de Protección
                </h3>
                
                {/* Gauge Visualization */}
                <div className="relative w-full max-w-xs mx-auto">
                    <svg 
                        viewBox="0 0 200 120" 
                        className="w-full h-auto"
                    >
                        {/* Background arc */}
                        <path
                            d="M 20 100 A 80 80 0 0 1 180 100"
                            fill="none"
                            stroke="var(--color-bg-tertiary)"
                            strokeWidth="12"
                            strokeLinecap="round"
                        />
                        
                        {/* Colored arc based on score */}
                        <path
                            d={`M 20 100 A 80 80 0 ${gaugeAngle > 90 ? 1 : 0} 1 ${20 + (gaugeAngle / 180) * 160} ${100 - 80 * Math.sin((gaugeAngle * Math.PI) / 180)}`}
                            fill="none"
                            stroke={getProtectionColor()}
                            strokeWidth="12"
                            strokeLinecap="round"
                        />
                        
                        {/* Needle */}
                        <line
                            x1="100"
                            y1="100"
                            x2={100 + 70 * Math.cos(((180 - gaugeAngle) * Math.PI) / 180)}
                            y2={100 - 70 * Math.sin(((180 - gaugeAngle) * Math.PI) / 180)}
                            stroke={getProtectionColor()}
                            strokeWidth="3"
                            strokeLinecap="round"
                        />
                        
                        {/* Center dot */}
                        <circle
                            cx="100"
                            cy="100"
                            r="6"
                            fill={getProtectionColor()}
                        />
                    </svg>
                    
                    {/* Score Display */}
                    <div 
                        className="absolute bottom-0 left-1/2 transform -translate-x-1/2 text-center"
                        style={{ width: '200px' }}
                    >
                        <div 
                            className="text-3xl font-bold"
                            style={{ 
                                color: getProtectionColor(),
                                fontFamily: 'var(--font-display)',
                            }}
                        >
                            {exposureScore.toFixed(1)}%
                        </div>
                        <div 
                            className="text-sm mt-1"
                            style={{ 
                                color: 'var(--color-text-secondary)',
                                fontFamily: 'var(--font-body)',
                            }}
                        >
                            {getProtectionLabel()}
                        </div>
                    </div>
                </div>
            </div>

            {/* Protection Level Badge */}
            <div className="flex justify-center">
                <div
                    className="px-4 py-2 rounded-full text-sm font-medium"
                    style={{
                        backgroundColor: `${getProtectionColor()}20`,
                        color: getProtectionColor(),
                        border: `1px solid ${getProtectionColor()}40`,
                    }}
                >
                    {getProtectionLabel()}
                </div>
            </div>
        </div>
    );
};


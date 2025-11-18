import React from 'react';

interface RegimeTestResultsProps {
    results: {
        regime: string;
        exposure_score: number;
        protection_level: 'high' | 'medium' | 'low';
        drawdown_estimate: {
            worst_case_p5: number;
            max_drawdown_pct: number;
            median: number;
            best_case_p95: number;
        };
        monte_carlo_results: any;
        recommendations: string[];
    };
}

const REGIME_LABELS: Record<string, string> = {
    expansion: 'Expansión',
    slowdown: 'Desaceleración',
    recession: 'Recesión',
    stress: 'Estrés',
};

export const RegimeTestResults: React.FC<RegimeTestResultsProps> = ({ results }) => {
    const formatCurrency = (value: number) => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 0,
            maximumFractionDigits: 0,
        }).format(value);
    };

    const formatPercent = (value: number) => {
        return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
    };

    return (
        <div className="space-y-4">
            {/* Regime Header */}
            <div className="text-center pb-4 border-b" style={{ borderColor: 'var(--color-bg-tertiary)' }}>
                <h3 
                    className="text-xl font-bold"
                    style={{ 
                        fontFamily: 'var(--font-display)',
                        color: 'var(--color-cream)',
                    }}
                >
                    Resultados para: {REGIME_LABELS[results.regime] || results.regime}
                </h3>
            </div>

            {/* Drawdown Estimates */}
            <div>
                <h4 
                    className="text-sm font-semibold mb-3"
                    style={{ 
                        fontFamily: 'var(--font-display)',
                        color: 'var(--color-text-secondary)',
                    }}
                >
                    Estimaciones de Drawdown (12 meses)
                </h4>
                <div className="grid grid-cols-3 gap-3">
                    {/* Worst Case */}
                    <div 
                        className="p-3 rounded-lg text-center"
                        style={{ backgroundColor: 'var(--color-bg-secondary)' }}
                    >
                        <div 
                            className="text-xs mb-1"
                            style={{ color: 'var(--color-text-secondary)' }}
                        >
                            Peor Caso (P5)
                        </div>
                        <div 
                            className="text-lg font-bold"
                            style={{ color: '#ef4444' }}
                        >
                            {formatPercent(results.drawdown_estimate.max_drawdown_pct)}
                        </div>
                        <div 
                            className="text-xs mt-1"
                            style={{ color: 'var(--color-text-secondary)' }}
                        >
                            {formatCurrency(results.drawdown_estimate.worst_case_p5)}
                        </div>
                    </div>

                    {/* Median */}
                    <div 
                        className="p-3 rounded-lg text-center"
                        style={{ backgroundColor: 'var(--color-bg-secondary)' }}
                    >
                        <div 
                            className="text-xs mb-1"
                            style={{ color: 'var(--color-text-secondary)' }}
                        >
                            Mediana (P50)
                        </div>
                        <div 
                            className="text-lg font-bold"
                            style={{ color: 'var(--color-text-primary)' }}
                        >
                            {formatPercent(((results.drawdown_estimate.median - 100000) / 100000) * 100)}
                        </div>
                        <div 
                            className="text-xs mt-1"
                            style={{ color: 'var(--color-text-secondary)' }}
                        >
                            {formatCurrency(results.drawdown_estimate.median)}
                        </div>
                    </div>

                    {/* Best Case */}
                    <div 
                        className="p-3 rounded-lg text-center"
                        style={{ backgroundColor: 'var(--color-bg-secondary)' }}
                    >
                        <div 
                            className="text-xs mb-1"
                            style={{ color: 'var(--color-text-secondary)' }}
                        >
                            Mejor Caso (P95)
                        </div>
                        <div 
                            className="text-lg font-bold"
                            style={{ color: '#10b981' }}
                        >
                            {formatPercent(((results.drawdown_estimate.best_case_p95 - 100000) / 100000) * 100)}
                        </div>
                        <div 
                            className="text-xs mt-1"
                            style={{ color: 'var(--color-text-secondary)' }}
                        >
                            {formatCurrency(results.drawdown_estimate.best_case_p95)}
                        </div>
                    </div>
                </div>
            </div>

            {/* Recommendations */}
            {results.recommendations && results.recommendations.length > 0 && (
                <div>
                    <h4 
                        className="text-sm font-semibold mb-3"
                        style={{ 
                            fontFamily: 'var(--font-display)',
                            color: 'var(--color-text-secondary)',
                        }}
                    >
                        Recomendaciones
                    </h4>
                    <div className="space-y-2">
                        {results.recommendations.map((rec, index) => (
                            <div 
                                key={index}
                                className="p-3 rounded-lg flex items-start gap-2"
                                style={{ backgroundColor: 'var(--color-bg-secondary)' }}
                            >
                                <span 
                                    className="text-sm mt-0.5"
                                    style={{ color: 'var(--color-primary)' }}
                                >
                                    •
                                </span>
                                <p 
                                    className="text-sm flex-1"
                                    style={{ 
                                        color: 'var(--color-text-primary)',
                                        fontFamily: 'var(--font-body)',
                                    }}
                                >
                                    {rec}
                                </p>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};


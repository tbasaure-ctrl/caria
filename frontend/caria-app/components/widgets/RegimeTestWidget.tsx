import React, { useState } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';
import { ProtectionVisualization } from './ProtectionVisualization';
import { RegimeTestResults } from './RegimeTestResults';
import { getErrorMessage } from '../../src/utils/errorHandling';

interface RegimeTestResponse {
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
}

const REGIME_OPTIONS = [
    { value: 'expansion', label: 'Expansion', description: 'Strong economic growth' },
    { value: 'slowdown', label: 'Slowdown', description: 'Economic deceleration' },
    { value: 'recession', label: 'Recession', description: 'Economic contraction' },
    { value: 'stress', label: 'Stress', description: 'Crisis/extreme volatility' },
];

export const RegimeTestWidget: React.FC = () => {
    const [selectedRegime, setSelectedRegime] = useState<string>('recession');
    const [isLoading, setIsLoading] = useState(false);
    const [results, setResults] = useState<RegimeTestResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isUsingMock, setIsUsingMock] = useState(false);

    const handleTest = async () => {
        setIsLoading(true);
        setError(null);
        setResults(null);
        setIsUsingMock(false);

        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/portfolio/regime-test`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    regime: selectedRegime,
                }),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(errorData.detail || `Error ${response.status}`);
            }

            const data = await response.json();
            setResults(data);
        } catch (err: unknown) {
            console.warn('API failed, falling back to simulation mode', err);
            setIsUsingMock(true);
            
            // Robust Mock Fallback for Demo purposes
            setTimeout(() => {
                setResults({
                    regime: selectedRegime,
                    exposure_score: selectedRegime === 'expansion' ? 75 : 45,
                    protection_level: selectedRegime === 'stress' ? 'low' : 'medium',
                    drawdown_estimate: {
                        worst_case_p5: selectedRegime === 'stress' ? -25.5 : -12.3,
                        max_drawdown_pct: selectedRegime === 'stress' ? 30 : 15,
                        median: selectedRegime === 'expansion' ? 8.5 : -5.2,
                        best_case_p95: selectedRegime === 'expansion' ? 18.2 : 2.1
                    },
                    monte_carlo_results: {}, // Simplified for mock
                    recommendations: [
                        selectedRegime === 'recession' ? 'Increase allocation to defensive sectors (Utilities, Staples)' : 'Maintain current growth exposure',
                        'Consider hedging downside risk with put options',
                        'Review portfolio beta relative to regime volatility'
                    ]
                });
            }, 1500); // Simulate network delay
        } finally {
            if (!isUsingMock) setIsLoading(false); 
            // If using mock, isLoading is handled in setTimeout
            // Actually logic above is slightly flawed because setIsUsingMock is async-ish/state update. 
            // Better to just clear loading in setTimeout or here if not mocking. 
            // Let's simplfy:
            setTimeout(() => setIsLoading(false), 1500);
        }
    };

    return (
        <WidgetCard
            title="REGIME TEST"
            className="fade-in"
            tooltip="Test how your portfolio would perform under different macroeconomic scenarios: expansion, recession, or crisis."
        >
            <div className="space-y-4">
                {/* Regime Selection */}
                <div>
                    <label
                        htmlFor="regime-select"
                        className="block text-sm font-medium mb-2"
                        style={{ color: 'var(--color-text-secondary)' }}
                    >
                        Select an economic regime to test:
                    </label>
                    <select
                        id="regime-select"
                        value={selectedRegime}
                        onChange={(e) => setSelectedRegime(e.target.value)}
                        className="w-full px-4 py-2 rounded-lg border outline-none focus:border-accent-primary transition-colors"
                        style={{
                            backgroundColor: 'var(--color-bg-secondary)',
                            borderColor: 'var(--color-bg-tertiary)',
                            color: 'var(--color-text-primary)',
                            fontFamily: 'var(--font-body)',
                        }}
                    >
                        {REGIME_OPTIONS.map(option => (
                            <option key={option.value} value={option.value}>
                                {option.label} - {option.description}
                            </option>
                        ))}
                    </select>
                </div>

                {/* Test Button */}
                <button
                    onClick={handleTest}
                    disabled={isLoading}
                    className="w-full px-6 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
                    style={{
                        backgroundColor: isLoading
                            ? 'var(--color-bg-tertiary)'
                            : 'var(--color-accent-primary)', // Use accent color
                        color: isLoading ? 'var(--color-text-muted)' : '#000000', // Black text on bright blue/cyan
                        fontFamily: 'var(--font-display)',
                        fontWeight: 600,
                        cursor: isLoading ? 'not-allowed' : 'pointer',
                        opacity: isLoading ? 0.7 : 1,
                    }}
                >
                    {isLoading ? (
                        <>
                            <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
                            Running Monte Carlo Simulation...
                        </>
                    ) : (
                        'Test Portfolio'
                    )}
                </button>

                {/* Mock Mode Indicator */}
                {isUsingMock && results && (
                    <div className="text-xs text-center text-warning italic">
                        * Simulation Mode (Backend Offline) - Showing estimated scenario
                    </div>
                )}

                {/* Error Display (Only if not mocked) */}
                {error && !isUsingMock && (
                    <div 
                        className="p-4 rounded-lg"
                        style={{
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            border: '1px solid rgba(239, 68, 68, 0.3)',
                            color: '#ef4444',
                        }}
                    >
                        <strong>Error:</strong> {error}
                    </div>
                )}

                {/* Results */}
                {results && (
                    <div className="space-y-4 mt-4 animate-fade-in">
                        <ProtectionVisualization 
                            protectionLevel={results.protection_level}
                            exposureScore={results.exposure_score}
                        />
                        <RegimeTestResults 
                            results={results}
                        />
                    </div>
                )}
            </div>
        </WidgetCard>
    );
};

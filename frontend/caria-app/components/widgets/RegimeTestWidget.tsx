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

    const handleTest = async () => {
        setIsLoading(true);
        setError(null);
        setResults(null);

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
            setError('Coming soon... Regime testing features are being enhanced to provide better portfolio stress analysis.');
        } finally {
            setIsLoading(false);
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
                        className="w-full px-4 py-2 rounded-lg border"
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
                    className="w-full px-6 py-3 rounded-lg font-medium transition-all"
                    style={{
                        backgroundColor: isLoading
                            ? 'var(--color-bg-tertiary)'
                            : 'var(--color-primary)',
                        color: 'var(--color-cream)',
                        fontFamily: 'var(--font-display)',
                        cursor: isLoading ? 'not-allowed' : 'pointer',
                        opacity: isLoading ? 0.6 : 1,
                    }}
                >
                    {isLoading ? 'Testing...' : 'Test Portfolio'}
                </button>

                {/* Error Display */}
                {error && (
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
                    <div className="space-y-4 mt-4">
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


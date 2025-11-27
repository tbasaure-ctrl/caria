import React, { useState } from 'react';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';

interface ProjectionData {
    [key: string]: {
        [year: number]: number;
    };
}

interface ProjectionValuationResponse {
    ticker: string;
    current_price: number;
    target_price_2029: number;
    upside: number;
    base_revenue: number;
    projection_data: ProjectionData;
    base_data: {
        revenue: number;
        operating_income: number;
        net_income: number;
        shares_outstanding: number;
        price: number;
        fcf: number;
    };
}

export const ProjectionValuation: React.FC = () => {
    const [ticker, setTicker] = useState('');
    const [data, setData] = useState<ProjectionValuationResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [macroRisk, setMacroRisk] = useState(0);
    const [industryRisk, setIndustryRisk] = useState(0);

    const handleCalculate = async () => {
        if (!ticker.trim()) {
            setError('Please enter a ticker symbol');
            return;
        }

        setLoading(true);
        setError(null);
        try {
            const response = await fetchWithAuth(
                `${API_BASE_URL}/api/valuation/projection/${ticker.toUpperCase()}?macro_risk=${macroRisk}&industry_risk=${industryRisk}`
            );
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch valuation' }));
                throw new Error(errorData.detail || 'Failed to fetch valuation');
            }
            
            const result = await response.json();
            setData(result);
        } catch (err: any) {
            console.error("Failed to fetch valuation", err);
            setError(err.message || 'Failed to calculate valuation');
        } finally {
            setLoading(false);
        }
    };

    const formatMoney = (value: number) => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        }).format(value);
    };

    const formatNumber = (value: number) => {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        }).format(value);
    };

    return (
        <div className="space-y-4">
            {/* Input Section */}
            <div className="flex flex-col gap-4">
                <div className="flex gap-3 items-end">
                    <div className="flex-1">
                        <label 
                            className="block text-xs font-medium mb-1.5 uppercase tracking-wider"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            Ticker Symbol
                        </label>
                        <input
                            type="text"
                            value={ticker}
                            onChange={(e) => setTicker(e.target.value.toUpperCase())}
                            placeholder="e.g. NVDA"
                            className="w-full px-3 py-2 rounded-lg text-sm font-mono"
                            style={{
                                backgroundColor: 'var(--color-bg-tertiary)',
                                border: '1px solid var(--color-border-subtle)',
                                color: 'var(--color-text-primary)',
                            }}
                            onKeyPress={(e) => {
                                if (e.key === 'Enter') {
                                    handleCalculate();
                                }
                            }}
                        />
                    </div>
                    <button
                        onClick={handleCalculate}
                        disabled={loading || !ticker.trim()}
                        className="px-5 py-2 rounded-lg text-sm font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                        style={{
                            backgroundColor: 'var(--color-accent-primary)',
                            color: '#FFFFFF',
                        }}
                    >
                        {loading ? 'Calculating...' : 'Run Valuation Model'}
                    </button>
                </div>

                {/* Risk Adjustments - Sliders */}
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <div className="flex justify-between mb-1">
                            <label 
                                className="text-xs font-medium uppercase tracking-wider"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Macro Risk
                            </label>
                            <span 
                                className="text-xs font-mono"
                                style={{ color: 'var(--color-text-secondary)' }}
                            >
                                {Math.round(macroRisk * 100)}%
                            </span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={macroRisk}
                            onChange={(e) => setMacroRisk(parseFloat(e.target.value))}
                            className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                            style={{
                                backgroundColor: 'var(--color-bg-tertiary)',
                                accentColor: 'var(--color-accent-primary)',
                            }}
                        />
                    </div>
                    <div>
                        <div className="flex justify-between mb-1">
                            <label 
                                className="text-xs font-medium uppercase tracking-wider"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Industry Risk
                            </label>
                            <span 
                                className="text-xs font-mono"
                                style={{ color: 'var(--color-text-secondary)' }}
                            >
                                {Math.round(industryRisk * 100)}%
                            </span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={industryRisk}
                            onChange={(e) => setIndustryRisk(parseFloat(e.target.value))}
                            className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                            style={{
                                backgroundColor: 'var(--color-bg-tertiary)',
                                accentColor: 'var(--color-positive)',
                            }}
                        />
                    </div>
                </div>
            </div>

            {/* Error Message */}
            {error && (
                <div 
                    className="rounded-xl p-4"
                    style={{
                        backgroundColor: 'var(--color-negative-muted)',
                        border: '1px solid var(--color-negative)',
                        color: 'var(--color-negative)',
                    }}
                >
                    {error}
                </div>
            )}

            {/* Results */}
            {data && (
                <div className="space-y-4">
                    {/* Summary Row */}
                    <div 
                        className="grid grid-cols-3 gap-3 p-4 rounded-lg"
                        style={{
                            backgroundColor: 'var(--color-bg-tertiary)',
                            border: '1px solid var(--color-border-subtle)',
                        }}
                    >
                        <div className="text-center">
                            <div 
                                className="text-[10px] font-medium uppercase tracking-wider mb-1"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Current
                            </div>
                            <div 
                                className="text-lg font-bold font-mono"
                                style={{ color: 'var(--color-text-primary)' }}
                            >
                                {formatMoney(data.current_price)}
                            </div>
                        </div>
                        <div className="text-center">
                            <div 
                                className="text-[10px] font-medium uppercase tracking-wider mb-1"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                2029 Target
                            </div>
                            <div 
                                className="text-lg font-bold font-mono"
                                style={{ color: 'var(--color-positive)' }}
                            >
                                {formatMoney(data.target_price_2029)}
                            </div>
                        </div>
                        <div className="text-center">
                            <div 
                                className="text-[10px] font-medium uppercase tracking-wider mb-1"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Upside
                            </div>
                            <div 
                                className="text-lg font-bold font-mono"
                                style={{ 
                                    color: data.upside >= 0 
                                        ? 'var(--color-positive)' 
                                        : 'var(--color-negative)' 
                                }}
                            >
                                {data.upside >= 0 ? '+' : ''}{formatNumber(data.upside)}%
                            </div>
                        </div>
                    </div>

                    {/* Projection Table - Collapsible */}
                    {data.projection_data && (
                        <details 
                            className="rounded-lg overflow-hidden"
                            style={{
                                backgroundColor: 'var(--color-bg-tertiary)',
                                border: '1px solid var(--color-border-subtle)',
                            }}
                        >
                            <summary 
                                className="px-4 py-3 cursor-pointer text-sm font-medium flex items-center justify-between"
                                style={{ color: 'var(--color-text-secondary)' }}
                            >
                                <span>View Projection Details (2024-2029)</span>
                                <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>Click to expand</span>
                            </summary>
                            <div className="p-4 pt-0 overflow-x-auto">
                                <table className="w-full text-xs">
                                    <thead>
                                        <tr style={{ borderBottom: '1px solid var(--color-border-subtle)' }}>
                                            <th 
                                                className="text-left py-2 px-2 font-semibold"
                                                style={{ color: 'var(--color-text-muted)' }}
                                            >
                                                Metric
                                            </th>
                                            {[2024, 2025, 2026, 2027, 2028, 2029].map(year => (
                                                <th 
                                                    key={year}
                                                    className="text-right py-2 px-2 font-semibold"
                                                    style={{ color: 'var(--color-text-muted)' }}
                                                >
                                                    {year}
                                                </th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {Object.entries(data.projection_data).map(([metric, values]) => (
                                            <tr 
                                                key={metric}
                                                style={{ borderBottom: '1px solid var(--color-border-subtle)' }}
                                            >
                                                <td 
                                                    className="py-2 px-2 font-medium"
                                                    style={{ color: 'var(--color-text-primary)' }}
                                                >
                                                    {metric}
                                                </td>
                                                {[2024, 2025, 2026, 2027, 2028, 2029].map(year => {
                                                    const value = values[year];
                                                    const isPercentage = metric.includes('Growth') || metric.includes('Margin');
                                                    const isMoney = metric.includes('Revenue') || 
                                                                   metric.includes('Income') || 
                                                                   metric.includes('Cash Flow') ||
                                                                   metric.includes('Buybacks') ||
                                                                   metric.includes('Price') ||
                                                                   metric.includes('Target');
                                                    
                                                    return (
                                                        <td 
                                                            key={year}
                                                            className="text-right py-2 px-2 font-mono"
                                                            style={{ color: 'var(--color-text-secondary)' }}
                                                        >
                                                            {isMoney 
                                                                ? formatMoney(value)
                                                                : isPercentage
                                                                ? `${formatNumber(value * 100)}%`
                                                                : formatNumber(value)
                                                            }
                                                        </td>
                                                    );
                                                })}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </details>
                    )}
                </div>
            )}
        </div>
    );
};


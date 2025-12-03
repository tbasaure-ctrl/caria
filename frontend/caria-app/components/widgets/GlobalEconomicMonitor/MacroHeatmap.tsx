import React, { useMemo } from 'react';
import { HeatmapCell } from '../../../services/apiService';

interface MacroHeatmapProps {
    data: HeatmapCell[];
    loading: boolean;
    onCountrySelect?: (countryCode: string) => void;
}

const STATUS_COLORS = {
    health: '#10b981', // Green
    warning: '#f59e0b', // Amber
    deterioration: '#ef4444', // Red
};

export const MacroHeatmap: React.FC<MacroHeatmapProps> = ({
    data,
    loading,
    onCountrySelect,
}) => {
    const { countries, indicators, gridData } = useMemo(() => {
        if (!data || data.length === 0) {
            return { countries: [], indicators: [], gridData: {} };
        }

        const countrySet = new Set<string>();
        const indicatorSet = new Set<string>();
        
        data.forEach(cell => {
            countrySet.add(cell.country_code);
            indicatorSet.add(cell.indicator_name);
        });

        const countries = Array.from(countrySet).sort();
        const indicators = Array.from(indicatorSet).sort();

        // Create grid data structure
        const grid: Record<string, Record<string, HeatmapCell>> = {};
        countries.forEach(country => {
            grid[country] = {};
            indicators.forEach(indicator => {
                const cell = data.find(
                    c => c.country_code === country && c.indicator_name === indicator
                );
                if (cell) {
                    grid[country][indicator] = cell;
                }
            });
        });

        return { countries, indicators, gridData: grid };
    }, [data]);

    if (loading) {
        return (
            <div className="flex items-center justify-center h-[400px]">
                <div className="text-text-muted text-sm">Loading heatmap data...</div>
            </div>
        );
    }

    if (countries.length === 0 || indicators.length === 0) {
        return (
            <div className="flex items-center justify-center h-[400px]">
                <div className="text-text-muted text-sm">No data available</div>
            </div>
        );
    }

    const getCellColor = (cell: HeatmapCell | undefined) => {
        if (!cell) return 'rgba(255,255,255,0.05)';
        return STATUS_COLORS[cell.status] || 'rgba(255,255,255,0.1)';
    };

    const getCellOpacity = (cell: HeatmapCell | undefined) => {
        if (!cell) return 0.1;
        // Use normalized_value for opacity (0-1 range)
        return Math.max(0.3, Math.min(1, cell.normalized_value));
    };

    return (
        <div className="space-y-4">
            <div className="text-xs text-text-muted">
                <p>Color intensity represents Z-score deviation from historical average</p>
            </div>

            <div className="overflow-x-auto">
                <div className="inline-block min-w-full">
                    <table className="w-full border-collapse">
                        <thead>
                            <tr>
                                <th className="sticky left-0 z-10 bg-bg-primary border border-white/10 px-3 py-2 text-left text-xs font-medium text-text-secondary">
                                    Country
                                </th>
                                {indicators.map(indicator => (
                                    <th
                                        key={indicator}
                                        className="border border-white/10 px-2 py-2 text-center text-[10px] font-medium text-text-secondary min-w-[100px]"
                                    >
                                        {indicator}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {countries.map(country => {
                                const countryName = data.find(c => c.country_code === country)?.country_name || country;
                                return (
                                    <tr
                                        key={country}
                                        className="hover:bg-white/5 cursor-pointer transition-colors"
                                        onClick={() => onCountrySelect?.(country)}
                                    >
                                        <td className="sticky left-0 z-10 bg-bg-primary border border-white/10 px-3 py-2 text-xs text-white font-medium">
                                            {countryName}
                                        </td>
                                        {indicators.map(indicator => {
                                            const cell = gridData[country]?.[indicator];
                                            const color = getCellColor(cell);
                                            const opacity = getCellOpacity(cell);
                                            
                                            return (
                                                <td
                                                    key={indicator}
                                                    className="border border-white/10 px-2 py-2 text-center"
                                                    style={{
                                                        backgroundColor: color,
                                                        opacity: opacity,
                                                    }}
                                                    title={cell ? `Z-score: ${cell.z_score.toFixed(2)}, Value: ${cell.value.toFixed(2)}` : ''}
                                                >
                                                    {cell && (
                                                        <span className="text-[10px] text-white font-mono">
                                                            {cell.z_score.toFixed(1)}
                                                        </span>
                                                    )}
                                                </td>
                                            );
                                        })}
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Legend */}
            <div className="flex flex-wrap gap-4 justify-center text-xs">
                {Object.entries(STATUS_COLORS).map(([status, color]) => (
                    <div key={status} className="flex items-center gap-2">
                        <div className="w-4 h-4 rounded" style={{ backgroundColor: color }} />
                        <span className="text-text-muted capitalize">{status}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};


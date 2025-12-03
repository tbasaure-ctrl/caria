import React, { useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from 'recharts';
import { BusinessCyclePoint } from '../../../services/apiService';

interface BusinessCycleClockProps {
    data: BusinessCyclePoint[];
    loading: boolean;
    onCountrySelect?: (countryCode: string) => void;
}

const PHASE_COLORS = {
    expansion: '#10b981', // Green
    slowdown: '#f59e0b', // Amber/Yellow
    recession: '#ef4444', // Red
    recovery: '#3b82f6', // Blue
};

export const BusinessCycleClock: React.FC<BusinessCycleClockProps> = ({
    data,
    loading,
    onCountrySelect,
}) => {
    const chartData = useMemo(() => {
        return data.map(point => ({
            x: point.x,
            y: point.y,
            country: point.country_name,
            code: point.country_code,
            phase: point.phase,
            trajectory: point.trajectory,
        }));
    }, [data]);

    if (loading) {
        return (
            <div className="flex items-center justify-center h-[400px]">
                <div className="text-text-muted text-sm">Loading business cycle data...</div>
            </div>
        );
    }

    if (chartData.length === 0) {
        return (
            <div className="flex items-center justify-center h-[400px]">
                <div className="text-text-muted text-sm">No data available</div>
            </div>
        );
    }

    // Calculate axis ranges
    const xValues = chartData.map(d => d.x);
    const yValues = chartData.map(d => d.y);
    const xMin = Math.min(...xValues, -2);
    const xMax = Math.max(...xValues, 2);
    const yMin = Math.min(...yValues, -2);
    const yMax = Math.max(...yValues, 2);

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            return (
                <div className="bg-bg-primary border border-white/20 rounded p-3 shadow-lg">
                    <p className="font-bold text-white">{data.country}</p>
                    <p className="text-xs text-text-secondary">Phase: <span className="capitalize">{data.phase}</span></p>
                    <p className="text-xs text-text-secondary">Momentum: {data.x.toFixed(3)}</p>
                    <p className="text-xs text-text-secondary">Deviation: {data.y.toFixed(3)}</p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="space-y-4">
            <div className="text-xs text-text-muted space-y-1">
                <p><strong>X-Axis (Momentum):</strong> Rate of change of economic activity</p>
                <p><strong>Y-Axis (Deviation):</strong> Deviation from long-term trend</p>
            </div>

            <ResponsiveContainer width="100%" height={400}>
                <ScatterChart
                    margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis
                        type="number"
                        dataKey="x"
                        name="Momentum"
                        domain={[xMin, xMax]}
                        stroke="rgba(255,255,255,0.5)"
                        tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 10 }}
                        label={{ value: 'Momentum', position: 'insideBottom', offset: -5, fill: 'rgba(255,255,255,0.7)' }}
                    />
                    <YAxis
                        type="number"
                        dataKey="y"
                        name="Deviation"
                        domain={[yMin, yMax]}
                        stroke="rgba(255,255,255,0.5)"
                        tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 10 }}
                        label={{ value: 'Deviation from Trend', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.7)' }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    
                    {/* Reference lines for quadrants */}
                    <ReferenceLine x={0} stroke="rgba(255,255,255,0.3)" strokeDasharray="2 2" />
                    <ReferenceLine y={0} stroke="rgba(255,255,255,0.3)" strokeDasharray="2 2" />
                    
                    {/* Quadrant labels */}
                    <text x="5%" y="10%" fill="rgba(16,185,129,0.7)" fontSize={10} fontWeight="bold">
                        Expansion
                    </text>
                    <text x="75%" y="10%" fill="rgba(245,158,11,0.7)" fontSize={10} fontWeight="bold">
                        Slowdown
                    </text>
                    <text x="75%" y="90%" fill="rgba(239,68,68,0.7)" fontSize={10} fontWeight="bold">
                        Recession
                    </text>
                    <text x="5%" y="90%" fill="rgba(59,130,246,0.7)" fontSize={10} fontWeight="bold">
                        Recovery
                    </text>

                    <Scatter
                        name="Countries"
                        data={chartData}
                        fill="#8884d8"
                        onClick={(data: any) => {
                            if (onCountrySelect && data.code) {
                                onCountrySelect(data.code);
                            }
                        }}
                        style={{ cursor: onCountrySelect ? 'pointer' : 'default' }}
                    >
                        {chartData.map((entry, index) => (
                            <Cell
                                key={`cell-${index}`}
                                fill={PHASE_COLORS[entry.phase as keyof typeof PHASE_COLORS] || '#8884d8'}
                            />
                        ))}
                    </Scatter>
                </ScatterChart>
            </ResponsiveContainer>

            {/* Legend */}
            <div className="flex flex-wrap gap-4 justify-center text-xs">
                {Object.entries(PHASE_COLORS).map(([phase, color]) => (
                    <div key={phase} className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
                        <span className="text-text-muted capitalize">{phase}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};


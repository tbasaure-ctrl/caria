
import React, { useState, useRef, useMemo, useEffect } from 'react';
import { fetchHoldingsWithPrices, HoldingsWithPrices, HoldingWithPrice } from '../../services/apiService';
import { ArrowUpIcon, ArrowDownIcon } from '../Icons';
import { WidgetCard } from './WidgetCard';

type AllocationData = { name: string; value: number; color: string };

const PieChart: React.FC<{ data: AllocationData[] }> = ({ data }) => {
    // ... (unchanged)
    const [activeSlice, setActiveSlice] = useState<AllocationData | null>(data.length > 0 ? data[0] : null);
    const total = data.reduce((acc, item) => acc + item.value, 0);
    
    const radius = 1;
    const innerRadius = 0.6;
    let cumulative = 0;

    return (
        <div className="flex items-center gap-4">
            <div className="relative w-24 h-24">
                 <svg viewBox="-1 -1 2 2" className="w-full h-full transform -rotate-90">
                    {data.map(item => {
                        const angleStart = (cumulative / total) * 2 * Math.PI;
                        cumulative += item.value;
                        const angleEnd = (cumulative / total) * 2 * Math.PI;
                        
                        const startX = Math.cos(angleStart) * radius;
                        const startY = Math.sin(angleStart) * radius;
                        const endX = Math.cos(angleEnd) * radius;
                        const endY = Math.sin(angleEnd) * radius;

                        const innerStartX = Math.cos(angleStart) * innerRadius;
                        const innerStartY = Math.sin(angleStart) * innerRadius;
                        const innerEndX = Math.cos(angleEnd) * innerRadius;
                        const innerEndY = Math.sin(angleEnd) * innerRadius;

                        const largeArcFlag = (angleEnd - angleStart) > Math.PI ? 1 : 0;
                        
                        const pathData = [
                            `M ${startX} ${startY}`,
                            `A ${radius} ${radius} 0 ${largeArcFlag} 1 ${endX} ${endY}`,
                            `L ${innerEndX} ${innerEndY}`,
                            `A ${innerRadius} ${innerRadius} 0 ${largeArcFlag} 0 ${innerStartX} ${innerStartY}`,
                            'Z'
                        ].join(' ');

                        return (
                            <path 
                                key={item.name} 
                                d={pathData} 
                                fill={item.color} 
                                onClick={() => setActiveSlice(item)}
                                className="cursor-pointer transition-opacity duration-200"
                                style={{opacity: activeSlice && activeSlice.name !== item.name ? 0.6 : 1}}
                            />
                        );
                    })}
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center text-center select-none">
                    {activeSlice ? (
                        <>
                            <span className="text-xl font-bold font-mono text-slate-100">{activeSlice.value}%</span>
                            <span className="text-xs text-slate-400">{activeSlice.name}</span>
                        </>
                    ) : (
                         <>
                            <span className="text-xs text-slate-400">Click a</span>
                            <span className="text-xs text-slate-400">slice</span>
                        </>
                    )}
                </div>
            </div>
            <div className="text-xs space-y-1">
                {data.map(item => (
                    <div key={item.name} className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: item.color }}></div>
                        <span className="text-slate-300">{item.name}</span>
                        <span className="font-mono text-slate-400">{item.value}%</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

type PerformanceGraphPoint = { date: string, value: number };
type MappedPoint = PerformanceGraphPoint & { x: number; y: number };

const PerformanceGraph: React.FC<{data: PerformanceGraphPoint[]}> = ({ data }) => {
    const [hoveredPoint, setHoveredPoint] = useState<MappedPoint | null>(null);
    const svgRef = useRef<SVGSVGElement>(null);
    
    const width = 500;
    const height = 150;
    const padding = { top: 20, right: 10, bottom: 20, left: 10 };
    
    const { dataPoints, path, areaPath, lastValue, valueChange, percentChange, isPositive } = useMemo(() => {
        if (data.length === 0) return { dataPoints: [], path: '', areaPath: '', lastValue: 0, valueChange: 0, percentChange: 0, isPositive: true };

        const maxValue = Math.max(...data.map(p => p.value));
        const minValue = Math.min(...data.map(p => p.value));
        const valueRange = maxValue - minValue > 0 ? maxValue - minValue : 1;

        const points: MappedPoint[] = data.map((point, i) => {
            const x = (i / (data.length - 1)) * (width - padding.left - padding.right) + padding.left;
            const y = height - padding.bottom - ((point.value - minValue) / valueRange * (height - padding.top - padding.bottom));
            return { ...point, x, y };
        });

        const pathD = points.map((p, i) => (i === 0 ? 'M' : 'L') + `${p.x},${p.y}`).join(' ');
        const areaPathD = `M ${padding.left},${height - padding.bottom} ` + pathD + ` L ${width - padding.right},${height - padding.bottom} Z`;

        const startValue = data[0].value;
        const endValue = data[data.length - 1].value;
        const valChange = endValue - startValue;
        const pctChange = startValue !== 0 ? (valChange / startValue) * 100 : 0;
        
        return {
            dataPoints: points,
            path: pathD,
            areaPath: areaPathD,
            lastValue: endValue,
            valueChange: valChange,
            percentChange: pctChange,
            isPositive: valChange >= 0
        };
    }, [data]);
    
    const handleMouseMove = (event: React.MouseEvent<SVGSVGElement>) => {
        if (!svgRef.current) return;
        const svgRect = svgRef.current.getBoundingClientRect();
        const mouseX = event.clientX - svgRect.left;

        const closestPoint = dataPoints.reduce((closest, point) => 
            Math.abs(point.x - mouseX) < Math.abs(closest.x - mouseX) ? point : closest
        );
        setHoveredPoint(closestPoint);
    };

    const handleMouseLeave = () => setHoveredPoint(null);
    
    return (
        <div>
            <div className="mb-4">
                <p className="text-3xl font-bold mb-1"
                   style={{fontFamily: 'var(--font-mono)', color: 'var(--color-cream)'}}>
                    ${lastValue.toLocaleString()}
                </p>
                <div className="flex items-center gap-2 text-sm"
                     style={{
                       color: isPositive ? 'var(--color-accent)' : 'var(--color-primary)',
                       fontFamily: 'var(--font-mono)'
                     }}>
                    {isPositive ? <ArrowUpIcon className="w-4 h-4" /> : <ArrowDownIcon className="w-4 h-4" />}
                    <span className="font-semibold">${valueChange.toFixed(2)}</span>
                    <span className="font-semibold">({percentChange >= 0 ? '+' : ''}{percentChange.toFixed(2)}%)</span>
                    <span style={{color: 'var(--color-text-muted)', fontFamily: 'var(--font-body)'}}>
                        {data.length} data points
                    </span>
                </div>
            </div>
            <div className="relative">
                <svg ref={svgRef} viewBox={`0 0 ${width} ${height}`} className="w-full h-auto cursor-crosshair" onMouseMove={handleMouseMove} onMouseLeave={handleMouseLeave}>
                    <defs>
                        <linearGradient id="gradient" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" style={{stopColor: isPositive ? '#3A5A40' : '#5A2A27', stopOpacity: 0.3}} />
                            <stop offset="100%" style={{stopColor: isPositive ? '#3A5A40' : '#5A2A27', stopOpacity: 0}} />
                        </linearGradient>
                    </defs>
                    <path d={areaPath} fill="url(#gradient)" />
                    <path d={path} fill="none" stroke={isPositive ? '#3A5A40' : '#5A2A27'} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />

                    {hoveredPoint && (
                        <g>
                            <line x1={hoveredPoint.x} y1={height} x2={hoveredPoint.x} y2={0} stroke="var(--color-text-muted)" strokeWidth="1" strokeDasharray="3,3" />
                            <circle cx={hoveredPoint.x} cy={hoveredPoint.y} r="5" fill={isPositive ? '#3A5A40' : '#5A2A27'} stroke="var(--color-cream)" strokeWidth="2" />
                        </g>
                    )}
                </svg>
                {hoveredPoint && (
                    <div className="absolute p-3 text-xs rounded-md pointer-events-none"
                         style={{
                            left: hoveredPoint.x > width / 2 ? `${hoveredPoint.x - 120}px` : `${hoveredPoint.x + 15}px`,
                            top: `${hoveredPoint.y - 30}px`,
                            transform: 'translateY(-100%)',
                            backgroundColor: 'var(--color-bg-secondary)',
                            border: '1px solid var(--color-bg-tertiary)',
                            backdropFilter: 'blur(8px)'
                         }}>
                        <p className="font-bold mb-1" style={{fontFamily: 'var(--font-mono)', color: 'var(--color-cream)'}}>
                            ${hoveredPoint.value.toFixed(2)}
                        </p>
                        <p style={{fontFamily: 'var(--font-body)', color: 'var(--color-text-muted)'}}>
                            {new Date(hoveredPoint.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
};

const TimeRangeSelector: React.FC<{ selected: string; onSelect: (range: string) => void }> = ({ selected, onSelect }) => {
    const ranges = ['1D', '1W', '1M', 'YTD', 'START'];
    return (
        <div className="flex items-center rounded-md p-1 gap-1"
             style={{
               backgroundColor: 'var(--color-bg-tertiary)',
               border: '1px solid var(--color-bg-tertiary)'
             }}>
            {ranges.map(range => (
                <button
                    key={range}
                    onClick={() => onSelect(range)}
                    className="px-3 py-1.5 text-xs font-semibold rounded transition-all duration-200"
                    style={{
                      backgroundColor: selected === range ? 'var(--color-primary)' : 'transparent',
                      color: selected === range ? 'var(--color-cream)' : 'var(--color-text-muted)',
                      fontFamily: 'var(--font-mono)'
                    }}
                    onMouseEnter={(e) => {
                      if (selected !== range) {
                        e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (selected !== range) {
                        e.currentTarget.style.backgroundColor = 'transparent';
                      }
                    }}
                >
                    {range}
                </button>
            ))}
        </div>
    );
};

const POLLING_INTERVAL = 30000; // 30 segundos

export const Portfolio: React.FC<{ id?: string }> = ({ id }) => {
    const [portfolioData, setPortfolioData] = useState<HoldingsWithPrices | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [timeRange, setTimeRange] = useState('1M');

    useEffect(() => {
        const updatePortfolio = async () => {
            try {
                setError(null);
                const data = await fetchHoldingsWithPrices();
                setPortfolioData(data);
                setLoading(false);
            } catch (err) {
                console.error('Error fetching portfolio:', err);
                setError(err instanceof Error ? err.message : 'Error cargando portfolio');
                setLoading(false);
            }
        };

        updatePortfolio();
        const interval = setInterval(updatePortfolio, POLLING_INTERVAL);

        return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <WidgetCard
                id={id}
                title="PORTFOLIO SNAPSHOT"
                tooltip="Vista general de tu cartera de inversión con holdings actuales, valores de mercado, y rendimiento histórico."
            >
                <div className="text-slate-400 text-sm">Cargando portfolio...</div>
            </WidgetCard>
        );
    }

    if (error) {
        return (
            <WidgetCard
                id={id}
                title="PORTFOLIO SNAPSHOT"
                tooltip="Vista general de tu cartera de inversión con holdings actuales, valores de mercado, y rendimiento histórico."
            >
                <div className="text-red-400 text-sm">{error}</div>
                <div className="text-slate-500 text-xs mt-1">Inicia sesión para ver tu portfolio</div>
            </WidgetCard>
        );
    }

    if (!portfolioData || portfolioData.holdings.length === 0) {
        return (
            <WidgetCard
                id={id}
                title="PORTFOLIO SNAPSHOT"
                tooltip="Vista general de tu cartera de inversión con holdings actuales, valores de mercado, y rendimiento histórico."
            >
                <div className="space-y-4">
                    <div className="text-slate-400 text-sm">
                        No hay holdings. Agrega posiciones para ver tu portfolio.
                    </div>
                    <div className="text-xs text-slate-500">
                        Usa el endpoint /api/holdings para agregar posiciones.
                    </div>
                </div>
            </WidgetCard>
        );
    }

    // Calculate performance data based on holdings and time range
    const getPerformanceData = (): PerformanceGraphPoint[] => {
        if (!portfolioData || portfolioData.holdings.length === 0) return [];

        const now = new Date();
        let startDate: Date;
        let dataPoints = 30;

        switch (timeRange) {
            case '1D':
                startDate = new Date(now.getTime() - 1 * 24 * 60 * 60 * 1000);
                dataPoints = 24; // hourly
                break;
            case '1W':
                startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
                dataPoints = 7;
                break;
            case '1M':
                startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
                dataPoints = 30;
                break;
            case 'YTD':
                startDate = new Date(now.getFullYear(), 0, 1);
                dataPoints = Math.ceil((now.getTime() - startDate.getTime()) / (24 * 60 * 60 * 1000));
                break;
            case 'START':
                // Find earliest purchase date
                const purchaseDates = portfolioData.holdings
                    .map((h: any) => h.purchase_date ? new Date(h.purchase_date) : null)
                    .filter((d): d is Date => d !== null);
                startDate = purchaseDates.length > 0
                    ? new Date(Math.min(...purchaseDates.map(d => d.getTime())))
                    : new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000);
                dataPoints = Math.ceil((now.getTime() - startDate.getTime()) / (24 * 60 * 60 * 1000));
                break;
            default:
                startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
                dataPoints = 30;
        }

        // Generate simulated historical data based on current values
        const data: PerformanceGraphPoint[] = [];
        const totalCurrentValue = portfolioData.total_value;
        const totalCost = portfolioData.total_cost;

        for (let i = 0; i <= dataPoints; i++) {
            const date = new Date(startDate.getTime() + (i / dataPoints) * (now.getTime() - startDate.getTime()));
            // Interpolate value from cost to current value with some randomness
            const progress = i / dataPoints;
            const baseValue = totalCost + (totalCurrentValue - totalCost) * progress;
            // Add some volatility (±3%)
            const volatility = baseValue * 0.03 * (Math.random() - 0.5);
            const value = baseValue + volatility;

            data.push({
                date: date.toISOString(),
                value: Math.max(value, totalCost * 0.9) // Don't go below 90% of cost
            });
        }

        return data;
    };

    const performanceData = getPerformanceData();

    // Convertir a datos de allocation para el pie chart
    const allocationData = portfolioData.holdings.map((holding, i) => {
        const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];
        return {
            name: holding.ticker,
            value: (holding.current_value / portfolioData.total_value) * 100,
            color: colors[i % colors.length],
        };
    });

    return (
        <WidgetCard
            id={id}
            title="PORTFOLIO SNAPSHOT"
            tooltip="Vista general de tu cartera de inversión con holdings actuales, valores de mercado, y rendimiento histórico."
        >
            <div className="space-y-6">
                {/* Resumen */}
                <div>
                    <h4 className="text-xs text-slate-400 mb-2">Resumen</h4>
                    <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                            <span className="text-slate-300">Valor Total:</span>
                            <span className="font-mono text-slate-100">${portfolioData.total_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                            <span className="text-slate-300">Costo Total:</span>
                            <span className="font-mono text-slate-300">${portfolioData.total_cost.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                        </div>
                        <div className={`flex justify-between text-sm ${portfolioData.total_gain_loss >= 0 ? 'text-blue-400' : 'text-red-400'}`}>
                            <span>Ganancia/Pérdida:</span>
                            <span className="font-mono">
                                ${portfolioData.total_gain_loss.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ({portfolioData.total_gain_loss_pct >= 0 ? '+' : ''}{portfolioData.total_gain_loss_pct.toFixed(2)}%)
                            </span>
                        </div>
                    </div>
                </div>

                {/* Performance Graph */}
                {performanceData.length > 0 && (
                    <div>
                        <div className="flex justify-between items-center mb-4">
                            <h4 className="text-xs"
                                style={{
                                  fontFamily: 'var(--font-mono)',
                                  color: 'var(--color-text-muted)',
                                  letterSpacing: '0.05em'
                                }}>
                                PERFORMANCE
                            </h4>
                            <TimeRangeSelector selected={timeRange} onSelect={setTimeRange} />
                        </div>
                        <PerformanceGraph data={performanceData} />
                    </div>
                )}

                {/* Allocation */}
                {allocationData.length > 0 && (
                    <div>
                        <h4 className="text-xs text-slate-400 mb-2">Allocation</h4>
                        <PieChart data={allocationData} />
                    </div>
                )}

                {/* Holdings List */}
                <div>
                    <h4 className="text-xs text-slate-400 mb-2">Posiciones</h4>
                    <div className="space-y-2 max-h-48 overflow-y-auto">
                        {portfolioData.holdings.map((holding: HoldingWithPrice) => (
                            <div key={holding.ticker} className="flex justify-between items-center text-sm border-b border-slate-800 pb-2">
                                <div>
                                    <span className="text-slate-200 font-semibold">{holding.ticker}</span>
                                    <span className="text-slate-400 ml-2">{holding.quantity.toFixed(2)} acciones</span>
                                </div>
                                <div className="text-right">
                                    <div className="font-mono text-slate-100">${holding.current_price.toFixed(2)}</div>
                                    <div className={`text-xs ${holding.gain_loss >= 0 ? 'text-blue-400' : 'text-red-400'}`}>
                                        {holding.gain_loss >= 0 ? '+' : ''}{holding.gain_loss_pct.toFixed(2)}%
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </WidgetCard>
    );
};

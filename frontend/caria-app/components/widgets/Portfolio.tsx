
import React, { useState, useRef, useMemo, useEffect, useCallback } from 'react';
import { fetchHoldingsWithPrices, HoldingsWithPrices, HoldingWithPrice, createHolding, deleteHolding } from '../../services/apiService';
import { ArrowUpIcon, ArrowDownIcon } from '../Icons';
import { WidgetCard } from './WidgetCard';
import { getErrorMessage } from '../../src/utils/errorHandling';

type AllocationData = { name: string; value: number; color: string };

const PieChart: React.FC<{ data: AllocationData[] }> = ({ data }) => {
    // ... (unchanged)
    const [activeSlice, setActiveSlice] = useState<AllocationData | null>(data.length > 0 ? data[0] : null);
    const total = data.reduce((acc, item) => acc + (item.value || 0), 0);

    if (!data.length || total <= 0 || !Number.isFinite(total)) {
        return (
            <div className="flex flex-col gap-2 text-slate-400 text-sm">
                <p>No allocation data yet.</p>
                <p>Add positions to visualize your weights.</p>
            </div>
        );
    }

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
                                style={{ opacity: activeSlice && activeSlice.name !== item.name ? 0.6 : 1 }}
                            />
                        );
                    })}
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center text-center select-none">
                    {activeSlice ? (
                        <>
                            <span className="text-xl font-bold font-mono text-slate-100">{activeSlice.value.toFixed(0)}%</span>
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
                        <span className="font-mono text-slate-400">{item.value.toFixed(0)}%</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

type PerformanceGraphPoint = { date: string, value: number };
type MappedPoint = PerformanceGraphPoint & { x: number; y: number };

const PerformanceGraph: React.FC<{ data: PerformanceGraphPoint[] }> = ({ data }) => {
    const [hoveredPoint, setHoveredPoint] = useState<MappedPoint | null>(null);
    const svgRef = useRef<SVGSVGElement>(null);

    const width = 500;
    const height = 150;
    const padding = { top: 20, right: 10, bottom: 20, left: 10 };

    const { dataPoints, path, areaPath, lastValue, valueChange, percentChange, isPositive } = useMemo(() => {
        if (!data || data.length === 0) return { dataPoints: [], path: '', areaPath: '', lastValue: 0, valueChange: 0, percentChange: 0, isPositive: true };

        // Filter out invalid data points
        const validData = data.filter(p => Number.isFinite(p.value));
        if (validData.length === 0) return { dataPoints: [], path: '', areaPath: '', lastValue: 0, valueChange: 0, percentChange: 0, isPositive: true };

        const maxValue = Math.max(...validData.map(p => p.value));
        const minValue = Math.min(...validData.map(p => p.value));
        const valueRange = maxValue - minValue > 0 ? maxValue - minValue : 1;

        const points: MappedPoint[] = validData.map((point, i) => {
            const x = (i / (validData.length - 1)) * (width - padding.left - padding.right) + padding.left;
            const y = height - padding.bottom - ((point.value - minValue) / valueRange * (height - padding.top - padding.bottom));
            return { ...point, x: Number.isFinite(x) ? x : 0, y: Number.isFinite(y) ? y : 0 };
        });

        const pathD = points.map((p, i) => (i === 0 ? 'M' : 'L') + `${p.x},${p.y}`).join(' ');
        const areaPathD = `M ${padding.left},${height - padding.bottom} ` + pathD + ` L ${width - padding.right},${height - padding.bottom} Z`;

        const startValue = validData[0].value;
        const endValue = validData[validData.length - 1].value;
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

    const chartColor = isPositive ? '#10b981' : '#ef4444'; // green or red like Yahoo Finance
    const gridColor = 'rgba(232, 230, 227, 0.1)';

    return (
        <div>
            <div className="mb-4">
                <p className="text-3xl font-bold mb-1"
                    style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-cream)' }}>
                    ${lastValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </p>
                <div className="flex items-center gap-2 text-sm"
                    style={{
                        color: chartColor,
                        fontFamily: 'var(--font-mono)'
                    }}>
                    {isPositive ? <ArrowUpIcon className="w-4 h-4" /> : <ArrowDownIcon className="w-4 h-4" />}
                    <span className="font-semibold">${Math.abs(valueChange).toFixed(2)}</span>
                    <span className="font-semibold">({percentChange >= 0 ? '+' : ''}{percentChange.toFixed(2)}%)</span>
                    <span style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-body)' }}>
                        All Time
                    </span>
                </div>
            </div>
            <div className="relative" style={{ backgroundColor: 'rgba(0,0,0,0.2)', borderRadius: '8px', padding: '16px' }}>
                <svg ref={svgRef} viewBox={`0 0 ${width} ${height}`} className="w-full h-auto cursor-crosshair" onMouseMove={handleMouseMove} onMouseLeave={handleMouseLeave}>
                    <defs>
                        <linearGradient id="areaGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" style={{ stopColor: chartColor, stopOpacity: 0.4 }} />
                            <stop offset="100%" style={{ stopColor: chartColor, stopOpacity: 0 }} />
                        </linearGradient>
                    </defs>

                    {/* Horizontal gridlines */}
                    {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
                        const y = padding.top + (height - padding.top - padding.bottom) * ratio;
                        return (
                            <line
                                key={ratio}
                                x1={padding.left}
                                y1={y}
                                x2={width - padding.right}
                                y2={y}
                                stroke={gridColor}
                                strokeWidth="1"
                                strokeDasharray="2,2"
                            />
                        );
                    })}

                    {/* Area fill */}
                    <path d={areaPath} fill="url(#areaGradient)" />

                    {/* Main line with smooth curve */}
                    <path
                        d={path}
                        fill="none"
                        stroke={chartColor}
                        strokeWidth="3"
                        strokeLinejoin="round"
                        strokeLinecap="round"
                        style={{ filter: 'drop-shadow(0 0 4px rgba(0,0,0,0.3))' }}
                    />

                    {hoveredPoint && (
                        <g>
                            <line
                                x1={hoveredPoint.x}
                                y1={height - padding.bottom}
                                x2={hoveredPoint.x}
                                y2={padding.top}
                                stroke="rgba(232, 230, 227, 0.5)"
                                strokeWidth="1"
                                strokeDasharray="4,4"
                            />
                            <circle
                                cx={hoveredPoint.x}
                                cy={hoveredPoint.y}
                                r="6"
                                fill={chartColor}
                                stroke="var(--color-cream)"
                                strokeWidth="2"
                                style={{ filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.5))' }}
                            />
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
                        <p className="font-bold mb-1" style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-cream)' }}>
                            ${hoveredPoint.value.toFixed(2)}
                        </p>
                        <p style={{ fontFamily: 'var(--font-body)', color: 'var(--color-text-muted)' }}>
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

type HoldingFormState = {
    ticker: string;
    quantity: string;
    average_cost: string;
    purchase_date: string;
    notes: string;
};

const createDefaultFormState = (): HoldingFormState => ({
    ticker: '',
    quantity: '',
    average_cost: '',
    purchase_date: new Date().toISOString().split('T')[0],
    notes: '',
});

export const Portfolio: React.FC<{ id?: string }> = ({ id }) => {
    const [portfolioData, setPortfolioData] = useState<HoldingsWithPrices | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [timeRange, setTimeRange] = useState('1M');
    const [formData, setFormData] = useState<HoldingFormState>(createDefaultFormState());
    const [formError, setFormError] = useState<string | null>(null);
    const [sortOption, setSortOption] = useState<'value' | 'return' | 'ticker'>('value');
    const [showForm, setShowForm] = useState(false);
    const [actionLoading, setActionLoading] = useState(false);
    const [isRefreshing, setIsRefreshing] = useState(false);

    const loadPortfolio = useCallback(
        async (showSpinner = true) => {
            // Always start loading state to prevent hook mismatches
            if (showSpinner) {
                setLoading(true);
            } else {
                setIsRefreshing(true);
            }
            setError(null); // Reset error state

            try {
                const data = await fetchHoldingsWithPrices();
                setPortfolioData(data);
            } catch (err: unknown) {
                setError(getErrorMessage(err) || 'Could not load your portfolio. Please try again.');
            } finally {
                if (showSpinner) {
                    setLoading(false);
                } else {
                    setIsRefreshing(false);
                }
            }
        },
        []
    );

    useEffect(() => {
        loadPortfolio();
        const interval = setInterval(() => loadPortfolio(false), POLLING_INTERVAL);
        return () => clearInterval(interval);
    }, [loadPortfolio]);

    useEffect(() => {
        if (portfolioData && portfolioData.holdings.length === 0) {
            setShowForm(true);
        }
    }, [portfolioData]);

    const hasHoldings = Boolean(portfolioData && portfolioData.holdings.length > 0);

    const getPerformanceData = useCallback((): PerformanceGraphPoint[] => {
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
    }, [portfolioData, timeRange]);

    const performanceData = useMemo(() => getPerformanceData(), [getPerformanceData]);

    const allocationData = useMemo(() => {
        if (!portfolioData || portfolioData.holdings.length === 0 || portfolioData.total_value <= 0) {
            return [];
        }
        const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];
        return portfolioData.holdings.map((holding, i) => {
            const holdingValue = holding.current_value ?? holding.current_price * holding.quantity;
            return {
                name: holding.ticker,
                value: portfolioData.total_value > 0 ? (holdingValue / portfolioData.total_value) * 100 : 0,
                color: colors[i % colors.length],
            };
        });
    }, [portfolioData]);

    const sortedHoldings = useMemo(() => {
        if (!portfolioData) return [];
        const holdings = [...portfolioData.holdings];
        switch (sortOption) {
            case 'return':
                return holdings.sort((a, b) => (b.gain_loss_pct || 0) - (a.gain_loss_pct || 0));
            case 'ticker':
                return holdings.sort((a, b) => a.ticker.localeCompare(b.ticker));
            case 'value':
            default:
                return holdings.sort((a, b) => (b.current_value || 0) - (a.current_value || 0));
        }
    }, [portfolioData, sortOption]);

    // Early returns MUST happen AFTER all hooks are declared
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

    const handleAddHolding = async (event: React.FormEvent) => {
        event.preventDefault();
        setFormError(null);

        const ticker = formData.ticker.trim().toUpperCase();
        const quantity = parseFloat(formData.quantity);
        const averageCost = parseFloat(formData.average_cost);

        if (!ticker) {
            setFormError('Ingresa un ticker válido.');
            return;
        }
        if (!Number.isFinite(quantity) || quantity <= 0) {
            setFormError('Please enter a valid quantity.');
            return;
        }
        if (!Number.isFinite(averageCost) || averageCost <= 0) {
            setFormError('Please enter a valid average cost.');
            return;
        }

        try {
            setActionLoading(true);
            await createHolding({
                ticker,
                quantity,
                average_cost: averageCost,
                purchase_date: formData.purchase_date,
                notes: formData.notes || undefined,
            });
            setFormData(createDefaultFormState());
            setShowForm(false);
            await loadPortfolio(false);
        } catch (err: unknown) {
            setFormError(getErrorMessage(err) || 'Could not save the position.');
        } finally {
            setActionLoading(false);
        }
    };

    const handleDeleteHolding = async (id: string) => {
        if (!window.confirm('Delete this position?')) {
            return;
        }
        try {
            setActionLoading(true);
            await deleteHolding(id);
            await loadPortfolio(false);
        } catch (err: unknown) {
            setFormError(getErrorMessage(err) || 'No se pudo eliminar la posición.');
        } finally {
            setActionLoading(false);
        }
    };

    return (
        <WidgetCard
            id={id}
            title="PORTFOLIO SNAPSHOT"
            tooltip="Vista general de tu cartera de inversión con holdings actuales, valores de mercado, y rendimiento histórico."
        >
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <div>
                        <h4 className="text-xs text-slate-400 mb-2">Resumen</h4>
                    </div>
                    {isRefreshing && (
                        <span className="text-xs text-slate-500 animate-pulse">Actualizando…</span>
                    )}
                </div>
                {portfolioData && (
                    <>
                        <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-300">Total Value:</span>
                                <span className="font-mono text-slate-100">
                                    ${portfolioData.total_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                </span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-300">Total Cost:</span>
                                <span className="font-mono text-slate-300">
                                    ${portfolioData.total_cost.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                </span>
                            </div>
                            <div className={`flex justify-between text-sm ${portfolioData.total_gain_loss >= 0 ? 'text-blue-400' : 'text-red-400'}`}>
                                <span>Gain/Loss:</span>
                                <span className="font-mono">
                                    ${portfolioData.total_gain_loss.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ({portfolioData.total_gain_loss_pct >= 0 ? '+' : ''}{portfolioData.total_gain_loss_pct.toFixed(2)}%)
                                </span>
                            </div>
                        </div>

                        {hasHoldings ? (
                            <>
                                {performanceData.length > 0 && (
                                    <div>
                                        <div className="flex justify-between items-center mb-4">
                                            <h4
                                                className="text-xs"
                                                style={{
                                                    fontFamily: 'var(--font-mono)',
                                                    color: 'var(--color-text-muted)',
                                                    letterSpacing: '0.05em',
                                                }}
                                            >
                                                PERFORMANCE
                                            </h4>
                                            <TimeRangeSelector selected={timeRange} onSelect={setTimeRange} />
                                        </div>
                                        <PerformanceGraph data={performanceData} />
                                    </div>
                                )}

                                {allocationData.length > 0 && (
                                    <div>
                                        <h4 className="text-xs text-slate-400 mb-2">Allocation</h4>
                                        <PieChart data={allocationData} />
                                    </div>
                                )}
                            </>
                        ) : (
                            <div className="rounded-lg border border-dashed border-slate-700 p-4 text-sm text-slate-400">
                                Your portfolio is empty. Add your first position to start tracking performance.
                            </div>
                        )}
                    </>
                )}

                <div className="border border-slate-800 rounded-lg p-4 space-y-4">
                    <div className="flex flex-wrap items-center gap-4 justify-between">
                        <div>
                            <h4 className="text-sm font-semibold" style={{ color: 'var(--color-cream)' }}>
                                Position Management
                            </h4>
                            <p className="text-xs text-slate-500">Track purchases, sales, and key notes.</p>
                        </div>
                        <div className="flex bg-slate-900/70 rounded-md overflow-hidden">
                            <button
                                onClick={() => setSortOption('value')}
                                className={`px-3 py-1 text-xs font-mono ${sortOption === 'value' ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-slate-200'}`}
                            >
                                $
                            </button>
                            <button
                                onClick={() => setSortOption('return')}
                                className={`px-3 py-1 text-xs font-mono ${sortOption === 'return' ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-slate-200'}`}
                            >
                                %
                            </button>
                            <button
                                onClick={() => setSortOption('ticker')}
                                className={`px-3 py-1 text-xs font-mono ${sortOption === 'ticker' ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-slate-200'}`}
                            >
                                AZ
                            </button>
                        </div>
                        <button
                            onClick={() => setShowForm((prev) => !prev)}
                            className="text-xs font-semibold px-4 py-2 rounded-md transition-colors"
                            style={{
                                backgroundColor: 'var(--color-primary)',
                                color: 'var(--color-cream)',
                            }}
                        >
                            {showForm ? 'Close Form' : 'Add Position'}
                        </button>
                    </div>

                    {showForm && (
                        <form onSubmit={handleAddHolding} className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-slate-900/50 p-4 rounded-lg border border-slate-800">
                            <div>
                                <label className="block text-xs text-slate-400 mb-1">Ticker</label>
                                <input
                                    type="text"
                                    value={formData.ticker}
                                    onChange={(e) => setFormData({ ...formData, ticker: e.target.value })}
                                    className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                                    placeholder="AAPL"
                                    required
                                />
                            </div>
                            <div>
                                <label className="block text-xs text-slate-400 mb-1">Quantity</label>
                                <input
                                    type="number"
                                    step="0.01"
                                    value={formData.quantity}
                                    onChange={(e) => setFormData({ ...formData, quantity: e.target.value })}
                                    className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                                    placeholder="10"
                                    required
                                />
                            </div>
                            <div>
                                <label className="block text-xs text-slate-400 mb-1">Average Cost</label>
                                <input
                                    type="number"
                                    step="0.01"
                                    value={formData.average_cost}
                                    onChange={(e) => setFormData({ ...formData, average_cost: e.target.value })}
                                    className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                                    placeholder="150.00"
                                    required
                                />
                            </div>
                            <div>
                                <label className="block text-xs text-slate-400 mb-1">Purchase Date</label>
                                <input
                                    type="date"
                                    value={formData.purchase_date}
                                    onChange={(e) => setFormData({ ...formData, purchase_date: e.target.value })}
                                    className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                                    required
                                />
                            </div>
                            <div className="md:col-span-2">
                                <label className="block text-xs text-slate-400 mb-1">Notes (optional)</label>
                                <input
                                    type="text"
                                    value={formData.notes}
                                    onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                                    className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                                    placeholder="Catalysts, thesis summary, etc."
                                />
                            </div>
                            <div className="md:col-span-2 flex gap-3">
                                <button
                                    type="submit"
                                    disabled={actionLoading}
                                    className="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold py-2 px-4 rounded transition-colors disabled:opacity-50"
                                >
                                    Save Position
                                </button>
                                <button
                                    type="button"
                                    onClick={() => {
                                        setShowForm(false);
                                        setFormData(createDefaultFormState());
                                        setFormError(null);
                                    }}
                                    className="px-4 py-2 rounded text-sm font-semibold text-slate-300 border border-slate-700 hover:border-slate-500"
                                >
                                    Cancel
                                </button>
                            </div>
                        </form>
                    )}

                    {formError && (
                        <div className="text-red-400 text-sm bg-red-900/20 border border-red-800 rounded p-2">
                            {formError}
                        </div>
                    )}

                    <div>
                        {sortedHoldings.length === 0 ? (
                            <div className="text-center text-sm text-slate-500 py-8">
                                No positions recorded yet.
                            </div>
                        ) : (
                            <div className="space-y-2 max-h-72 overflow-y-auto pr-1 custom-scrollbar">
                                {sortedHoldings.map((holding: HoldingWithPrice) => (
                                    <div
                                        key={holding.id}
                                        className="flex justify-between items-center p-2 bg-slate-900/40 rounded border border-slate-800"
                                    >
                                        <div>
                                            <div className="flex items-baseline gap-2">
                                                <span className="text-slate-200 font-semibold">{holding.ticker}</span>
                                                {holding.gain_loss_pct !== undefined && (
                                                    <span className={`text-xs ${holding.gain_loss_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                        {holding.gain_loss_pct >= 0 ? '+' : ''}
                                                        {holding.gain_loss_pct.toFixed(2)}%
                                                    </span>
                                                )}
                                            </div>
                                            <div className="text-xs text-slate-400">
                                                {holding.quantity} @ ${holding.average_cost.toFixed(2)}
                                            </div>
                                            {holding.notes && (
                                                <div className="text-xs text-slate-500 mt-1">{holding.notes}</div>
                                            )}
                                        </div>
                                        <div className="text-right space-y-1">
                                            <div className="font-mono text-slate-100">
                                                ${(holding.current_value ?? holding.current_price * holding.quantity).toLocaleString(undefined, {
                                                    minimumFractionDigits: 0,
                                                    maximumFractionDigits: 0,
                                                })}
                                            </div>
                                            <button
                                                type="button"
                                                onClick={() => holding.id && handleDeleteHolding(holding.id)}
                                                disabled={actionLoading}
                                                className="text-xs text-red-400 hover:text-red-300"
                                            >
                                                Eliminar
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </WidgetCard>
    );
};

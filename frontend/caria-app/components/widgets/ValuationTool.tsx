import React, { useState } from "react";
import Plot from "react-plotly.js";
import { fetchWithAuth, API_BASE_URL } from "../../services/apiService";

// Types
interface DcfAssumptions {
    fcf_yield_start: number;
    high_growth_rate: number;
    high_growth_years: number;
    fade_years: number;
    terminal_growth_rate: number;
    discount_rate: number;
    horizon_years: number;
    shares_outstanding?: number | null;
    net_debt?: number | null;
}

interface DcfBlock {
    method: string;
    fair_value_per_share: number;
    upside_percent: number;
    implied_return_cagr: number;
    intrinsic_value_equity?: number | null;
    assumptions: DcfAssumptions;
    explanation: string;
}

interface MultiplesBlock {
    method: string;
    multiples: Record<string, number>;
    explanation: string;
}

interface ReverseDcfBlock {
    implied_growth_rate: number;
    explanation: string;
}

interface MultiplesValuationBlock {
    method: string;
    fair_value: number;
    ev_sales_median?: number;
    ev_ebitda_median?: number;
    breakdown?: {
        ev_sales?: number;
        ev_ebitda?: number;
    };
    explanation: string;
}

interface QuickValuationResponse {
    ticker: string;
    currency: string;
    current_price: number;
    dcf?: DcfBlock;
    reverse_dcf?: ReverseDcfBlock;
    multiples_valuation?: MultiplesValuationBlock;
    multiples?: MultiplesBlock;
}

interface MonteCarloResult {
    paths: number[][];
    final_values: number[];
    percentiles: {
        p5: number;
        p10: number;
        p25: number;
        p50: number;
        p75: number;
        p90: number;
        p95: number;
    };
    metrics: {
        mean: number;
        median: number;
        std: number;
        var_5pct: number;
        cvar_5pct: number;
        prob_final_less_invested: number;
        moic_median: number;
    };
    plotly_data: any;
    histogram: any;
    visualization_data?: {
        ticker: string;
        raw_values: number[];
        metrics: {
            p10: number;
            p50: number;
            p90: number;
        };
        visual_range: [number, number];
    };
    simulation_params: {
        initial_value: number;
        mu: number;
        sigma: number;
        years: number;
        simulations: number;
    };
}


const formatMoney = (v: number) =>
    `$${v.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;

export const ValuationTool: React.FC = () => {
    const [ticker, setTicker] = useState("AAPL");
    const [valuation, setValuation] = useState<QuickValuationResponse | null>(null);
    const [isLoadingValuation, setIsLoadingValuation] = useState(false);
    const [valError, setValError] = useState<string | null>(null);

    // Monte Carlo - Fixed to 2 years
    const years = 2;
    const [simulations] = useState(10_000);
    const [mcResult, setMcResult] = useState<MonteCarloResult | null>(null);
    const [isLoadingMC, setIsLoadingMC] = useState(false);
    const [mcError, setMcError] = useState<string | null>(null);

    const handleAnalyze = async () => {
        setIsLoadingValuation(true);
        setValError(null);
        setMcResult(null);
        setMcError(null);

        try {
            const cleanTicker = ticker.trim().toUpperCase();
            if (!cleanTicker) throw new Error("Please enter a ticker symbol.");

            // Get current price
            const priceResp = await fetchWithAuth(`${API_BASE_URL}/api/prices/realtime/${cleanTicker}`);
            if (!priceResp.ok) throw new Error("Could not fetch current price");
            const priceData = await priceResp.json();
            const currentPrice = priceData.price ?? priceData.current_price;

            if (!currentPrice || currentPrice <= 0) {
                throw new Error("Invalid current price from API.");
            }

            // Get valuation (quick multiples)
            const valResp = await fetchWithAuth(`${API_BASE_URL}/api/valuation/${cleanTicker}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ current_price: currentPrice }),
            });

            if (!valResp.ok) {
                const errData = await valResp.json().catch(() => ({ detail: "Valuation failed" }));
                throw new Error(errData.detail || "Valuation failed");
            }

            const valData: QuickValuationResponse = await valResp.json();
            setValuation(valData);

            // Run Monte Carlo with fixed 2 years
            await runMonteCarlo();
        } catch (err: any) {
            console.error("Valuation error:", err);
            setValError(err.message || "An unexpected error occurred.");
        } finally {
            setIsLoadingValuation(false);
        }
    };

    const runMonteCarlo = async () => {
        setIsLoadingMC(true);
        setMcError(null);
        setMcResult(null);

        try {
            const resp = await fetchWithAuth(`${API_BASE_URL}/api/montecarlo/forecast/stock`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    ticker: ticker.toUpperCase(),
                    horizon_years: years, // Fixed to 2 years
                    simulations: simulations,
                }),
            });

            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({ detail: "Simulation failed" }));
                throw new Error(errData.detail || "Simulation failed");
            }

            setMcResult(await resp.json());
        } catch (err: any) {
            console.error("Monte Carlo error:", err);
            setMcError(err.message || "Simulation failed");
        } finally {
            setIsLoadingMC(false);
        }
    };

    // Plotly layouts
    const mcLayout = {
        title: { text: "", font: { color: "#F2F4F7", size: 14 } },
        xaxis: { title: "Years", gridcolor: "#1E2733", color: "#6B7A8F", tickfont: { size: 11 } },
        yaxis: { title: "Price ($)", gridcolor: "#1E2733", color: "#6B7A8F", tickfont: { size: 11 } },
        plot_bgcolor: "#0F1419",
        paper_bgcolor: "#0F1419",
        font: { color: "#6B7A8F" },
        showlegend: false,
        margin: { l: 50, r: 20, t: 20, b: 40 },
    };

    const [showMonteCarloInfo, setShowMonteCarloInfo] = useState(false);

    return (
        <div className="rounded-xl p-6 h-full" style={{ backgroundColor: 'var(--color-bg-secondary)', border: '1px solid var(--color-border-subtle)' }}>
            {/* Widget Header with Title */}
            <div className="mb-6">
                <h3 className="text-lg font-display font-bold text-white mb-2">
                    Monte Carlo Simulation: Project Where the Price of the Stock Will Be in 2 Years
                </h3>
                <button
                    onClick={() => setShowMonteCarloInfo(!showMonteCarloInfo)}
                    className="text-xs font-medium transition-all duration-200 flex items-center gap-2"
                    style={{ color: 'var(--color-accent-cyan)' }}
                >
                    <span>{showMonteCarloInfo ? 'Hide explanation' : 'Learn more about Monte Carlo Simulation'}</span>
                    <span className="text-[10px]">{showMonteCarloInfo ? '\u25B2' : '\u25BC'}</span>
                </button>

                {showMonteCarloInfo && (
                    <div
                        className="mt-3 p-4 rounded-lg text-xs leading-relaxed animate-fade-in"
                        style={{
                            backgroundColor: 'var(--color-bg-tertiary)',
                            border: '1px solid var(--color-border-subtle)',
                            color: 'var(--color-text-secondary)'
                        }}
                    >
                        <p className="mb-2">
                            <strong style={{ color: 'var(--color-accent-cyan)' }}>Monte Carlo Simulation</strong> is a powerful statistical technique used to model the probability of different outcomes in a process that cannot easily be predicted due to random variables.
                        </p>
                        <p className="mb-2">
                            For stock valuation, it runs <strong>10,000 simulations</strong> of possible price paths based on historical volatility and returns, giving you a range of potential future prices rather than a single estimate.
                        </p>
                        <p>
                            The results show you the <strong>Bear case (10th percentile)</strong>, <strong>Base case (50th percentile/median)</strong>, and <strong>Bull case (90th percentile)</strong> - helping you understand both the upside potential and downside risk of an investment.
                        </p>
                    </div>
                )}
            </div>

            <div className="space-y-4">
            {/* Ticker Input */}
            <div className="flex gap-3 items-end">
                <div className="flex-1">
                    <label
                        className="block text-xs font-medium tracking-wider uppercase mb-1.5"
                        style={{ color: 'var(--color-text-muted)' }}
                    >
                        Ticker Symbol
                    </label>
                    <input
                        value={ticker}
                        onChange={(e) => setTicker(e.target.value.toUpperCase())}
                        className="w-full px-3 py-2 rounded-lg text-sm font-mono"
                        style={{
                            backgroundColor: 'var(--color-bg-tertiary)',
                            border: '1px solid var(--color-border-subtle)',
                            color: 'var(--color-text-primary)',
                        }}
                        placeholder="AAPL"
                    />
                </div>
                <button
                    onClick={handleAnalyze}
                    disabled={isLoadingValuation || !ticker.trim()}
                    className="px-5 py-2 rounded-lg font-semibold text-sm transition-all duration-200 disabled:opacity-50"
                    style={{
                        backgroundColor: 'var(--color-accent-primary)',
                        color: '#FFFFFF',
                    }}
                >
                    {isLoadingValuation ? "Analyzing..." : "Analyze"}
                </button>
            </div>

            {valError && (
                <div 
                    className="px-4 py-3 rounded-lg text-sm"
                    style={{
                        backgroundColor: 'var(--color-negative-muted)',
                        color: 'var(--color-negative)',
                        border: '1px solid var(--color-negative)',
                    }}
                >
                    {valError}
                </div>
            )}

            {/* Valuation Results */}
            {valuation && (
                <div className="space-y-6">
                    {/* Quick Multiples Valuation */}
                    <div className="grid md:grid-cols-2 gap-4">

                        {/* Multiples Valuation */}
                        {valuation.multiples_valuation && (
                            <div 
                                className="rounded-xl p-5"
                                style={{
                                    backgroundColor: 'var(--color-positive-muted)',
                                    border: '1px solid rgba(0, 200, 83, 0.25)',
                                }}
                            >
                                <div className="flex items-center justify-between mb-3">
                                    <span 
                                        className="text-[10px] font-semibold tracking-widest uppercase"
                                        style={{ color: 'var(--color-positive)' }}
                                    >
                                        Historical Multiples
                                    </span>
                                    <span 
                                        className="text-[10px]"
                                        style={{ color: 'rgba(0, 200, 83, 0.7)' }}
                                    >
                                        Fair Value
                                    </span>
                                </div>
                                {valuation.multiples_valuation.fair_value != null && valuation.multiples_valuation.fair_value > 0 ? (
                                    <>
                                        <div 
                                            className="text-3xl font-bold font-mono mb-2"
                                            style={{ color: 'var(--color-positive)' }}
                                        >
                                            {formatMoney(valuation.multiples_valuation.fair_value)}
                                        </div>
                                        <p 
                                            className="text-xs"
                                            style={{ color: 'rgba(0, 200, 83, 0.8)' }}
                                        >
                                            EV/Sales: {valuation.multiples_valuation.ev_sales_median?.toFixed(1) ?? '—'}x • 
                                            EV/EBITDA: {valuation.multiples_valuation.ev_ebitda_median?.toFixed(1) ?? '—'}x
                                        </p>
                                    </>
                                ) : (
                                    <div 
                                        className="text-lg font-medium"
                                        style={{ color: 'rgba(0, 200, 83, 0.5)' }}
                                    >
                                        Insufficient data
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Current Multiples Grid */}
                    {valuation.multiples?.multiples && Object.keys(valuation.multiples.multiples).length > 0 && (
                        <div>
                            <div 
                                className="text-[10px] font-semibold tracking-widest uppercase mb-3"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Current Market Multiples
                            </div>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                {Object.entries(valuation.multiples.multiples).map(([key, value]) => {
                                    const num = Number(value);
                                    return (
                                        <div
                                            key={key}
                                            className="px-3 py-2 rounded-lg"
                                            style={{
                                                backgroundColor: 'var(--color-bg-tertiary)',
                                                border: '1px solid var(--color-border-subtle)',
                                            }}
                                        >
                                            <div 
                                                className="text-[10px] mb-1"
                                                style={{ color: 'var(--color-text-muted)' }}
                                            >
                                                {key}
                                            </div>
                                            <div 
                                                className="text-sm font-mono font-medium"
                                                style={{ color: 'var(--color-text-primary)' }}
                                            >
                                                {key.includes("yield") || key.includes("rate")
                                                    ? `${(num * 100).toFixed(1)}%`
                                                    : num.toFixed(2)}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {/* Monte Carlo Section */}
                    <div 
                        className="pt-6 border-t"
                        style={{ borderColor: 'var(--color-border-subtle)' }}
                    >
                        <div className="mb-4">
                            <div 
                                className="text-sm font-semibold"
                                style={{ color: 'var(--color-text-primary)' }}
                            >
                                Monte Carlo Price Forecast (2-Year Horizon)
                            </div>
                            <div 
                                className="text-xs"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                {simulations.toLocaleString()} simulations based on historical volatility
                            </div>
                        </div>

                        {mcError && (
                            <div 
                                className="px-4 py-2 rounded-lg text-xs mb-4"
                                style={{
                                    backgroundColor: 'var(--color-negative-muted)',
                                    color: 'var(--color-negative)',
                                }}
                            >
                                {mcError}
                            </div>
                        )}

                        {mcResult && (
                            <div className="space-y-4">
                                {/* Percentiles Summary */}
                                <div className="grid grid-cols-3 gap-3">
                                    {[
                                        { label: 'Bear (P10)', value: mcResult.percentiles.p10, color: 'var(--color-negative)' },
                                        { label: 'Base (P50)', value: mcResult.percentiles.p50, color: 'var(--color-text-primary)' },
                                        { label: 'Bull (P90)', value: mcResult.percentiles.p90, color: 'var(--color-positive)' },
                                    ].map((p) => (
                                        <div 
                                            key={p.label}
                                            className="px-4 py-3 rounded-lg text-center"
                                            style={{
                                                backgroundColor: 'var(--color-bg-tertiary)',
                                                border: '1px solid var(--color-border-subtle)',
                                            }}
                                        >
                                            <div 
                                                className="text-[10px] font-medium tracking-wider uppercase mb-1"
                                                style={{ color: 'var(--color-text-muted)' }}
                                            >
                                                {p.label}
                                            </div>
                                            <div 
                                                className="text-xl font-bold font-mono"
                                                style={{ color: p.color }}
                                            >
                                                {formatMoney(p.value)}
                                            </div>
                                        </div>
                                    ))}
                                </div>

                                {/* Histogram - Main Chart */}
                                {(() => {
                                    // Calculate dynamic X-axis range based on price distribution
                                    const finalValues = mcResult.final_values;
                                    const minPrice = Math.min(...finalValues);
                                    const maxPrice = Math.max(...finalValues);
                                    
                                    // Use percentiles for a focused view (P5 to P95 covers 90% of outcomes)
                                    const p5 = mcResult.percentiles.p5 || minPrice;
                                    const p95 = mcResult.percentiles.p95 || maxPrice;
                                    
                                    // Calculate range and add 15% padding on each side for readability
                                    const percentileRange = p95 - p5;
                                    const padding = percentileRange * 0.15;
                                    
                                    // Set axis bounds with padding, ensuring we don't go below 0
                                    let xAxisMin = Math.max(0, p5 - padding);
                                    let xAxisMax = p95 + padding;
                                    
                                    // If current price is available, ensure it's visible in context
                                    const currentPrice = valuation?.current_price;
                                    if (currentPrice) {
                                        // Expand range if current price is outside the percentile range
                                        if (currentPrice < p5) {
                                            xAxisMin = Math.max(0, currentPrice * 0.8);
                                        }
                                        if (currentPrice > p95) {
                                            xAxisMax = currentPrice * 1.2;
                                        }
                                    }
                                    
                                    const adjustedMin = xAxisMin;
                                    const adjustedMax = xAxisMax;
                                    
                                    // Use visualization_data if available, otherwise fallback to old format
                                    const vizData = mcResult.visualization_data;
                                    const values = vizData?.raw_values || mcResult.final_values;
                                    const metrics = vizData?.metrics || {
                                        p10: mcResult.percentiles.p10,
                                        p50: mcResult.percentiles.p50,
                                        p90: mcResult.percentiles.p90
                                    };
                                    const visualRange = vizData?.visual_range || [adjustedMin, adjustedMax];
                                    const tickerName = vizData?.ticker || ticker;

                                    // 1. EL HISTOGRAMA (Amarillo con bordes negros)
                                    const traceHistogram = {
                                        x: values,
                                        type: 'histogram',
                                        histnorm: 'count',
                                        marker: {
                                            color: '#F4D03F',      // El amarillo de la imagen
                                            line: {
                                                color: '#333333',  // Borde negro fino
                                                width: 1
                                            }
                                        },
                                        opacity: 0.9,
                                        showlegend: false,
                                        nbinsx: 40 // Ajusta el ancho de las barras
                                    };

                                    // 2. TRUCO PARA LA LEYENDA (Dummy Traces)
                                    const legendP10 = {
                                        x: [null], 
                                        y: [null],
                                        mode: 'lines',
                                        name: `10th %ile: $${metrics.p10.toFixed(2)}`,
                                        line: { color: '#E74C3C', width: 3, dash: 'dash' } // Rojo discontinuo
                                    };

                                    const legendP50 = {
                                        x: [null], 
                                        y: [null],
                                        mode: 'lines',
                                        name: `Median: $${metrics.p50.toFixed(2)}`,
                                        line: { color: '#27AE60', width: 3, dash: 'dash' } // Verde discontinuo
                                    };

                                    const legendP90 = {
                                        x: [null], 
                                        y: [null],
                                        mode: 'lines',
                                        name: `90th %ile: $${metrics.p90.toFixed(2)}`,
                                        line: { color: '#2980B9', width: 3, dash: 'dash' } // Azul discontinuo
                                    };

                                    // 3. EL LAYOUT (Líneas verticales reales y estilos)
                                    const horizonYears = mcResult.simulation_params?.years || 1;
                                    const horizonText = horizonYears === 1 ? '12-month horizon' : `${horizonYears}-year horizon`;
                                    const layout = {
                                        title: {
                                            text: `${tickerName} Macro Monte Carlo Simulation (${horizonText})`,
                                            font: { size: 18, color: '#333' }
                                        },
                                        xaxis: {
                                            title: 'Valuation ($/share)',
                                            range: visualRange,
                                            gridcolor: '#eee'
                                        },
                                        yaxis: {
                                            title: '',
                                            gridcolor: '#eee',
                                            zeroline: false
                                        },
                                        plot_bgcolor: 'white',
                                        paper_bgcolor: 'white',
                                        
                                        // Aquí dibujamos las líneas verticales reales
                                        shapes: [
                                            { // Línea P10 (Roja)
                                                type: 'line',
                                                x0: metrics.p10, 
                                                x1: metrics.p10,
                                                y0: 0, 
                                                y1: 1, 
                                                yref: 'paper',
                                                line: { color: '#E74C3C', width: 3, dash: 'dash' }
                                            },
                                            { // Línea Mediana (Verde)
                                                type: 'line',
                                                x0: metrics.p50, 
                                                x1: metrics.p50,
                                                y0: 0, 
                                                y1: 1, 
                                                yref: 'paper',
                                                line: { color: '#27AE60', width: 3, dash: 'dash' }
                                            },
                                            { // Línea P90 (Azul)
                                                type: 'line',
                                                x0: metrics.p90, 
                                                x1: metrics.p90,
                                                y0: 0, 
                                                y1: 1, 
                                                yref: 'paper',
                                                line: { color: '#2980B9', width: 3, dash: 'dash' }
                                            }
                                        ],
                                        legend: {
                                            x: 1,
                                            xanchor: 'right',
                                            y: 1,
                                            bgcolor: 'rgba(255, 255, 255, 0.8)',
                                            bordercolor: '#ccc',
                                            borderwidth: 1
                                        },
                                        bargap: 0.05,
                                        height: 400
                                    };

                                    return (
                                        <div 
                                            className="rounded-lg overflow-hidden"
                                            style={{
                                                backgroundColor: 'var(--color-bg-tertiary)',
                                                border: '1px solid var(--color-border-subtle)',
                                            }}
                                        >
                                            <Plot
                                                data={[traceHistogram, legendP10, legendP50, legendP90] as any}
                                                layout={layout}
                                                config={{ displayModeBar: false, responsive: true }}
                                                style={{ width: "100%", height: "400px" }}
                                                useResizeHandler
                                            />
                                        </div>
                                    );
                                })()}

                                {/* Probability Explanation */}
                                <div 
                                    className="rounded-lg p-4"
                                    style={{
                                        backgroundColor: 'var(--color-bg-tertiary)',
                                        border: '1px solid var(--color-border-subtle)',
                                    }}
                                >
                                    <div 
                                        className="text-xs font-semibold mb-2"
                                        style={{ color: 'var(--color-text-primary)' }}
                                    >
                                        Probabilistic Interpretation
                                    </div>
                                    <div 
                                        className="text-xs leading-relaxed space-y-1"
                                        style={{ color: 'var(--color-text-secondary)' }}
                                    >
                                        <p>
                                            • <strong>10% probability</strong> the price falls below <strong>{formatMoney(mcResult.percentiles.p10)}</strong> (bear case)
                                        </p>
                                        <p>
                                            • <strong>50% probability</strong> the price is below <strong>{formatMoney(mcResult.percentiles.p50)}</strong> (median outcome)
                                        </p>
                                        <p>
                                            • <strong>90% probability</strong> the price is below <strong>{formatMoney(mcResult.percentiles.p90)}</strong> (bull case)
                                        </p>
                                        {mcResult.metrics && (
                                            <p className="mt-2 pt-2 border-t" style={{ borderColor: 'var(--color-border-subtle)' }}>
                                                Expected value: <strong>{formatMoney(mcResult.metrics.mean)}</strong> • 
                                                Std deviation: <strong>{formatMoney(mcResult.metrics.std)}</strong>
                                            </p>
                                        )}
                                    </div>
                                </div>

                                {/* Paths Chart - Secondary */}
                                <details className="rounded-lg overflow-hidden" style={{ backgroundColor: 'var(--color-bg-tertiary)', border: '1px solid var(--color-border-subtle)' }}>
                                    <summary 
                                        className="px-4 py-2 cursor-pointer text-xs font-medium"
                                        style={{ color: 'var(--color-text-muted)' }}
                                    >
                                        View Simulation Paths (Secondary)
                                    </summary>
                                    <div className="p-4 pt-2">
                                        <Plot
                                            data={[mcResult.plotly_data] as any}
                                            layout={mcLayout}
                                            config={{ displayModeBar: false, responsive: true }}
                                            style={{ width: "100%", height: "240px" }}
                                            useResizeHandler
                                        />
                                    </div>
                                </details>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Empty State */}
            {!valuation && !isLoadingValuation && !valError && (
                <div
                    className="text-center py-12"
                    style={{ color: 'var(--color-text-muted)' }}
                >
                    <p className="text-sm">
                        Enter a ticker symbol to run valuation analysis
                    </p>
                </div>
            )}
            </div>
        </div>
    );
};

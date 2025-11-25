import React, { useState } from "react";
import Plot from "react-plotly.js";
import { WidgetCard } from "./WidgetCard";
import { fetchWithAuth, API_BASE_URL } from "../../services/apiService";

// ---------- Tipos que deben calzar con el backend ----------

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
  dcf: DcfBlock;
  reverse_dcf: ReverseDcfBlock;
  multiples_valuation: MultiplesValuationBlock;
  multiples: MultiplesBlock;
}

// Monte Carlo (mismo shape que tu widget actual)
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
  plotly_data: {
    x: (number | null)[];
    y: (number | null)[];
    type: string;
    mode: string;
    line: { width: number; color: string };
    name: string;
  };
  histogram: {
    x: number[];
    type: string;
    nbinsx: number;
    marker: { color: string; line: { color: string; width: number } };
    name: string;
  };
  simulation_params: {
    initial_value: number;
    mu: number;
    sigma: number;
    years: number;
    simulations: number;
  };
}

interface ScoringResponse {
  ticker: string;
  qualityScore: number;
  valuationScore: number;
  momentumScore: number;
  compositeScore: number;
  valuation_upside_pct: number | null;
}

const formatMoney = (v: number) =>
  `$${v.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;

export const ValuationTool: React.FC = () => {
  const [ticker, setTicker] = useState("AAPL");
  const [valuation, setValuation] = useState<QuickValuationResponse | null>(
    null
  );
  const [scoring, setScoring] = useState<ScoringResponse | null>(null);
  const [scoringError, setScoringError] = useState<string | null>(null);
  const [isLoadingValuation, setIsLoadingValuation] = useState(false);
  const [valError, setValError] = useState<string | null>(null);

  // Monte Carlo (portfolio view)
  const [initialValue, setInitialValue] = useState(100_000);
  const [mu, setMu] = useState(0.1);
  const [sigma, setSigma] = useState(0.25);
  const [years, setYears] = useState(5);
  const [simulations, setSimulations] = useState(10_000);
  const [mcResult, setMcResult] = useState<MonteCarloResult | null>(null);
  const [isLoadingMC, setIsLoadingMC] = useState(false);
  const [mcError, setMcError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    setIsLoadingValuation(true);
    setValError(null);
    setScoring(null);
    setScoringError(null);
    setMcResult(null);
    setMcError(null);

    try {
      const cleanTicker = ticker.trim().toUpperCase();
      if (!cleanTicker) {
        throw new Error("Please enter a ticker symbol.");
      }

      // 1) Precio actual
      const priceResp = await fetchWithAuth(
        `${API_BASE_URL}/api/prices/realtime/${cleanTicker}`
      );
      if (!priceResp.ok) {
        const errData = await priceResp
          .json()
          .catch(() => ({ detail: "Could not fetch current price" }));
        throw new Error(errData.detail || "Could not fetch current price");
      }
      const priceData: any = await priceResp.json();
      const currentPrice: number = priceData.price ?? priceData.current_price;

      if (!currentPrice || currentPrice <= 0) {
        throw new Error("Backend did not return a valid current price.");
      }

      // 2) Valuación DCF / múltiplos
      const dcfBody = {
        current_price: currentPrice, // el resto usa defaults del backend
      };

      const valResp = await fetchWithAuth(
        `${API_BASE_URL}/api/valuation/${cleanTicker}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(dcfBody),
        }
      );

      if (!valResp.ok) {
        const errData = await valResp
          .json()
          .catch(() => ({ detail: "Valuation failed" }));
        throw new Error(errData.detail || "Valuation failed");
      }

      const valData: QuickValuationResponse = await valResp.json();
      setValuation(valData);

      try {
        const scoringResp = await fetchWithAuth(`${API_BASE_URL}/api/analysis/scoring/${cleanTicker}`);
        if (scoringResp.ok) {
          const scoringData: ScoringResponse = await scoringResp.json();
          setScoring(scoringData);
        } else {
          throw new Error('Scoring failed');
        }
      } catch (scoringErr: any) {
        console.error('Scoring error:', scoringErr);
        setScoringError('No se pudieron calcular los puntajes de Quality/Valuation/Momentum.');
      }

      // 3) usar el CAGR implícito del DCF como μ y horizonte para Monte Carlo
      const impliedMu = valData.dcf.implied_return_cagr;
      const horizon = valData.dcf.assumptions.horizon_years;

      setMu(impliedMu);
      setYears(horizon);

      await runMonteCarlo(impliedMu, horizon);
    } catch (err: any) {
      console.error("Quick valuation error:", err);
      setValError(err.message || "An unexpected error occurred during valuation.");
    } finally {
      setIsLoadingValuation(false);
    }
  };

  const runMonteCarlo = async (muForSim?: number, yearsForSim?: number) => {
    // const muToUse = muForSim ?? mu; // Not used for stock forecast
    const yearsToUse = yearsForSim ?? years;

    setIsLoadingMC(true);
    setMcError(null);
    setMcResult(null);

    try {
      const resp = await fetchWithAuth(`${API_BASE_URL}/api/montecarlo/forecast/stock`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ticker: ticker.toUpperCase(),
          horizon_years: yearsToUse,
          simulations: simulations,
        }),
      });

      if (!resp.ok) {
        const errData = await resp
          .json()
          .catch(() => ({ detail: "Simulation failed" }));
        throw new Error(errData.detail || "Simulation failed");
      }

      const data: MonteCarloResult = await resp.json();
      setMcResult(data);
    } catch (err: any) {
      console.error("Monte Carlo valuation error:", err);
      setMcError(err.message || "Simulation failed");
    } finally {
      setIsLoadingMC(false);
    }
  };

  // Layouts Plotly
  const mcLayout = {
    title: {
      text: `Monte Carlo – portfolio in ${valuation?.ticker ?? ticker.toUpperCase()
        } (${mcResult?.simulation_params.simulations.toLocaleString() ??
        simulations.toLocaleString()
        } paths)`,
      font: { color: "#E0E1DD", size: 14 },
      pad: { b: 10 }
    },
    xaxis: {
      title: "Years",
      gridcolor: "#334155",
      color: "#94a3b8",
    },
    yaxis: {
      title: "Portfolio Value ($)",
      gridcolor: "#334155",
      color: "#94a3b8",
    },
    plot_bgcolor: "#0f172a",
    paper_bgcolor: "#0f172a",
    font: { color: "#94a3b8" },
    showlegend: false,
    margin: { l: 60, r: 20, t: 50, b: 50 },
  };

  const histLayout = {
    title: {
      text: "Distribution of Final Values",
      font: { color: "#E0E1DD", size: 14 },
    },
    xaxis: {
      title: "Final Value ($)",
      gridcolor: "#334155",
      color: "#94a3b8",
    },
    yaxis: {
      title: "Frequency",
      gridcolor: "#334155",
      color: "#94a3b8",
    },
    plot_bgcolor: "#0f172a",
    paper_bgcolor: "#0f172a",
    font: { color: "#94a3b8" },
    margin: { l: 60, r: 20, t: 50, b: 50 },
  };

  return (
    <WidgetCard
      title="QUICK VALUATION"
      tooltip="Valuación rápida con DCF y múltiplos. Analiza precio objetivo, upside potencial y simulaciones Monte Carlo para cualquier ticker."
    >
      <div className="space-y-6">
        {/* Ticker + botón */}
        <section className="space-y-3">
          <div className="flex flex-col md:flex-row gap-3 items-stretch">
            <div className="flex-1">
              <label className="block text-xs text-slate-400 mb-1">Ticker</label>
              <input
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                className="w-full bg-gray-800 border border-slate-700 rounded-md py-2 px-3 text-slate-100"
                placeholder="AAPL, MSFT, NVDA..."
              />
            </div>
            <button
              onClick={handleAnalyze}
              disabled={isLoadingValuation || !ticker.trim()}
              className="px-6 py-2 rounded-md bg-slate-700 hover:bg-slate-600 text-sm font-semibold"
            >
              {isLoadingValuation ? "Analyzing..." : "Analyze"}
            </button>
          </div>
          {valError && (
            <div className="text-sm text-red-400 bg-red-900/30 p-2 rounded-md">
              {valError}
            </div>
          )}
        </section>

        {scoring && (
          <section className="border border-slate-800 rounded-lg p-6 bg-gray-950/50 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-blue-500/10 to-transparent rounded-bl-full pointer-events-none"></div>
            
            <div className="flex flex-col md:flex-row gap-8 items-center">
              {/* Main C-Score */}
              <div className="text-center min-w-[180px]">
                <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-2">
                  Caria C-Score
                </h3>
                <div 
                  className="text-6xl font-bold mb-2"
                  style={{
                    color: scoring.compositeScore >= 80 ? '#10b981' : scoring.compositeScore >= 60 ? '#fbbf24' : '#9ca3af',
                    textShadow: '0 0 20px rgba(0,0,0,0.5)'
                  }}
                >
                  {scoring.compositeScore.toFixed(0)}
                </div>
                <div className="px-3 py-1 rounded-full text-xs font-bold inline-block"
                  style={{
                    backgroundColor: scoring.compositeScore >= 80 ? 'rgba(16, 185, 129, 0.2)' : scoring.compositeScore >= 60 ? 'rgba(251, 191, 36, 0.2)' : 'rgba(156, 163, 175, 0.2)',
                    color: scoring.compositeScore >= 80 ? '#10b981' : scoring.compositeScore >= 60 ? '#fbbf24' : '#9ca3af',
                  }}
                >
                  {scoring.compositeScore >= 80 ? 'PROBABLE OUTLIER' : scoring.compositeScore >= 60 ? 'HIGH-QUALITY' : 'STANDARD'}
                </div>
              </div>

              {/* Divider */}
              <div className="hidden md:block w-px h-24 bg-slate-800"></div>

              {/* Breakdown */}
              <div className="flex-1 w-full grid grid-cols-3 gap-4">
                <div className="space-y-1">
                  <div className="text-xs text-slate-500 font-medium">QUALITY (35%)</div>
                  <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div 
                      className="h-full rounded-full" 
                      style={{ width: `${scoring.qualityScore}%`, backgroundColor: '#3b82f6' }}
                    ></div>
                  </div>
                  <div className="text-lg font-bold text-slate-200">{scoring.qualityScore.toFixed(0)}<span className="text-xs text-slate-500 font-normal">/100</span></div>
                </div>

                <div className="space-y-1">
                  <div className="text-xs text-slate-500 font-medium">VALUATION (25%)</div>
                  <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div 
                      className="h-full rounded-full" 
                      style={{ width: `${scoring.valuationScore}%`, backgroundColor: '#8b5cf6' }}
                    ></div>
                  </div>
                  <div className="text-lg font-bold text-slate-200">{scoring.valuationScore.toFixed(0)}<span className="text-xs text-slate-500 font-normal">/100</span></div>
                </div>

                <div className="space-y-1">
                  <div className="text-xs text-slate-500 font-medium">MOMENTUM (20%)</div>
                  <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div 
                      className="h-full rounded-full" 
                      style={{ width: `${scoring.momentumScore}%`, backgroundColor: '#f59e0b' }}
                    ></div>
                  </div>
                  <div className="text-lg font-bold text-slate-200">{scoring.momentumScore.toFixed(0)}<span className="text-xs text-slate-500 font-normal">/100</span></div>
                </div>
                
                <div className="col-span-3 text-xs text-slate-500 mt-1 flex justify-between">
                   <span>* Moat Score (20%) not yet available</span>
                   <span>Weights re-normalized to 100%</span>
                </div>
              </div>
            </div>
          </section>
        )}

        {scoringError && (
          <div className="text-xs text-amber-300 bg-amber-900/20 border border-amber-500/20 p-2 rounded-md">
            {scoringError}
          </div>
        )}

        {/* Contenido de valuación */}
        {valuation && (
          <section className="space-y-6">
            {/* Reverse DCF & Multiples Valuation (New) */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Reverse DCF */}
              <div className="bg-indigo-900/20 border border-indigo-500/30 rounded-lg p-4 space-y-2">
                <div className="flex justify-between items-start">
                  <div className="text-xs text-indigo-300 font-semibold uppercase tracking-wider">Reverse DCF</div>
                  <div className="text-xs text-indigo-400/70">Implied Growth</div>
                </div>
                <div className="text-3xl font-bold text-indigo-100">
                  {(valuation.reverse_dcf.implied_growth_rate * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-indigo-300/80 leading-relaxed">
                  {valuation.reverse_dcf.explanation}
                </div>
              </div>

              {/* Multiples Valuation */}
              <div className="bg-emerald-900/20 border border-emerald-500/30 rounded-lg p-4 space-y-2">
                <div className="flex justify-between items-start">
                  <div className="text-xs text-emerald-300 font-semibold uppercase tracking-wider">Historical Multiples</div>
                  <div className="text-xs text-emerald-400/70">Fair Value</div>
                </div>
                <div className="text-3xl font-bold text-emerald-100">
                  {formatMoney(valuation.multiples_valuation.fair_value)}
                </div>
                <div className="text-xs text-emerald-300/80 leading-relaxed space-y-1">
                  <div>
                    Medianas: EV/Sales {valuation.multiples_valuation.ev_sales_median?.toFixed(1) ?? '—'}x • EV/EBITDA{' '}
                    {valuation.multiples_valuation.ev_ebitda_median?.toFixed(1) ?? '—'}x
                  </div>
                  {valuation.multiples_valuation.breakdown && (
                    <div className="text-[11px] text-emerald-200/80">
                      {valuation.multiples_valuation.breakdown.ev_sales !== undefined && (
                        <span>EV/Sales → {formatMoney(valuation.multiples_valuation.breakdown.ev_sales)} </span>
                      )}
                      {valuation.multiples_valuation.breakdown.ev_ebitda !== undefined && (
                        <span className="ml-1">
                          EV/EBITDA → {formatMoney(valuation.multiples_valuation.breakdown.ev_ebitda)}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Multiples Table */}
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-slate-300">Current Market Multiples</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                {Object.entries(valuation.multiples.multiples).map(
                  ([key, value]) => {
                    const num = Number(value);
                    return (
                      <div
                        key={key}
                        className="bg-gray-900/40 p-2 rounded border border-slate-800"
                      >
                        <div className="text-slate-500 mb-1">{key}</div>
                        <div className="text-slate-200 font-mono">
                          {key.includes("yield") || key.includes("rate")
                            ? `${(num * 100).toFixed(1)}%`
                            : num.toFixed(2)}
                        </div>
                      </div>
                    );
                  }
                )}
              </div>
            </div>

            {/* Monte Carlo */}
            <div className="border-t border-slate-800 pt-4 space-y-3">
              <h2 className="text-sm font-semibold text-slate-300">
                Monte Carlo – Stock Price Forecast
              </h2>
              <p className="text-xs text-slate-500">
                Projecting future stock price based on historical volatility.
              </p>

              {/* Parámetros MC */}
              <div className="flex items-center gap-3 text-xs">
                <div className="w-24">
                  <label className="text-slate-400 block mb-1">Years</label>
                  <input
                    type="number"
                    value={years}
                    onChange={(e) => setYears(parseInt(e.target.value) || 1)}
                    className="w-full bg-gray-800 border border-slate-700 rounded py-1 px-2 text-slate-100"
                    disabled={isLoadingMC}
                  />
                </div>
                <div className="flex-1 flex items-end">
                  <button
                    onClick={() => runMonteCarlo()}
                    disabled={isLoadingMC}
                    className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-1.5 rounded transition-colors text-xs font-medium"
                  >
                    {isLoadingMC ? "Running..." : "Re-run Simulation"}
                  </button>
                </div>
              </div>

              {mcError && (
                <div className="text-xs text-red-400 bg-red-900/20 p-2 rounded">
                  {mcError}
                </div>
              )}

              {mcResult && (
                <div className="space-y-4 mt-2">
                  {/* Percentiles */}
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div className="bg-gray-900/40 p-2 rounded border border-slate-800">
                      <div className="text-slate-500">P10 (Bear)</div>
                      <div className="text-slate-200 font-mono">
                        {formatMoney(mcResult.percentiles.p10)}
                      </div>
                    </div>
                    <div className="bg-gray-900/40 p-2 rounded border border-slate-800">
                      <div className="text-slate-500">P50 (Base)</div>
                      <div className="text-slate-200 font-mono">
                        {formatMoney(mcResult.percentiles.p50)}
                      </div>
                    </div>
                    <div className="bg-gray-900/40 p-2 rounded border border-slate-800">
                      <div className="text-slate-500">P90 (Bull)</div>
                      <div className="text-slate-200 font-mono">
                        {formatMoney(mcResult.percentiles.p90)}
                      </div>
                    </div>
                  </div>

                  {/* Paths */}
                  <div className="bg-gray-900/20 rounded border border-slate-800/50 p-1">
                    <Plot
                      data={[mcResult.plotly_data] as any}
                      layout={mcLayout}
                      config={{ displayModeBar: false, responsive: true }}
                      style={{ width: "100%", height: "250px" }}
                      useResizeHandler
                    />
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {!valuation && !isLoadingValuation && !valError && (
          <div className="text-center text-xs text-slate-500 py-8 italic">
            Enter a ticker to see Reverse DCF, Multiples Valuation, and Monte Carlo forecasts.
          </div>
        )}
      </div>
    </WidgetCard>
  );
};

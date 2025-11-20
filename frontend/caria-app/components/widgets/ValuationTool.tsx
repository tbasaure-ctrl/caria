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

interface QuickValuationResponse {
  ticker: string;
  currency: string;
  current_price: number;
  dcf: DcfBlock;
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

const formatMoney = (v: number) =>
  `$${v.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;

export const ValuationTool: React.FC = () => {
  const [ticker, setTicker] = useState("AAPL");
  const [valuation, setValuation] = useState<QuickValuationResponse | null>(
    null
  );
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
    const muToUse = muForSim ?? mu;
    const yearsToUse = yearsForSim ?? years;

    setIsLoadingMC(true);
    setMcError(null);
    setMcResult(null);

    try {
      const resp = await fetchWithAuth(`${API_BASE_URL}/api/montecarlo/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          initial_value: initialValue,
          mu: muToUse,
          sigma: sigma,
          years: yearsToUse,
          simulations: simulations,
          contributions_per_year: 0.0,
          annual_fee: 0.0,
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
      setMcError("Coming soon... Monte Carlo simulations are being enhanced for more accurate predictions.");
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

        {/* Contenido de valuación */}
        {valuation && (
          <section className="space-y-6">
            {/* Fair value + precio actual */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-900/70 rounded-lg p-4 space-y-1">
                <div className="text-xs text-slate-400">Fair Value</div>
                <div className="text-2xl font-semibold text-slate-100">
                  {formatMoney(valuation.dcf.fair_value_per_share)}
                </div>
                <div className="text-xs text-slate-500">
                  via DCF scorecard · Upside{" "}
                  <span
                    className={
                      valuation.dcf.upside_percent >= 0
                        ? "text-emerald-400"
                        : "text-red-400"
                    }
                  >
                    {valuation.dcf.upside_percent.toFixed(1)}%
                  </span>
                </div>
              </div>

              <div className="bg-gray-900/70 rounded-lg p-4 space-y-1">
                <div className="text-xs text-slate-400">Current Price</div>
                <div className="text-2xl font-semibold text-slate-100">
                  {formatMoney(valuation.current_price)}
                </div>
                <div className="text-xs text-slate-500">
                  Currency: {valuation.currency}
                </div>
              </div>
            </div>

            {/* DCF method + assumptions */}
            <div className="space-y-3">
              <h2 className="text-lg font-semibold text-slate-100">
                1. DCF – Method & Assumptions
              </h2>
              <p className="text-sm text-slate-400">{valuation.dcf.explanation}</p>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
                <div className="bg-gray-900/60 p-3 rounded-md">
                  <div className="text-slate-400">FCF Yield start</div>
                  <div className="text-slate-100 font-mono">
                    {(valuation.dcf.assumptions.fcf_yield_start * 100).toFixed(
                      1
                    )}
                    %
                  </div>
                </div>
                <div className="bg-gray-900/60 p-3 rounded-md">
                  <div className="text-slate-400">High growth</div>
                  <div className="text-slate-100 font-mono">
                    {(
                      valuation.dcf.assumptions.high_growth_rate * 100
                    ).toFixed(1)}
                    % · {valuation.dcf.assumptions.high_growth_years} yrs
                  </div>
                </div>
                <div className="bg-gray-900/60 p-3 rounded-md">
                  <div className="text-slate-400">Fade years</div>
                  <div className="text-slate-100 font-mono">
                    {valuation.dcf.assumptions.fade_years}
                  </div>
                </div>
                <div className="bg-gray-900/60 p-3 rounded-md">
                  <div className="text-slate-400">Terminal growth</div>
                  <div className="text-slate-100 font-mono">
                    {(
                      valuation.dcf.assumptions.terminal_growth_rate * 100
                    ).toFixed(1)}
                    %
                  </div>
                </div>
                <div className="bg-gray-900/60 p-3 rounded-md">
                  <div className="text-slate-400">Discount rate</div>
                  <div className="text-slate-100 font-mono">
                    {(valuation.dcf.assumptions.discount_rate * 100).toFixed(
                      1
                    )}
                    %
                  </div>
                </div>
                <div className="bg-gray-900/60 p-3 rounded-md">
                  <div className="text-slate-400">Implied return (CAGR)</div>
                  <div className="text-slate-100 font-mono">
                    {(valuation.dcf.implied_return_cagr * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>

            {/* Multiples derivados */}
            <div className="space-y-3">
              <h2 className="text-lg font-semibold text-slate-100">
                2. Multiples – sanity check
              </h2>
              <p className="text-sm text-slate-400">
                {valuation.multiples.explanation}
              </p>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
                {Object.entries(valuation.multiples.multiples).map(
                  ([key, value]) => {
                    const num = Number(value); // cast para contentar a TS
                    return (
                      <div
                        key={key}
                        className="bg-gray-900/60 p-3 rounded-md space-y-1"
                      >
                        <div className="text-slate-400">{key}</div>
                        <div className="text-slate-100 font-mono">
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

            {/* Monte Carlo como retorno del portfolio */}
            <div className="space-y-3">
              <h2 className="text-lg font-semibold text-slate-100">
                3. Monte Carlo – portfolio invested in {valuation.ticker}
              </h2>
              <p className="text-sm text-slate-400">
                Se usa el retorno anual implícito del DCF como μ inicial y se
                simula la distribución de posibles trayectorias del valor de un
                portfolio invertido en {valuation.ticker}. Puedes ajustar μ y σ
                para explorar escenarios más agresivos o conservadores.
              </p>

              {/* Parámetros MC */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                <div>
                  <div className="text-slate-400">Initial Value ($)</div>
                  <input
                    type="number"
                    value={initialValue}
                    onChange={(e) =>
                      setInitialValue(parseFloat(e.target.value) || 0)
                    }
                    className="w-full mt-1 bg-gray-800 border border-slate-700 rounded-md py-1 px-2 text-slate-100"
                    disabled={isLoadingMC}
                  />
                </div>
                <div>
                  <div className="text-slate-400">Expected Return μ</div>
                  <input
                    type="number"
                    step="0.01"
                    value={mu}
                    onChange={(e) => setMu(parseFloat(e.target.value) || 0)}
                    className="w-full mt-1 bg-gray-800 border border-slate-700 rounded-md py-1 px-2 text-slate-100"
                    disabled={isLoadingMC}
                  />
                </div>
                <div>
                  <div className="text-slate-400">Volatility σ</div>
                  <input
                    type="number"
                    step="0.01"
                    value={sigma}
                    onChange={(e) => setSigma(parseFloat(e.target.value) || 0)}
                    className="w-full mt-1 bg-gray-800 border border-slate-700 rounded-md py-1 px-2 text-slate-100"
                    disabled={isLoadingMC}
                  />
                </div>
                <div>
                  <div className="text-slate-400">Years</div>
                  <input
                    type="number"
                    value={years}
                    onChange={(e) => setYears(parseInt(e.target.value) || 1)}
                    className="w-full mt-1 bg-gray-800 border border-slate-700 rounded-md py-1 px-2 text-slate-100"
                    disabled={isLoadingMC}
                  />
                </div>
              </div>

              <button
                onClick={() => runMonteCarlo()}
                disabled={isLoadingMC || sigma <= 0}
                className="w-full bg-slate-700 text-white font-bold py-2 px-4 rounded-md hover:bg-slate-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed text-sm"
              >
                {isLoadingMC ? "Running Monte Carlo..." : "Re-run Monte Carlo"}
              </button>

              {mcError && (
                <div className="text-sm text-red-400 bg-red-900/30 p-2 rounded-md">
                  {mcError}
                </div>
              )}

              {mcResult && (
                <div className="space-y-4">
                  {/* Percentiles */}
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div className="bg-gray-900/70 p-2 rounded-md">
                      <div className="text-slate-400">P10</div>
                      <div className="text-slate-100 font-mono">
                        {formatMoney(mcResult.percentiles.p10)}
                      </div>
                    </div>
                    <div className="bg-gray-900/70 p-2 rounded-md">
                      <div className="text-slate-400">P50 (Median)</div>
                      <div className="text-slate-100 font-mono">
                        {formatMoney(mcResult.percentiles.p50)}
                      </div>
                    </div>
                    <div className="bg-gray-900/70 p-2 rounded-md">
                      <div className="text-slate-400">P90</div>
                      <div className="text-slate-100 font-mono">
                        {formatMoney(mcResult.percentiles.p90)}
                      </div>
                    </div>
                  </div>

                  {/* Riesgo */}
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Expected Value:</span>
                      <span className="text-slate-100 font-mono">
                        {formatMoney(mcResult.metrics.mean)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">VaR (5%):</span>
                      <span className="text-red-400 font-mono">
                        {formatMoney(mcResult.metrics.var_5pct)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">CVaR (5%):</span>
                      <span className="text-red-400 font-mono">
                        {formatMoney(mcResult.metrics.cvar_5pct)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Prob. Loss:</span>
                      <span className="text-slate-300 font-mono">
                        {(
                          mcResult.metrics.prob_final_less_invested * 100
                        ).toFixed(1)}
                        %
                      </span>
                    </div>
                  </div>

                  {/* Paths */}
                  <div className="bg-gray-900/50 rounded-md p-2">
                    <Plot
                      data={[mcResult.plotly_data] as any}
                      layout={mcLayout}
                      config={{ displayModeBar: false, responsive: true }}
                      style={{ width: "100%", height: "300px" }}
                      useResizeHandler
                    />
                  </div>

                  {/* Histograma */}
                  {mcResult.histogram && (
                    <div className="bg-gray-900/50 rounded-md p-2">
                      <Plot
                        data={[mcResult.histogram] as any}
                        layout={histLayout}
                        config={{ displayModeBar: false, responsive: true }}
                        style={{ width: "100%", height: "250px" }}
                        useResizeHandler
                      />
                    </div>
                  )}
                </div>
              )}

              {!mcResult && !isLoadingMC && !mcError && (
                <div className="text-center text-xs text-slate-500 py-4">
                  Run Monte Carlo to see the distribution of possible portfolio
                  values given this valuation.
                </div>
              )}
            </div>
          </section>
        )}

        {!valuation && !isLoadingValuation && !valError && (
          <div className="text-center text-xs text-slate-500 py-4">
            Enter a ticker and click Analyze to get a DCF-based fair value and
            a portfolio Monte Carlo projection.
          </div>
        )}
      </div>
    </WidgetCard>
  );
};

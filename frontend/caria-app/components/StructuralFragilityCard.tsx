
import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ScatterChart, Scatter, ReferenceLine } from "recharts";
import { useCariaSR } from "../hooks/useCariaSR"; // Check path alias

interface Props {
    ticker: string;
}

export default function StructuralFragilityCard({ ticker }: Props) {
    const { series, status, loading, error } = useCariaSR(ticker);

    if (loading) return <div className="p-4 text-white/60 animate-pulse">Analizando fragilidad estructural...</div>;
    if (error || !status) return <div className="p-4 text-white/40">Datos no disponibles para {ticker} (Run SR Job)</div>;

    const scatterData = series.map(p => ({
        e4: p.e4,
        sr: p.sr,
        regime: p.regime,
    }));

    // Helper for regime color
    const regimeColor = status.last_regime === 1 ? "text-red-400" : "text-green-400";
    const regimeLabel = status.last_regime === 1 ? "FRÁGIL" : "NORMAL";

    return (
        <div className="bg-white/5 border border-white/10 rounded-2xl shadow-lg p-6 space-y-6">
            <div className="flex justify-between items-start">
                <div>
                    <h2 className="text-xl font-bold text-white flex items-center gap-2">
                        Structural Fragility
                        <span className="text-xs bg-white/10 px-2 py-1 rounded text-white/60">{ticker}</span>
                    </h2>
                    <p className="text-xs text-white/40 mt-1">Análisis de correlación vol-crédito y riesgo oculto</p>
                </div>
                <div className="text-right">
                    <div className={`text-2xl font-bold ${regimeColor}`}>{regimeLabel}</div>
                    <div className="text-xs text-white/50">SR: {status.last_sr.toFixed(2)} | AUC: {status.auc.toFixed(2)}</div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                {/* Time Series */}
                <div className="h-64 bg-black/20 rounded-xl p-4 border border-white/5">
                    <h3 className="text-xs font-semibold text-white/60 mb-2 uppercase">Evolución SR (Histórico)</h3>
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={series}>
                            <XAxis dataKey="date" hide />
                            <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: '#666' }} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#111', borderColor: '#333', color: '#fff' }}
                                itemStyle={{ color: '#ccc' }}
                            />
                            <ReferenceLine y={0.8} stroke="red" strokeDasharray="3 3" opacity={0.5} />
                            <Line type="monotone" dataKey="sr" stroke="#fbbf24" dot={false} strokeWidth={2} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                {/* Scatter Plot */}
                <div className="h-64 bg-black/20 rounded-xl p-4 border border-white/5">
                    <h3 className="text-xs font-semibold text-white/60 mb-2 uppercase">Mapa de Fragilidad (E4 vs SR)</h3>
                    <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart>
                            <XAxis dataKey="e4" type="number" name="E4 (Vol Mix)" tick={{ fontSize: 10, fill: '#666' }} />
                            <YAxis dataKey="sr" type="number" name="SR Score" domain={[0, 1]} tick={{ fontSize: 10, fill: '#666' }} />
                            <Tooltip
                                cursor={{ strokeDasharray: '3 3' }}
                                contentStyle={{ backgroundColor: '#111', borderColor: '#333', color: '#fff' }}
                            />
                            {/* Scatter points color coded by regime? simplistic for now */}
                            <Scatter name="Normal" data={scatterData.filter(d => d.regime === 0)} fill="#4ade80" fillOpacity={0.6} shape="circle" />
                            <Scatter name="Fragile" data={scatterData.filter(d => d.regime === 1)} fill="#f87171" fillOpacity={0.6} shape="triangle" />
                        </ScatterChart>
                    </ResponsiveContainer>
                </div>

            </div>

            <div className="text-xs text-white/30 pt-2 border-t border-white/5 flex gap-4">
                <span>Mean Return (Normal): {(status.mean_normal * 100 * 252).toFixed(1)}% ann.</span>
                <span>Mean Return (Fragile): {(status.mean_fragile * 100 * 252).toFixed(1)}% ann.</span>
            </div>
        </div>
    );
}

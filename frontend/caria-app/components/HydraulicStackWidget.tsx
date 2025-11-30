import React, { useEffect, useState } from 'react';
import { Activity, Eye, Shield, Crosshair, Droplets, Info } from 'lucide-react';
import { API_BASE_URL } from '../services/apiService';

interface StackStatus {
    liquidity: {
        score: number;
        state: string;
        trend: number;
    };
    llm: {
        mode: string;
        operational: boolean;
    };
    volatility: {
        signal: string;
        ratio: number;
    };
    execution: {
        position: string;
        risk: string;
    };
}

export default function HydraulicStackWidget() {
    const [status, setStatus] = useState<StackStatus | null>(null);
    const [pulse, setPulse] = useState(0);

    useEffect(() => {
        fetchStackStatus();
        const interval = setInterval(fetchStackStatus, 10000); // Update every 10s
        const pulseInterval = setInterval(() => setPulse(p => (p + 1) % 3), 2000);
        return () => {
            clearInterval(interval);
            clearInterval(pulseInterval);
        };
    }, []);

    const fetchStackStatus = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/liquidity/status`);
            const data = await response.json();

            setStatus({
                liquidity: {
                    score: data.score || 50,
                    state: data.state || 'NEUTRAL',
                    trend: data.trend_roc_4w || 0
                },
                llm: {
                    mode: data.score > 60 ? 'GROWTH' : data.score < 40 ? 'BEAR' : 'BALANCED',
                    operational: true
                },
                volatility: {
                    signal: 'NORMAL',
                    ratio: 1.0
                },
                execution: {
                    position: data.score > 60 ? 'MAX' : data.score < 40 ? 'SMALL' : 'HALF',
                    risk: data.score > 60 ? 'LOW' : data.score < 40 ? 'HIGH' : 'MEDIUM'
                }
            });
        } catch (error) {
            console.error('Error fetching stack status:', error);
        }
    };

    if (!status) {
        return (
            <div className="w-full bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 border border-purple-500/20 rounded-xl p-6">
                <div className="text-purple-300/50 text-center">Loading Hydraulic Stack...</div>
            </div>
        );
    }

    const getTextColor = (score: number) => {
        if (score >= 60) return 'text-green-400';
        if (score <= 40) return 'text-red-400';
        return 'text-yellow-400';
    };

    return (
        <div className="w-full bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-950 border border-purple-500/30 shadow-2xl shadow-purple-500/20 relative overflow-hidden rounded-xl">
            {/* Animated background effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500/5 via-transparent to-cyan-500/5 animate-pulse" />

            {/* Mystic glow orb */}
            <div className={`absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-gradient-radial from-purple-500/20 to-transparent blur-3xl transition-opacity duration-1000 ${pulse === 0 ? 'opacity-40' : pulse === 1 ? 'opacity-60' : 'opacity-30'}`} />

            {/* Header */}
            <div className="relative z-10 p-4 pb-3 border-b border-purple-500/20">
                <div className="flex items-center gap-2 text-purple-100 font-bold">
                    <Activity className="h-5 w-5 text-purple-400 animate-pulse" />
                    <span className="bg-gradient-to-r from-purple-200 via-cyan-200 to-purple-200 bg-clip-text text-transparent">
                        AI-Hydraulic Stack
                    </span>
                    <div className="group relative ml-1">
                        <Info className="h-4 w-4 text-purple-400/60 hover:text-purple-300 cursor-help" />
                        <div className="absolute left-0 top-6 w-72 bg-slate-900 border border-purple-500/50 rounded-lg p-3 text-xs text-gray-300 font-normal opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-opacity z-50">
                            <p className="font-semibold text-purple-300 mb-1">AI-Hydraulic Stack</p>
                            <p>Monitors Fed liquidity (Assets - TGA - RRP) and yield curve to determine market regime. Score &gt;60 = Expansion (growth mode), &lt;40 = Contraction (defensive mode). Guides strategy allocation in real-time.</p>
                        </div>
                    </div>
                    <span className="text-xs text-purple-400/60 font-normal ml-auto">LIVE</span>
                </div>
            </div>

            {/* Content */}
            <div className="relative z-10 p-4 space-y-3">
                {/* Core Score Display */}
                <div className={`relative p-4 rounded-lg bg-gradient-to-r ${status.liquidity.score >= 60 ? 'from-green-500/20 via-emerald-500/30 to-green-500/20' :
                    status.liquidity.score <= 40 ? 'from-red-500/20 via-rose-500/30 to-red-500/20' :
                        'from-yellow-500/20 via-amber-500/30 to-yellow-500/20'
                    } border border-purple-400/20 backdrop-blur-sm`}>
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <Droplets className={`h-8 w-8 ${getTextColor(status.liquidity.score)}`} />
                            <div>
                                <div className={`text-2xl font-bold ${getTextColor(status.liquidity.score)}`}>
                                    {status.liquidity.score}
                                    <span className="text-sm ml-1 opacity-60">/100</span>
                                </div>
                                <div className="text-xs text-purple-300/80">Hydraulic Core</div>
                            </div>
                        </div>
                        <span className={`inline-flex items-center px-2 py-1 rounded-md text-xs font-semibold ${getTextColor(status.liquidity.score)} bg-black/30 border border-current`}>
                            {status.liquidity.state}
                        </span>
                    </div>
                </div>

                {/* 4 Layers Grid */}
                <div className="grid grid-cols-2 gap-2">
                    {/* Layer 2: Scout (LLM) */}
                    <div className="p-3 rounded-lg bg-gradient-to-br from-cyan-500/10 to-transparent border border-cyan-500/20 backdrop-blur-sm hover:border-cyan-400/40 transition-all">
                        <div className="flex items-center gap-2 mb-1">
                            <Eye className="h-4 w-4 text-cyan-400" />
                            <span className="text-xs font-semibold text-cyan-300">Scout</span>
                        </div>
                        <div className="text-xs text-cyan-100/80">{status.llm.mode}</div>
                        <div className={`text-[10px] mt-1 ${status.llm.operational ? 'text-green-400' : 'text-red-400'}`}>
                            ● {status.llm.operational ? 'Active' : 'Offline'}
                        </div>
                    </div>

                    {/* Layer 3: Guard (Vol) */}
                    <div className="p-3 rounded-lg bg-gradient-to-br from-amber-500/10 to-transparent border border-amber-500/20 backdrop-blur-sm hover:border-amber-400/40 transition-all">
                        <div className="flex items-center gap-2 mb-1">
                            <Shield className="h-4 w-4 text-amber-400" />
                            <span className="text-xs font-semibold text-amber-300">Guard</span>
                        </div>
                        <div className="text-xs text-amber-100/80">{status.volatility.signal}</div>
                        <div className="text-[10px] text-amber-300/70 mt-1">
                            {status.volatility.ratio.toFixed(2)}x vol
                        </div>
                    </div>

                    {/* Layer 4: Sniper (Execution) */}
                    <div className="p-3 rounded-lg bg-gradient-to-br from-rose-500/10 to-transparent border border-rose-500/20 backdrop-blur-sm hover:border-rose-400/40 transition-all col-span-2">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Crosshair className="h-4 w-4 text-rose-400" />
                                <span className="text-xs font-semibold text-rose-300">Sniper</span>
                            </div>
                            <div className="text-right">
                                <div className="text-xs text-rose-100/80">{status.execution.position} Position</div>
                                <div className={`text-[10px] mt-0.5 ${status.execution.risk === 'LOW' ? 'text-green-400' :
                                    status.execution.risk === 'HIGH' ? 'text-red-400' : 'text-yellow-400'
                                    }`}>
                                    {status.execution.risk} Risk
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* System Philosophy tagline */}
                <div className="text-center pt-2 border-t border-purple-500/20">
                    <p className="text-[10px] text-purple-300/50 italic font-light">
                        "Intelligence Respecting Liquidity"
                    </p>
                </div>
            </div>
        </div>
    );
}

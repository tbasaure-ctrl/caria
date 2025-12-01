import React, { useEffect, useState } from 'react';
import { Activity, Eye, Shield, Crosshair, Droplets, Info, Zap } from 'lucide-react';
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
            if (!response.ok) {
                throw new Error('API not available');
            }
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
            // Use fallback data when API is unavailable
            setStatus({
                liquidity: { score: 55, state: 'NEUTRAL', trend: 0.02 },
                llm: { mode: 'BALANCED', operational: true },
                volatility: { signal: 'NORMAL', ratio: 1.0 },
                execution: { position: 'HALF', risk: 'MEDIUM' }
            });
        }
    };

    if (!status) {
        return (
            <div className="w-full bg-bg-tertiary border border-white/5 rounded-xl p-6">
                <div className="text-text-muted text-center text-xs animate-pulse">Initializing Hydraulic Stack...</div>
            </div>
        );
    }

    const getTextColor = (score: number) => {
        if (score >= 60) return 'text-positive';
        if (score <= 40) return 'text-negative';
        return 'text-warning';
    };

    const getBgColor = (score: number) => {
        if (score >= 60) return 'bg-positive/10 border-positive/20';
        if (score <= 40) return 'bg-negative/10 border-negative/20';
        return 'bg-warning/10 border-warning/20';
    };

    const getTacticalInsight = () => {
        const { score } = status.liquidity;
        if (score >= 65) return "Liquidity is abundant. Risk-on environment favored. Increase exposure to growth and beta.";
        if (score >= 45) return "Liquidity is neutral. Market is choppy. Focus on stock selection and quality factors.";
        if (score >= 30) return "Liquidity is tightening. Risk-off environment. Reduce leverage and increase cash or defensive allocations.";
        return "Liquidity crisis detected. Maximum defensive posture recommended. Cash is king.";
    };

    return (
        <div className="w-full bg-[#0B1221] border border-white/5 shadow-lg relative overflow-hidden rounded-xl h-full flex flex-col">
            {/* Header */}
            <div className="relative z-10 p-4 pb-3 border-b border-white/5 flex justify-between items-center">
                <div className="flex items-center gap-2">
                    <Activity className="h-4 w-4 text-accent-cyan animate-pulse" />
                    <span className="font-display font-bold text-white tracking-wide text-sm">
                        AI-Hydraulic Stack
                    </span>
                    <div className="group relative ml-1">
                        <Info className="h-3 w-3 text-text-muted hover:text-accent-cyan cursor-help" />
                        <div className="absolute left-0 top-6 w-64 bg-bg-surface border border-white/10 rounded-lg p-3 text-[10px] text-text-secondary opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-opacity z-50 shadow-xl">
                            <p className="font-bold text-white mb-1">Liquidity Engine</p>
                            <p>Monitors Fed liquidity (Assets - TGA - RRP) to determine market regime. Score &gt;60 = Expansion, &lt;40 = Contraction.</p>
                        </div>
                    </div>
                </div>
                <span className="text-[10px] font-mono text-accent-cyan/80 flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan animate-pulse"></span>
                    LIVE
                </span>
            </div>

            {/* Content */}
            <div className="relative z-10 p-4 space-y-3 flex-1 overflow-y-auto custom-scrollbar">
                {/* Core Score Display */}
                <div className={`relative p-4 rounded-lg border backdrop-blur-sm transition-colors duration-500 ${getBgColor(status.liquidity.score)}`}>
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <Droplets className={`h-8 w-8 ${getTextColor(status.liquidity.score)}`} />
                            <div>
                                <div className={`text-3xl font-mono font-bold ${getTextColor(status.liquidity.score)}`}>
                                    {status.liquidity.score}
                                    <span className="text-xs ml-1 opacity-60 text-text-muted">/100</span>
                                </div>
                                <div className="text-[10px] text-text-muted uppercase tracking-wider">Liquidity Score</div>
                            </div>
                        </div>
                        <span className={`inline-flex items-center px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider ${getTextColor(status.liquidity.score)} bg-black/20 border border-white/5`}>
                            {status.liquidity.state}
                        </span>
                    </div>
                </div>

                {/* 4 Layers Grid */}
                <div className="grid grid-cols-2 gap-2">
                    {/* Layer 2: Scout (LLM) */}
                    <div className="p-3 rounded-lg bg-bg-tertiary border border-white/5 hover:border-accent-cyan/30 transition-all">
                        <div className="flex items-center gap-2 mb-1">
                            <Eye className="h-3 w-3 text-accent-cyan" />
                            <span className="text-[10px] font-bold text-text-secondary uppercase tracking-wider">Scout</span>
                        </div>
                        <div className="text-xs font-bold text-white">{status.llm.mode}</div>
                        <div className={`text-[9px] mt-1 ${status.llm.operational ? 'text-positive' : 'text-negative'}`}>
                            ● {status.llm.operational ? 'Active' : 'Offline'}
                        </div>
                    </div>

                    {/* Layer 3: Guard (Vol) */}
                    <div className="p-3 rounded-lg bg-bg-tertiary border border-white/5 hover:border-warning/30 transition-all">
                        <div className="flex items-center gap-2 mb-1">
                            <Shield className="h-3 w-3 text-warning" />
                            <span className="text-[10px] font-bold text-text-secondary uppercase tracking-wider">Guard</span>
                        </div>
                        <div className="text-xs font-bold text-white">{status.volatility.signal}</div>
                        <div className="text-[9px] text-text-muted mt-1 font-mono">
                            {status.volatility.ratio.toFixed(2)}x vol
                        </div>
                    </div>

                    {/* Layer 4: Sniper (Execution) */}
                    <div className="p-3 rounded-lg bg-bg-tertiary border border-white/5 hover:border-accent-primary/30 transition-all col-span-2">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Crosshair className="h-3 w-3 text-accent-primary" />
                                <span className="text-[10px] font-bold text-text-secondary uppercase tracking-wider">Sniper Execution</span>
                            </div>
                            <div className="text-right flex items-center gap-3">
                                <div>
                                    <div className="text-xs font-bold text-white">{status.execution.position}</div>
                                    <div className="text-[9px] text-text-muted">Position</div>
                                </div>
                                <div className={`text-right pl-3 border-l border-white/10`}>
                                    <div className={`text-xs font-bold ${status.execution.risk === 'LOW' ? 'text-positive' : status.execution.risk === 'HIGH' ? 'text-negative' : 'text-warning'}`}>
                                        {status.execution.risk}
                                    </div>
                                    <div className="text-[9px] text-text-muted">Risk</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Tactical Insight Footer */}
            <div className="relative z-10 bg-black/40 border-t border-white/5 p-3">
                <div className="flex items-start gap-3">
                    <div className="p-1.5 bg-accent-cyan/10 rounded border border-accent-cyan/30 mt-0.5">
                        <Zap className="w-3 h-3 text-accent-cyan" />
                    </div>
                    <div>
                        <div className="text-[10px] text-accent-cyan uppercase tracking-wider font-bold mb-0.5">Tactical Insight</div>
                        <p className="text-xs text-text-secondary leading-snug">
                            {getTacticalInsight()}
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}

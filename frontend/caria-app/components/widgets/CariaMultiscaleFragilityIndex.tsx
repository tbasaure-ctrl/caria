/**
 * CARIA: COGNITIVE ANALYSIS AND RISK INVESTMENT ASSISTANT
 * 
 * "Einstein's Universe" Visualization.
 * - Fragility is visualized as the Topology of Social Consensus.
 * - High Synchronization = Deep Well (Total Consensus).
 * - Low Synchronization = Flat Field (Healthy Disagreement).
 */

import React, { useEffect, useState, useRef, useCallback } from 'react';
import { BookOpen, X, Play, Zap, Info, HelpCircle } from 'lucide-react';
import { API_BASE_URL } from '../../services/apiService';

interface MSFIData {
    version: string;
    metrics: {
        msfi: number;     // "Depth" -> Strength of Consensus
        clock_sync: number; // "Distortion" -> Uniformity of Opinion
        resonance: number;  // "Energy" -> Glow intensity
        trend_signal: number; // "Momentum" -> Direction (-1 to 1)
    };
    status: 'STABLE' | 'WARNING' | 'CRITICAL';
    auc_score: number;
}

const DEMO_DATA: MSFIData = {
    version: 'CARIA 1.0',
    status: 'WARNING',
    metrics: { msfi: 2.1, clock_sync: 0.85, resonance: 0.75, trend_signal: 0.15 },
    auc_score: 0.72
};

export default function CariaMultiscaleFragilityIndex() {
    const [data, setData] = useState<MSFIData>(DEMO_DATA);
    const [loading, setLoading] = useState(true);
    const [showExplanation, setShowExplanation] = useState(false);
    const [explanationMode, setExplanationMode] = useState<'SIMPLE' | 'ADVANCED'>('SIMPLE');
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const fetchData = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/fragility/msfi`);
            if (response.ok) {
                const result = await response.json();
                setData(result);
            }
        } catch {
            setData(DEMO_DATA);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { fetchData(); }, [fetchData]);

    // --- EINSTEIN SPACETIME VISUALIZATION (Gravity Well) ---
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let animationFrameId: number;
        let time = 0;

        const render = () => {
            time += 0.05;
            const width = canvas.width;
            const height = canvas.height;
            const centerX = width / 2;
            const centerY = height / 2;

            // Clear with deep space bg
            ctx.fillStyle = '#020205'; // Deep void
            ctx.fillRect(0, 0, width, height);

            // Stars
            for (let i = 0; i < 50; i++) {
                const x = (Math.sin(i * 132 + time * 0.01) * width + width) % width;
                const y = (Math.cos(i * 523) * height + height) % height;
                ctx.fillStyle = `rgba(255, 255, 255, ${Math.random() * 0.5})`;
                ctx.fillRect(x, y, 1, 1);
            }

            // --- SPACETIME GRID ---
            // The grid distorts based on "Mass" (MSFI) and "Curvature" (Sync)
            const mass = data.metrics.msfi * 1.5; // Depth
            const curvature = data.metrics.clock_sync; // Tightness

            ctx.strokeStyle = `rgba(34, 211, 238, ${0.1 + curvature * 0.4})`; // Cyan glow increases with sync
            ctx.lineWidth = 1;

            const gridSize = 30;
            const horizonRadius = 30 + mass * 10;

            // Vertical Lines
            for (let x = 0; x <= width; x += gridSize) {
                ctx.beginPath();
                for (let y = 0; y <= height; y += 5) {
                    // Calculate distortion
                    const dx = x - centerX;
                    const dy = y - centerY;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    // Gravity Well Function
                    // Identify how close we are to the "Singularity"
                    const pull = Math.max(0, (200 - dist) / 200);
                    const warp = Math.pow(pull, 2) * mass * 20 * Math.sin(time + dist * 0.05);

                    // If inside event horizon, chaotic distortion
                    if (dist < horizonRadius && curvature > 0.8) {
                        ctx.lineTo(x + (Math.random() - 0.5) * 5, y);
                    } else {
                        // Pull towards center
                        const pullX = (dx / (dist + 1)) * warp;
                        const pullY = (dy / (dist + 1)) * warp;
                        ctx.lineTo(x - pullX, y - pullY);
                    }
                }
                ctx.stroke();
            }

            // Horizontal Lines
            for (let y = 0; y <= height; y += gridSize) {
                ctx.beginPath();
                for (let x = 0; x <= width; x += 5) {
                    const dx = x - centerX;
                    const dy = y - centerY;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    const pull = Math.max(0, (200 - dist) / 200);
                    const warp = Math.pow(pull, 2) * mass * 20 * Math.cos(time + dist * 0.05);

                    if (dist < horizonRadius && curvature > 0.8) {
                        ctx.lineTo(x, y + (Math.random() - 0.5) * 5);
                    } else {
                        const pullX = (dx / (dist + 1)) * warp;
                        const pullY = (dy / (dist + 1)) * warp;
                        ctx.lineTo(x - pullX, y - pullY);
                    }
                }
                ctx.stroke();
            }

            // --- THE SINGULARITY (Center) ---
            // Representing the "Underlying Fragility"
            const coreRadius = 10 + mass * 5;

            // Outer Glow (Resonance)
            const gradient = ctx.createRadialGradient(centerX, centerY, coreRadius * 0.5, centerX, centerY, coreRadius * 4);
            gradient.addColorStop(0, 'rgba(255, 255, 255, 0.8)');
            gradient.addColorStop(0.2, data.status === 'CRITICAL' ? 'rgba(239, 68, 68, 0.5)' : 'rgba(34, 211, 238, 0.3)');
            gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(centerX, centerY, coreRadius * 4, 0, Math.PI * 2);
            ctx.fill();

            // Black Hole (Event Horizon)
            ctx.fillStyle = '#000';
            ctx.beginPath();
            ctx.arc(centerX, centerY, coreRadius, 0, Math.PI * 2);
            ctx.fill();

            // Accretion Disk (Energy)
            ctx.strokeStyle = data.status === 'CRITICAL' ? '#ef4444' : '#fbbf24';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.ellipse(centerX, centerY, coreRadius * 2.5, coreRadius * 0.8, time, 0, Math.PI * 2);
            ctx.stroke();

            ctx.strokeStyle = data.status === 'CRITICAL' ? '#ef4444' : '#22d3ee';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.ellipse(centerX, centerY, coreRadius * 2.5, coreRadius * 0.8, -time * 0.8, 0, Math.PI * 2);
            ctx.stroke();

            animationFrameId = requestAnimationFrame(render);
        };

        render();
        return () => cancelAnimationFrame(animationFrameId);
    }, [data]);

    const getStatusColor = (status: string) => {
        if (status === 'CRITICAL') return '#ef4444';
        if (status === 'WARNING') return '#f59e0b';
        return '#10b981';
    };

    if (loading) return <div className="p-10 text-center font-mono text-cyan-500 animate-pulse">WARPING SPACETIME...</div>;

    return (
        <div className="relative w-full h-[450px] bg-[#020205] border border-gray-800 rounded-2xl overflow-hidden font-sans group">
            {/* --- HEADER --- */}
            <div className="absolute top-0 left-0 right-0 z-10 p-4 flex justify-between items-start bg-gradient-to-b from-black/80 to-transparent">
                <div>
                    <h2 className="text-2xl font-light tracking-widest text-white flex items-center gap-2">
                        CARIA <span className="text-[10px] bg-cyan-900/30 text-cyan-400 px-2 py-0.5 rounded border border-cyan-800/50">RELATIVITY CORE</span>
                    </h2>
                    <div className="flex items-center gap-2 mt-1">
                        <div className="text-xs text-gray-400 uppercase tracking-wider">Underlying Fragility Index</div>
                    </div>
                </div>
                <div className="flex flex-col items-end gap-1">
                    <div className="text-3xl font-thin text-white tabular-nums">
                        {data.metrics.msfi.toFixed(2)}
                    </div>
                    <div className={`text-[10px] font-bold px-2 py-0.5 rounded tracking-widest ${data.status === 'CRITICAL' ? 'bg-red-500 text-black' : 'bg-cyan-500 text-black'}`}>
                        {data.status} STATE
                    </div>
                </div>
            </div>

            {/* --- CANVAS --- */}
            <canvas ref={canvasRef} width={800} height={450} className="w-full h-full object-cover opacity-80" />

            {/* --- SMART STRATEGY HUD --- */}
            <div className="absolute bottom-6 left-6 z-20 flex gap-6">

                {/* 1. Sync Gauge (The Structure) */}
                <div className="bg-black/40 backdrop-blur-md p-3 rounded-xl border border-white/10">
                    <div className="text-[9px] text-gray-400 uppercase tracking-widest mb-1">Structure (Consensus)</div>
                    <div className="flex items-end gap-2">
                        <div className={`text-2xl font-light ${data.metrics.clock_sync > 0.8 ? 'text-red-400' : 'text-cyan-400'}`}>
                            {(data.metrics.clock_sync * 100).toFixed(0)}%
                        </div>
                        <div className="text-[9px] text-gray-500 mb-1">
                            {data.metrics.clock_sync > 0.8 ? 'FRAGILE' : 'STABLE'}
                        </div>
                    </div>
                    <div className="w-24 h-1 bg-gray-800 mt-1 rounded-full overflow-hidden">
                        <div
                            className={`h-full ${data.metrics.clock_sync > 0.8 ? 'bg-red-500 shadow-[0_0_10px_red]' : 'bg-cyan-500'}`}
                            style={{ width: `${data.metrics.clock_sync * 100}%` }}
                        />
                    </div>
                </div>

                {/* 2. Trend Gauge (The Trigger) */}
                <div className="bg-black/40 backdrop-blur-md p-3 rounded-xl border border-white/10">
                    <div className="text-[9px] text-gray-400 uppercase tracking-widest mb-1">Momentum (Trend)</div>
                    <div className="flex items-end gap-2">
                        <div className={`text-2xl font-light ${data.metrics.trend_signal > 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {data.metrics.trend_signal > 0 ? '↗' : '↘'}
                        </div>
                        <div className="text-[9px] text-gray-500 mb-1">
                            {data.metrics.trend_signal > 0 ? 'POSITIVE' : 'BREAKING'}
                        </div>
                    </div>
                    <div className="w-24 h-1 bg-gray-800 mt-1 rounded-full overflow-hidden flex">
                        <div className="w-1/2 h-full border-r border-black/50 relative">
                            {/* Negative Bar */}
                            {data.metrics.trend_signal < 0 && (
                                <div className="absolute right-0 h-full bg-red-500" style={{ width: `${Math.min(Math.abs(data.metrics.trend_signal) * 100, 100)}%` }} />
                            )}
                        </div>
                        <div className="w-1/2 h-full relative">
                            {/* Positive Bar */}
                            {data.metrics.trend_signal > 0 && (
                                <div className="absolute left-0 h-full bg-green-500" style={{ width: `${Math.min(data.metrics.trend_signal * 100, 100)}%` }} />
                            )}
                        </div>
                    </div>
                </div>

                {/* 3. The "Smart Action" Signal */}
                <div className={`p-3 rounded-xl border backdrop-blur-md flex flex-col justify-center min-w-[100px] ${data.metrics.clock_sync > 0.8 && data.metrics.trend_signal < 0
                        ? 'bg-red-500/20 border-red-500/50 text-red-100 shadow-[0_0_20px_rgba(239,68,68,0.3)]'
                        : data.metrics.clock_sync > 0.8
                            ? 'bg-yellow-500/20 border-yellow-500/50 text-yellow-100 shadow-[0_0_20px_rgba(234,179,8,0.3)]'
                            : 'bg-green-500/10 border-green-500/30 text-green-100'
                    }`}>
                    <div className="text-[9px] uppercase tracking-widest opacity-80 mb-1 text-center">Protocol Action</div>
                    <div className="text-xl font-bold text-center tracking-wider">
                        {data.metrics.clock_sync > 0.8 && data.metrics.trend_signal < 0
                            ? 'EXIT NOW'
                            : data.metrics.clock_sync > 0.8
                                ? 'RIDE BUBBLE'
                                : 'HOLD'}
                    </div>
                </div>

            </div>

            {/* --- EXPLAIN BUTTON --- */}
            <button
                onClick={() => setShowExplanation(true)}
                className="absolute bottom-6 right-6 z-10 p-3 rounded-full bg-white/5 border border-white/10 text-white/50 hover:bg-white/10 hover:text-white transition-all backdrop-blur-md group-hover:scale-110"
            >
                <HelpCircle className="w-5 h-5" />
            </button>

            {/* --- EXPLANATION OVERLAY --- */}
            {showExplanation && (
                <div className="absolute inset-0 z-50 bg-black/90 backdrop-blur-xl flex flex-col animate-in fade-in duration-300">
                    {/* Toolbar */}
                    <div className="p-6 border-b border-white/10 flex justify-between items-center">
                        <div className="flex gap-4">
                            <button
                                onClick={() => setExplanationMode('SIMPLE')}
                                className={`text-sm tracking-widest px-4 py-2 rounded-full transition-all ${explanationMode === 'SIMPLE' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50' : 'text-gray-500 hover:text-white'}`}
                            >
                                SIMPLIFIED
                            </button>
                            <button
                                onClick={() => setExplanationMode('ADVANCED')}
                                className={`text-sm tracking-widest px-4 py-2 rounded-full transition-all ${explanationMode === 'ADVANCED' ? 'bg-purple-500/20 text-purple-300 border border-purple-500/50' : 'text-gray-500 hover:text-white'}`}
                            >
                                ADVANCED
                            </button>
                        </div>
                        <button onClick={() => setShowExplanation(false)}><X className="w-6 h-6 text-gray-400 hover:text-white" /></button>
                    </div>

                    {/* Content */}
                    <div className="flex-1 p-8 overflow-y-auto">
                        <h3 className="text-3xl font-thin text-white mb-6">
                            {explanationMode === 'SIMPLE' ? 'The Traffic Jam Analogy' : 'Structural Dynamics & Entropy'}
                        </h3>

                        {explanationMode === 'SIMPLE' ? (
                            <div className="prose prose-invert prose-lg max-w-none">
                                <p className="text-gray-300 leading-relaxed">
                                    Imagine a highway with three lanes:
                                </p>
                                <ul className="list-disc pl-6 space-y-2 text-gray-400">
                                    <li><strong>Fast Lane (HFTs):</strong> Ferraris zooming at 200mph.</li>
                                    <li><strong>Middle Lane (Hedge Funds):</strong> Sedans going 60mph.</li>
                                    <li><strong>Slow Lane (Pensions):</strong> Trucks moving at 40mph.</li>
                                </ul>
                                <p className="text-gray-300 leading-relaxed mt-4">
                                    Usually, they move at their own speeds. This is a <strong>Healthy Market</strong>.
                                    <br /><br />
                                    But sometimes, everyone slams on the brakes at once. The Ferraris are stuck behind the Trucks.
                                    Everyone is moving at the same slow, dangerous speed.
                                    This is <strong className="text-cyan-400">Synchronization</strong>.
                                    <br /><br />
                                    CARIA measures this "Traffic Jam Effect". When we see the lanes locking up, we know a crash is coming—even if the cars explicitly haven't hit each other yet.
                                </p>
                            </div>
                        ) : (
                            <div className="prose prose-invert prose-lg max-w-none">
                                <p className="text-gray-300 leading-relaxed">
                                    We model the Global Market as a <strong>Complex Social System</strong> of interacting time horizons.
                                </p>
                                <div className="grid grid-cols-2 gap-8 my-6">
                                    <div className="bg-white/5 p-6 rounded-xl border border-white/10">
                                        <h4 className="text-purple-300 m-0 text-sm uppercase tracking-widest mb-2">The Metric</h4>
                                        <p className="text-sm text-gray-400">
                                            <p className="text-sm text-gray-400">
                                                We use the <strong>Kuramoto Order Parameter ($r$)</strong> to measure Consensus.
                                                <br />
                                                $$ r(t) = | \frac{1}{N} \sum e^{i\phi_k(t)} | $$
                                            </p>
                                            <div className="mt-4 p-3 bg-black/40 rounded border border-white/10">
                                                <h5 className="text-xs text-cyan-400 uppercase tracking-widest mb-2">The Algorithm</h5>
                                                <ul className="text-[10px] text-gray-400 space-y-1">
                                                    <li className="flex justify-between"><span>Condition 1:</span> <span className="text-white">Is Sync {'>'} 80%?</span></li>
                                                    <li className="flex justify-between"><span>Condition 2:</span> <span className="text-white">Is Trend Negative?</span></li>
                                                    <li className="flex justify-between border-t border-white/10 pt-1 mt-1 font-bold">
                                                        <span>Action:</span>
                                                        <span className={data.metrics.trend_signal < 0 && data.metrics.clock_sync > 0.8 ? "text-red-400" : "text-green-400"}>
                                                            {data.metrics.trend_signal < 0 && data.metrics.clock_sync > 0.8 ? "EXIT" : "HOLD"}
                                                        </span>
                                                    </li>
                                                </ul>
                                            </div>
                                    </div>
                                    <div className="bg-white/5 p-6 rounded-xl border border-white/10">
                                        <h4 className="text-yellow-300 m-0 text-sm uppercase tracking-widest mb-2">The Dynamics</h4>
                                        <p className="text-sm text-gray-400">
                                            As $r \to 1$, the <strong>Entropy</strong> of the system collapses. The diversity of opinion vanishes, creating a "Gravity Well" of Total Consensus.
                                            <br />
                                            The grid helps visualize this topological collapse. A deep well represents a market that is structurally fragile because everyone is doing the same thing.
                                        </p>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}

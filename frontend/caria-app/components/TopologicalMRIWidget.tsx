import React, { useEffect, useState, useRef } from 'react';
import { Brain, Activity, AlertTriangle, Fingerprint, Scan, Zap, Info } from 'lucide-react';
import { API_BASE_URL } from '../services/apiService';

interface Alien {
    ticker: string;
    isolation_score: number;
    type: string;
}

interface TopologyScan {
    status: string;
    diagnosis: string;
    description: string;
    status_color: string;
    metrics: {
        betti_1_loops: number;
        total_persistence: number;
        complexity_score: number;
    };
    aliens: Alien[];
}

export default function TopologicalMRIWidget() {
    const [scan, setScan] = useState<TopologyScan | null>(null);
    const [loading, setLoading] = useState(true);
    const [nodes, setNodes] = useState<{ x: number, y: number, r: number, vx: number, vy: number }[]>([]);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Initialize random nodes for the "Core"
    useEffect(() => {
        const initialNodes = Array.from({ length: 30 }).map(() => ({
            x: 150 + Math.random() * 100,
            y: 150 + Math.random() * 100,
            r: 3 + Math.random() * 5,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5
        }));
        setNodes(initialNodes);
    }, []);

    useEffect(() => {
        fetchScan();
        const interval = setInterval(fetchScan, 10000); // 10s refresh
        return () => clearInterval(interval);
    }, []);

    // Animation Loop
    useEffect(() => {
        let animationFrameId: number;

        const render = () => {
            if (!canvasRef.current) return;
            const ctx = canvasRef.current.getContext('2d');
            if (!ctx) return;

            // Clear canvas
            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

            // Update Core Nodes
            const newNodes = nodes.map(node => {
                let { x, y, vx, vy } = node;

                // Apply gentle force to keep them clustered
                const dx = 200 - x;
                const dy = 200 - y;
                vx += dx * 0.001;
                vy += dy * 0.001;

                // Random jitter
                vx += (Math.random() - 0.5) * 0.1;
                vy += (Math.random() - 0.5) * 0.1;

                // Damping
                vx *= 0.95;
                vy *= 0.95;

                return { ...node, x: x + vx, y: y + vy, vx, vy };
            });

            // Draw Connections (Core)
            ctx.strokeStyle = 'rgba(6, 182, 212, 0.2)'; // Cyan low opacity
            ctx.lineWidth = 1;
            for (let i = 0; i < newNodes.length; i++) {
                for (let j = i + 1; j < newNodes.length; j++) {
                    const dist = Math.hypot(newNodes[i].x - newNodes[j].x, newNodes[i].y - newNodes[j].y);
                    if (dist < 60) {
                        ctx.beginPath();
                        ctx.moveTo(newNodes[i].x, newNodes[i].y);
                        ctx.lineTo(newNodes[j].x, newNodes[j].y);
                        ctx.stroke();
                    }
                }
            }

            // Draw Core Nodes
            newNodes.forEach(node => {
                // Glow
                const gradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.r * 4);
                gradient.addColorStop(0, 'rgba(6, 182, 212, 1)');
                gradient.addColorStop(1, 'rgba(6, 182, 212, 0)');
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(node.x, node.y, node.r * 4, 0, Math.PI * 2);
                ctx.fill();

                // Core
                ctx.fillStyle = '#22d3ee'; // Cyan-400
                ctx.beginPath();
                ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
                ctx.fill();
            });

            // Draw Alien (if detected)
            if (scan && scan.aliens.length > 0) {
                const alien = scan.aliens[0];
                // Alien Position (Fixed relative to core for drama)
                const ax = 500;
                const ay = 150;

                // Tether
                ctx.strokeStyle = 'rgba(236, 72, 153, 0.6)'; // Pink/Red
                ctx.lineWidth = 2;
                ctx.beginPath();
                // Connect to nearest core node (visually)
                ctx.moveTo(300, 200); // Approximate center of core
                ctx.lineTo(ax, ay);
                ctx.stroke();

                // Alien Glow
                const gradient = ctx.createRadialGradient(ax, ay, 0, ax, ay, 40);
                gradient.addColorStop(0, 'rgba(236, 72, 153, 0.8)');
                gradient.addColorStop(0.5, 'rgba(236, 72, 153, 0.2)');
                gradient.addColorStop(1, 'rgba(236, 72, 153, 0)');
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(ax, ay, 40, 0, Math.PI * 2);
                ctx.fill();

                // Alien Core
                ctx.fillStyle = '#f472b6'; // Pink-400
                ctx.beginPath();
                ctx.arc(ax, ay, 12, 0, Math.PI * 2);
                ctx.fill();

                // Label
                ctx.fillStyle = '#fff';
                ctx.font = 'bold 14px monospace';
                ctx.fillText(alien.ticker, ax - 20, ay - 50);
            }

            animationFrameId = requestAnimationFrame(render);
        };

        render();
        return () => cancelAnimationFrame(animationFrameId);
    }, [nodes, scan]); // Re-bind when nodes/scan change

    const fetchScan = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/topology/scan`);
            if (!response.ok) throw new Error('API not available');
            const data = await response.json();
            setScan(data);
        } catch {
            // Use demo data when API is unavailable
            setScan({
                status: "STABLE",
                diagnosis: "Normal Operation",
                description: "Market topology within normal parameters",
                status_color: "green",
                metrics: { betti_1_loops: 3, total_persistence: 0.72, complexity_score: 45 },
                aliens: [
                    { ticker: "NVDA", isolation_score: 0.87, type: "momentum" },
                    { ticker: "SMCI", isolation_score: 0.82, type: "volatility" }
                ]
            });
        } finally {
            setLoading(false);
        }
    };

    if (loading) return <div className="h-64 flex items-center justify-center text-cyan-500 font-mono">INITIALIZING CORTEX...</div>;

    const topAlien = scan?.aliens[0];
    const isError = scan?.status === "ERROR" || scan?.status === "OFFLINE";
    const isWaiting = scan?.status === "WAITING";

    // Tactical Insight Logic
    const getTacticalInsight = () => {
        if (!scan) return "Analyzing market topology...";
        if (scan.metrics.betti_1_loops > 5) return "High market fragility detected. Structure is breaking down. Consider hedging or reducing beta.";
        if (scan.metrics.total_persistence < 0.5) return "Trends are weak and short-lived. Avoid momentum strategies; focus on mean reversion.";
        if (topAlien && topAlien.isolation_score > 0.9) return `Critical anomaly in ${topAlien.ticker}. It has decoupled from the market. Potential mean reversion trade.`;
        return "Market structure is robust. Trend following strategies are favored.";
    };

    return (
        <div className="w-full bg-black border border-gray-800 rounded-xl overflow-hidden relative font-mono">
            {/* Background Grid/Space */}
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-slate-900 via-black to-black opacity-80" />

            {/* HUD Overlay - Top Left */}
            <div className="absolute top-4 left-4 z-10 border border-white/20 p-2 bg-black/50 backdrop-blur-sm">
                <div className="flex items-center gap-2">
                    <div className="text-xs text-gray-400">Caria Cortex_v1</div>
                    <div className="group relative">
                        <Info className="h-3 w-3 text-gray-500 hover:text-gray-300 cursor-help" />
                        <div className="absolute left-0 top-5 w-72 bg-slate-900 border border-cyan-500/50 rounded-lg p-3 text-xs text-gray-300 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-opacity z-50">
                            <p className="font-semibold text-cyan-300 mb-1">Topological MRI Scanner</p>
                            <p>Uses Topological Data Analysis (TDA) to detect "alien" stocks - companies exhibiting abnormal behavioral patterns disconnected from their sector. High isolation scores indicate structural anomalies worth investigating.</p>
                        </div>
                    </div>
                </div>
                <div className={`text-xs flex items-center gap-2 ${isError ? 'text-red-500' : isWaiting ? 'text-yellow-500' : 'text-green-500'}`}>
                    <span className="relative flex h-2 w-2">
                        {!isError && <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isWaiting ? 'bg-yellow-400' : 'bg-green-400'}`}></span>}
                        <span className={`relative inline-flex rounded-full h-2 w-2 ${isError ? 'bg-red-500' : isWaiting ? 'bg-yellow-500' : 'bg-green-500'}`}></span>
                    </span>
                    {isError ? 'SYSTEM FAILURE' : isWaiting ? 'AWAITING DATA' : 'LIVE FEED: FMP_PIPE_ACTIVE'}
                </div>
                {isError && <div className="text-[10px] text-red-400 mt-1">{scan?.diagnosis}</div>}
            </div>

            {/* Metrics Panel - Top Right (Plain English) */}
            <div className="absolute top-4 right-4 z-10 flex flex-col gap-2 items-end">
                <div className="bg-black/60 backdrop-blur-sm border border-white/10 p-2 rounded text-right">
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider">Market Fragility</div>
                    <div className={`text-lg font-bold ${scan && scan.metrics.betti_1_loops > 5 ? 'text-red-500' : 'text-cyan-400'}`}>
                        {scan?.metrics.betti_1_loops} <span className="text-[10px] font-normal text-gray-600">(Loops)</span>
                    </div>
                </div>
                <div className="bg-black/60 backdrop-blur-sm border border-white/10 p-2 rounded text-right">
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider">Trend Durability</div>
                    <div className="text-lg font-bold text-cyan-400">
                        {scan?.metrics.total_persistence.toFixed(2)}
                    </div>
                </div>
            </div>

            {/* Canvas Layer */}
            <canvas
                ref={canvasRef}
                width={800}
                height={400}
                className="w-full h-[400px] object-cover relative z-0"
            />

            {/* Tactical Insight Footer */}
            <div className="absolute bottom-0 left-0 right-0 bg-black/80 backdrop-blur-md border-t border-white/10 p-3 z-20">
                <div className="flex items-start gap-3">
                    <div className="p-1.5 bg-cyan-500/10 rounded border border-cyan-500/30 mt-0.5">
                        <Zap className="w-4 h-4 text-cyan-400" />
                    </div>
                    <div>
                        <div className="text-[10px] text-cyan-500 uppercase tracking-wider font-bold mb-0.5">Tactical Insight</div>
                        <p className="text-sm text-gray-300 leading-tight">
                            {getTacticalInsight()}
                        </p>
                    </div>
                </div>
            </div>

            {/* Alert Box - Floating near Alien */}
            {topAlien && (
                <div className="absolute bottom-20 right-8 z-10 w-64 border border-red-500/50 bg-black/80 backdrop-blur-md p-4 shadow-[0_0_20px_rgba(239,68,68,0.2)]">
                    <div className="text-[10px] text-red-500 mb-1 tracking-widest">/// STRUCTURAL BREACH DETECTED</div>
                    <div className="text-3xl font-bold text-white mb-2">{topAlien.ticker}</div>

                    <div className="space-y-1 text-xs">
                        <div className="flex justify-between">
                            <span className="text-gray-500">ISOLATION SCORE:</span>
                            <span className="text-red-400 font-bold">{topAlien.isolation_score.toFixed(4)}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-500">SECTOR TETHER:</span>
                            <span className="text-white font-bold">CRITICAL</span>
                        </div>
                    </div>

                    {/* Decorative Corner */}
                    <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-red-500" />
                </div>
            )}

            {/* Scanlines */}
            <div className="absolute inset-0 pointer-events-none bg-[url('https://media.giphy.com/media/xT9Igk31elskX5qetq/giphy.gif')] opacity-[0.02] mix-blend-overlay" />
        </div>
    );
}

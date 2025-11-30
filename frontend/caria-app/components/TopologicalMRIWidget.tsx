import React, { useEffect, useState } from 'react';
import { Brain, Activity, AlertTriangle, Fingerprint } from 'lucide-react';

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

    useEffect(() => {
        fetchScan();
        const interval = setInterval(fetchScan, 30000); // 30s refresh
        return () => clearInterval(interval);
    }, []);

    const fetchScan = async () => {
        try {
            const response = await fetch('/api/topology/scan');
            const data = await response.json();
            setScan(data);
        } catch (error) {
            console.error('Error fetching topology scan:', error);
        } finally {
            setLoading(false);
        }
    };

    if (loading || !scan) {
        return (
            <div className="w-full h-full min-h-[200px] bg-black border border-cyan-900/50 rounded-xl p-4 flex flex-col items-center justify-center relative overflow-hidden">
                <div className="absolute inset-0 bg-[url('https://media.giphy.com/media/3o7TKs2tYdC4F6x8wE/giphy.gif')] opacity-10 bg-cover mix-blend-screen"></div>
                <Activity className="h-8 w-8 text-cyan-500 animate-pulse mb-2" />
                <div className="text-cyan-500 font-mono text-sm animate-pulse">INITIALIZING TDA SEQUENCE...</div>
            </div>
        );
    }

    const isCritical = scan.status_color === 'RED';
    const isWarning = scan.status_color === 'YELLOW';

    const borderColor = isCritical ? 'border-red-500/50' : isWarning ? 'border-yellow-500/50' : 'border-cyan-500/50';
    const glowColor = isCritical ? 'shadow-red-500/20' : isWarning ? 'shadow-yellow-500/20' : 'shadow-cyan-500/20';
    const textColor = isCritical ? 'text-red-400' : isWarning ? 'text-yellow-400' : 'text-cyan-400';

    return (
        <div className={`w-full bg-black ${borderColor} border shadow-lg ${glowColor} rounded-xl p-0 overflow-hidden relative`}>
            {/* Scan Line Animation */}
            <div className="absolute inset-0 pointer-events-none z-0">
                <div className="w-full h-[2px] bg-cyan-500/50 shadow-[0_0_10px_rgba(6,182,212,0.8)] animate-[scan_3s_ease-in-out_infinite]" />
            </div>

            {/* Header */}
            <div className="relative z-10 bg-slate-900/80 p-3 border-b border-white/10 flex items-center justify-between backdrop-blur-md">
                <div className="flex items-center gap-2">
                    <Brain className={`h-5 w-5 ${textColor}`} />
                    <span className="font-bold text-white tracking-wider">TOPOLOGICAL MRI</span>
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-[10px] text-gray-400 font-mono">GUDHI ENGINE</span>
                    <div className={`h-2 w-2 rounded-full ${isCritical ? 'bg-red-500 animate-ping' : 'bg-green-500'}`} />
                </div>
            </div>

            <div className="relative z-10 p-4 space-y-4">
                {/* Diagnosis Section */}
                <div className="flex items-start gap-4">
                    <div className="flex-1">
                        <div className={`text-lg font-bold ${textColor} mb-1`}>
                            {scan.diagnosis}
                        </div>
                        <p className="text-xs text-gray-400 leading-relaxed">
                            {scan.description}
                        </p>
                    </div>
                    {/* Betti Numbers Display */}
                    <div className="bg-slate-900/50 rounded-lg p-2 border border-white/5 text-center min-w-[80px]">
                        <div className="text-[10px] text-gray-500 uppercase tracking-widest">Betti-1</div>
                        <div className="text-2xl font-mono font-bold text-white">{scan.metrics.betti_1_loops}</div>
                        <div className="text-[9px] text-gray-600">LOOPS</div>
                    </div>
                </div>

                {/* Barcode Visualization (Abstract) */}
                <div className="space-y-1">
                    <div className="flex justify-between text-[10px] text-gray-500 font-mono">
                        <span>PERSISTENCE BARCODE</span>
                        <span>COMPLEXITY: {scan.metrics.complexity_score.toFixed(0)}%</span>
                    </div>
                    <div className="h-12 bg-slate-900/80 rounded border border-white/5 flex items-center px-1 gap-[2px] overflow-hidden">
                        {/* Generate fake bars based on complexity for visual effect */}
                        {Array.from({ length: 20 }).map((_, i) => (
                            <div
                                key={i}
                                className={`h-full w-1 rounded-full ${i < (scan.metrics.complexity_score / 5) ? textColor : 'bg-gray-800'}`}
                                style={{
                                    height: `${30 + Math.random() * 70}%`,
                                    opacity: 0.5 + Math.random() * 0.5
                                }}
                            />
                        ))}
                    </div>
                </div>

                {/* Topological Aliens */}
                <div>
                    <div className="flex items-center gap-2 mb-2">
                        <Fingerprint className="h-4 w-4 text-purple-400" />
                        <span className="text-xs font-bold text-purple-300 tracking-wider">TOPOLOGICAL ALIENS DETECTED</span>
                    </div>
                    <div className="space-y-2">
                        {scan.aliens.map((alien, idx) => (
                            <div key={idx} className="bg-purple-900/10 border border-purple-500/20 rounded p-2 flex items-center justify-between hover:bg-purple-900/20 transition-colors">
                                <div className="flex items-center gap-2">
                                    <span className="font-mono font-bold text-purple-200">{alien.ticker}</span>
                                    <span className="text-[10px] bg-purple-500/20 text-purple-300 px-1 rounded">
                                        {alien.type}
                                    </span>
                                </div>
                                <div className="text-xs font-mono text-purple-400">
                                    ISO: {alien.isolation_score.toFixed(2)}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

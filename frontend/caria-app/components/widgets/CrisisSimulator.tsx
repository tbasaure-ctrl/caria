import React, { useState, useEffect } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ComposedChart, Line } from 'recharts';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';
import { AlertTriangle, Activity, Calendar, TrendingDown, Clock, ShieldAlert, ChevronDown, MousePointer2 } from 'lucide-react';

const CRISES = [
    { id: '1987_black_monday', name: 'Black Monday', year: '1987', type: 'Acute' },
    { id: '1929_depression', name: '1929 Great Depression', year: '1929', type: 'Recession' },
    { id: '1939_wwii', name: 'WWII', year: '1939', type: 'Geopolitical' },
    { id: '1962_cuban_missile', name: 'Cuban Missile Crisis', year: '1962', type: 'Geopolitical' },
    { id: '1963_jfk', name: 'JFK Assassination', year: '1963', type: 'Shock' },
    { id: '2001_911', name: '9/11', year: '2001', type: 'Shock' },
    { id: '2008_gfc', name: '2008 GFC', year: '2008', type: 'Recession' },
    { id: '2020_covid', name: 'COVID-19 Crash', year: '2020', type: 'Pandemic' },
];

interface SimulationResult {
    crisis_name: string;
    dates: string[];
    portfolio_values: number[];
    benchmark_values: number[];
    metrics: {
        max_drawdown: number;
        total_return: number;
        benchmark_return: number;
        recovery_time: string | number;
        volatility: string;
    };
}

export const CrisisSimulator: React.FC = () => {
    const [selectedCrisis, setSelectedCrisis] = useState(CRISES[0].id); // Default to Black Monday
    const [result, setResult] = useState<SimulationResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleSimulate = async (crisisId: string) => {
        setSelectedCrisis(crisisId);
        setLoading(true);
        setError(null);
        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/simulation/crisis`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ portfolio: [], crisis_id: crisisId }),
            });

            if (!response.ok) throw new Error('Simulation failed');

            const data = await response.json();
            setResult(data);
        } catch (err: any) {
            setError(err.message || 'Failed to run simulation');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        handleSimulate(selectedCrisis);
    }, []);

    const chartData = result ? result.dates.map((date, i) => ({
        date,
        portfolio: result.portfolio_values[i],
        benchmark: result.benchmark_values[i],
    })) : [];

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (active && payload && payload.length) {
            return (
                <div className="bg-[#0B1221]/95 backdrop-blur-xl border border-accent-cyan/40 p-3 rounded-lg shadow-2xl flex flex-col gap-1">
                    <p className="text-[10px] text-text-muted font-bold uppercase tracking-wider mb-1">{label}</p>
                    {payload.map((entry: any, index: number) => (
                        <div key={index} className="flex items-center justify-between gap-4">
                            <span className="flex items-center gap-1.5">
                                <div className={`w-1.5 h-1.5 rounded-full ${entry.dataKey === 'portfolio' ? 'bg-accent-cyan' : 'bg-gray-500'}`} />
                                <span className="text-[10px] text-text-secondary uppercase">{entry.name}:</span>
                            </span>
                            <span className={`text-[11px] font-mono font-bold ${entry.value >= 0 ? 'text-positive' : 'text-negative'}`}>
                                {entry.value >= 0 ? '+' : ''}{entry.value.toFixed(1)}%
                            </span>
                        </div>
                    ))}
                </div>
            );
        }
        return null;
    };

    return (
        <div className="w-full h-full bg-[#0A0C14] border border-white/5 rounded-3xl overflow-hidden flex flex-col shadow-[0_20px_50px_rgba(0,0,0,0.5)] font-sans">
            {/* Header Area */}
            <div className="p-8 pb-4 flex justify-between items-start">
                <div className="space-y-1">
                    <h2 className="text-3xl font-black text-white tracking-tight">
                        CRISIS SIMULATOR
                    </h2>
                    <p className="text-xs text-white/40 font-medium">Simulate historical market events on your current portfolio.</p>
                </div>
                
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-6 text-[10px] uppercase tracking-widest font-black">
                        <div className="flex items-center gap-2 text-accent-cyan">
                            <span className="w-2 h-2 rounded-full bg-accent-cyan shadow-[0_0_8px_rgba(34,211,238,0.8)]"></span>
                            Your Portfolio
                        </div>
                        <div className="flex items-center gap-2 text-white/30">
                            <span className="w-2 h-2 rounded-full bg-white/20"></span>
                            S&P 500
                        </div>
                    </div>
                    <div className="bg-white/5 border border-green-500/40 rounded-xl px-4 py-2 text-xs font-bold text-white flex items-center gap-3 cursor-pointer hover:bg-white/10 transition-all">
                        {CRISES.find(c => c.id === selectedCrisis)?.name.toUpperCase()}
                        <ChevronDown className="w-4 h-4 text-green-500" />
                    </div>
                </div>
            </div>

            <div className="flex flex-1 min-h-0">
                {/* Chart Visualization */}
                <div className="flex-1 p-8 pt-2 flex flex-col">
                    <div className="flex-1 w-full min-h-[350px]">
                        {loading ? (
                            <div className="w-full h-full flex flex-col items-center justify-center gap-4">
                                <div className="w-16 h-16 border-4 border-accent-cyan/20 border-t-accent-cyan rounded-full animate-spin"></div>
                                <span className="text-[10px] text-accent-cyan animate-pulse uppercase tracking-[0.2em] font-black">Processing Data Points...</span>
                            </div>
                        ) : (
                            <ResponsiveContainer width="100%" height="100%">
                                <ComposedChart data={chartData} margin={{ top: 20, right: 20, left: 0, bottom: 0 }}>
                                    <defs>
                                        <linearGradient id="colorPortfolio" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#22D3EE" stopOpacity={0.2}/>
                                            <stop offset="95%" stopColor="#22D3EE" stopOpacity={0}/>
                                        </linearGradient>
                                        <filter id="glow">
                                            <feGaussianBlur stdDeviation="3" result="blur" />
                                            <feComposite in="SourceGraphic" in2="blur" operator="over" />
                                        </filter>
                                    </defs>
                                    <CartesianGrid strokeDasharray="0" stroke="rgba(255,255,255,0.03)" vertical={true} horizontal={true} />
                                    <XAxis 
                                        dataKey="date" 
                                        axisLine={false} 
                                        tickLine={false} 
                                        tick={{fill: 'rgba(255,255,255,0.2)', fontSize: 10, fontWeight: 600}}
                                        dy={15}
                                        minTickGap={60}
                                    />
                                    <YAxis 
                                        axisLine={false} 
                                        tickLine={false} 
                                        tick={{fill: 'rgba(255,255,255,0.2)', fontSize: 10, fontWeight: 600}}
                                        tickFormatter={(val) => `${val > 0 ? '+' : ''}${val}%`}
                                        dx={-10}
                                    />
                                    <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 1 }} />
                                    
                                    <Area 
                                        name="S&P 500"
                                        type="monotone" 
                                        dataKey="benchmark" 
                                        stroke="rgba(255,255,255,0.2)" 
                                        strokeWidth={2}
                                        fill="transparent"
                                        dot={false}
                                        activeDot={{ r: 4, fill: '#64748B' }}
                                    />
                                    
                                    <Area 
                                        name="Your Portfolio"
                                        type="monotone" 
                                        dataKey="portfolio" 
                                        stroke="#22D3EE" 
                                        strokeWidth={3}
                                        fill="url(#colorPortfolio)"
                                        dot={false}
                                        activeDot={{ r: 6, fill: '#22D3EE', stroke: '#fff', strokeWidth: 2 }}
                                        style={{ filter: 'url(#glow)' }}
                                    />
                                </ComposedChart>
                            </ResponsiveContainer>
                        )}
                    </div>
                </div>

                {/* Sidebar Navigation */}
                <div className="w-72 border-l border-white/5 bg-black/20 flex flex-col p-6 gap-3">
                    <h3 className="text-[10px] text-white/30 font-black uppercase tracking-[0.2em] mb-3">Select Crisis</h3>
                    <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 space-y-2">
                        {CRISES.map((c) => (
                            <button
                                key={c.id}
                                onClick={() => handleSimulate(c.id)}
                                className={`w-full text-left px-4 py-3 rounded-2xl border transition-all duration-300 group ${
                                    selectedCrisis === c.id 
                                    ? 'bg-white/5 border-green-500/50 shadow-[0_0_20px_rgba(34,197,94,0.1)]' 
                                    : 'bg-transparent border-white/5 hover:border-white/20'
                                }`}
                            >
                                <div className="flex flex-col gap-0.5">
                                    <span className={`text-[11px] font-black leading-tight ${selectedCrisis === c.id ? 'text-white' : 'text-white/60 group-hover:text-white'}`}>
                                        {c.name.toUpperCase()}
                                    </span>
                                    <span className="text-[9px] text-white/20 font-bold group-hover:text-white/40">{c.year}</span>
                                </div>
                            </button>
                        ))}
                    </div>
                    
                    <button className="w-full mt-4 py-3 bg-transparent border border-accent-cyan/30 rounded-2xl text-[10px] font-black text-accent-cyan uppercase tracking-widest hover:bg-accent-cyan hover:text-black transition-all duration-500 shadow-[0_0_15px_rgba(34,211,238,0.1)]">
                        Custom Simulation
                    </button>
                </div>
            </div>

            {/* Metrics Panel Footer */}
            <div className="p-8 pt-0 mt-2">
                <div className="bg-[#11141D] border border-white/5 rounded-[2rem] p-8 shadow-inner relative overflow-hidden">
                    <div className="relative z-10">
                        <h3 className="text-[10px] text-white/30 font-black uppercase tracking-[0.2em] mb-6">Metrics Panel</h3>
                        <div className="grid grid-cols-3 gap-12">
                            <div className="space-y-2">
                                <div className="text-[10px] text-white/40 font-bold uppercase tracking-widest flex items-center gap-2">
                                    Max Drawdown <TrendingDown className="w-3 h-3 text-red-500" />
                                </div>
                                <div className="text-5xl font-black text-[#FF4D4D] tracking-tighter tabular-nums flex items-baseline gap-1">
                                    {result ? (result.metrics.max_drawdown * 100).toFixed(1) : '--'}
                                    <span className="text-2xl font-bold opacity-60">%</span>
                                </div>
                            </div>
                            
                            <div className="space-y-2 border-l border-white/5 pl-12">
                                <div className="text-[10px] text-white/40 font-bold uppercase tracking-widest flex items-center gap-2">
                                    Recovery Time <Clock className="w-3 h-3 text-accent-cyan" />
                                </div>
                                <div className="text-5xl font-black text-white tracking-tighter tabular-nums flex items-baseline gap-2">
                                    {result?.metrics.recovery_time || '--'} 
                                    <span className="text-xl font-black text-white/30 uppercase tracking-widest">Months</span>
                                </div>
                            </div>

                            <div className="space-y-2 border-l border-white/5 pl-12">
                                <div className="text-[10px] text-white/40 font-bold uppercase tracking-widest flex items-center gap-2">
                                    Volatility <Activity className="w-3 h-3 text-green-400" />
                                </div>
                                <div className="text-5xl font-black text-[#4DFFB4] tracking-tight drop-shadow-[0_0_15px_rgba(77,255,180,0.4)]">
                                    {result?.metrics.volatility || 'MEDIUM'}
                                </div>
                            </div>
                        </div>
                        <p className="text-[10px] text-white/20 mt-8 font-medium italic">
                            High volatility indicates significant price fluctuations compared to the market.
                        </p>
                    </div>
                    
                    {/* Inner decorative glow */}
                    <div className="absolute -bottom-24 -right-24 w-64 h-64 bg-accent-cyan/5 blur-[100px] rounded-full"></div>
                </div>
            </div>

            {/* Global background effects */}
            <div className="absolute inset-0 pointer-events-none overflow-hidden rounded-3xl opacity-20">
                <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-accent-cyan/10 blur-[150px] -translate-y-1/2 translate-x-1/2 rounded-full"></div>
                <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-purple-500/5 blur-[150px] translate-y-1/2 -translate-x-1/2 rounded-full"></div>
            </div>
        </div>
    );
};

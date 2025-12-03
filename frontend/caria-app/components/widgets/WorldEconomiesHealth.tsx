import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { WidgetCard } from './WidgetCard';
import { fetchHeatmap } from '../../services/apiService';

// Mock data for fallback
const MOCK_WORLD_DATA = [
    { code: 'USA', name: 'United States', score: 75, status: 'Stable/Slow', gdp: 2.1, inflation: 3.2, debt: 122 },
    { code: 'CHN', name: 'China', score: 60, status: 'Slowdown', gdp: 4.5, inflation: 0.7, debt: 83 },
    { code: 'DEU', name: 'Germany', score: 45, status: 'Stagflation/Risk', gdp: -0.1, inflation: 2.9, debt: 66 },
    { code: 'JPN', name: 'Japan', score: 55, status: 'Stable/Slow', gdp: 1.0, inflation: 2.2, debt: 260 },
    { code: 'IND', name: 'India', score: 85, status: 'Healthy Expansion', gdp: 6.5, inflation: 5.1, debt: 81 },
    { code: 'GBR', name: 'United Kingdom', score: 50, status: 'Stagflation/Risk', gdp: 0.5, inflation: 3.4, debt: 101 },
    { code: 'FRA', name: 'France', score: 52, status: 'Stagflation/Risk', gdp: 0.7, inflation: 2.8, debt: 110 },
    { code: 'BRA', name: 'Brazil', score: 70, status: 'Overheating', gdp: 2.9, inflation: 4.5, debt: 74 },
    { code: 'CAN', name: 'Canada', score: 65, status: 'Stable/Slow', gdp: 1.1, inflation: 2.9, debt: 106 },
    { code: 'RUS', name: 'Russia', score: 40, status: 'Stagflation/Risk', gdp: 2.6, inflation: 7.4, debt: 18 },
    { code: 'ITA', name: 'Italy', score: 48, status: 'Stagflation/Risk', gdp: 0.7, inflation: 1.3, debt: 140 },
    { code: 'AUS', name: 'Australia', score: 68, status: 'Stable/Slow', gdp: 1.5, inflation: 3.4, debt: 36 },
    { code: 'KOR', name: 'South Korea', score: 62, status: 'Stable/Slow', gdp: 2.2, inflation: 2.8, debt: 54 },
    { code: 'ESP', name: 'Spain', score: 55, status: 'Stable/Slow', gdp: 1.7, inflation: 2.8, debt: 109 },
    { code: 'MEX', name: 'Mexico', score: 60, status: 'Stable/Slow', gdp: 2.3, inflation: 4.4, debt: 49 },
    { code: 'IDN', name: 'Indonesia', score: 80, status: 'Healthy Expansion', gdp: 5.0, inflation: 2.6, debt: 39 },
    { code: 'TUR', name: 'Turkey', score: 65, status: 'Stable/Slow', gdp: 4.0, inflation: 67.0, debt: 32 },
    { code: 'SAU', name: 'Saudi Arabia', score: 72, status: 'Overheating', gdp: 2.7, inflation: 1.6, debt: 24 },
    { code: 'ZAF', name: 'South Africa', score: 45, status: 'Stagflation/Risk', gdp: 0.6, inflation: 5.3, debt: 71 },
    { code: 'ARG', name: 'Argentina', score: 30, status: 'Stagflation/Risk', gdp: -2.8, inflation: 254.2, debt: 85 },
];

const STATUS_COLORS = {
    'Healthy Expansion': '#10b981', // Green
    'Stable/Slow': '#3b82f6',       // Blue
    'Overheating': '#f59e0b',       // Amber
    'Stagflation/Risk': '#ef4444',  // Red
    'Slowdown': '#6366f1'           // Indigo
};

interface CountryData {
    code: string;
    name: string;
    score: number;
    status: string;
    gdp?: number;
    inflation?: number;
    debt?: number;
}

type InfoTab = 'guide' | 'details' | 'metrics';

export const WorldEconomiesHealth: React.FC<{ id?: string }> = ({ id }) => {
    const [data, setData] = useState<CountryData[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedCountry, setSelectedCountry] = useState<CountryData | null>(null);
    const [activeTab, setActiveTab] = useState<InfoTab>('guide');

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        setLoading(true);
        try {
            const response = await fetchHeatmap();
            
            if (response.cells && response.cells.length > 0) {
                const countryScores: Record<string, { total: number, count: number, name: string }> = {};
                
                response.cells.forEach(cell => {
                    if (!countryScores[cell.country_code]) {
                        countryScores[cell.country_code] = { total: 0, count: 0, name: cell.country_name };
                    }
                    countryScores[cell.country_code].total += cell.normalized_value * 100;
                    countryScores[cell.country_code].count += 1;
                });

                const processedData = Object.entries(countryScores).map(([code, stats]) => {
                    const avgScore = stats.total / stats.count;
                    let status = 'Stable/Slow';
                    if (avgScore > 75) status = 'Healthy Expansion';
                    else if (avgScore > 60) status = 'Overheating';
                    else if (avgScore < 40) status = 'Stagflation/Risk';
                    
                    const isoMap: Record<string, string> = {
                        'US': 'USA', 'CN': 'CHN', 'JP': 'JPN', 'DE': 'DEU', 'IN': 'IND',
                        'GB': 'GBR', 'FR': 'FRA', 'BR': 'BRA', 'IT': 'ITA', 'CA': 'CAN',
                        'KR': 'KOR', 'RU': 'RUS', 'AU': 'AUS', 'ES': 'ESP', 'MX': 'MEX',
                        'ID': 'IDN', 'TR': 'TUR', 'SA': 'SAU', 'ZA': 'ZAF', 'AR': 'ARG'
                    };
                    
                    const mockMatch = MOCK_WORLD_DATA.find(m => m.code === (isoMap[code] || code) || m.name === stats.name);
                    
                    return {
                        code: isoMap[code] || code,
                        name: stats.name,
                        score: avgScore,
                        status,
                        gdp: mockMatch?.gdp,
                        inflation: mockMatch?.inflation,
                        debt: mockMatch?.debt
                    };
                });
                
                setData(processedData);
            } else {
                setData(MOCK_WORLD_DATA);
            }
        } catch (err) {
            console.error("Using mock data for World Map due to API error:", err);
            setData(MOCK_WORLD_DATA);
        } finally {
            setLoading(false);
        }
    };

    const locations = data.map(d => d.code);
    const z = data.map(d => d.score);
    const text = data.map(d => `${d.name}<br>Status: ${d.status}<br>Health Score: ${d.score.toFixed(1)}`);
    
    const colorscale = [
        [0, '#ef4444'],   // Red
        [0.4, '#f59e0b'], // Amber
        [0.6, '#3b82f6'], // Blue
        [1, '#10b981']    // Green
    ];

    useEffect(() => {
        if (selectedCountry) {
            setActiveTab('details');
        }
    }, [selectedCountry]);

    return (
        <WidgetCard
            id={id}
            title="WORLD ECONOMIES HEALTH MONITOR"
            tooltip="3D Mission Control: Global economic health visualization."
            className="min-h-[600px]"
        >
            <div className="flex flex-col xl:flex-row gap-6 h-full">
                {/* Left: 3D Globe (Main Visual) - Orbital Theme */}
                <div className="flex-1 relative h-[500px] xl:h-[600px] rounded-xl overflow-hidden bg-black border border-white/10 shadow-[inset_0_0_100px_rgba(0,0,0,1)] group">
                    {/* Deep Space Background */}
                    <div className="absolute inset-0 z-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-[#0B1221] via-[#020408] to-black"></div>
                    
                    {/* Star Field */}
                    <div className="absolute inset-0 z-0 opacity-60" 
                         style={{ 
                             backgroundImage: 'radial-gradient(white 1px, transparent 1px)', 
                             backgroundSize: '50px 50px' 
                         }}>
                    </div>

                    {loading ? (
                        <div className="absolute inset-0 flex items-center justify-center z-20">
                            <div className="flex flex-col items-center gap-4">
                                <div className="w-12 h-12 border-4 border-accent-cyan border-t-transparent rounded-full animate-spin shadow-[0_0_20px_rgba(34,211,238,0.5)]"></div>
                                <div className="text-accent-cyan font-mono text-sm tracking-[0.2em] animate-pulse">INITIALIZING ORBITAL FEED...</div>
                            </div>
                        </div>
                    ) : (
                        <Plot
                            data={[
                                {
                                    type: 'choropleth',
                                    locationmode: 'ISO-3',
                                    locations: locations,
                                    z: z,
                                    text: text,
                                    colorscale: colorscale,
                                    autocolorscale: false,
                                    reversescale: false,
                                    marker: {
                                        line: {
                                            color: '#000000',
                                            width: 0.5
                                        }
                                    },
                                    colorbar: {
                                        title: { text: 'HEALTH', font: { color: '#94a3b8', size: 10, family: 'monospace' } },
                                        thickness: 8,
                                        len: 0.4,
                                        x: 0.95,
                                        y: 0.5,
                                        tickfont: { color: '#94a3b8', size: 10, family: 'monospace' },
                                        bgcolor: 'rgba(0,0,0,0.8)',
                                        outlinecolor: 'rgba(255,255,255,0.1)'
                                    },
                                    hoverinfo: 'text',
                                    hovertemplate: '<b>%{text}</b><extra></extra>'
                                }
                            ]}
                            layout={{
                                geo: {
                                    showframe: false,
                                    showcoastlines: false,
                                    projection: {
                                        type: 'orthographic'
                                    },
                                    bgcolor: 'rgba(0,0,0,0)',
                                    showland: true,
                                    landcolor: '#050505',
                                    showocean: true,
                                    oceancolor: '#000000',
                                    showlakes: true,
                                    lakecolor: '#000000',
                                    lonaxis: { showgrid: true, gridcolor: '#1e293b', gridwidth: 0.5 },
                                    lataxis: { showgrid: true, gridcolor: '#1e293b', gridwidth: 0.5 },
                                },
                                paper_bgcolor: 'rgba(0,0,0,0)',
                                plot_bgcolor: 'rgba(0,0,0,0)',
                                margin: { l: 0, r: 0, t: 0, b: 0 },
                                showlegend: false,
                                hoverlabel: {
                                    bgcolor: '#0f172a',
                                    bordercolor: '#334155',
                                    font: { color: '#ffffff', family: 'monospace' }
                                }
                            }}
                            style={{ width: '100%', height: '100%' }}
                            config={{ displayModeBar: false, scrollZoom: true }}
                            onClick={(data) => {
                                const point = data.points[0];
                                // @ts-ignore
                                const country = MOCK_WORLD_DATA.find(c => c.code === point.location) || 
                                                // @ts-ignore
                                                data.find(c => c.code === point.location);
                                if (country) setSelectedCountry(country as CountryData);
                            }}
                        />
                    )}

                    {/* Orbital HUD Overlay */}
                    <div className="absolute top-6 left-6 z-10 pointer-events-none">
                        <div className="backdrop-blur-md bg-black/60 border border-white/10 rounded-lg p-4 shadow-[0_0_30px_rgba(0,0,0,0.8)] border-l-4 border-l-accent-cyan">
                            <div className="flex items-center gap-3 mb-2">
                                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse shadow-[0_0_10px_rgba(34,197,94,0.8)]"></div>
                                <span className="text-white text-xs font-display uppercase tracking-widest">Global Monitor Online</span>
                            </div>
                            <div className="text-[10px] text-text-muted font-mono space-y-1">
                                <div className="flex justify-between gap-4"><span>SYSTEM:</span> <span className="text-accent-cyan">NOMINAL</span></div>
                                <div className="flex justify-between gap-4"><span>COVERAGE:</span> <span className="text-white">92% GDP</span></div>
                            </div>
                        </div>
                    </div>

                    {/* Legend Overlay - Orbital Style */}
                    <div className="absolute bottom-6 left-6 z-10 pointer-events-none">
                        <div className="backdrop-blur-md bg-black/60 border border-white/10 rounded-lg p-4 shadow-[0_0_30px_rgba(0,0,0,0.8)]">
                            <h4 className="text-text-muted text-[10px] font-mono uppercase tracking-widest mb-3 border-b border-white/10 pb-2">Regime Classification</h4>
                            <div className="space-y-2">
                                {Object.entries(STATUS_COLORS).map(([status, color]) => (
                                    <div key={status} className="flex items-center gap-3">
                                        <div className="w-2 h-2 rounded-full shadow-[0_0_8px_currentColor]" style={{ backgroundColor: color, color: color }}></div>
                                        <span className="text-[10px] text-white font-mono uppercase opacity-80">{status}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right: Detailed Analysis Panel (Sidebar) */}
                <div className="xl:w-96 flex flex-col gap-4 h-[600px]">
                    {/* Tab Navigation */}
                    <div className="flex bg-[#0B1221] border border-white/10 rounded-lg p-1">
                        <button
                            onClick={() => setActiveTab('guide')}
                            className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${activeTab === 'guide' ? 'bg-white/10 text-white shadow-glow-sm' : 'text-text-muted hover:text-white'}`}
                        >
                            Manual
                        </button>
                        <button
                            onClick={() => setActiveTab('details')}
                            className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${activeTab === 'details' ? 'bg-white/10 text-white shadow-glow-sm' : 'text-text-muted hover:text-white'}`}
                        >
                            Analysis
                        </button>
                        <button
                            onClick={() => setActiveTab('metrics')}
                            className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${activeTab === 'metrics' ? 'bg-white/10 text-white shadow-glow-sm' : 'text-text-muted hover:text-white'}`}
                        >
                            Metrics
                        </button>
                    </div>

                    {/* Panel Content */}
                    <div className="bg-[#0B1221] border border-white/10 rounded-lg flex-1 overflow-y-auto custom-scrollbar relative shadow-[0_0_30px_rgba(0,0,0,0.3)]">
                        
                        {/* TAB: GUIDE (Manual) */}
                        {activeTab === 'guide' && (
                            <div className="p-6 space-y-6 animate-fade-in">
                                <div>
                                    <h3 className="text-lg font-display text-white mb-2">System Interpretation</h3>
                                    <div className="h-0.5 w-12 bg-accent-cyan mb-4"></div>
                                    <p className="text-xs text-text-secondary leading-relaxed">
                                        The World Economies Health Monitor aggregates over 50 macroeconomic data points into a single <span className="text-white font-bold">Health Score</span>. This score functions as a leading indicator for market performance and economic stability.
                                    </p>
                                </div>

                                <div className="space-y-4">
                                    <h4 className="text-xs font-bold text-white uppercase tracking-wider border-b border-white/10 pb-2">The Economic Compass</h4>
                                    
                                    <div className="space-y-3">
                                        <div className="p-3 bg-white/5 rounded border border-white/5">
                                            <div className="flex items-center gap-2 mb-1">
                                                <span className="w-2 h-2 rounded-full bg-green-500"></span>
                                                <span className="text-xs font-bold text-white">Expansion (Score 75-100)</span>
                                            </div>
                                            <p className="text-[10px] text-text-muted">
                                                Robust growth, controlled inflation. Ideal environment for equities and risk assets.
                                            </p>
                                        </div>

                                        <div className="p-3 bg-white/5 rounded border border-white/5">
                                            <div className="flex items-center gap-2 mb-1">
                                                <span className="w-2 h-2 rounded-full bg-amber-500"></span>
                                                <span className="text-xs font-bold text-white">Overheating (Score 60-75)</span>
                                            </div>
                                            <p className="text-[10px] text-text-muted">
                                                Growth exceeding potential, rising inflation risks. Central banks likely to tighten.
                                            </p>
                                        </div>

                                        <div className="p-3 bg-white/5 rounded border border-white/5">
                                            <div className="flex items-center gap-2 mb-1">
                                                <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                                                <span className="text-xs font-bold text-white">Stable/Slow (Score 40-60)</span>
                                            </div>
                                            <p className="text-[10px] text-text-muted">
                                                Trend growth or slight deceleration. Neutral policy environment.
                                            </p>
                                        </div>

                                        <div className="p-3 bg-white/5 rounded border border-white/5">
                                            <div className="flex items-center gap-2 mb-1">
                                                <span className="w-2 h-2 rounded-full bg-red-500"></span>
                                                <span className="text-xs font-bold text-white">Stagflation/Risk (Score &lt; 40)</span>
                                            </div>
                                            <p className="text-[10px] text-text-muted">
                                                Low growth combined with high inflation or structural weakness. Defensive positioning advised.
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* TAB: DETAILS (Analysis) */}
                        {activeTab === 'details' && (
                            <div className="p-6 space-y-6 animate-fade-in">
                                {selectedCountry ? (
                                    <>
                                        <div className="flex justify-between items-start">
                                            <div>
                                                <div className="text-[10px] font-mono text-accent-cyan mb-1">TARGET LOCKED</div>
                                                <h2 className="text-3xl font-display text-white">{selectedCountry.name}</h2>
                                                <div className="inline-block px-2 py-0.5 rounded bg-white/10 text-[10px] font-mono text-text-muted mt-2">
                                                    ISO: {selectedCountry.code}
                                                </div>
                                            </div>
                                            <div className="text-right">
                                                <div className="text-4xl font-mono text-white font-bold tracking-tighter">{selectedCountry.score}</div>
                                                <div className="text-[10px] text-text-muted uppercase tracking-widest">Health Index</div>
                                            </div>
                                        </div>

                                        <div className="p-4 bg-white/5 rounded border border-white/10">
                                            <div className="text-xs text-text-muted uppercase tracking-widest mb-2">Regime Status</div>
                                            <div className="text-lg font-medium text-white flex items-center gap-2">
                                                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: STATUS_COLORS[selectedCountry.status as keyof typeof STATUS_COLORS] || '#fff' }}></span>
                                                {selectedCountry.status}
                                            </div>
                                            <div className="mt-3 h-1 w-full bg-white/10 rounded-full overflow-hidden">
                                                <div 
                                                    className="h-full transition-all duration-500" 
                                                    style={{ 
                                                        width: `${selectedCountry.score}%`,
                                                        backgroundColor: STATUS_COLORS[selectedCountry.status as keyof typeof STATUS_COLORS] || '#fff'
                                                    }}
                                                ></div>
                                            </div>
                                        </div>

                                        <div className="space-y-3">
                                            <h4 className="text-xs font-bold text-white uppercase tracking-wider border-b border-white/10 pb-2">Economic Vector</h4>
                                            <p className="text-xs text-text-secondary leading-relaxed">
                                                {selectedCountry.score > 60 
                                                    ? "Economy is showing signs of robust expansion. Leading indicators suggest continued momentum, though inflation risks should be monitored."
                                                    : "Economy is facing structural headwinds. Leading indicators point to deceleration or below-trend growth. Caution is warranted."
                                                }
                                            </p>
                                        </div>
                                    </>
                                ) : (
                                    <div className="h-full flex flex-col items-center justify-center text-center text-text-muted space-y-4 opacity-60">
                                        <div className="w-16 h-16 rounded-full border border-dashed border-white/20 flex items-center justify-center animate-pulse-slow">
                                            <span className="text-2xl">⊕</span>
                                        </div>
                                        <div>
                                            <p className="text-sm font-mono text-white">NO TARGET SELECTED</p>
                                            <p className="text-xs mt-1">Click a region on the globe to initiate analysis.</p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* TAB: METRICS (Data) */}
                        {activeTab === 'metrics' && (
                            <div className="p-6 space-y-6 animate-fade-in">
                                {selectedCountry ? (
                                    <>
                                        <div>
                                            <h3 className="text-lg font-display text-white mb-1">{selectedCountry.name}</h3>
                                            <p className="text-xs text-text-muted font-mono">Key Performance Indicators</p>
                                        </div>

                                        <div className="space-y-4">
                                            {/* GDP */}
                                            <div className="space-y-1">
                                                <div className="flex justify-between text-xs">
                                                    <span className="text-text-secondary">GDP Growth (YoY)</span>
                                                    <span className={`font-mono font-bold ${selectedCountry.gdp && selectedCountry.gdp > 0 ? 'text-positive' : 'text-negative'}`}>
                                                        {selectedCountry.gdp ? `${selectedCountry.gdp > 0 ? '+' : ''}${selectedCountry.gdp}%` : 'N/A'}
                                                    </span>
                                                </div>
                                                <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden relative">
                                                    <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-white/30"></div>
                                                    <div 
                                                        className={`h-full ${selectedCountry.gdp && selectedCountry.gdp > 0 ? 'bg-positive' : 'bg-negative'}`} 
                                                        style={{ 
                                                            width: `${Math.min(50, Math.abs(selectedCountry.gdp || 0) * 10)}%`,
                                                            marginLeft: selectedCountry.gdp && selectedCountry.gdp > 0 ? '50%' : `${50 - Math.min(50, Math.abs(selectedCountry.gdp || 0) * 10)}%`
                                                        }}
                                                    ></div>
                                                </div>
                                            </div>

                                            {/* Inflation */}
                                            <div className="space-y-1">
                                                <div className="flex justify-between text-xs">
                                                    <span className="text-text-secondary">Inflation (CPI)</span>
                                                    <span className={`font-mono font-bold ${selectedCountry.inflation && selectedCountry.inflation > 3 ? 'text-warning' : 'text-white'}`}>
                                                        {selectedCountry.inflation ? `${selectedCountry.inflation}%` : 'N/A'}
                                                    </span>
                                                </div>
                                                <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                                                    <div className={`h-full ${selectedCountry.inflation && selectedCountry.inflation > 5 ? 'bg-negative' : 'bg-accent-cyan'}`} style={{ width: `${Math.min(100, (selectedCountry.inflation || 0) * 10)}%` }}></div>
                                                </div>
                                            </div>

                                            {/* Debt */}
                                            <div className="space-y-1">
                                                <div className="flex justify-between text-xs">
                                                    <span className="text-text-secondary">Debt-to-GDP</span>
                                                    <span className={`font-mono font-bold ${selectedCountry.debt && selectedCountry.debt > 100 ? 'text-negative' : 'text-white'}`}>
                                                        {selectedCountry.debt ? `${selectedCountry.debt}%` : 'N/A'}
                                                    </span>
                                                </div>
                                                <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                                                    <div className="bg-purple-500 h-full" style={{ width: `${Math.min(100, (selectedCountry.debt || 0) / 2)}%` }}></div>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded mt-4">
                                            <h4 className="text-[10px] font-bold text-blue-400 uppercase tracking-widest mb-2">Caria Insight</h4>
                                            <p className="text-xs text-blue-200/80 leading-relaxed">
                                                Debt levels relative to growth suggest {selectedCountry.debt && selectedCountry.debt > 90 ? "limited fiscal space for stimulus." : "adequate fiscal capacity for future shocks."} 
                                                Inflation trend requires {selectedCountry.inflation && selectedCountry.inflation > 3 ? "continued restrictive monetary policy." : "monitoring but allows for policy flexibility."}
                                            </p>
                                        </div>
                                    </>
                                ) : (
                                    <div className="h-full flex flex-col items-center justify-center text-center text-text-muted space-y-4 opacity-60">
                                        <div className="w-16 h-16 rounded-full border border-dashed border-white/20 flex items-center justify-center animate-pulse-slow">
                                            <span className="text-2xl">📊</span>
                                        </div>
                                        <div>
                                            <p className="text-sm font-mono text-white">AWAITING DATA STREAM</p>
                                            <p className="text-xs mt-1">Select a jurisdiction to load telemetry.</p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </WidgetCard>
    );
};

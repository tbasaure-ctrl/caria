import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { WidgetCard } from './WidgetCard';
import { CountryState, CyclePhase, EconomicFlowData, EconomicArc } from '../../types/worldEconomies';

// Visual constants
const PHASE_COLORS: Record<CyclePhase, string> = {
    'expansion': '#10b981', // Green
    'slowdown': '#f59e0b',  // Amber
    'recession': '#ef4444', // Red
    'recovery': '#3b82f6'   // Blue
};

// Connection strength colors (matching the HUD aesthetic)
const CONNECTION_COLORS: Record<string, { color: string; glow: string }> = {
    strong: { color: '#ef4444', glow: 'rgba(239, 68, 68, 0.6)' },      // Hot Red
    medium: { color: '#f59e0b', glow: 'rgba(245, 158, 11, 0.5)' },     // Amber
    weak: { color: '#22d3ee', glow: 'rgba(34, 211, 238, 0.4)' },       // Cyan
    'usa-link': { color: '#a855f7', glow: 'rgba(168, 85, 247, 0.5)' }  // Purple for USA
};

type InfoTab = 'guide' | 'details' | 'metrics' | 'flows';
type ConnectionFilter = 'all' | 'strong' | 'medium' | 'usa';

export const WorldEconomiesHealth: React.FC<{ id?: string }> = ({ id }) => {
    const [data, setData] = useState<CountryState[]>([]);
    const [flowData, setFlowData] = useState<EconomicFlowData | null>(null);
    const [loading, setLoading] = useState(true);
    const [selectedCountry, setSelectedCountry] = useState<CountryState | null>(null);
    const [selectedArc, setSelectedArc] = useState<EconomicArc | null>(null);
    const [activeTab, setActiveTab] = useState<InfoTab>('guide');
    const [showConnections, setShowConnections] = useState(true);
    const [connectionFilter, setConnectionFilter] = useState<ConnectionFilter>('all');

    useEffect(() => {
        loadData();
    }, []);

    useEffect(() => {
        if (selectedCountry) {
            setActiveTab('details');
        }
    }, [selectedCountry]);

    useEffect(() => {
        if (selectedArc) {
            setActiveTab('flows');
        }
    }, [selectedArc]);

    const loadData = async () => {
        setLoading(true);
        try {
            // Fetch country data
            const response = await fetch('/data/world_economies.json');
            if (!response.ok) throw new Error('Failed to load data');
            const jsonData: CountryState[] = await response.json();
            setData(jsonData);

            // Fetch CARIA economic flows
            try {
                const flowResponse = await fetch('/data/caria_flows.json');
                if (flowResponse.ok) {
                    const flowJson: EconomicFlowData = await flowResponse.json();
                    setFlowData(flowJson);
                    console.log(`🌐 CARIA: Loaded ${flowJson.arcs.length} economic connections`);
                }
            } catch (flowErr) {
                console.warn("CARIA flows not available:", flowErr);
            }
        } catch (err) {
            console.error("Failed to load world economies data:", err);
        } finally {
            setLoading(false);
        }
    };

    // Filter arcs based on user selection
    const filteredArcs = flowData?.arcs.filter(arc => {
        if (!showConnections) return false;
        if (connectionFilter === 'all') return true;
        if (connectionFilter === 'strong') return arc.strength === 'strong';
        if (connectionFilter === 'medium') return arc.strength === 'strong' || arc.strength === 'medium';
        if (connectionFilter === 'usa') return arc.isUSA === true;
        return true;
    }) || [];

    // --- Prepare Plotly Traces ---

    // 1. Base Map (Dark) - handled by layout.geo

    // 2. Halos (Outer Glow) - Size based on Structural Risk
    const haloTrace: Partial<Plotly.Data> = {
        type: 'scattergeo',
        mode: 'markers',
        lat: data.map(d => d.lat),
        lon: data.map(d => d.lon),
        text: data.map(d => d.name),
        hoverinfo: 'skip', // Hover on core orb instead
        marker: {
            size: data.map(d => 15 + (d.structuralRisk / 3)), // 15 to ~48
            color: data.map(d => PHASE_COLORS[d.cyclePhase]),
            opacity: 0.2, // Faint glow
            symbol: 'circle',
            line: { width: 0 }
        },
        name: 'Risk Halo'
    };

    // 3. Core Orbs - Color by Phase, Opacity by Stress
    const orbTrace: Partial<Plotly.Data> = {
        type: 'scattergeo',
        mode: 'markers',
        lat: data.map(d => d.lat),
        lon: data.map(d => d.lon),
        text: data.map(d =>
            `${d.name}<br>` +
            `Phase: ${d.cyclePhase.toUpperCase()}<br>` +
            `Stress: ${d.stressLevel}/100<br>` +
            `Risk: ${d.structuralRisk}/100`
        ),
        hoverinfo: 'text',
        hovertemplate: '<b>%{text}</b><extra></extra>',
        marker: {
            size: 12, // Standard core size
            color: data.map(d => PHASE_COLORS[d.cyclePhase]),
            opacity: data.map(d => 0.6 + (d.stressLevel / 250)), // 0.6 to 1.0 (Higher stress = more solid)
            symbol: 'circle',
            line: {
                // Ring thickness based on External Vulnerability
                width: data.map(d => 1 + (d.externalVulnerability / 10)), // 1px to 11px
                color: '#ffffff'
            }
        },
        name: 'Economies'
    };

    // 4. CARIA Economic Connections - Lines showing influence flows
    const connectionTraces: Partial<Plotly.Data>[] = filteredArcs.map((arc, idx) => {
        const colorConfig = CONNECTION_COLORS[arc.strength];
        const opacity = Math.min(0.4 + arc.weight * 2, 0.9);
        
        return {
            type: 'scattergeo',
            mode: 'lines',
            lat: [arc.startLat, arc.endLat],
            lon: [arc.startLng, arc.endLng],
            line: {
                width: 1 + arc.weight * 8, // 1 to ~4px based on weight
                color: colorConfig.color,
            },
            opacity: opacity,
            hoverinfo: 'text',
            text: `<b>${arc.sourceName}</b> → <b>${arc.targetName}</b><br>` +
                  `Influence: ${(arc.weight * 100).toFixed(1)}%<br>` +
                  `Strength: ${arc.strength.toUpperCase()}`,
            hovertemplate: '%{text}<extra></extra>',
            name: arc.label,
            showlegend: false,
        } as Partial<Plotly.Data>;
    });

    return (
        <WidgetCard
            id={id}
            title="GLOBAL ECONOMIC MONITOR"
            tooltip="Real-time macroeconomic surveillance system."
            className="min-h-[600px]"
        >
            <div className="flex flex-col xl:flex-row gap-6 h-full">
                {/* Left: 3D Globe */}
                <div className="flex-1 relative h-[500px] xl:h-[600px] rounded-xl overflow-hidden bg-black border border-white/10 shadow-[inset_0_0_100px_rgba(0,0,0,1)] group">
                    {/* Deep Space Background */}
                    <div className="absolute inset-0 z-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-[#0B1221] via-[#020408] to-black"></div>
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
                                <div className="text-accent-cyan font-mono text-sm tracking-[0.2em] animate-pulse">ESTABLISHING UPLINK...</div>
                            </div>
                        </div>
                    ) : (
                        <Plot
                            data={[...connectionTraces, haloTrace, orbTrace]}
                            layout={{
                                geo: {
                                    showframe: false,
                                    showcoastlines: true,
                                    coastlinecolor: '#334155',
                                    projection: {
                                        type: 'orthographic',
                                        rotation: { lon: 10, lat: 10 } // Initial view
                                    },
                                    bgcolor: 'rgba(0,0,0,0)',
                                    showland: true,
                                    landcolor: '#0f172a', // Dark slate land
                                    showocean: true,
                                    oceancolor: 'rgba(0,0,0,0)', // Transparent ocean to see background
                                    showlakes: false,
                                    showcountries: true,
                                    countrycolor: '#1e293b',
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
                            onClick={(event) => {
                                const point = event.points[0];
                                // Match by index since we mapped directly
                                if (point && point.pointIndex !== undefined) {
                                    const country = data[point.pointIndex];
                                    if (country) setSelectedCountry(country);
                                }
                            }}
                        />
                    )}

                    {/* HUD Overlay */}
                    <div className="absolute top-6 left-6 z-10 pointer-events-auto">
                        <div className="backdrop-blur-md bg-black/60 border border-white/10 rounded-lg p-4 shadow-[0_0_30px_rgba(0,0,0,0.8)] border-l-4 border-l-accent-cyan">
                            <div className="flex items-center gap-3 mb-2">
                                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse shadow-[0_0_10px_rgba(34,197,94,0.8)]"></div>
                                <span className="text-white text-xs font-display uppercase tracking-widest">Live Feed</span>
                            </div>
                            <div className="text-[10px] text-text-muted font-mono space-y-1">
                                <div className="flex justify-between gap-4"><span>NODES:</span> <span className="text-accent-cyan">{data.length} ACTIVE</span></div>
                                <div className="flex justify-between gap-4"><span>SYNC:</span> <span className="text-white">T-00:00:00</span></div>
                            </div>
                            
                            {/* CARIA Connection Controls */}
                            {flowData && (
                                <div className="mt-3 pt-3 border-t border-white/10">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-[10px] text-accent-cyan font-mono">CARIA V12</span>
                                        <button
                                            onClick={() => setShowConnections(!showConnections)}
                                            className={`text-[10px] px-2 py-0.5 rounded font-mono transition-all ${
                                                showConnections 
                                                    ? 'bg-accent-cyan/20 text-accent-cyan' 
                                                    : 'bg-white/5 text-text-muted'
                                            }`}
                                        >
                                            {showConnections ? 'ON' : 'OFF'}
                                        </button>
                                    </div>
                                    {showConnections && (
                                        <div className="space-y-1">
                                            <div className="flex justify-between gap-4">
                                                <span>FLOWS:</span>
                                                <span className="text-white">{filteredArcs.length}</span>
                                            </div>
                                            <div className="flex gap-1 mt-1 flex-wrap">
                                                {(['all', 'strong', 'medium', 'usa'] as ConnectionFilter[]).map(filter => (
                                                    <button
                                                        key={filter}
                                                        onClick={() => setConnectionFilter(filter)}
                                                        className={`text-[8px] px-1.5 py-0.5 rounded uppercase font-mono transition-all ${
                                                            connectionFilter === filter
                                                                ? filter === 'usa' 
                                                                    ? 'bg-purple-500/30 text-purple-300'
                                                                    : 'bg-white/20 text-white'
                                                                : 'text-text-muted hover:text-white'
                                                        }`}
                                                    >
                                                        {filter === 'usa' ? '🇺🇸' : filter}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Legend */}
                    <div className="absolute bottom-6 left-6 z-10 pointer-events-none">
                        <div className="backdrop-blur-md bg-black/60 border border-white/10 rounded-lg p-4 shadow-[0_0_30px_rgba(0,0,0,0.8)]">
                            <h4 className="text-text-muted text-[10px] font-mono uppercase tracking-widest mb-3 border-b border-white/10 pb-2">Cycle Phase</h4>
                            <div className="space-y-2">
                                {Object.entries(PHASE_COLORS).map(([phase, color]) => (
                                    <div key={phase} className="flex items-center gap-3">
                                        <div className="w-2 h-2 rounded-full shadow-[0_0_8px_currentColor]" style={{ backgroundColor: color, color: color }}></div>
                                        <span className="text-[10px] text-white font-mono uppercase opacity-80">{phase}</span>
                                    </div>
                                ))}
                            </div>
                            
                            {/* Connection Strength Legend */}
                            {showConnections && flowData && (
                                <>
                                    <h4 className="text-text-muted text-[10px] font-mono uppercase tracking-widest mt-4 mb-3 border-b border-white/10 pb-2">Influence Flow</h4>
                                    <div className="space-y-2">
                                        <div className="flex items-center gap-3">
                                            <div className="w-4 h-0.5 rounded" style={{ backgroundColor: CONNECTION_COLORS.strong.color }}></div>
                                            <span className="text-[10px] text-white font-mono uppercase opacity-80">Strong &gt;15%</span>
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <div className="w-4 h-0.5 rounded" style={{ backgroundColor: CONNECTION_COLORS.medium.color }}></div>
                                            <span className="text-[10px] text-white font-mono uppercase opacity-80">Medium 10-15%</span>
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <div className="w-4 h-0.5 rounded" style={{ backgroundColor: CONNECTION_COLORS.weak.color }}></div>
                                            <span className="text-[10px] text-white font-mono uppercase opacity-80">Weak 6-10%</span>
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <div className="w-4 h-0.5 rounded" style={{ backgroundColor: CONNECTION_COLORS['usa-link'].color }}></div>
                                            <span className="text-[10px] text-white font-mono uppercase opacity-80">🇺🇸 USA Link</span>
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                </div>

                {/* Right: Analysis Panel */}
                <div className="xl:w-96 flex flex-col gap-4 h-[600px]">
                    {/* Tab Navigation */}
                    <div className="flex bg-[#0B1221] border border-white/10 rounded-lg p-1">
                        <button onClick={() => setActiveTab('guide')} className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${activeTab === 'guide' ? 'bg-white/10 text-white shadow-glow-sm' : 'text-text-muted hover:text-white'}`}>Manual</button>
                        <button onClick={() => setActiveTab('details')} className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${activeTab === 'details' ? 'bg-white/10 text-white shadow-glow-sm' : 'text-text-muted hover:text-white'}`}>Analysis</button>
                        <button onClick={() => setActiveTab('metrics')} className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${activeTab === 'metrics' ? 'bg-white/10 text-white shadow-glow-sm' : 'text-text-muted hover:text-white'}`}>Metrics</button>
                        <button onClick={() => setActiveTab('flows')} className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${activeTab === 'flows' ? 'bg-accent-cyan/20 text-accent-cyan shadow-glow-sm' : 'text-accent-cyan/60 hover:text-accent-cyan'}`}>
                            <span className="flex items-center justify-center gap-1">
                                <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan animate-pulse"></span>
                                CARIA
                            </span>
                        </button>
                    </div>

                    {/* Panel Content */}
                    <div className="bg-[#0B1221] border border-white/10 rounded-lg flex-1 overflow-y-auto custom-scrollbar relative shadow-[0_0_30px_rgba(0,0,0,0.3)]">

                        {/* GUIDE TAB */}
                        {activeTab === 'guide' && (
                            <div className="p-6 space-y-6 animate-fade-in">
                                <div>
                                    <h3 className="text-lg font-display text-white mb-2">Visual Decoding</h3>
                                    <div className="h-0.5 w-12 bg-accent-cyan mb-4"></div>
                                    <p className="text-xs text-text-secondary leading-relaxed">
                                        Each node represents a national economy. Its visual properties encode real-time macroeconomic stress and cycle positioning.
                                    </p>
                                </div>
                                <div className="space-y-4">
                                    <div className="p-3 bg-white/5 rounded border border-white/5">
                                        <div className="text-xs font-bold text-white mb-1">Core Color</div>
                                        <p className="text-[10px] text-text-muted">Indicates the current Business Cycle Phase (Expansion, Slowdown, Recession, Recovery).</p>
                                    </div>
                                    <div className="p-3 bg-white/5 rounded border border-white/5">
                                        <div className="text-xs font-bold text-white mb-1">Halo Size</div>
                                        <p className="text-[10px] text-text-muted">Proportional to <span className="text-white">Structural Risk</span> (Debt, Deficits, Inflation).</p>
                                    </div>
                                    <div className="p-3 bg-white/5 rounded border border-white/5">
                                        <div className="text-xs font-bold text-white mb-1">Outer Ring Thickness</div>
                                        <p className="text-[10px] text-text-muted">Indicates <span className="text-white">External Vulnerability</span> (FX reserves, Terms of Trade).</p>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* DETAILS TAB */}
                        {activeTab === 'details' && (
                            <div className="p-6 space-y-6 animate-fade-in">
                                {selectedCountry ? (
                                    <>
                                        <div className="flex justify-between items-start">
                                            <div>
                                                <div className="text-[10px] font-mono text-accent-cyan mb-1">TARGET LOCKED</div>
                                                <h2 className="text-3xl font-display text-white">{selectedCountry.name}</h2>
                                                <div className="inline-block px-2 py-0.5 rounded bg-white/10 text-[10px] font-mono text-text-muted mt-2">
                                                    ISO: {selectedCountry.isoCode}
                                                </div>
                                            </div>
                                            <div className="text-right">
                                                <div className="text-4xl font-mono text-white font-bold tracking-tighter">{selectedCountry.stressLevel}</div>
                                                <div className="text-[10px] text-text-muted uppercase tracking-widest">Stress Index</div>
                                            </div>
                                        </div>

                                        <div className="p-4 bg-white/5 rounded border border-white/10">
                                            <div className="text-xs text-text-muted uppercase tracking-widest mb-2">Cycle Phase</div>
                                            <div className="text-lg font-medium text-white flex items-center gap-2">
                                                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: PHASE_COLORS[selectedCountry.cyclePhase] }}></span>
                                                {selectedCountry.cyclePhase.toUpperCase()}
                                            </div>
                                            <div className="mt-2 text-[10px] text-text-secondary">
                                                Momentum: <span className={selectedCountry.cycleMomentum > 0 ? 'text-positive' : 'text-negative'}>
                                                    {selectedCountry.cycleMomentum > 0 ? 'Accelerating' : 'Decelerating'} ({selectedCountry.cycleMomentum})
                                                </span>
                                            </div>
                                        </div>

                                        <div className="space-y-3">
                                            <h4 className="text-xs font-bold text-white uppercase tracking-wider border-b border-white/10 pb-2">Risk Profile</h4>

                                            <div className="space-y-1">
                                                <div className="flex justify-between text-xs">
                                                    <span className="text-text-secondary">Structural Risk</span>
                                                    <span className="text-white font-mono">{selectedCountry.structuralRisk}/100</span>
                                                </div>
                                                <div className="w-full bg-white/10 h-1 rounded-full overflow-hidden">
                                                    <div className="bg-amber-500 h-full" style={{ width: `${selectedCountry.structuralRisk}%` }}></div>
                                                </div>
                                            </div>

                                            <div className="space-y-1">
                                                <div className="flex justify-between text-xs">
                                                    <span className="text-text-secondary">External Vulnerability</span>
                                                    <span className="text-white font-mono">{selectedCountry.externalVulnerability}/100</span>
                                                </div>
                                                <div className="w-full bg-white/10 h-1 rounded-full overflow-hidden">
                                                    <div className="bg-red-500 h-full" style={{ width: `${selectedCountry.externalVulnerability}%` }}></div>
                                                </div>
                                            </div>

                                            <div className="space-y-1">
                                                <div className="flex justify-between text-xs">
                                                    <span className="text-text-secondary">Instability Risk</span>
                                                    <span className="text-white font-mono">{selectedCountry.instabilityRisk}/100</span>
                                                </div>
                                                <div className="w-full bg-white/10 h-1 rounded-full overflow-hidden">
                                                    <div className="bg-purple-500 h-full" style={{ width: `${selectedCountry.instabilityRisk}%` }}></div>
                                                </div>
                                            </div>
                                        </div>
                                    </>
                                ) : (
                                    <div className="h-full flex flex-col items-center justify-center text-center text-text-muted space-y-4 opacity-60">
                                        <div className="w-16 h-16 rounded-full border border-dashed border-white/20 flex items-center justify-center animate-pulse-slow">
                                            <span className="text-2xl">⊕</span>
                                        </div>
                                        <div>
                                            <p className="text-sm font-mono text-white">NO TARGET SELECTED</p>
                                            <p className="text-xs mt-1">Click a node on the globe to initiate analysis.</p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* METRICS TAB */}
                        {activeTab === 'metrics' && (
                            <div className="p-6 space-y-6 animate-fade-in">
                                {selectedCountry && selectedCountry.metrics ? (
                                    <>
                                        <div>
                                            <h3 className="text-lg font-display text-white mb-1">{selectedCountry.name}</h3>
                                            <p className="text-xs text-text-muted font-mono">Macro Telemetry</p>
                                        </div>

                                        <div className="grid grid-cols-2 gap-4">
                                            <div className="p-3 bg-white/5 rounded border border-white/5">
                                                <div className="text-[10px] text-text-muted uppercase">GDP Growth</div>
                                                <div className={`text-lg font-mono font-bold ${selectedCountry.metrics.gdpGrowth! > 0 ? 'text-positive' : 'text-negative'}`}>
                                                    {selectedCountry.metrics.gdpGrowth}%
                                                </div>
                                            </div>
                                            <div className="p-3 bg-white/5 rounded border border-white/5">
                                                <div className="text-[10px] text-text-muted uppercase">Inflation</div>
                                                <div className={`text-lg font-mono font-bold ${selectedCountry.metrics.inflation! > 5 ? 'text-negative' : 'text-white'}`}>
                                                    {selectedCountry.metrics.inflation}%
                                                </div>
                                            </div>
                                            <div className="p-3 bg-white/5 rounded border border-white/5">
                                                <div className="text-[10px] text-text-muted uppercase">Unemployment</div>
                                                <div className="text-lg font-mono font-bold text-white">
                                                    {selectedCountry.metrics.unemployment}%
                                                </div>
                                            </div>
                                            <div className="p-3 bg-white/5 rounded border border-white/5">
                                                <div className="text-[10px] text-text-muted uppercase">Debt/GDP</div>
                                                <div className={`text-lg font-mono font-bold ${selectedCountry.metrics.debtToGdp! > 90 ? 'text-warning' : 'text-white'}`}>
                                                    {selectedCountry.metrics.debtToGdp}%
                                                </div>
                                            </div>
                                        </div>

                                        <div className="mt-4">
                                            <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-2">Behavioral Signal</h4>
                                            <div className="flex items-center gap-4">
                                                <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden relative">
                                                    <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-white/50"></div>
                                                    {/* Map -1 to 1 range to 0 to 100% */}
                                                    <div
                                                        className={`h-full absolute top-0 bottom-0 w-2 rounded-full ${selectedCountry.behavioralSignal > 0 ? 'bg-negative' : 'bg-positive'}`}
                                                        style={{
                                                            left: `${((selectedCountry.behavioralSignal + 1) / 2) * 100}%`,
                                                            transform: 'translateX(-50%)'
                                                        }}
                                                    ></div>
                                                </div>
                                                <div className="text-xs font-mono text-white w-12 text-right">{selectedCountry.behavioralSignal}</div>
                                            </div>
                                            <div className="flex justify-between text-[10px] text-text-muted mt-1">
                                                <span>Optimistic</span>
                                                <span>Fearful</span>
                                            </div>
                                        </div>
                                    </>
                                ) : (
                                    <div className="h-full flex flex-col items-center justify-center text-center text-text-muted space-y-4 opacity-60">
                                        <div className="w-16 h-16 rounded-full border border-dashed border-white/20 flex items-center justify-center animate-pulse-slow">
                                            <span className="text-2xl">📊</span>
                                        </div>
                                        <div>
                                            <p className="text-sm font-mono text-white">NO METRICS AVAILABLE</p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* CARIA FLOWS TAB */}
                        {activeTab === 'flows' && (
                            <div className="p-6 space-y-6 animate-fade-in">
                                {flowData ? (
                                    <>
                                        {/* Header */}
                                        <div>
                                            <div className="flex items-center gap-2">
                                                <div className="w-2 h-2 rounded-full bg-accent-cyan animate-pulse"></div>
                                                <span className="text-xs font-mono text-accent-cyan">{flowData.modelVersion}</span>
                                            </div>
                                            <h3 className="text-xl font-display text-white mt-1">Economic Influence Graph</h3>
                                            <p className="text-[11px] text-text-secondary mt-2 leading-relaxed">
                                                Economic dependencies discovered through neural network analysis of macro indicators, 
                                                market returns, and cross-border data.
                                            </p>
                                        </div>

                                        {/* Explanation Box */}
                                        <div className="p-3 bg-gradient-to-r from-accent-cyan/5 to-purple-500/5 rounded-lg border border-white/10">
                                            <div className="flex items-start gap-2">
                                                <span className="text-lg">🧠</span>
                                                <div>
                                                    <h4 className="text-xs font-bold text-white mb-1">How to read the connections</h4>
                                                    <p className="text-[10px] text-text-muted leading-relaxed mb-2">
                                                        Each line represents a <span className="text-white">predictive relationship</span> between two economies. 
                                                        The direction (Country A → Country B) means changes in Country A's markets tend to 
                                                        <span className="text-accent-cyan"> precede</span> changes in Country B's markets. 
                                                        Line thickness indicates the strength of this predictive signal.
                                                    </p>
                                                    <p className="text-[10px] text-text-muted leading-relaxed">
                                                        Connections can reveal two types of dynamics:
                                                    </p>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Demand Pull Explanation */}
                                        <div className="p-3 bg-green-500/5 rounded-lg border border-green-500/20">
                                            <h4 className="text-xs font-bold text-white mb-2 flex items-center gap-2">
                                                <span>📈</span>
                                                <span>Demand Pull Signals</span>
                                            </h4>
                                            <p className="text-[10px] text-text-muted leading-relaxed">
                                                When <span className="text-white">Country X → Country Y</span> appears, it often means 
                                                <span className="text-green-300"> Country X (the Leader)</span> is a major consumer, and 
                                                <span className="text-green-300"> Country Y (the Lagger)</span> is a supplier. 
                                                Here's why: When a large economy accelerates, its financial markets react immediately, 
                                                anticipating future commodity needs. The supplier's markets react later, when capital flows 
                                                in to secure that supply. This creates a <span className="text-white">lead-lag relationship</span> 
                                                where demand-side movements predict supply-side responses.
                                            </p>
                                        </div>

                                        {/* Supply Shock Explanation */}
                                        <div className="p-3 bg-red-500/5 rounded-lg border border-red-500/20">
                                            <h4 className="text-xs font-bold text-white mb-2 flex items-center gap-2">
                                                <span>⚡</span>
                                                <span>Supply Shock Signals</span>
                                            </h4>
                                            <p className="text-[10px] text-text-muted leading-relaxed">
                                                If the connection were reversed (<span className="text-white">Country Y → Country X</span>), 
                                                it would indicate <span className="text-red-300">supply-side dynamics</span> — disruptions in 
                                                production or export capacity affecting downstream consumers. Natural disasters, strikes, or 
                                                policy changes in supplier nations would create this pattern. The model learns which type 
                                                of dynamic is currently dominant based on historical market behavior.
                                            </p>
                                        </div>

                                        {/* Stats Grid */}
                                        <div className="grid grid-cols-4 gap-2">
                                            <div className="p-2 bg-red-500/10 rounded border border-red-500/20 text-center">
                                                <div className="text-lg font-mono font-bold text-red-400">{flowData.stats.strongConnections}</div>
                                                <div className="text-[8px] text-red-400/70 uppercase tracking-wider">Strong</div>
                                            </div>
                                            <div className="p-2 bg-amber-500/10 rounded border border-amber-500/20 text-center">
                                                <div className="text-lg font-mono font-bold text-amber-400">{flowData.stats.mediumConnections}</div>
                                                <div className="text-[8px] text-amber-400/70 uppercase tracking-wider">Medium</div>
                                            </div>
                                            <div className="p-2 bg-cyan-500/10 rounded border border-cyan-500/20 text-center">
                                                <div className="text-lg font-mono font-bold text-cyan-400">{flowData.stats.weakConnections}</div>
                                                <div className="text-[8px] text-cyan-400/70 uppercase tracking-wider">Weak</div>
                                            </div>
                                            <div className="p-2 bg-purple-500/10 rounded border border-purple-500/20 text-center">
                                                <div className="text-lg font-mono font-bold text-purple-400">{flowData.stats.usaConnections || 0}</div>
                                                <div className="text-[8px] text-purple-400/70 uppercase tracking-wider">🇺🇸 USA</div>
                                            </div>
                                        </div>

                                        {/* Key Discoveries */}
                                        <div>
                                            <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-3 flex items-center gap-2">
                                                <span>🔬 Hidden Patterns Discovered</span>
                                            </h4>
                                            <div className="space-y-2 max-h-[320px] overflow-y-auto custom-scrollbar pr-2">
                                                {/* IND-CHL Connection */}
                                                <div className="p-2.5 rounded bg-gradient-to-r from-red-500/10 to-transparent border-l-2 border-red-500">
                                                    <div className="flex items-center justify-between mb-1">
                                                        <span className="text-xs font-bold text-white">🇮🇳 India ↔ 🇨🇱 Chile</span>
                                                        <span className="text-[10px] font-mono text-red-400">31.5% / 22.4%</span>
                                                    </div>
                                                    <p className="text-[10px] text-text-muted leading-relaxed">
                                                        <span className="text-red-300">Strongest bidirectional connection.</span> This represents a 
                                                        <span className="text-white"> demand pull dynamic</span>: Chile is commodity-driven (copper, lithium), 
                                                        while India is a major importer with growing manufacturing. Both economies respond to the 
                                                        global commodity supercycle, but India's demand signals precede Chile's supply responses. 
                                                        The connection reveals: <span className="text-red-200">commodity cycle → emerging market demand → 
                                                        resource exporter revenues</span>. This type of non-linear, asymmetric relationship is typically 
                                                        missed by traditional econometric models.
                                                    </p>
                                                </div>

                                                {/* IDN-MEX Connection */}
                                                <div className="p-2.5 rounded bg-gradient-to-r from-red-500/10 to-transparent border-l-2 border-red-500">
                                                    <div className="flex items-center justify-between mb-1">
                                                        <span className="text-xs font-bold text-white">🇮🇩 Indonesia ↔ 🇲🇽 Mexico</span>
                                                        <span className="text-[10px] font-mono text-red-400">18.7% / 17.3%</span>
                                                    </div>
                                                    <p className="text-[10px] text-text-muted leading-relaxed">
                                                        Both economies share exposure to global risk sentiment, USD/Fed rate sensitivity, 
                                                        and manufacturing integration into global supply chains. The bidirectional connection 
                                                        reflects synchronized responses to capital flow shifts: 
                                                        <span className="text-red-200">financial shocks → USD strength → emerging market FX pressure → 
                                                        correlated movements</span>.
                                                    </p>
                                                </div>

                                                {/* DEU-FRA Connection */}
                                                <div className="p-2.5 rounded bg-gradient-to-r from-amber-500/10 to-transparent border-l-2 border-amber-500">
                                                    <div className="flex items-center justify-between mb-1">
                                                        <span className="text-xs font-bold text-white">🇩🇪 Germany ↔ 🇫🇷 France</span>
                                                        <span className="text-[10px] font-mono text-amber-400">14.9% / 14.0%</span>
                                                    </div>
                                                    <p className="text-[10px] text-text-muted leading-relaxed">
                                                        High economic integration through shared currency, integrated supply chains, and 
                                                        synchronized industrial cycles. The connection demonstrates that Europe functions 
                                                        as a cohesive economic region within the global network.
                                                    </p>
                                                </div>

                                                {/* BRA-KOR Connection */}
                                                <div className="p-2.5 rounded bg-gradient-to-r from-amber-500/10 to-transparent border-l-2 border-amber-500">
                                                    <div className="flex items-center justify-between mb-1">
                                                        <span className="text-xs font-bold text-white">🇧🇷 Brazil ↔ 🇰🇷 S. Korea</span>
                                                        <span className="text-[10px] font-mono text-amber-400">14.1% / 12.5%</span>
                                                    </div>
                                                    <p className="text-[10px] text-text-muted leading-relaxed">
                                                        Illustrates a complementary relationship: Brazil as commodity exporter and Korea 
                                                        as industrial importer. Both are synchronized to the global manufacturing cycle, 
                                                        but from opposite positions in the value chain. This demonstrates the model's ability 
                                                        to capture relationships across different stages of global production networks.
                                                    </p>
                                                </div>

                                                {/* MEX Multi-connection */}
                                                <div className="p-2.5 rounded bg-gradient-to-r from-amber-500/10 to-transparent border-l-2 border-amber-500">
                                                    <div className="flex items-center justify-between mb-1">
                                                        <span className="text-xs font-bold text-white">🇲🇽 Mexico → KOR / BRA</span>
                                                        <span className="text-[10px] font-mono text-amber-400">12.4% / 10.0%</span>
                                                    </div>
                                                    <p className="text-[10px] text-text-muted leading-relaxed">
                                                        Mexico's economy is influenced by multiple factors: proximity to the US business cycle, 
                                                        integration into global manufacturing chains, and sensitivity to emerging market capital flows. 
                                                        The connections to Korea (manufacturing) and Brazil (commodities) reflect these different 
                                                        transmission channels.
                                                    </p>
                                                </div>

                                                {/* CHN-AUS Connection */}
                                                <div className="p-2.5 rounded bg-gradient-to-r from-cyan-500/10 to-transparent border-l-2 border-cyan-500">
                                                    <div className="flex items-center justify-between mb-1">
                                                        <span className="text-xs font-bold text-white">🇨🇳 China → 🇦🇺 Australia</span>
                                                        <span className="text-[10px] font-mono text-cyan-400">11.4%</span>
                                                    </div>
                                                    <p className="text-[10px] text-text-muted leading-relaxed">
                                                        Represents a clear demand pull dynamic: China's industrial demand drives Australia's 
                                                        mineral exports. As China accounts for 30-40% of global industrial commodity consumption, 
                                                        changes in Chinese economic activity predict movements in resource-dependent economies.
                                                    </p>
                                                </div>

                                                {/* GBR-EU Connection */}
                                                <div className="p-2.5 rounded bg-gradient-to-r from-cyan-500/10 to-transparent border-l-2 border-cyan-500">
                                                    <div className="flex items-center justify-between mb-1">
                                                        <span className="text-xs font-bold text-white">🇬🇧 UK → 🇩🇪🇫🇷 EU</span>
                                                        <span className="text-[10px] font-mono text-cyan-400">~10.7%</span>
                                                    </div>
                                                    <p className="text-[10px] text-text-muted leading-relaxed">
                                                        Despite political separation, trade flows and financial market correlations maintain 
                                                        economic connections. This demonstrates that economic integration can persist even 
                                                        when political relationships change.
                                                    </p>
                                                </div>
                                            </div>
                                        </div>

                                        {/* USA Analysis */}
                                        {flowData.usaAnalysis && (
                                            <div className="border-t border-white/10 pt-4">
                                                <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-3 flex items-center gap-2">
                                                    <span>🇺🇸</span>
                                                    <span>USA Influence Profile</span>
                                                </h4>
                                                <div className="grid grid-cols-2 gap-4">
                                                    <div>
                                                        <div className="text-[10px] text-text-muted uppercase mb-2">Influenced By</div>
                                                        <div className="space-y-1">
                                                            {flowData.usaAnalysis.influencedBy.slice(0, 4).map((item, idx) => (
                                                                <div key={idx} className="flex justify-between text-[10px]">
                                                                    <span className="text-text-secondary">{item.country}</span>
                                                                    <span className="text-white font-mono">{(item.weight * 100).toFixed(1)}%</span>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </div>
                                                    <div>
                                                        <div className="text-[10px] text-text-muted uppercase mb-2">Influences</div>
                                                        <div className="space-y-1">
                                                            {flowData.usaAnalysis.influences.slice(0, 4).map((item, idx) => (
                                                                <div key={idx} className="flex justify-between text-[10px]">
                                                                    <span className="text-text-secondary">{item.country}</span>
                                                                    <span className="text-white font-mono">{(item.weight * 100).toFixed(1)}%</span>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </div>
                                                </div>

                                                {/* USA Insight */}
                                                <div className="mt-3 p-2.5 bg-purple-500/10 rounded border border-purple-500/20">
                                                    <p className="text-[10px] text-purple-200/80 leading-relaxed">
                                                        <span className="text-purple-300 font-bold">💡 Interpreting USA's role:</span><br/>
                                                        The USA's influence appears distributed rather than concentrated in strong 
                                                        directional connections. This occurs because US factors (currency, interest rates, 
                                                        volatility indices, commodity markets) are embedded in global financial conditions. 
                                                        The model treats US conditions as part of the baseline environment rather than 
                                                        a differentiating signal. This reflects that the US economy moves 
                                                        <span className="text-white"> with</span> global risk cycles rather than 
                                                        consistently <span className="text-white">before</span> them.
                                                    </p>
                                                </div>
                                            </div>
                                        )}

                                        {/* Updated timestamp */}
                                        <div className="text-[10px] text-text-muted font-mono text-center pt-2 border-t border-white/5">
                                            Last updated: {flowData.date}
                                        </div>
                                    </>
                                ) : (
                                    <div className="h-full flex flex-col items-center justify-center text-center text-text-muted space-y-4 opacity-60">
                                        <div className="w-16 h-16 rounded-full border border-dashed border-accent-cyan/30 flex items-center justify-center">
                                            <span className="text-2xl">🌐</span>
                                        </div>
                                        <div>
                                            <p className="text-sm font-mono text-white">CARIA MODEL LOADING...</p>
                                            <p className="text-xs mt-1">Economic influence data not available</p>
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

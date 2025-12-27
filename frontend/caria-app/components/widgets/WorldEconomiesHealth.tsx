import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { WidgetCard } from './WidgetCard';
import { CountryState, EconomicFlowData, EconomicArc, DirectionPredictionsData } from '../../types/worldEconomies';

// Country unique colors - 22 countries
const COUNTRY_COLORS: Record<string, string> = {
    'USA': '#ef4444',   // Red
    'CHN': '#f59e0b',   // Amber
    'JPN': '#3b82f6',   // Blue
    'DEU': '#10b981',   // Green
    'GBR': '#8b5cf6',   // Purple
    'FRA': '#ec4899',   // Pink
    'IND': '#f97316',   // Orange
    'BRA': '#22d3ee',   // Cyan
    'CAN': '#06b6d4',   // Sky
    'KOR': '#a855f7',   // Violet
    'AUS': '#14b8a6',   // Teal
    'MEX': '#eab308',   // Yellow
    'IDN': '#84cc16',   // Lime
    'ZAF': '#64748b',   // Slate
    'CHL': '#dc2626',   // Dark Red
    'SGP': '#6366f1',   // Indigo
    'NLD': '#f43f5e',   // Rose
    'HKG': '#0ea5e9',   // Light Blue
    'CHE': '#d946ef',   // Fuchsia
    'TWN': '#14532d',   // Dark Green
    'VNM': '#fbbf24',   // Gold
    'NOR': '#0284c7',   // Ocean Blue
};

// Connection strength colors
const CONNECTION_COLORS: Record<string, { color: string; glow: string }> = {
    strong: { color: '#ef4444', glow: 'rgba(239, 68, 68, 0.6)' },
    medium: { color: '#f59e0b', glow: 'rgba(245, 158, 11, 0.5)' },
    weak: { color: '#22d3ee', glow: 'rgba(34, 211, 238, 0.4)' },
    'usa-link': { color: '#a855f7', glow: 'rgba(168, 85, 247, 0.5)' }
};

type InfoTab = 'guide' | 'analysis' | 'metrics';
type ConnectionFilter = 'all' | 'strong' | 'medium' | 'usa';

interface CountryWithPrediction extends CountryState {
    prediction?: {
        direction: 'UP' | 'DOWN';
        confidence: number;
        rawValue: number;
    };
}

export const WorldEconomiesHealth: React.FC<{ id?: string }> = ({ id }) => {
    const [data, setData] = useState<CountryWithPrediction[]>([]);
    const [flowData, setFlowData] = useState<EconomicFlowData | null>(null);
    const [directionPredictions, setDirectionPredictions] = useState<DirectionPredictionsData | null>(null);
    const [loading, setLoading] = useState(true);
    const [selectedCountry, setSelectedCountry] = useState<CountryWithPrediction | null>(null);
    const [selectedArc, setSelectedArc] = useState<EconomicArc | null>(null);
    const [activeTab, setActiveTab] = useState<InfoTab>('guide');
    const [showConnections, setShowConnections] = useState(true);
    const [connectionFilter, setConnectionFilter] = useState<ConnectionFilter>('all');

    useEffect(() => {
        loadData();
    }, []);

    useEffect(() => {
        if (selectedCountry) {
            setActiveTab('analysis');
        }
    }, [selectedCountry]);

    const loadData = async () => {
        setLoading(true);
        try {
            // Fetch country data
            const response = await fetch('/data/world_economies.json');
            if (!response.ok) throw new Error('Failed to load data');
            const jsonData: CountryState[] = await response.json();
            
            // Fetch direction predictions
            try {
                const predResponse = await fetch('/data/caria_direction_predictions.json');
                if (predResponse.ok) {
                    const predData: DirectionPredictionsData = await predResponse.json();
                    setDirectionPredictions(predData);
                    
                    // Merge predictions with country data
                    const countriesWithPreds = jsonData.map(country => {
                        const predIdx = predData.countries.indexOf(country.isoCode);
                        if (predIdx >= 0) {
                            return {
                                ...country,
                                prediction: {
                                    direction: predData.directions[predIdx],
                                    confidence: predData.confidences[predIdx],
                                    rawValue: predData.predictions[predIdx]
                                }
                            };
                        }
                        return country;
                    });
                    setData(countriesWithPreds);
                } else {
                    setData(jsonData);
                }
            } catch (predErr) {
                console.warn("Direction predictions not available:", predErr);
                setData(jsonData);
            }

            // Fetch economic flows (V22 format)
            try {
                const flowResponse = await fetch('/data/caria_flows.json');
                if (flowResponse.ok) {
                    const flowJson: EconomicFlowData = await flowResponse.json();
                    
                    // Transform V22 connections to arcs with coordinates
                    const arcs: EconomicArc[] = flowJson.connections.map(conn => {
                        const fromCoord = flowJson.coordinates[conn.from];
                        const toCoord = flowJson.coordinates[conn.to];
                        return {
                            source: conn.from,
                            target: conn.to,
                            sourceName: fromCoord?.name || conn.from,
                            targetName: toCoord?.name || conn.to,
                            startLat: fromCoord?.lat || 0,
                            startLng: fromCoord?.lon || 0,
                            endLat: toCoord?.lat || 0,
                            endLng: toCoord?.lon || 0,
                            weight: conn.weight,
                            raw: conn.raw,
                            strength: conn.strength,
                            label: conn.label
                        };
                    });
                    
                    // Build USA analysis
                    const usaInfluences = arcs.filter(a => a.source === 'USA').map(a => ({
                        country: a.target,
                        name: a.targetName,
                        weight: a.weight
                    }));
                    const usaInfluencedBy = arcs.filter(a => a.target === 'USA').map(a => ({
                        country: a.source,
                        name: a.sourceName,
                        weight: a.weight
                    }));
                    
                    flowJson.arcs = arcs;
                    flowJson.usaAnalysis = {
                        influences: usaInfluences.sort((a, b) => b.weight - a.weight),
                        influencedBy: usaInfluencedBy.sort((a, b) => b.weight - a.weight)
                    };
                    
                    setFlowData(flowJson);
                    console.log(`🌐 Economic Monitor: Loaded ${arcs.length} connections across ${flowJson.stats.totalCountries} countries`);
                }
            } catch (flowErr) {
                console.warn("Economic flows not available:", flowErr);
            }
        } catch (err) {
            console.error("Failed to load world economies data:", err);
        } finally {
            setLoading(false);
        }
    };

    // Filter arcs
    const filteredArcs = flowData?.arcs?.filter(arc => {
        if (!showConnections) return false;
        if (connectionFilter === 'all') return true;
        if (connectionFilter === 'strong') return arc.strength === 'strong';
        if (connectionFilter === 'medium') return arc.strength === 'strong' || arc.strength === 'medium';
        if (connectionFilter === 'usa') return arc.source === 'USA' || arc.target === 'USA';
        return true;
    }) || [];

    // Get country color
    const getCountryColor = (isoCode: string): string => {
        return COUNTRY_COLORS[isoCode] || '#64748b';
    };

    // Get GDP growth for ring thickness (positive = thicker)
    const getGdpRingWidth = (country: CountryWithPrediction): number => {
        const gdpGrowth = country.metrics?.gdpGrowth || 0;
        // Positive GDP = thicker ring (1px to 8px)
        if (gdpGrowth > 0) {
            return Math.min(1 + (gdpGrowth / 5), 8); // Scale: 0% GDP = 1px, 35%+ GDP = 8px
        }
        return 1; // Minimum width for negative/zero GDP
    };

    // Get confidence for halo size
    const getConfidenceHaloSize = (country: CountryWithPrediction): number => {
        if (country.prediction) {
            // Halo size based on confidence: 15px to 50px
            // Normalize confidence to 0-1 range first, then scale to 15-50px
            const maxConf = Math.max(...(data.filter(d => d.prediction).map(d => d.prediction!.confidence) || [1]));
            const normalizedConf = maxConf > 0 ? country.prediction.confidence / maxConf : 0;
            return 15 + (normalizedConf * 35); // 15px to 50px range
        }
        return 15; // Default size if no prediction
    };

    // --- Prepare Plotly Traces ---

    // 1. Halos - Size based on confidence
    const haloTrace: Partial<Plotly.Data> = {
        type: 'scattergeo',
        mode: 'markers',
        lat: data.map(d => d.lat),
        lon: data.map(d => d.lon),
        text: data.map(d => d.name),
        hoverinfo: 'skip',
        marker: {
            size: data.map(d => getConfidenceHaloSize(d)),
            color: data.map(d => getCountryColor(d.isoCode)),
            opacity: 0.25, // Faint glow
            symbol: 'circle',
            line: { width: 0 }
        },
        name: 'Confidence Halo'
    };

    // 2. Core Orbs - Color by country, ring thickness by GDP
    const orbTrace: Partial<Plotly.Data> = {
        type: 'scattergeo',
        mode: 'markers',
        lat: data.map(d => d.lat),
        lon: data.map(d => d.lon),
        text: data.map(d => {
            const pred = d.prediction;
            if (pred) {
                return `${d.name}<br>` +
                       `Direction: ${pred.direction}<br>` +
                       `Confidence: ${(pred.confidence * 100).toFixed(1)}%`;
            }
            return `${d.name}<br>Direction: Not Available`;
        }),
        hoverinfo: 'text',
        hovertemplate: '<b>%{text}</b><extra></extra>',
        marker: {
            size: 12,
            color: data.map(d => getCountryColor(d.isoCode)),
            opacity: 0.9,
            symbol: 'circle',
            line: {
                width: data.map(d => getGdpRingWidth(d)),
                color: '#ffffff'
            }
        },
        name: 'Economies'
    };

    // 3. Economic Connections
    const connectionTraces: Partial<Plotly.Data>[] = filteredArcs.map((arc) => {
        const colorConfig = CONNECTION_COLORS[arc.strength];
        const opacity = Math.min(0.4 + arc.weight * 2, 0.9);
        
        return {
            type: 'scattergeo',
            mode: 'lines',
            lat: [arc.startLat, arc.endLat],
            lon: [arc.startLng, arc.endLng],
            line: {
                width: 1 + arc.weight * 8,
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
            tooltip="Economic Direction Predictions & Dependencies"
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
                                        rotation: { lon: 10, lat: 10 }
                                    },
                                    bgcolor: 'rgba(0,0,0,0)',
                                    showland: true,
                                    landcolor: '#0f172a',
                                    showocean: true,
                                    oceancolor: 'rgba(0,0,0,0)',
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
                                {flowData && (
                                    <div className="flex justify-between gap-4">
                                        <span>LINKS:</span> 
                                        <span className="text-white">{flowData.arcs.length}</span>
                                    </div>
                                )}
                            </div>
                            
                            {/* Connection Controls */}
                            {flowData && (
                                <div className="mt-3 pt-3 border-t border-white/10">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-[10px] text-accent-cyan font-mono">V12 ENGINE</span>
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
                            <h4 className="text-text-muted text-[10px] font-mono uppercase tracking-widest mb-3 border-b border-white/10 pb-2">Visual Guide</h4>
                            <div className="space-y-2 text-[10px] text-white font-mono">
                                <div className="flex items-center gap-3">
                                    <div className="w-3 h-3 rounded-full border-2 border-white" style={{ backgroundColor: 'rgba(255,255,255,0.2)' }}></div>
                                    <span className="opacity-80">Halo = Confidence</span>
                                </div>
                                <div className="flex items-center gap-3">
                                    <div className="w-2 h-2 rounded-full border border-white"></div>
                                    <span className="opacity-80">Ring = GDP Growth</span>
                                </div>
                            </div>
                            
                            {showConnections && flowData && (
                                <>
                                    <h4 className="text-text-muted text-[10px] font-mono uppercase tracking-widest mt-4 mb-3 border-b border-white/10 pb-2">Influence Flow</h4>
                            <div className="space-y-2">
                                        {Object.entries(CONNECTION_COLORS).filter(([key]) => key !== 'usa-link').map(([key, config]) => (
                                            <div key={key} className="flex items-center gap-3">
                                                <div className="w-4 h-0.5 rounded" style={{ backgroundColor: config.color }}></div>
                                                <span className="text-[10px] text-white font-mono uppercase opacity-80">{key}</span>
                                    </div>
                                ))}
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
                        <button onClick={() => setActiveTab('analysis')} className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${activeTab === 'analysis' ? 'bg-white/10 text-white shadow-glow-sm' : 'text-text-muted hover:text-white'}`}>Analysis</button>
                        <button onClick={() => setActiveTab('metrics')} className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${activeTab === 'metrics' ? 'bg-white/10 text-white shadow-glow-sm' : 'text-text-muted hover:text-white'}`}>Metrics</button>
                    </div>

                    {/* Panel Content */}
                    <div className="bg-[#0B1221] border border-white/10 rounded-lg flex-1 overflow-y-auto custom-scrollbar relative shadow-[0_0_30px_rgba(0,0,0,0.3)]">

                        {/* MANUAL TAB */}
                        {activeTab === 'guide' && (
                            <div className="p-6 space-y-6 animate-fade-in">
                                <div>
                                    <h3 className="text-lg font-display text-white mb-2">How to Read the Map</h3>
                                    <div className="h-0.5 w-12 bg-accent-cyan mb-4"></div>
                                </div>

                                <div className="space-y-4">
                                    <div className="p-3 bg-white/5 rounded border border-white/5">
                                        <div className="text-xs font-bold text-white mb-1">Nodes</div>
                                        <p className="text-[10px] text-text-muted">Each node represents a country. Colors are unique per country for easy identification.</p>
                                    </div>

                                    <div className="p-3 bg-white/5 rounded border border-white/5">
                                        <div className="text-xs font-bold text-white mb-1">Connections</div>
                                        <p className="text-[10px] text-text-muted">Lines between countries show discovered economic relationships. <span className="text-white">A → B</span> means changes in A tend to precede changes in B.</p>
                                    </div>

                                    <div className="p-3 bg-white/5 rounded border border-white/5">
                                        <div className="text-xs font-bold text-white mb-1">Line Strength</div>
                                        <p className="text-[10px] text-text-muted">Thicker, brighter lines = stronger relationships. Colors indicate strength: <span className="text-red-400">strong</span>, <span className="text-amber-400">medium</span>, <span className="text-cyan-400">weak</span>.</p>
                                    </div>

                                    <div className="p-3 bg-white/5 rounded border border-white/5">
                                        <div className="text-xs font-bold text-white mb-1">Interaction</div>
                                        <p className="text-[10px] text-text-muted">Click a country to see its connections. Use filters to show specific relationship types.</p>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* ANALYSIS TAB */}
                        {activeTab === 'analysis' && (
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
                                            {selectedCountry.prediction && (
                                            <div className="text-right">
                                                    <div className={`text-4xl font-mono font-bold tracking-tighter ${selectedCountry.prediction.direction === 'UP' ? 'text-green-400' : 'text-red-400'}`}>
                                                        {selectedCountry.prediction.direction === 'UP' ? '▲' : '▼'}
                                                    </div>
                                                    <div className="text-[10px] text-text-muted uppercase tracking-widest">Direction</div>
                                            </div>
                                            )}
                                        </div>


                                        {/* Connected Countries */}
                                        {flowData && flowData.arcs && (
                                            <div>
                                                <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-3 border-b border-white/10 pb-2">
                                                    Economic Connections
                                                </h4>
                                                <div className="space-y-2 max-h-[200px] overflow-y-auto custom-scrollbar">
                                                    {flowData.arcs
                                                        .filter(arc => arc.source === selectedCountry.isoCode || arc.target === selectedCountry.isoCode)
                                                        .sort((a, b) => b.weight - a.weight)
                                                        .map((arc, idx) => {
                                                            const isInfluencer = arc.source === selectedCountry.isoCode;
                                                            const otherCountry = isInfluencer ? arc.targetName : arc.sourceName;
                                                            const weight = arc.weight;
                                                            
                                                            return (
                                                                <div key={idx} className="p-2 bg-white/5 rounded border border-white/5">
                                                                    <div className="flex justify-between items-center mb-1">
                                                                        <span className="text-xs text-white">
                                                                            {isInfluencer ? '→' : '←'} {otherCountry}
                                                                        </span>
                                                                        <span className="text-[10px] font-mono text-accent-cyan">
                                                                            {(weight * 100).toFixed(1)}%
                                                                        </span>
                                                                    </div>
                                                                    <div className="w-full bg-white/10 h-1 rounded-full overflow-hidden">
                                                                        <div 
                                                                            className="h-full bg-accent-cyan"
                                                                            style={{ width: `${Math.min(weight / (flowData.stats.maxWeight || 1) * 100, 100)}%` }}
                                                                        ></div>
                                                                    </div>
                                                                </div>
                                                            );
                                                        })}
                                                    {flowData.arcs.filter(arc => arc.source === selectedCountry.isoCode || arc.target === selectedCountry.isoCode).length === 0 && (
                                                        <p className="text-xs text-text-muted">No strong connections detected for this country.</p>
                                                    )}
                                            </div>
                                            </div>
                                        )}
                                    </>
                                ) : (
                                    <>
                                        {/* Show all relationships and predictions */}
                                        {/* Show all economic relationships */}
                                        {flowData && flowData.arcs && (
                                            <>
                                                <div>
                                                    <h3 className="text-lg font-display text-white mb-2">All Connections</h3>
                                                    <p className="text-xs text-text-muted">{flowData.arcs.length} relationships discovered</p>
                                        </div>

                                                <div className="space-y-2 max-h-[300px] overflow-y-auto custom-scrollbar">
                                                    {flowData.arcs
                                                        .sort((a, b) => b.weight - a.weight)
                                                        .map((arc, idx) => (
                                                            <div key={idx} className="p-2.5 bg-white/5 rounded border border-white/5">
                                                                <div className="flex justify-between items-center mb-1">
                                                                    <span className="text-xs font-bold text-white">
                                                                        {arc.sourceName} → {arc.targetName}
                                                                    </span>
                                                                    <span className="text-[10px] font-mono text-accent-cyan">
                                                                        {(arc.weight * 100).toFixed(1)}%
                                                                    </span>
                                                                </div>
                                                                <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                                                                    <div 
                                                                        className="h-full bg-accent-cyan"
                                                                        style={{ width: `${Math.min((arc.weight / (flowData.stats.maxWeight || 1)) * 100, 100)}%` }}
                                                                    ></div>
                                                                </div>
                                                                <div className="text-[10px] text-text-muted mt-1">
                                                                    Strength: {arc.strength.toUpperCase()} | Raw: {arc.raw.toFixed(2)}
                                                                </div>
                                                            </div>
                                                        ))}
                                            </div>
                                            </>
                                        )}

                                        {!flowData && (
                                            <div className="h-full flex flex-col items-center justify-center text-center text-text-muted space-y-4 opacity-60">
                                                <div className="w-16 h-16 rounded-full border border-dashed border-white/20 flex items-center justify-center animate-pulse-slow">
                                                    <span className="text-2xl">⊕</span>
                                                </div>
                                                <div>
                                                    <p className="text-sm font-mono text-white">NO DATA</p>
                                                    <p className="text-xs mt-1">Click a country to see its connections.</p>
                                                </div>
                                            </div>
                                        )}
                                    </>
                                )}
                            </div>
                        )}

                        {/* METRICS TAB - Model Info */}
                        {activeTab === 'metrics' && (
                            <div className="p-6 space-y-6 animate-fade-in">
                                <div>
                                    <h3 className="text-lg font-display text-white mb-2">Model Information</h3>
                                    <p className="text-xs text-text-muted">Technical details about the relationship discovery model</p>
                                </div>

                                {/* Model Stats */}
                                <div className="space-y-4">
                                    <div className="p-3 bg-white/5 rounded border border-white/5">
                                        <div className="text-xs font-bold text-white mb-2">Network Analysis</div>
                                        <div className="space-y-1 text-[10px]">
                                            <div className="flex justify-between">
                                                <span className="text-text-secondary">Countries Analyzed:</span>
                                                <span className="text-white font-mono">{flowData?.stats.totalCountries || 22}</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-text-secondary">Features per Country:</span>
                                                <span className="text-white font-mono">79</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-text-secondary">Time Series:</span>
                                                <span className="text-white font-mono">2000-2024</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-text-secondary">Connections Found:</span>
                                                <span className="text-white font-mono">{flowData?.stats.totalConnections || 0}</span>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="p-3 bg-white/5 rounded border border-white/5">
                                        <div className="text-xs font-bold text-white mb-2">Data Sources</div>
                                        <div className="space-y-1 text-[10px]">
                                            <div className="flex justify-between">
                                                <span className="text-text-secondary">Market Data:</span>
                                                <span className="text-white">Yahoo Finance</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-text-secondary">US Indicators:</span>
                                                <span className="text-white">FRED</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-text-secondary">Global Macro:</span>
                                                <span className="text-white">Trading Economics</span>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="p-3 bg-white/5 rounded border border-white/5">
                                        <div className="text-xs font-bold text-white mb-2">Relationship Types</div>
                                        <div className="space-y-1 text-[10px]">
                                            <div className="flex justify-between">
                                                <span className="text-text-secondary">Asymmetric:</span>
                                                <span className="text-accent-cyan">A → B ≠ B → A</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-text-secondary">Temporal Lead:</span>
                                                <span className="text-white">Changes in A precede B</span>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Direction Finding */}
                                    {flowData?.directionAccuracy && (
                                        <div className="p-3 bg-green-500/5 rounded border border-green-500/20">
                                            <div className="text-xs font-bold text-green-400 mb-2">Auxiliary Finding</div>
                                            <div className="space-y-1 text-[10px]">
                                                <div className="flex justify-between">
                                                    <span className="text-text-secondary">Direction Accuracy:</span>
                                                    <span className="text-green-400 font-mono">{(flowData.directionAccuracy * 100).toFixed(1)}%</span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-text-secondary">Edge over Random:</span>
                                                    <span className="text-green-400 font-mono">+{((flowData.directionAccuracy - 0.5) * 100).toFixed(1)}pp</span>
                                                </div>
                                            </div>
                                            <p className="text-[9px] text-text-muted mt-2 italic">
                                                Discovered as a byproduct of relationship learning
                                            </p>
                                        </div>
                                    )}
                                </div>

                                {/* Model Architecture */}
                                {flowData && (
                                    <div className="pt-4 border-t border-white/10">
                                        <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-3">Model Architecture</h4>
                                        <div className="space-y-2 text-[10px]">
                                            <div className="flex justify-between">
                                                <span className="text-text-secondary">Parameters:</span>
                                                <span className="text-white font-mono">881,431</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-text-secondary">Nodes (Countries):</span>
                                                <span className="text-white font-mono">{flowData.stats.totalCountries}</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-text-secondary">Features per Node:</span>
                                                <span className="text-white font-mono">79</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-text-secondary">Version:</span>
                                                <span className="text-accent-cyan font-mono">{flowData.modelVersion}</span>
                                            </div>
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


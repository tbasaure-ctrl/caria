import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { MessageSquare, Activity, Target, ChevronDown, ChevronRight, Terminal, Search } from 'lucide-react';
import { fetchWithAuth, API_BASE_URL, getToken } from '../../services/apiService';
import { ProjectionValuation } from '../widgets/ProjectionValuation';
import { MonteCarloSimulation } from '../widgets/MonteCarloSimulation';
import { RiskRewardWidget } from '../widgets/RiskRewardWidget';
import { ChatWindow } from '../ChatWindow';
import { HiddenGemsScreener } from '../widgets/HiddenGemsScreener';
import { HiddenRiskReport } from '../widgets/HiddenRiskReport';
import { ProtectedWidget } from '../ProtectedWidget';
import { AlphaStockPicker } from '../widgets/AlphaStockPicker';
import { ValuationTool } from '../widgets/ValuationTool';
import { ValuationWorkshop } from '../widgets/ValuationWorkshop';

// Components for "Progressive Disclosure" UI

const SectionHeader: React.FC<{ title: string; isOpen: boolean; onToggle: () => void }> = ({ title, isOpen, onToggle }) => (
    <button 
        onClick={onToggle}
        className="w-full flex items-center justify-between p-4 bg-bg-secondary border-b border-white/5 hover:bg-white/5 transition-colors"
    >
        <span className="text-sm font-display font-bold text-white tracking-wide">{title}</span>
        {isOpen ? <ChevronDown className="w-4 h-4 text-text-muted" /> : <ChevronRight className="w-4 h-4 text-text-muted" />}
    </button>
);

const ScorecardPillar: React.FC<{ label: string; value: string; subtext: string; status: 'positive' | 'negative' | 'neutral' }> = ({ label, value, subtext, status }) => {
    let color = 'text-white';
    if (status === 'positive') color = 'text-positive';
    if (status === 'negative') color = 'text-negative';
    if (status === 'neutral') color = 'text-warning';

    return (
        <div className="flex-1 p-4 border-r border-white/10 last:border-r-0">
            <div className="text-[10px] text-text-muted uppercase tracking-widest mb-1">{label}</div>
            <div className={`text-lg font-bold ${color}`}>{value}</div>
            <div className="text-xs text-text-secondary mt-1">{subtext}</div>
        </div>
    );
};

export const AnalysisPage: React.FC = () => {
    const [searchParams, setSearchParams] = useSearchParams();
    const ticker = searchParams.get('ticker') || '';
    const [tsmomData, setTsmomData] = useState<any>(null);
    const [showLogs, setShowLogs] = useState(false);
    const [searchInput, setSearchInput] = useState('');
    const [activeSidebarSection, setActiveSection] = useState<'main' | 'valuation' | 'screener'>('main');

    useEffect(() => {
        if (!ticker) return; 

        const fetchTsmom = async () => {
            try {
                const token = getToken();
                const headers: HeadersInit = { 'Content-Type': 'application/json' };
                if (token) headers['Authorization'] = `Bearer ${token}`;
                
                const response = await fetch(`${API_BASE_URL}/api/analysis/tsmom/${ticker}`, { headers });
                if (response.ok) {
                    const data = await response.json();
                    setTsmomData(data);
                }
            } catch (e) {
                console.error(e);
            }
        };
        fetchTsmom();
    }, [ticker]);

    const handleSearchSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (searchInput.trim()) {
            setSearchParams({ ticker: searchInput.trim().toUpperCase() });
        }
    };

    // --- MARKET VIEW (No Ticker Selected) ---
    if (!ticker) {
        return (
            <div className="flex gap-8 h-[calc(100vh-100px)] animate-fade-in">
                {/* Left Sidebar - Navigation */}
                <div className="w-64 border-r border-white/10 pr-6 hidden md:block">
                    <div className="space-y-6">
                        <div>
                            <h3 className="text-lg font-display text-white mb-2">Analysis Center</h3>
                            <p className="text-xs text-text-secondary leading-relaxed">
                                Want a starter position? Looking for a deep dive? Here you can find all the tools you need to make a well-informed decision. We'll show you all the components involved in your future investment and, using our skeptical and critical point of view (that made us so popular in high school), challenge your conviction so we arrive together at a solid, second-order thinking-like investment thesis you will be proud of.
                            </p>
                        </div>
                        <div className="space-y-1">
                            <button 
                                onClick={() => setActiveSection('main')}
                                className={`w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors ${activeSidebarSection === 'main' ? 'bg-white/10 text-white' : 'text-text-muted hover:text-white hover:bg-white/5'}`}
                            >
                                Overview
                            </button>
                            <button 
                                onClick={() => setActiveSection('valuation')}
                                className={`w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors ${activeSidebarSection === 'valuation' ? 'bg-white/10 text-white' : 'text-text-muted hover:text-white hover:bg-white/5'}`}
                            >
                                Valuation Tools
                            </button>
                            <button 
                                onClick={() => setActiveSection('screener')}
                                className={`w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors ${activeSidebarSection === 'screener' ? 'bg-white/10 text-white' : 'text-text-muted hover:text-white hover:bg-white/5'}`}
                            >
                                Screener Tools
                            </button>
                        </div>
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 pb-20">
                    {activeSidebarSection === 'main' && (
                        <div className="space-y-8">
                            {/* Search Banner */}
                            <div className="bg-bg-secondary border border-white/10 rounded-lg p-6">
                                <form onSubmit={handleSearchSubmit} className="flex gap-4 items-center">
                                    <div className="flex-1">
                                        <input
                                            type="text"
                                            value={searchInput}
                                            onChange={(e) => setSearchInput(e.target.value)}
                                            placeholder="Enter ticker symbol (e.g., AAPL, TSLA)"
                                            className="w-full bg-bg-tertiary border border-white/10 rounded px-4 py-2 text-sm text-white placeholder-text-muted focus:border-accent-primary focus:outline-none"
                                        />
                                    </div>
                                    <button
                                        type="submit"
                                        className="bg-accent-primary text-black font-bold text-sm px-6 py-2 rounded hover:bg-accent-primary/90 transition-colors"
                                    >
                                        Analyze Stock
                                    </button>
                                </form>
                            </div>

                            {/* ZONE 1: The Synthesis (Immediate Value) */}
                            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 min-h-[400px]">
                                {/* Left: Caria Chat */}
                                <div className="border border-white/10 rounded-lg overflow-hidden bg-bg-secondary flex flex-col">
                                    <div className="p-3 border-b border-white/10 bg-bg-tertiary flex items-center gap-2">
                                        <MessageSquare className="w-4 h-4 text-accent-primary" />
                                        <span className="text-xs font-bold text-white">Qualitative Synthesis</span>
                                    </div>
                                    <div className="flex-1 relative">
                                        <ChatWindow />
                                    </div>
                                </div>

                                {/* Right: Starter Position Scorecard */}
                                <div className="border border-white/10 rounded-lg overflow-hidden bg-bg-secondary flex flex-col">
                                    <div className="p-3 border-b border-white/10 bg-bg-tertiary flex items-center gap-2">
                                        <Target className="w-4 h-4 text-accent-gold" />
                                        <span className="text-xs font-bold text-white">Starter Position Scorecard</span>
                                    </div>
                                    
                                    {/* Pillars */}
                                    <div className="flex border-b border-white/10">
                                        <ScorecardPillar 
                                            label="Value" 
                                            value="--" 
                                            subtext="Enter ticker to analyze"
                                            status="neutral" 
                                        />
                                        <ScorecardPillar 
                                            label="Risk" 
                                            value="--" 
                                            subtext="Enter ticker to analyze"
                                            status="neutral" 
                                        />
                                        <ScorecardPillar 
                                            label="Momentum (TSMOM)" 
                                            value="--" 
                                            subtext="Enter ticker to analyze"
                                            status="neutral" 
                                        />
                                    </div>

                                    {/* Verdict */}
                                    <div className="p-6 flex-1 flex flex-col justify-center items-center text-center">
                                        <div className="text-xs text-text-muted mb-2 uppercase tracking-widest">Ready to Analyze</div>
                                        <p className="text-lg text-white font-display leading-relaxed">
                                            Enter a ticker symbol using the search above or click on a holding in your portfolio to begin your analysis. We'll provide a comprehensive evaluation combining Value, Risk, and Momentum metrics.
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {/* ZONE 2: The Evidence (Scroll for Details) */}
                            <div className="space-y-6">
                                <h4 className="text-sm font-display text-white border-b border-white/10 pb-2">Deep Dive Evidence</h4>
                                
                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                    <div className="h-[400px]">
                                        <RiskRewardWidget />
                                    </div>
                                    <div className="h-[400px]">
                                        <MonteCarloSimulation />
                                    </div>
                                </div>

                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                    <div className="h-[400px]">
                                        <ProjectionValuation />
                                    </div>
                                    <div className="h-[400px]">
                                        <ProtectedWidget featureName="Hidden Risk Scanner">
                                            <HiddenRiskReport />
                                        </ProtectedWidget>
                                    </div>
                                </div>
                            </div>

                            {/* ZONE 3: The Engine Room (System Logs) */}
                            <div className="border border-white/10 rounded-lg bg-[#050912] overflow-hidden">
                                <SectionHeader 
                                    title="System Logs (Engine Room)" 
                                    isOpen={showLogs} 
                                    onToggle={() => setShowLogs(!showLogs)} 
                                />
                                
                                {showLogs && (
                                    <div className="p-4 font-mono text-xs text-text-muted space-y-2">
                                        <div className="flex items-center gap-2 text-accent-primary mb-2">
                                            <Terminal className="w-3 h-3" />
                                            <span>Waiting for ticker input...</span>
                                        </div>
                                        <div>Enter a ticker symbol to see TSMOM calculations and system logs.</div>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {activeSidebarSection === 'valuation' && (
                        <div className="space-y-8">
                            <h2 className="text-lg font-display text-white border-b border-white/10 pb-2">Valuation Suite</h2>
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                <div className="h-[600px]">
                                    <ValuationWorkshop />
                                </div>
                                <div className="h-[600px]">
                                    <ValuationTool />
                                </div>
                            </div>
                            <div className="h-[400px]">
                                <RiskRewardWidget />
                            </div>
                        </div>
                    )}

                    {activeSidebarSection === 'screener' && (
                        <div className="space-y-8">
                            <h2 className="text-lg font-display text-white border-b border-white/10 pb-2">Idea Generation</h2>
                            
                            <div className="h-[500px]">
                                <ProtectedWidget featureName="Alpha Stock Picker">
                                    <AlphaStockPicker />
                                </ProtectedWidget>
                                <div className="mt-4 p-4 bg-bg-secondary border border-white/5 rounded-lg text-xs text-text-secondary">
                                    <strong className="text-white block mb-2">About the C-Score:</strong>
                                    <p className="mb-2">The C-Score is a proprietary multi-factor model that ranks stocks based on three key dimensions:</p>
                                    <ul className="list-disc list-inside space-y-1 ml-2">
                                        <li><strong>Quality:</strong> Measures business fundamentals including Return on Invested Capital (ROIC), profit margins, and operational efficiency. High-quality companies generate superior returns on capital.</li>
                                        <li><strong>Value:</strong> Assesses valuation attractiveness through Free Cash Flow (FCF) Yield and Enterprise Value to EBIT (EV/EBIT) ratios. Identifies companies trading below intrinsic value.</li>
                                        <li><strong>Momentum:</strong> Evaluates price strength and trend persistence. Captures stocks with positive momentum that may continue outperforming.</li>
                                    </ul>
                                    <p className="mt-2">A score above 80 indicates an "Investable" candidate with strong fundamentals, attractive valuation, and positive price momentum—a combination that historically outperforms the market.</p>
                                </div>
                            </div>

                            <div className="h-[500px]">
                                <ProtectedWidget featureName="Hidden Gems Screener">
                                    <HiddenGemsScreener />
                                </ProtectedWidget>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        );
    }

    // --- ASSET VIEW (Ticker Selected) ---
    return (
        <div className="flex gap-8 animate-fade-in h-[calc(100vh-100px)]">
            {/* Left Sidebar - Context */}
            <div className="w-64 hidden lg:block border-r border-white/10 pr-6 overflow-y-auto custom-scrollbar pb-20">
                <div className="space-y-8">
                    <div>
                        <div className="flex items-center justify-between mb-1">
                            <h3 className="text-2xl font-display text-white">{ticker}</h3>
                            <button onClick={() => setSearchParams({})} className="text-xs text-accent-primary hover:text-white">Change</button>
                        </div>
                        <div className="text-xs text-text-muted">Stock Analysis</div>
                        <p className="text-[10px] text-text-secondary mt-2 italic leading-relaxed">
                            "Challenge your conviction. Use the tools on the right to stress-test this asset before committing capital."
                        </p>
                    </div>

                    <div className="space-y-4">
                        <div>
                            <div className="text-[10px] text-text-muted uppercase tracking-widest mb-2">Valuation Details</div>
                            <div className="space-y-2 text-sm text-text-secondary">
                                <div className="flex justify-between"><span>P/E Ratio</span> <span className="text-white">--</span></div>
                                <div className="flex justify-between"><span>EV/EBITDA</span> <span className="text-white">--</span></div>
                                <div className="flex justify-between"><span>FCF Yield</span> <span className="text-white">--</span></div>
                            </div>
                        </div>

                        <div>
                            <div className="text-[10px] text-text-muted uppercase tracking-widest mb-2">Risk Profile</div>
                            <div className="space-y-2 text-sm text-text-secondary">
                                <div className="flex justify-between"><span>Beta</span> <span className="text-white">--</span></div>
                                <div className="flex justify-between"><span>Volatility</span> <span className="text-white">{(tsmomData?.annualized_volatility * 100)?.toFixed(1) || '--'}%</span></div>
                                <div className="flex justify-between"><span>Drawdown</span> <span className="text-negative">--</span></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content - Progressive Disclosure */}
            <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar space-y-8 pb-20">
                
                {/* ZONE 1: The Synthesis (Immediate Value) */}
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 min-h-[400px]">
                    {/* Left: Caria Chat */}
                    <div className="border border-white/10 rounded-lg overflow-hidden bg-bg-secondary flex flex-col">
                        <div className="p-3 border-b border-white/10 bg-bg-tertiary flex items-center gap-2">
                            <MessageSquare className="w-4 h-4 text-accent-primary" />
                            <span className="text-xs font-bold text-white">Qualitative Synthesis</span>
                        </div>
                        <div className="flex-1 relative">
                            <ChatWindow initialMessage={`Analyze ${ticker} for me. Is it a buy?`} />
                        </div>
                    </div>

                    {/* Right: Starter Position Scorecard */}
                    <div className="border border-white/10 rounded-lg overflow-hidden bg-bg-secondary flex flex-col">
                        <div className="p-3 border-b border-white/10 bg-bg-tertiary flex items-center gap-2">
                            <Target className="w-4 h-4 text-accent-gold" />
                            <span className="text-xs font-bold text-white">Starter Position Scorecard</span>
                        </div>
                        
                        {/* Pillars */}
                        <div className="flex border-b border-white/10">
                            <ScorecardPillar 
                                label="Value" 
                                value="--" 
                                subtext="Run Valuation Model"
                                status="neutral" 
                            />
                            <ScorecardPillar 
                                label="Risk" 
                                value="--" 
                                subtext="Check Risk Engine"
                                status="neutral" 
                            />
                            <ScorecardPillar 
                                label="Momentum (TSMOM)" 
                                value={tsmomData?.trend_direction || "Loading..."} 
                                subtext={tsmomData ? `12m Return: ${(tsmomData.trend_strength_12m * 100).toFixed(1)}%` : "Calculating..."}
                                status={tsmomData?.trend_direction === 'Bullish' ? 'positive' : 'negative'} 
                            />
                        </div>

                        {/* Verdict */}
                        <div className="p-6 flex-1 flex flex-col justify-center items-center text-center">
                            <div className="text-xs text-text-muted mb-2 uppercase tracking-widest">Algorithm Verdict</div>
                            <p className="text-lg text-white font-display leading-relaxed">
                                {tsmomData 
                                    ? "TSMOM signal generated. Combine with Fundamental Valuation (DCF) and Risk Assessment below to form a complete thesis." 
                                    : "Initializing TSMOM Engine..."}
                            </p>
                        </div>
                    </div>
                </div>

                {/* ZONE 2: The Evidence (Scroll for Details) */}
                <div className="space-y-6">
                    <h4 className="text-sm font-display text-white border-b border-white/10 pb-2">Deep Dive Evidence</h4>
                    
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div className="h-[400px]">
                            <RiskRewardWidget />
                        </div>
                        <div className="h-[400px]">
                            <MonteCarloSimulation />
                        </div>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div className="h-[400px]">
                            <ProjectionValuation />
                        </div>
                        <div className="h-[400px]">
                            <ProtectedWidget featureName="Hidden Risk Scanner">
                                <HiddenRiskReport />
                            </ProtectedWidget>
                        </div>
                    </div>
                </div>

                {/* ZONE 3: The Engine Room (System Logs) */}
                <div className="border border-white/10 rounded-lg bg-[#050912] overflow-hidden">
                    <SectionHeader 
                        title="System Logs (Engine Room)" 
                        isOpen={showLogs} 
                        onToggle={() => setShowLogs(!showLogs)} 
                    />
                    
                    {showLogs && (
                        <div className="p-4 font-mono text-xs text-text-muted space-y-2">
                            <div className="flex items-center gap-2 text-accent-primary mb-2">
                                <Terminal className="w-3 h-3" />
                                <span>TSMOM Calculation Log</span>
                            </div>
                            {tsmomData ? (
                                <>
                                    <div>[INFO] Fetching monthly adjusted prices for {ticker}... OK</div>
                                    <div>[CALC] Price(t): ${tsmomData.raw_data.current_price.toFixed(2)}</div>
                                    <div>[CALC] Price(t-12): ${tsmomData.raw_data.price_t_minus_12.toFixed(2)}</div>
                                    <div>[MATH] Excess Return r(t-12,t) = ({tsmomData.raw_data.current_price.toFixed(2)} / {tsmomData.raw_data.price_t_minus_12.toFixed(2)}) - 1 = {(tsmomData.trend_strength_12m * 100).toFixed(2)}%</div>
                                    <div>[CALC] Ex-ante Volatility (sigma_t): {(tsmomData.annualized_volatility * 100).toFixed(2)}% (Annualized)</div>
                                    <div>[LOGIC] Trend &gt; 0 ? YES. Volatility &gt; 30% ? {tsmomData.volatility_context === 'High' ? 'YES' : 'NO'}.</div>
                                    <div className="text-white">[RESULT] Signal: {tsmomData.trend_direction.toUpperCase()}</div>
                                </>
                            ) : (
                                <div>Waiting for data stream...</div>
                            )}
                        </div>
                    )}
                </div>

            </div>
        </div>
    );
};

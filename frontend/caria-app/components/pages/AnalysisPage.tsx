import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { MessageSquare, Activity, Target, ChevronDown, ChevronRight, Terminal } from 'lucide-react';
import { fetchWithAuth, API_BASE_URL, getToken } from '../../services/apiService';
import { ProjectionValuation } from '../widgets/ProjectionValuation';
import { CrisisSimulator } from '../widgets/CrisisSimulator';
import { RiskRewardWidget } from '../widgets/RiskRewardWidget';
import { ChatWindow } from '../ChatWindow';

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
    const [searchParams] = useSearchParams();
    const ticker = searchParams.get('ticker') || 'AAPL';
    const [tsmomData, setTsmomData] = useState<any>(null);
    const [showLogs, setShowLogs] = useState(false);

    useEffect(() => {
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

    return (
        <div className="flex gap-8 animate-fade-in h-[calc(100vh-100px)]">
            {/* Left Sidebar - Context */}
            <div className="w-64 hidden lg:block border-r border-white/10 pr-6">
                <div className="sticky top-24 space-y-8">
                    <div>
                        <h3 className="text-2xl font-display text-white mb-1">{ticker}</h3>
                        <div className="text-xs text-text-muted">Equity Asset</div>
                    </div>

                    <div className="space-y-4">
                        <div>
                            <div className="text-[10px] text-text-muted uppercase tracking-widest mb-2">Valuation Details</div>
                            <div className="space-y-2 text-sm text-text-secondary">
                                <div className="flex justify-between"><span>P/E Ratio</span> <span className="text-white">28.5x</span></div>
                                <div className="flex justify-between"><span>EV/EBITDA</span> <span className="text-white">22.1x</span></div>
                                <div className="flex justify-between"><span>FCF Yield</span> <span className="text-white">3.2%</span></div>
                            </div>
                        </div>

                        <div>
                            <div className="text-[10px] text-text-muted uppercase tracking-widest mb-2">Risk Profile</div>
                            <div className="space-y-2 text-sm text-text-secondary">
                                <div className="flex justify-between"><span>Beta</span> <span className="text-white">1.12</span></div>
                                <div className="flex justify-between"><span>Volatility</span> <span className="text-white">{(tsmomData?.annualized_volatility * 100)?.toFixed(1)}%</span></div>
                                <div className="flex justify-between"><span>Drawdown</span> <span className="text-negative">-12%</span></div>
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
                                value="Fair" 
                                subtext="Within 10% of fair value"
                                status="neutral" 
                            />
                            <ScorecardPillar 
                                label="Risk" 
                                value="Moderate" 
                                subtext="1.5 : 1 Reward/Risk"
                                status="neutral" 
                            />
                            <ScorecardPillar 
                                label="Momentum (TSMOM)" 
                                value={tsmomData?.trend_direction || "Loading..."} 
                                subtext={`12m Return: ${(tsmomData?.trend_strength_12m * 100)?.toFixed(1)}%`}
                                status={tsmomData?.trend_direction === 'Bullish' ? 'positive' : 'negative'} 
                            />
                        </div>

                        {/* Verdict */}
                        <div className="p-6 flex-1 flex flex-col justify-center items-center text-center">
                            <div className="text-xs text-text-muted mb-2 uppercase tracking-widest">Algorithm Verdict</div>
                            <p className="text-lg text-white font-display leading-relaxed">
                                "Fundamentals are solid but momentum is overheated. TSMOM indicates a positive trend, but volatility suggests caution. Consider a <strong>half-sized starter position</strong>."
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
                            <ProjectionValuation />
                        </div>
                    </div>

                    <div className="h-[400px]">
                        <CrisisSimulator />
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


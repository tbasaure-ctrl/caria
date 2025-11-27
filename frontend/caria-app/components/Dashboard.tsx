import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Portfolio } from './widgets/Portfolio';
import { ThesisIcon } from './Icons';
import { CommunityFeed } from './widgets/CommunityFeed';
import { RankingsWidget } from './widgets/RankingsWidget';
import { PortfolioAnalytics } from './widgets/PortfolioAnalytics';
import { RegimeTestWidget } from './widgets/RegimeTestWidget';
import { IndustryResearch } from './widgets/IndustryResearch';
import { ThesisArena } from './widgets/ThesisArena';
import { CrisisSimulator } from './widgets/CrisisSimulator';
import { MacroSimulator } from './widgets/MacroSimulator';
import { ValuationTool } from './widgets/ValuationTool';
import { ValuationWorkshop } from './widgets/ValuationWorkshop';
import { ProjectionValuation } from './widgets/ProjectionValuation';
import { AlphaStockPicker } from './widgets/AlphaStockPicker';
import { HiddenGemsScreener } from './widgets/HiddenGemsScreener';
import { WeeklyMedia } from './widgets/WeeklyMedia';
import { RedditSentiment } from './widgets/RedditSentiment';
import { OpportunityRadar } from './widgets/OpportunityRadar';
import { Resources } from './widgets/Resources';
import { GlobalMarketBar } from './widgets/GlobalMarketBar';
import { ModelOutlook } from './widgets/ModelOutlook';
import { FearGreedIndex } from './widgets/FearGreedIndex';
import { ProtectedWidget } from './ProtectedWidget';
import { fetchWithAuth, API_BASE_URL } from '../services/apiService';

// Analysis CTA Component
const AnalysisCTA: React.FC<{ onStartAnalysis: () => void; onEnterArena: () => void }> = ({ 
    onStartAnalysis, 
    onEnterArena 
}) => (
    <div
        className="rounded-lg p-8 transition-all duration-300 cursor-pointer group h-full flex flex-col justify-center"
        style={{
            backgroundColor: 'var(--color-bg-secondary)',
            border: '1px solid var(--color-border-subtle)',
        }}
        onClick={onStartAnalysis}
        onMouseEnter={(e) => {
            e.currentTarget.style.borderColor = 'var(--color-border-emphasis)';
        }}
        onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = 'var(--color-border-subtle)';
        }}
    >
        <div className="flex flex-col items-center text-center">
            <div 
                className="w-16 h-16 rounded-full flex items-center justify-center mb-6"
                style={{ backgroundColor: 'rgba(56, 189, 248, 0.1)' }}
            >
                <ThesisIcon className="w-8 h-8" style={{ color: 'var(--color-accent-primary)' }} />
            </div>
            
            <h3 
                className="text-2xl font-display font-medium mb-3 text-text-primary"
            >
                Challenge Your Thesis
            </h3>
            
            <p 
                className="text-base text-text-secondary leading-relaxed mb-8 max-w-md font-light"
            >
                Test your investment ideas against Caria's AI analysis. Uncover biases and strengthen your conviction.
            </p>
            
            <div className="flex flex-col gap-4 w-full max-w-xs">
                <button
                    onClick={(e) => {
                        e.stopPropagation();
                        onStartAnalysis();
                    }}
                    className="py-3 px-6 rounded-lg font-medium text-sm transition-all duration-300 bg-accent-primary text-white shadow-glow-sm hover:shadow-glow-md transform hover:-translate-y-0.5"
                >
                    Start Analysis
                </button>
                
                <button
                    onClick={(e) => {
                        e.stopPropagation();
                        onEnterArena();
                    }}
                    className="py-3 px-6 rounded-lg text-sm font-medium transition-all duration-300 text-text-secondary border border-white/10 hover:border-accent-primary/50 hover:text-white"
                >
                    Enter Thesis Arena →
                </button>
            </div>
        </div>
    </div>
);


interface DashboardProps {
    onStartAnalysis: () => void;
}

interface RegimeData {
    regime: string;
    confidence: number;
}

type DashboardTab = 'portfolio' | 'analysis' | 'research';

export const Dashboard: React.FC<DashboardProps> = ({ onStartAnalysis }) => {
    const [searchParams, setSearchParams] = useSearchParams();
    const [regimeData, setRegimeData] = useState<RegimeData | null>(null);
    const [isLoadingRegime, setIsLoadingRegime] = useState(true);
    const [showArena, setShowArena] = useState(false);
    
    // Get tab from URL or default to 'portfolio'
    const tabFromUrl = searchParams.get('tab') as DashboardTab;
    const [activeTab, setActiveTab] = useState<DashboardTab>(
        tabFromUrl && ['portfolio', 'analysis', 'research'].includes(tabFromUrl) 
            ? tabFromUrl 
            : 'portfolio'
    );

    // Update tab when URL changes
    useEffect(() => {
        const tabFromUrl = searchParams.get('tab') as DashboardTab;
        if (tabFromUrl && ['portfolio', 'analysis', 'research'].includes(tabFromUrl)) {
            setActiveTab(tabFromUrl);
        } else if (!tabFromUrl) {
            // If no tab in URL, set default to portfolio
            setSearchParams({ tab: 'portfolio' }, { replace: true });
        }
    }, [searchParams, setSearchParams]);

    // Update URL when tab changes
    const handleTabChange = (tab: DashboardTab) => {
        setActiveTab(tab);
        setSearchParams({ tab });
    };

    useEffect(() => {
        const fetchRegimeData = async () => {
            setIsLoadingRegime(true);
            try {
                const response = await fetchWithAuth(`${API_BASE_URL}/api/regime/current`);
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                const data = await response.json();
                setRegimeData({ regime: data.regime, confidence: data.confidence });
            } catch (error) {
                console.error("Failed to fetch regime data:", error);
                // Set to null so ModelOutlook shows default state
                setRegimeData(null);
            } finally {
                setIsLoadingRegime(false);
            }
        };

        fetchRegimeData();
    }, []);

    const tabs = [
        { id: 'portfolio' as DashboardTab, label: 'Portfolio' },
        { id: 'analysis' as DashboardTab, label: 'Analysis' },
        { id: 'research' as DashboardTab, label: 'Research' },
    ];

    return (
        <main 
            className="flex-1 overflow-y-auto relative min-h-screen bg-bg-primary"
        >
            {/* Dashboard Header */}
            <div 
                className="sticky top-0 z-40 border-b border-white/5 bg-bg-primary/95 backdrop-blur-xl"
            >
                <div className="w-full px-6 lg:px-12 py-4">
                    <div className="flex items-center justify-between">
                        <h1 className="text-2xl font-display text-white tracking-wide hidden md:block">
                            Terminal
                        </h1>
                        {/* Tab Navigation - Centered & Minimal */}
                        <div className="flex gap-8">
                            {tabs.map((tab) => (
                                <button
                                    key={tab.id}
                                    onClick={() => handleTabChange(tab.id)}
                                    className={`
                                        relative py-2 text-sm font-medium tracking-widest uppercase transition-colors duration-300
                                        ${activeTab === tab.id ? 'text-accent-cyan' : 'text-text-muted hover:text-text-secondary'}
                                    `}
                                >
                                    {tab.label}
                                    {activeTab === tab.id && (
                                        <div className="absolute -bottom-5 left-0 right-0 h-0.5 bg-accent-cyan shadow-glow-sm" />
                                    )}
                                </button>
                            ))}
                        </div>
                        <div className="w-24 hidden md:block"></div> {/* Spacer for centering */}
                    </div>
                </div>
            </div>

            {/* Tab Content - Expanded Layout */}
            <div className="w-full px-6 lg:px-12 py-12 relative z-10">
                
                {/* PORTFOLIO TAB */}
                {activeTab === 'portfolio' && (
                    <div className="space-y-16 animate-fade-in max-w-[1800px] mx-auto">
                        {/* Section: Market Overview */}
                        <section>
                            <div className="mb-8">
                                <h2 className="text-3xl font-display text-white mb-2">Market Overview</h2>
                                <p className="text-text-muted font-light">Global indices and macro sentiment</p>
                            </div>
                            
                            <div className="mb-8">
                                <GlobalMarketBar id="market-bar-widget" />
                            </div>
                            
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                                <div className="min-h-[400px]">
                                    <ModelOutlook regimeData={regimeData} isLoading={isLoadingRegime} />
                                </div>
                                <div className="min-h-[400px]">
                                    <FearGreedIndex />
                                </div>
                            </div>
                        </section>

                        <div className="h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent" />

                        {/* Section: Portfolio Management */}
                        <section>
                            <div className="mb-8">
                                <h2 className="text-3xl font-display text-white mb-2">Portfolio Management</h2>
                                <p className="text-text-muted font-light">Track holdings, allocation and performance</p>
                            </div>

                            <div className="grid grid-cols-1 xl:grid-cols-2 gap-10">
                                <div className="xl:col-span-1 min-h-[500px]">
                                    <ProtectedWidget featureName="Portfolio Management">
                                        <Portfolio id="portfolio-widget" />
                                    </ProtectedWidget>
                                </div>
                                <div className="xl:col-span-1 min-h-[500px]">
                                    <ProtectedWidget featureName="Portfolio Analytics">
                                        <PortfolioAnalytics />
                                    </ProtectedWidget>
                                </div>
                            </div>
                        </section>

                        <div className="h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent" />

                        {/* Section: Risk & Scenario */}
                        <section>
                            <div className="mb-8">
                                <h2 className="text-3xl font-display text-white mb-2">Risk & Scenarios</h2>
                                <p className="text-text-muted font-light">Stress testing and macro simulation</p>
                            </div>
                            
                            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                                <div className="lg:col-span-2 min-h-[450px]">
                                    <ProtectedWidget featureName="Crisis Simulator">
                                        <CrisisSimulator />
                                    </ProtectedWidget>
                                </div>
                                <div className="lg:col-span-1 min-h-[450px]">
                                    <MacroSimulator />
                                </div>
                            </div>
                            
                            <div className="mt-8">
                                <ProtectedWidget featureName="Regime Test">
                                    <RegimeTestWidget />
                                </ProtectedWidget>
                            </div>
                        </section>
                    </div>
                )}

                {/* ANALYSIS TAB */}
                {activeTab === 'analysis' && (
                    <div className="space-y-16 animate-fade-in max-w-[1800px] mx-auto">
                        {/* Section: Valuation Tools */}
                        <section>
                            <div className="mb-8">
                                <h2 className="text-3xl font-display text-white mb-2">Valuation Intelligence</h2>
                                <p className="text-text-muted font-light">Deep fundamental analysis and probabilistic forecasting</p>
                            </div>
                            
                            <div className="grid grid-cols-1 xl:grid-cols-2 gap-10">
                                <div className="bg-bg-secondary border border-white/5 rounded-lg p-8 min-h-[600px]">
                                    <h3 className="text-xl font-display text-white mb-1">DCF & Multiples</h3>
                                    <p className="text-sm text-text-muted mb-6">5-year projection model with risk adjustments</p>
                                    <ProjectionValuation />
                                </div>

                                <div className="bg-bg-secondary border border-white/5 rounded-lg p-8 min-h-[600px]">
                                    <h3 className="text-xl font-display text-white mb-1">Monte Carlo Simulation</h3>
                                    <p className="text-sm text-text-muted mb-6">Probabilistic price discovery based on volatility</p>
                                    <ValuationTool />
                                </div>
                            </div>
                        </section>

                        <div className="h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent" />

                        {/* Section: Investment Thesis */}
                        <section>
                            <div className="mb-8">
                                <h2 className="text-3xl font-display text-white mb-2">Thesis Arena</h2>
                                <p className="text-text-muted font-light">Challenge your conviction against AI counter-arguments</p>
                            </div>
                            
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                                <ProtectedWidget 
                                    featureName="Investment Thesis Analysis"
                                    description="Challenge your investment ideas against Caria's AI analysis."
                                >
                                    <AnalysisCTA
                                        onStartAnalysis={onStartAnalysis}
                                        onEnterArena={() => setShowArena(true)}
                                    />
                                </ProtectedWidget>
                                <div className="min-h-[400px]">
                                    <ProtectedWidget featureName="Valuation Workshop">
                                        <ValuationWorkshop />
                                    </ProtectedWidget>
                                </div>
                            </div>
                        </section>

                        <div className="h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent" />

                        {/* Section: Screeners */}
                        <section>
                            <div className="mb-8">
                                <h2 className="text-3xl font-display text-white mb-2">Discovery Engines</h2>
                                <p className="text-text-muted font-light">Algorithmic screening and hidden gem detection</p>
                            </div>
                            
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                                <ProtectedWidget featureName="Alpha Stock Picker">
                                    <AlphaStockPicker />
                                </ProtectedWidget>
                                <ProtectedWidget featureName="Hidden Gems Screener">
                                    <HiddenGemsScreener />
                                </ProtectedWidget>
                            </div>
                        </section>
                    </div>
                )}

                {/* RESEARCH TAB */}
                {activeTab === 'research' && (
                    <div className="space-y-16 animate-fade-in max-w-[1800px] mx-auto">
                        {/* Section: Market Signals */}
                        <section>
                            <div className="mb-8">
                                <h2 className="text-3xl font-display text-white mb-2">Market Signals</h2>
                                <p className="text-text-muted font-light">Real-time volume anomalies and social sentiment</p>
                            </div>
                            
                            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                                <div className="lg:col-span-2">
                                    <OpportunityRadar />
                                </div>
                                <div className="lg:col-span-1">
                                    <WeeklyMedia compact={false} />
                                </div>
                            </div>
                        </section>

                        <div className="h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent" />

                        {/* Section: Deep Research */}
                        <section>
                            <div className="mb-8">
                                <h2 className="text-3xl font-display text-white mb-2">Deep Research</h2>
                                <p className="text-text-muted font-light">Industry analysis and educational resources</p>
                            </div>
                            
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                                <ProtectedWidget featureName="Industry Research">
                                    <IndustryResearch />
                                </ProtectedWidget>
                                <Resources />
                            </div>
                        </section>

                        <div className="h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent" />

                        {/* Section: Community */}
                        <section>
                            <div className="mb-8">
                                <h2 className="text-3xl font-display text-white mb-2">Community Intelligence</h2>
                                <p className="text-text-muted font-light">Peer insights and rankings</p>
                            </div>
                            
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                                <ProtectedWidget featureName="Community">
                                    <CommunityFeed />
                                </ProtectedWidget>
                                <RankingsWidget />
                            </div>
                        </section>
                    </div>
                )}
            </div>

            {/* Thesis Arena Modal */}
            {showArena && (
                <div
                    className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-md"
                    onClick={() => setShowArena(false)}
                >
                    <div
                        className="rounded-xl max-w-5xl w-full max-h-[90vh] overflow-y-auto bg-bg-secondary border border-white/10 shadow-2xl"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="sticky top-0 flex justify-between items-center px-8 py-6 border-b border-white/5 bg-bg-secondary/95 backdrop-blur">
                            <h2 className="text-2xl font-display text-white">
                                Thesis Arena
                            </h2>
                            <button
                                onClick={() => setShowArena(false)}
                                className="w-8 h-8 rounded-full flex items-center justify-center text-xl text-text-muted hover:text-white hover:bg-white/5 transition-colors"
                            >
                                ×
                            </button>
                        </div>
                        <div className="p-8">
                            <ThesisArena onClose={() => setShowArena(false)} />
                        </div>
                    </div>
                </div>
            )}
        </main>
    );
};

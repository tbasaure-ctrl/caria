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
        className="rounded-lg p-8 transition-all duration-300 cursor-pointer group h-full flex flex-col justify-center items-center relative overflow-hidden"
    >
        {/* Background Glow */}
        <div className="absolute inset-0 bg-accent-cyan/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
        <div className="absolute inset-0 border border-white/5 group-hover:border-accent-cyan/30 transition-colors duration-500 rounded-lg" />

        <div 
            className="w-16 h-16 rounded-full flex items-center justify-center mb-6 bg-accent-cyan/10 group-hover:bg-accent-cyan/20 transition-colors relative z-10"
        >
            <ThesisIcon className="w-8 h-8 text-accent-cyan" />
        </div>
        
        <h3 className="text-2xl font-display font-medium mb-3 text-white text-center relative z-10">
            Challenge Your Thesis
        </h3>
        
        <p className="text-sm text-text-secondary leading-relaxed mb-8 max-w-md text-center font-light relative z-10">
            Test your investment ideas against Caria's AI analysis. Uncover biases and strengthen your conviction.
        </p>
        
        <div className="flex flex-col gap-4 w-full max-w-xs relative z-10">
            <button
                onClick={(e) => {
                    e.stopPropagation();
                    onStartAnalysis();
                }}
                className="py-3 px-6 rounded font-bold text-xs uppercase tracking-widest transition-all duration-300 bg-accent-primary text-white shadow-[0_0_15px_rgba(56,189,248,0.3)] hover:shadow-[0_0_25px_rgba(56,189,248,0.5)] transform hover:-translate-y-0.5"
            >
                Start Analysis
            </button>
            
            <button
                onClick={(e) => {
                    e.stopPropagation();
                    onEnterArena();
                }}
                className="py-3 px-6 rounded text-xs font-bold uppercase tracking-widest transition-all duration-300 text-text-muted border border-white/10 hover:border-accent-primary/50 hover:text-white"
            >
                Enter Thesis Arena
            </button>
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
    
    const tabFromUrl = searchParams.get('tab') as DashboardTab;
    const [activeTab, setActiveTab] = useState<DashboardTab>(
        tabFromUrl && ['portfolio', 'analysis', 'research'].includes(tabFromUrl) 
            ? tabFromUrl 
            : 'portfolio'
    );

    useEffect(() => {
        const tabFromUrl = searchParams.get('tab') as DashboardTab;
        if (tabFromUrl && ['portfolio', 'analysis', 'research'].includes(tabFromUrl)) {
            setActiveTab(tabFromUrl);
        } else if (!tabFromUrl) {
            setSearchParams({ tab: 'portfolio' }, { replace: true });
        }
    }, [searchParams, setSearchParams]);

    const handleTabChange = (tab: DashboardTab) => {
        setActiveTab(tab);
        setSearchParams({ tab });
    };

    useEffect(() => {
        const fetchRegimeData = async () => {
            setIsLoadingRegime(true);
            try {
                const response = await fetchWithAuth(`${API_BASE_URL}/api/regime/current`);
                if (!response.ok) throw new Error('Network response was not ok');
                const data = await response.json();
                setRegimeData({ regime: data.regime, confidence: data.confidence });
            } catch (error) {
                console.error("Failed to fetch regime data:", error);
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
        <main className="flex-1 overflow-y-auto relative min-h-screen bg-transparent custom-scrollbar">
            {/* Dashboard Header - Floating Glass Bar */}
            <div className="sticky top-0 z-40 border-b border-white/5 bg-[#020408]/80 backdrop-blur-xl">
                <div className="w-full px-6 lg:px-8 py-3 flex items-center justify-between">
                    
                    {/* Breadcrumb / Title */}
                    <div className="flex items-center gap-4">
                        <h1 className="text-xl font-display text-white tracking-wide hidden md:block">
                            Terminal
                        </h1>
                        <span className="text-white/20 text-xl font-light hidden md:block">/</span>
                        <span className="text-accent-cyan font-mono text-xs uppercase tracking-widest">
                            {activeTab}
                        </span>
                    </div>

                    {/* Tab Navigation - Capsules */}
                    <div className="flex bg-white/5 rounded-full p-1 border border-white/5">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => handleTabChange(tab.id)}
                                className={`
                                    px-6 py-1.5 rounded-full text-[10px] font-bold uppercase tracking-widest transition-all duration-300
                                    ${activeTab === tab.id 
                                        ? 'bg-accent-primary text-white shadow-[0_0_10px_rgba(56,189,248,0.4)]' 
                                        : 'text-text-muted hover:text-white hover:bg-white/5'
                                    }
                                `}
                            >
                                {tab.label}
                            </button>
                        ))}
                    </div>
                    
                    {/* Right Spacer or Actions */}
                    <div className="w-24 hidden md:flex justify-end items-center gap-3">
                        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]"></div>
                        <span className="text-[10px] text-text-muted font-mono uppercase">Live</span>
                    </div>
                </div>
            </div>

            {/* Tab Content - Ultra Dense Layout */}
            <div className="w-full px-4 lg:px-6 py-6 relative z-10 max-w-[2400px] mx-auto">
                
                {/* PORTFOLIO TAB */}
                {activeTab === 'portfolio' && (
                    <div className="grid grid-cols-12 gap-4 animate-fade-in">
                        {/* Row 1: Markets */}
                        <div className="col-span-12">
                            <GlobalMarketBar id="market-bar-widget" />
                        </div>

                        {/* Row 2: Indicators & Portfolio Main */}
                        <div className="col-span-12 lg:col-span-3 xl:col-span-2 space-y-4">
                            <div className="h-[200px]">
                                <ModelOutlook regimeData={regimeData} isLoading={isLoadingRegime} />
                            </div>
                            <div className="h-[200px]">
                                <FearGreedIndex />
                            </div>
                        </div>
                        
                        <div className="col-span-12 lg:col-span-9 xl:col-span-7 min-h-[600px]">
                            <ProtectedWidget featureName="Portfolio Management">
                                <Portfolio id="portfolio-widget" />
                            </ProtectedWidget>
                        </div>

                        <div className="col-span-12 lg:col-span-12 xl:col-span-3 min-h-[600px]">
                            <ProtectedWidget featureName="Portfolio Analytics">
                                <PortfolioAnalytics />
                            </ProtectedWidget>
                        </div>

                        {/* Row 3: Risk Tools */}
                        <div className="col-span-12 lg:col-span-8 min-h-[400px]">
                            <ProtectedWidget featureName="Crisis Simulator">
                                <CrisisSimulator />
                            </ProtectedWidget>
                        </div>
                        <div className="col-span-12 lg:col-span-4 space-y-4">
                            <div className="h-[190px]">
                                <MacroSimulator />
                            </div>
                            <div className="h-[190px]">
                                <ProtectedWidget featureName="Regime Test">
                                    <RegimeTestWidget />
                                </ProtectedWidget>
                            </div>
                        </div>
                    </div>
                )}

                {/* ANALYSIS TAB */}
                {activeTab === 'analysis' && (
                    <div className="grid grid-cols-12 gap-4 animate-fade-in">
                        <div className="col-span-12 lg:col-span-6 xl:col-span-4 min-h-[600px]">
                            <ProjectionValuation />
                        </div>
                        <div className="col-span-12 lg:col-span-6 xl:col-span-4 min-h-[600px]">
                            <ValuationTool />
                        </div>
                        
                        {/* Central Thesis Column */}
                        <div className="col-span-12 lg:col-span-12 xl:col-span-4 space-y-4">
                            <div className="h-[290px]">
                                <ProtectedWidget featureName="Investment Thesis Analysis">
                                    <AnalysisCTA onStartAnalysis={onStartAnalysis} onEnterArena={() => setShowArena(true)} />
                                </ProtectedWidget>
                            </div>
                            <div className="h-[290px]">
                                <ProtectedWidget featureName="Valuation Workshop">
                                    <ValuationWorkshop />
                                </ProtectedWidget>
                            </div>
                        </div>

                        {/* Bottom Screeners */}
                        <div className="col-span-12 lg:col-span-6 h-full">
                            <ProtectedWidget featureName="Alpha Stock Picker">
                                <AlphaStockPicker />
                            </ProtectedWidget>
                        </div>
                        <div className="col-span-12 lg:col-span-6 h-full">
                            <ProtectedWidget featureName="Hidden Gems Screener">
                                <HiddenGemsScreener />
                            </ProtectedWidget>
                        </div>
                    </div>
                )}

                {/* RESEARCH TAB */}
                {activeTab === 'research' && (
                    <div className="grid grid-cols-12 gap-4 animate-fade-in">
                        {/* Featured Research */}
                        <div className="col-span-12 xl:col-span-8">
                            <ProtectedWidget featureName="Industry Research">
                                <IndustryResearch />
                            </ProtectedWidget>
                        </div>
                        
                        {/* Right Column: Signals */}
                        <div className="col-span-12 xl:col-span-4 space-y-4">
                            <OpportunityRadar />
                            <WeeklyMedia compact={false} />
                        </div>

                        {/* Bottom Row */}
                        <div className="col-span-12 lg:col-span-4">
                            <ProtectedWidget featureName="Community">
                                <CommunityFeed />
                            </ProtectedWidget>
                        </div>
                        <div className="col-span-12 lg:col-span-4">
                            <Resources />
                        </div>
                        <div className="col-span-12 lg:col-span-4">
                            <RankingsWidget />
                        </div>
                    </div>
                )}
            </div>

            {/* Thesis Arena Modal */}
            {showArena && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-md" onClick={() => setShowArena(false)}>
                    <div className="rounded-xl max-w-5xl w-full max-h-[90vh] overflow-y-auto bg-[#050A14] border border-accent-cyan/30 shadow-[0_0_50px_rgba(34,211,238,0.1)] custom-scrollbar relative" onClick={(e) => e.stopPropagation()}>
                        {/* Decorative corner lines */}
                        <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-accent-cyan"></div>
                        <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-accent-cyan"></div>
                        <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-accent-cyan"></div>
                        <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-accent-cyan"></div>

                        <div className="sticky top-0 flex justify-between items-center px-8 py-6 border-b border-white/10 bg-[#050A14]/95 backdrop-blur z-10">
                            <h2 className="text-2xl font-display text-white tracking-wide">Thesis Arena</h2>
                            <button onClick={() => setShowArena(false)} className="w-8 h-8 rounded-full flex items-center justify-center text-xl text-text-muted hover:text-white hover:bg-white/5 transition-colors">×</button>
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

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
        className="rounded-lg p-8 transition-all duration-300 cursor-pointer group h-full flex flex-col justify-center items-center bg-[#0B1221] border border-white/5 hover:border-accent-cyan/30"
        onClick={onStartAnalysis}
    >
        <div 
            className="w-16 h-16 rounded-full flex items-center justify-center mb-6 bg-accent-cyan/10 group-hover:bg-accent-cyan/20 transition-colors"
        >
            <ThesisIcon className="w-8 h-8 text-accent-cyan" />
        </div>
        
        <h3 className="text-2xl font-display font-medium mb-3 text-white text-center">
            Challenge Your Thesis
        </h3>
        
        <p className="text-base text-text-secondary leading-relaxed mb-8 max-w-md text-center font-light">
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
        <main className="flex-1 overflow-y-auto relative min-h-screen bg-bg-primary custom-scrollbar">
            {/* Dashboard Header */}
            <div className="sticky top-0 z-40 border-b border-white/5 bg-bg-primary/95 backdrop-blur-xl">
                <div className="w-full px-6 lg:px-12 py-4">
                    <div className="flex items-center justify-between">
                        <h1 className="text-2xl font-display text-white tracking-wide hidden md:block">
                            Terminal
                        </h1>
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
                        <div className="w-24 hidden md:block"></div>
                    </div>
                </div>
            </div>

            {/* Tab Content - Compact Dense Layout */}
            <div className="w-full px-4 lg:px-8 py-8 relative z-10 max-w-[1920px] mx-auto">
                
                {/* PORTFOLIO TAB */}
                {activeTab === 'portfolio' && (
                    <div className="grid grid-cols-12 gap-6 animate-fade-in">
                        {/* Top Row: Global Market Bar (Span Full) */}
                        <div className="col-span-12">
                            <GlobalMarketBar id="market-bar-widget" />
                        </div>

                        {/* Row 2: Outlook & Fear (Span 6 each on LG) */}
                        <div className="col-span-12 lg:col-span-6 h-full min-h-[400px]">
                            <ModelOutlook regimeData={regimeData} isLoading={isLoadingRegime} />
                        </div>
                        <div className="col-span-12 lg:col-span-6 h-full min-h-[400px]">
                            <FearGreedIndex />
                        </div>

                        {/* Row 3: Portfolio Main (Span 7) & Analytics (Span 5) */}
                        <div className="col-span-12 lg:col-span-7 h-full min-h-[600px]">
                            <ProtectedWidget featureName="Portfolio Management">
                                <Portfolio id="portfolio-widget" />
                            </ProtectedWidget>
                        </div>
                        <div className="col-span-12 lg:col-span-5 h-full min-h-[600px]">
                            <ProtectedWidget featureName="Portfolio Analytics">
                                <PortfolioAnalytics />
                            </ProtectedWidget>
                        </div>

                        {/* Row 4: Risk Tools (Dense Grid) */}
                        <div className="col-span-12 lg:col-span-8 h-full min-h-[500px]">
                            <ProtectedWidget featureName="Crisis Simulator">
                                <CrisisSimulator />
                            </ProtectedWidget>
                        </div>
                        <div className="col-span-12 lg:col-span-4 space-y-6">
                            <div className="min-h-[240px]">
                                <MacroSimulator />
                            </div>
                            <div className="min-h-[240px]">
                                <ProtectedWidget featureName="Regime Test">
                                    <RegimeTestWidget />
                                </ProtectedWidget>
                            </div>
                        </div>
                    </div>
                )}

                {/* ANALYSIS TAB */}
                {activeTab === 'analysis' && (
                    <div className="grid grid-cols-12 gap-6 animate-fade-in">
                        <div className="col-span-12 lg:col-span-6 h-full min-h-[600px] bg-[#0B1221] border border-white/5 rounded-lg p-1">
                            <ProjectionValuation />
                        </div>
                        <div className="col-span-12 lg:col-span-6 h-full min-h-[600px] bg-[#0B1221] border border-white/5 rounded-lg p-1">
                            <ValuationTool />
                        </div>

                        <div className="col-span-12 lg:col-span-7 h-full">
                            <ProtectedWidget featureName="Investment Thesis Analysis">
                                <AnalysisCTA onStartAnalysis={onStartAnalysis} onEnterArena={() => setShowArena(true)} />
                            </ProtectedWidget>
                        </div>
                        <div className="col-span-12 lg:col-span-5 h-full">
                            <ProtectedWidget featureName="Valuation Workshop">
                                <ValuationWorkshop />
                            </ProtectedWidget>
                        </div>

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
                    <div className="grid grid-cols-12 gap-6 animate-fade-in">
                        {/* Industry Research - Featured (Full Width or Large) */}
                        <div className="col-span-12">
                            <ProtectedWidget featureName="Industry Research">
                                <IndustryResearch />
                            </ProtectedWidget>
                        </div>

                        {/* Signals & Media */}
                        <div className="col-span-12 lg:col-span-8 h-full">
                            <OpportunityRadar />
                        </div>
                        <div className="col-span-12 lg:col-span-4 h-full">
                            <WeeklyMedia compact={false} />
                        </div>

                        {/* Community & Resources */}
                        <div className="col-span-12 lg:col-span-6 h-full">
                            <ProtectedWidget featureName="Community">
                                <CommunityFeed />
                            </ProtectedWidget>
                        </div>
                        <div className="col-span-12 lg:col-span-6 h-full">
                            <Resources />
                        </div>
                        <div className="col-span-12">
                            <RankingsWidget />
                        </div>
                    </div>
                )}
            </div>

            {/* Thesis Arena Modal */}
            {showArena && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-md" onClick={() => setShowArena(false)}>
                    <div className="rounded-xl max-w-5xl w-full max-h-[90vh] overflow-y-auto bg-bg-secondary border border-white/10 shadow-2xl custom-scrollbar" onClick={(e) => e.stopPropagation()}>
                        <div className="sticky top-0 flex justify-between items-center px-8 py-6 border-b border-white/5 bg-bg-secondary/95 backdrop-blur">
                            <h2 className="text-2xl font-display text-white">Thesis Arena</h2>
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

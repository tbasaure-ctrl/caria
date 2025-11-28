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

// Ask Caria Widget (Satellite for Portfolio)
const AskCariaWidget: React.FC<{ onClick: () => void }> = ({ onClick }) => (
    <div
        className="rounded-lg p-6 cursor-pointer group h-full flex flex-col justify-between relative overflow-hidden bg-[#0B1221] border border-white/5 hover:border-accent-cyan/30 transition-all duration-300"
        onClick={onClick}
    >
        <div className="absolute top-0 right-0 p-4 opacity-20 group-hover:opacity-40 transition-opacity">
            <ThesisIcon className="w-12 h-12 text-accent-cyan" />
        </div>
        
        <div>
            <h3 className="text-lg font-display font-bold text-white mb-1">Ask Caria</h3>
            <p className="text-xs text-text-muted">Senior Partner AI</p>
        </div>
        
        <div className="mt-4">
            <p className="text-sm text-text-secondary leading-snug mb-4">
                "¿Debería vender Apple ahora?"
                <br/>
                "Analiza mi exposición a China."
            </p>
            <button className="text-xs font-bold uppercase tracking-widest text-accent-cyan flex items-center gap-2 group-hover:gap-3 transition-all">
                Start Chat <span>→</span>
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

    const switchToAnalysis = () => handleTabChange('analysis');

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
            {/* Dashboard Header */}
            <div className="sticky top-0 z-40 border-b border-white/5 bg-[#020408]/80 backdrop-blur-xl">
                <div className="w-full px-6 lg:px-12 py-6 flex flex-col md:flex-row items-center justify-between gap-6">
                    <div className="flex items-center gap-4 shrink-0">
                        <h1 className="text-xl font-display text-white tracking-wide hidden md:block">
                            Terminal
                        </h1>
                        <span className="text-white/20 text-xl font-light hidden md:block">/</span>
                        <span className="text-accent-cyan font-mono text-xs uppercase tracking-widest">
                            {activeTab}
                        </span>
                    </div>

                    <div className="flex bg-white/5 rounded-full p-1.5 border border-white/5 gap-2 overflow-x-auto max-w-full">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => handleTabChange(tab.id)}
                                className={`
                                    px-8 py-2 rounded-full text-xs font-bold uppercase tracking-widest transition-all duration-300 whitespace-nowrap
                                    ${activeTab === tab.id 
                                        ? 'bg-accent-primary text-white shadow-[0_0_15px_rgba(56,189,248,0.4)]' 
                                        : 'text-text-muted hover:text-white hover:bg-white/5'
                                    }
                                `}
                            >
                                {tab.label}
                            </button>
                        ))}
                    </div>
                    
                    <div className="w-24 hidden md:flex justify-end items-center gap-3 shrink-0">
                        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]"></div>
                        <span className="text-[10px] text-text-muted font-mono uppercase">Live</span>
                    </div>
                </div>
            </div>

            {/* Tab Content */}
            <div className="w-full px-4 lg:px-8 py-8 relative z-10 max-w-[2400px] mx-auto">
                
                {/* PORTFOLIO TAB */}
                {activeTab === 'portfolio' && (
                    <div className="grid grid-cols-12 gap-8 animate-fade-in">
                        <div className="col-span-12">
                            <GlobalMarketBar id="market-bar-widget" />
                        </div>

                        {/* Indicators */}
                        <div className="col-span-12 lg:col-span-3 xl:col-span-2 space-y-6">
                            <div className="h-[220px]">
                                <ModelOutlook regimeData={regimeData} isLoading={isLoadingRegime} />
                            </div>
                            <div className="h-[220px]">
                                <FearGreedIndex />
                            </div>
                        </div>
                        
                        <div className="col-span-12 lg:col-span-9 xl:col-span-7 min-h-[650px]">
                            <ProtectedWidget featureName="Portfolio Management">
                                <Portfolio id="portfolio-widget" />
                            </ProtectedWidget>
                        </div>

                        <div className="col-span-12 lg:col-span-12 xl:col-span-3 flex flex-col gap-6">
                            <div className="flex-1">
                                <ProtectedWidget featureName="Portfolio Analytics">
                                    <PortfolioAnalytics />
                                </ProtectedWidget>
                            </div>
                            <div className="h-[180px]">
                                <AskCariaWidget onClick={switchToAnalysis} />
                            </div>
                        </div>

                        <div className="col-span-12 lg:col-span-8 min-h-[450px]">
                            <ProtectedWidget featureName="Crisis Simulator">
                                <CrisisSimulator />
                            </ProtectedWidget>
                        </div>
                        <div className="col-span-12 lg:col-span-4 space-y-8">
                            <div className="min-h-[220px]">
                                <MacroSimulator />
                            </div>
                            <div className="min-h-[220px]">
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
                        {/* CARIA ARENA - HERO POSITION */}
                        <div className="col-span-12 h-[650px]">
                            <ProtectedWidget featureName="Investment Thesis Analysis">
                                <ThesisArena />
                            </ProtectedWidget>
                        </div>

                        {/* Valuation Tools */}
                        <div className="col-span-12 lg:col-span-6 min-h-[600px]">
                            <ProjectionValuation />
                        </div>
                        <div className="col-span-12 lg:col-span-6 min-h-[600px]">
                            <ValuationTool />
                        </div>
                        
                        {/* Screeners */}
                        <div className="col-span-12 lg:col-span-6 h-full min-h-[500px]">
                            <ProtectedWidget featureName="Alpha Stock Picker">
                                <AlphaStockPicker />
                            </ProtectedWidget>
                        </div>
                        <div className="col-span-12 lg:col-span-6 h-full min-h-[500px]">
                            <ProtectedWidget featureName="Hidden Gems Screener">
                                <HiddenGemsScreener />
                            </ProtectedWidget>
                        </div>
                    </div>
                )}

                {/* RESEARCH TAB */}
                {activeTab === 'research' && (
                    <div className="grid grid-cols-12 gap-8 animate-fade-in">
                        <div className="col-span-12 xl:col-span-8 min-h-[500px]">
                            <ProtectedWidget featureName="Industry Research">
                                <IndustryResearch />
                            </ProtectedWidget>
                        </div>
                        
                        <div className="col-span-12 xl:col-span-4 space-y-8">
                            <OpportunityRadar />
                            <WeeklyMedia compact={false} />
                        </div>

                        <div className="col-span-12 lg:col-span-6 min-h-[400px]">
                            <ProtectedWidget featureName="Community">
                                <CommunityFeed />
                            </ProtectedWidget>
                        </div>
                        <div className="col-span-12 lg:col-span-6 min-h-[400px]">
                            <Resources />
                        </div>
                        <div className="col-span-12">
                            <RankingsWidget />
                        </div>
                    </div>
                )}
            </div>
        </main>
    );
};

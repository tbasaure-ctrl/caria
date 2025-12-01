

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
import { RiskRewardWidget } from './widgets/RiskRewardWidget';
import { WeeklyMedia } from './widgets/WeeklyMedia';
import { OpportunityRadar } from './widgets/OpportunityRadar';
import { Resources } from './widgets/Resources';
import { GlobalMarketBar } from './widgets/GlobalMarketBar';
import { ModelOutlook } from './widgets/ModelOutlook';
import { FearGreedIndex } from './widgets/FearGreedIndex';
import LiquidityGauge from './LiquidityGauge';
import HydraulicStackWidget from './HydraulicStackWidget';
import TopologicalMRIWidget from './TopologicalMRIWidget';
import TutorialPanel from './TutorialPanel';
import { ProtectedWidget } from './ProtectedWidget';
import { fetchWithAuth, API_BASE_URL } from '../services/apiService';

// Ask Caria Widget (Satellite for Portfolio) - Enhanced UX
const AskCariaWidget: React.FC<{ onClick: () => void }> = ({ onClick }) => (
    <div
        className="rounded-lg p-4 sm:p-6 cursor-pointer group h-full flex flex-col justify-between relative overflow-hidden bg-[#0B1221] border border-white/5 hover:border-accent-cyan/30 transition-all duration-300"
        onClick={onClick}
    >
        <div className="absolute top-0 right-0 p-3 sm:p-4 opacity-20 group-hover:opacity-40 transition-opacity">
            <ThesisIcon className="w-8 h-8 sm:w-12 sm:h-12 text-accent-cyan" />
        </div>

        <div>
            <h3 className="text-base sm:text-lg font-display font-bold text-white mb-0.5 sm:mb-1">Ask Caria</h3>
            <p className="text-[10px] sm:text-xs text-text-muted">Tu Socio de Inversiones con IA</p>
        </div>

        <div className="mt-3 sm:mt-4">
            <div className="space-y-1.5 sm:space-y-2 mb-3 sm:mb-4">
                <p className="text-[10px] sm:text-xs text-text-secondary leading-snug flex items-start gap-1.5">
                    <span className="text-accent-cyan">→</span> Analiza tesis de inversión
                </p>
                <p className="text-[10px] sm:text-xs text-text-secondary leading-snug flex items-start gap-1.5">
                    <span className="text-accent-cyan">→</span> Calcula valoraciones
                </p>
                <p className="text-[10px] sm:text-xs text-text-secondary leading-snug flex items-start gap-1.5">
                    <span className="text-accent-cyan">→</span> Desafía tus ideas
                </p>
            </div>
            <button className="text-[10px] sm:text-xs font-bold uppercase tracking-widest text-accent-cyan flex items-center gap-1.5 sm:gap-2 group-hover:gap-2 sm:group-hover:gap-3 transition-all">
                Iniciar Chat <span>→</span>
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

type DashboardTab = 'portfolio' | 'analysis' | 'research' | 'tutorial';

export const Dashboard: React.FC<DashboardProps> = ({ onStartAnalysis }) => {
    const [searchParams, setSearchParams] = useSearchParams();
    const [regimeData, setRegimeData] = useState<RegimeData | null>(null);
    const [isLoadingRegime, setIsLoadingRegime] = useState(true);
    const [showTutorial, setShowTutorial] = useState(false);

    const tabFromUrl = searchParams.get('tab') as DashboardTab;
    const [activeTab, setActiveTab] = useState<DashboardTab>(
        tabFromUrl && ['portfolio', 'analysis', 'research', 'tutorial'].includes(tabFromUrl)
            ? tabFromUrl
            : 'portfolio'
    );

    useEffect(() => {
        const tabFromUrl = searchParams.get('tab') as DashboardTab;
        if (tabFromUrl && ['portfolio', 'analysis', 'research', 'tutorial'].includes(tabFromUrl)) {
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
        { id: 'portfolio' as DashboardTab, label: 'Command Center' },
        { id: 'analysis' as DashboardTab, label: 'Deep Dive' },
        { id: 'research' as DashboardTab, label: 'Intel & Research' },
        { id: 'tutorial' as DashboardTab, label: 'Tutorial' },
    ];

    return (
        <main className="flex-1 overflow-y-auto relative min-h-screen bg-transparent custom-scrollbar">
            {/* Dashboard Header - Mobile First */}
            <div className="sticky top-0 z-40 border-b border-white/5 bg-[#020408]/80 backdrop-blur-xl">
                <div className="w-full px-3 sm:px-6 lg:px-12 py-3 sm:py-6 flex flex-col sm:flex-row items-center justify-between gap-3 sm:gap-6">
                    <div className="flex items-center gap-2 sm:gap-4 shrink-0">
                        <h1 className="text-lg sm:text-xl font-display text-white tracking-wide hidden sm:block">
                            Caria Terminal
                        </h1>
                        <span className="text-white/20 text-xl font-light hidden sm:block">/</span>
                        <span className="text-accent-cyan font-mono text-[10px] sm:text-xs uppercase tracking-widest">
                            {activeTab.replace('-', ' ')}
                        </span>
                    </div>

                    {/* Tabs - Responsive sizing */}
                    <div className="flex bg-white/5 rounded-full p-1 sm:p-1.5 border border-white/5 gap-1 sm:gap-2 overflow-x-auto max-w-full scrollbar-hide">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => handleTabChange(tab.id)}
                                className={`
                                    px-4 sm:px-6 lg:px-8 py-1.5 sm:py-2 rounded-full text-[10px] sm:text-xs font-bold uppercase tracking-wider sm:tracking-widest transition-all duration-300 whitespace-nowrap
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

                    <div className="flex items-center gap-3 shrink-0">
                        <button
                            onClick={() => handleTabChange('tutorial')}
                            className="p-2 rounded-lg hover:bg-white/5 transition-colors"
                            title="Tutorial"
                        >
                            <svg className="w-5 h-5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <circle cx="12" cy="12" r="10" strokeWidth="2" />
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                                <circle cx="12" cy="17" r="0.5" fill="currentColor" strokeWidth="0" />
                            </svg>
                        </button>
                        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]"></div>
                        <span className="text-[10px] text-text-muted font-mono uppercase">System Online</span>
                    </div>
                </div>
            </div>

            {/* Tab Content - Mobile First with responsive padding */}
            <div className="w-full px-3 sm:px-4 lg:px-8 py-4 sm:py-6 lg:py-8 relative z-10 max-w-[2400px] mx-auto">

                {/* PORTFOLIO TAB (COMMAND CENTER) */}
                {activeTab === 'portfolio' && (
                    <div className="flex flex-col gap-6 animate-fade-in">
                        {/* Section 1: Market Vitals (Full Width) */}
                        <div className="w-full">
                            <GlobalMarketBar id="market-bar-widget" />
                        </div>

                        {/* Section 2: Advanced Models (The "Brain") */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <div className="h-[300px] lg:h-[350px]">
                                <HydraulicStackWidget />
                            </div>
                            <div className="h-[300px] lg:h-[350px]">
                                <TopologicalMRIWidget />
                            </div>
                        </div>

                        {/* Section 3: Portfolio Core */}
                        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
                            {/* Left: Portfolio & Analytics */}
                            <div className="col-span-1 xl:col-span-8 flex flex-col gap-6">
                                <div className="min-h-[500px]">
                                    <ProtectedWidget featureName="Portfolio Management">
                                        <Portfolio id="portfolio-widget" />
                                    </ProtectedWidget>
                                </div>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div className="h-[300px]">
                                        <ProtectedWidget featureName="Portfolio Analytics">
                                            <PortfolioAnalytics />
                                        </ProtectedWidget>
                                    </div>
                                    <div className="h-[300px]">
                                        <AskCariaWidget onClick={switchToAnalysis} />
                                    </div>
                                </div>
                            </div>

                            {/* Right: Risk & Simulation */}
                            <div className="col-span-1 xl:col-span-4 flex flex-col gap-6">
                                <div className="h-[200px]">
                                    <ModelOutlook regimeData={regimeData} isLoading={isLoadingRegime} />
                                </div>
                                <div className="h-[200px]">
                                    <FearGreedIndex />
                                </div>
                                <div className="h-[200px]">
                                    <LiquidityGauge />
                                </div>
                                <div className="flex-1 min-h-[400px]">
                                    <ProtectedWidget featureName="Crisis Simulator">
                                        <CrisisSimulator />
                                    </ProtectedWidget>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* ANALYSIS TAB (DEEP DIVE) */}
                {activeTab === 'analysis' && (
                    <div className="flex flex-col gap-6 animate-fade-in">
                        {/* Hero Section: Thesis & Risk */}
                        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
                            <div className="col-span-1 xl:col-span-7 min-h-[600px]">
                                <ProtectedWidget featureName="Investment Thesis Analysis">
                                    <ThesisArena />
                                </ProtectedWidget>
                            </div>
                            <div className="col-span-1 xl:col-span-5 min-h-[600px]">
                                <RiskRewardWidget />
                            </div>
                        </div>

                        {/* Valuation Suite */}
                        <h2 className="text-xl font-display text-white mt-8 mb-4 border-b border-white/10 pb-2">Valuation Suite</h2>
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <div className="flex flex-col gap-6">
                                <div className="h-[400px]">
                                    <div className="rounded-xl p-4 sm:p-6 h-full" style={{ backgroundColor: 'var(--color-bg-secondary)', border: '1px solid var(--color-border-subtle)' }}>
                                        <h3 className="text-base sm:text-lg font-display font-bold text-white mb-3 sm:mb-4">DCF Valuation: 5-Year Target Price</h3>
                                        <ProjectionValuation />
                                    </div>
                                </div>
                                <div className="h-[600px]">
                                    <ValuationTool />
                                </div>
                            </div>
                            <div className="h-full min-h-[600px]">
                                <ValuationWorkshop />
                            </div>
                        </div>

                        {/* Screeners */}
                        <h2 className="text-xl font-display text-white mt-8 mb-4 border-b border-white/10 pb-2">Idea Generation</h2>
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <div className="h-[500px]">
                                <ProtectedWidget featureName="Alpha Stock Picker">
                                    <AlphaStockPicker />
                                </ProtectedWidget>
                            </div>
                            <div className="h-[500px]">
                                <ProtectedWidget featureName="Hidden Gems Screener">
                                    <HiddenGemsScreener />
                                </ProtectedWidget>
                            </div>
                        </div>
                    </div>
                )}

                {/* RESEARCH TAB (INTEL) */}
                {activeTab === 'research' && (
                    <div className="flex flex-col gap-6 animate-fade-in">
                        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
                            <div className="col-span-1 xl:col-span-8 min-h-[600px]">
                                <ProtectedWidget featureName="Industry Research">
                                    <IndustryResearch />
                                </ProtectedWidget>
                            </div>
                            <div className="col-span-1 xl:col-span-4 flex flex-col gap-6">
                                <OpportunityRadar />
                                <WeeklyMedia compact={false} />
                            </div>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <div className="h-[400px]">
                                <ProtectedWidget featureName="Community">
                                    <CommunityFeed />
                                </ProtectedWidget>
                            </div>
                            <div className="h-[400px]">
                                <Resources />
                            </div>
                        </div>

                        <div className="w-full">
                            <RankingsWidget />
                        </div>
                    </div>
                )}

                {/* TUTORIAL TAB */}
                {activeTab === 'tutorial' && (
                    <div className="flex flex-col gap-6 animate-fade-in">
                        <TutorialPanel onClose={() => handleTabChange('portfolio')} />
                    </div>
                )}
            </div>
        </main>
    );
};

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
        className="rounded-xl p-6 transition-all duration-300 cursor-pointer group"
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
                className="w-14 h-14 rounded-xl flex items-center justify-center mb-5"
                style={{ backgroundColor: 'rgba(46, 124, 246, 0.12)' }}
            >
                <ThesisIcon className="w-7 h-7" style={{ color: 'var(--color-accent-primary)' }} />
            </div>
            
            <h3 
                className="text-xl font-semibold mb-2"
                style={{ 
                    fontFamily: 'var(--font-display)',
                    color: 'var(--color-text-primary)' 
                }}
            >
                Challenge Your Thesis
            </h3>
            
            <p 
                className="text-sm leading-relaxed mb-6 max-w-sm"
                style={{ color: 'var(--color-text-secondary)' }}
            >
                Test your investment ideas against Caria's AI analysis. Uncover biases and strengthen your conviction.
            </p>
            
            <div className="flex flex-col gap-3 w-full max-w-xs">
                <button
                    onClick={(e) => {
                        e.stopPropagation();
                        onStartAnalysis();
                    }}
                    className="py-3 px-6 rounded-lg font-semibold text-sm transition-all duration-200"
                    style={{
                        backgroundColor: 'var(--color-accent-primary)',
                        color: '#FFFFFF',
                    }}
                    onMouseEnter={(e) => {
                        e.currentTarget.style.transform = 'translateY(-2px)';
                        e.currentTarget.style.boxShadow = '0 4px 12px rgba(46, 124, 246, 0.3)';
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.transform = 'translateY(0)';
                        e.currentTarget.style.boxShadow = 'none';
                    }}
                >
                    Start Analysis
                </button>
                
                <button
                    onClick={(e) => {
                        e.stopPropagation();
                        onEnterArena();
                    }}
                    className="py-2 px-4 rounded-lg text-sm font-medium transition-all duration-200"
                    style={{
                        backgroundColor: 'transparent',
                        color: 'var(--color-text-secondary)',
                        border: '1px solid var(--color-border-subtle)',
                    }}
                    onMouseEnter={(e) => {
                        e.currentTarget.style.borderColor = 'var(--color-accent-primary)';
                        e.currentTarget.style.color = 'var(--color-accent-primary)';
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.borderColor = 'var(--color-border-subtle)';
                        e.currentTarget.style.color = 'var(--color-text-secondary)';
                    }}
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
            className="flex-1 overflow-y-auto relative"
            style={{ backgroundColor: 'var(--color-bg-primary)' }}
        >
            {/* Dashboard Header */}
            <div 
                className="sticky top-0 z-40 border-b relative"
                style={{ 
                    backgroundColor: 'rgba(10, 14, 20, 0.95)',
                    backdropFilter: 'blur(12px)',
                    WebkitBackdropFilter: 'blur(12px)',
                    borderColor: 'var(--color-border-subtle)'
                }}
            >
                <div className="max-w-[1800px] mx-auto px-6 lg:px-10">
                    {/* Tab Navigation - Bloomberg Style */}
                    <div className="flex gap-1 -mb-px">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => handleTabChange(tab.id)}
                                className="relative px-5 py-3 font-medium text-sm transition-all duration-200"
                                style={{
                                    color: activeTab === tab.id 
                                        ? 'var(--color-text-primary)' 
                                        : 'var(--color-text-muted)',
                                    backgroundColor: activeTab === tab.id 
                                        ? 'var(--color-bg-secondary)' 
                                        : 'transparent',
                                    borderTopLeftRadius: '8px',
                                    borderTopRightRadius: '8px',
                                }}
                            >
                                {tab.label}
                                {activeTab === tab.id && (
                                    <div 
                                        className="absolute bottom-0 left-0 right-0 h-0.5"
                                        style={{ backgroundColor: 'var(--color-accent-primary)' }}
                                    />
                                )}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Tab Content */}
            <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-10 relative z-10">
                
                {/* Tagline Banner */}
                <div 
                    className="mb-6 md:mb-8 text-center py-3 px-4 rounded-lg"
                    style={{ 
                        backgroundColor: 'rgba(74, 144, 226, 0.08)',
                        border: '1px solid rgba(74, 144, 226, 0.15)'
                    }}
                >
                    <p 
                        className="text-sm md:text-base"
                        style={{ 
                            color: 'rgba(232, 230, 227, 0.7)',
                            fontFamily: "'Crimson Pro', Georgia, serif",
                            fontStyle: 'italic'
                        }}
                    >
                        We don't intend to offer financial advice — we want to join your journey to financial freedom.
                    </p>
                </div>

                {/* PORTFOLIO TAB */}
                {activeTab === 'portfolio' && (
                    <div className="space-y-12 animate-fade-in">
                        {/* Section: Market Overview */}
                        <section>
                            <h2 
                                className="text-2xl font-bold mb-2"
                                style={{ 
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-text-primary)' 
                                }}
                            >
                                Market Overview
                            </h2>
                            <p 
                                className="text-sm mb-6"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Global indices and market sentiment
                            </p>
                            
                            <GlobalMarketBar id="market-bar-widget" />
                            
                            <div className="grid lg:grid-cols-2 gap-8 mt-8">
                                <ModelOutlook regimeData={regimeData} isLoading={isLoadingRegime} />
                                <FearGreedIndex />
                            </div>
                        </section>

                        {/* Divider */}
                        <hr style={{ borderColor: 'var(--color-border-subtle)', borderTopWidth: '1px' }} />

                        {/* Section: Portfolio Management */}
                        <section>
                            <h2 
                                className="text-2xl font-bold mb-2"
                                style={{ 
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-text-primary)' 
                                }}
                            >
                                Portfolio Management
                            </h2>
                            <p 
                                className="text-sm mb-6"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Track holdings and analyze performance
                            </p>

                            <div className="grid lg:grid-cols-2 gap-8">
                                <ProtectedWidget featureName="Portfolio Management">
                                    <Portfolio id="portfolio-widget" />
                                </ProtectedWidget>
                                <ProtectedWidget featureName="Portfolio Analytics">
                                    <PortfolioAnalytics />
                                </ProtectedWidget>
                            </div>
                        </section>

                        {/* Divider */}
                        <hr style={{ borderColor: 'var(--color-border-subtle)', borderTopWidth: '1px' }} />

                        {/* Section: Crisis Simulator */}
                        <section>
                            <h2 
                                className="text-2xl font-bold mb-2"
                                style={{ 
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-text-primary)' 
                                }}
                            >
                                Crisis Simulator
                            </h2>
                            <p 
                                className="text-sm mb-6"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Test your portfolio against historical crashes
                            </p>
                            
                            <ProtectedWidget featureName="Crisis Simulator">
                                <CrisisSimulator />
                            </ProtectedWidget>
                        </section>

                        {/* Divider */}
                        <hr style={{ borderColor: 'var(--color-border-subtle)', borderTopWidth: '1px' }} />

                        {/* Section: Scenario Analysis */}
                        <section>
                            <h2 
                                className="text-2xl font-bold mb-2"
                                style={{ 
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-text-primary)' 
                                }}
                            >
                                Scenario Analysis
                            </h2>
                            <p 
                                className="text-sm mb-6"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Macro simulations and regime detection
                            </p>
                            
                            <div className="grid lg:grid-cols-2 gap-8">
                                <MacroSimulator />
                                <ProtectedWidget featureName="Regime Test">
                                    <RegimeTestWidget />
                                </ProtectedWidget>
                            </div>
                        </section>
                    </div>
                )}

                {/* ANALYSIS TAB */}
                {activeTab === 'analysis' && (
                    <div className="space-y-12 animate-fade-in">
                        {/* Section: Valuation Tools */}
                        <section>
                            <h2 
                                className="text-2xl font-bold mb-2"
                                style={{ 
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-text-primary)' 
                                }}
                            >
                                Valuation Tools
                            </h2>
                            <p 
                                className="text-sm mb-6"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Fundamental analysis and price simulations
                            </p>
                            
                            <div className="grid lg:grid-cols-2 gap-8">
                                <div 
                                    className="rounded-xl p-6"
                                    style={{
                                        backgroundColor: 'var(--color-bg-secondary)',
                                        border: '1px solid var(--color-border-subtle)',
                                    }}
                                >
                                    <h3 
                                        className="text-base font-semibold mb-1"
                                        style={{ color: 'var(--color-text-primary)' }}
                                    >
                                        Valuation Analysis
                                    </h3>
                                    <p 
                                        className="text-xs mb-5"
                                        style={{ color: 'var(--color-text-muted)' }}
                                    >
                                        5-year projection model with risk adjustments
                                    </p>
                                    <ProjectionValuation />
                                </div>

                                <div 
                                    className="rounded-xl p-6"
                                    style={{
                                        backgroundColor: 'var(--color-bg-secondary)',
                                        border: '1px solid var(--color-border-subtle)',
                                    }}
                                >
                                    <h3 
                                        className="text-base font-semibold mb-1"
                                        style={{ color: 'var(--color-text-primary)' }}
                                    >
                                        Monte Carlo Forecast
                                    </h3>
                                    <p 
                                        className="text-xs mb-5"
                                        style={{ color: 'var(--color-text-muted)' }}
                                    >
                                        2-year price simulations based on volatility
                                    </p>
                                    <ValuationTool />
                                </div>
                            </div>
                        </section>

                        {/* Divider */}
                        <hr style={{ borderColor: 'var(--color-border-subtle)', borderTopWidth: '1px' }} />

                        {/* Section: Investment Thesis */}
                        <section>
                            <h2 
                                className="text-2xl font-bold mb-2"
                                style={{ 
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-text-primary)' 
                                }}
                            >
                                Investment Thesis
                            </h2>
                            <p 
                                className="text-sm mb-6"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Challenge your ideas and learn valuation
                            </p>
                            
                            <div className="grid lg:grid-cols-2 gap-8">
                                <ProtectedWidget 
                                    featureName="Investment Thesis Analysis"
                                    description="Challenge your investment ideas against Caria's AI analysis."
                                >
                                    <AnalysisCTA
                                        onStartAnalysis={onStartAnalysis}
                                        onEnterArena={() => setShowArena(true)}
                                    />
                                </ProtectedWidget>
                                <ProtectedWidget featureName="Valuation Workshop">
                                    <ValuationWorkshop />
                                </ProtectedWidget>
                            </div>
                        </section>

                        {/* Divider */}
                        <hr style={{ borderColor: 'var(--color-border-subtle)', borderTopWidth: '1px' }} />

                        {/* Section: Stock Screeners */}
                        <section>
                            <h2 
                                className="text-2xl font-bold mb-2"
                                style={{ 
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-text-primary)' 
                                }}
                            >
                                Stock Screeners
                            </h2>
                            <p 
                                className="text-sm mb-6"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Discover opportunities with curated screens
                            </p>
                            
                            <div className="grid lg:grid-cols-2 gap-8">
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
                    <div className="space-y-12 animate-fade-in">
                        {/* Section: Weekly Content */}
                        <section>
                            <h2 
                                className="text-2xl font-bold mb-2"
                                style={{ 
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-text-primary)' 
                                }}
                            >
                                Weekly Picks
                            </h2>
                            <p 
                                className="text-sm mb-6"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Curated content for the week
                            </p>
                            
                            <WeeklyMedia compact={false} />
                        </section>

                        {/* Divider */}
                        <hr style={{ borderColor: 'var(--color-border-subtle)', borderTopWidth: '1px' }} />

                        {/* Section: Industry Research */}
                        <section>
                            <h2 
                                className="text-2xl font-bold mb-2"
                                style={{ 
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-text-primary)' 
                                }}
                            >
                                Industry Research
                            </h2>
                            <p 
                                className="text-sm mb-6"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Sector deep dives and industry analysis
                            </p>
                            
                            <ProtectedWidget featureName="Industry Research">
                                <IndustryResearch />
                            </ProtectedWidget>
                        </section>

                        {/* Divider */}
                        <hr style={{ borderColor: 'var(--color-border-subtle)', borderTopWidth: '1px' }} />

                        {/* Section: Social & Resources */}
                        <section>
                            <h2 
                                className="text-2xl font-bold mb-2"
                                style={{ 
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-text-primary)' 
                                }}
                            >
                                Market Signals & Resources
                            </h2>
                            <p 
                                className="text-sm mb-6"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Real-time volume anomalies and learning materials
                            </p>
                            
                            <div className="grid lg:grid-cols-2 gap-8">
                                <OpportunityRadar />
                                <Resources />
                            </div>
                        </section>

                        {/* Divider */}
                        <hr style={{ borderColor: 'var(--color-border-subtle)', borderTopWidth: '1px' }} />

                        {/* Section: Community */}
                        <section>
                            <h2 
                                className="text-2xl font-bold mb-2"
                                style={{ 
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-text-primary)' 
                                }}
                            >
                                Community
                            </h2>
                            <p 
                                className="text-sm mb-6"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Connect with other investors
                            </p>
                            
                            <div className="grid lg:grid-cols-2 gap-8">
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
                    className="fixed inset-0 z-50 flex items-center justify-center p-4"
                    style={{ backgroundColor: 'rgba(0, 0, 0, 0.85)' }}
                    onClick={() => setShowArena(false)}
                >
                    <div
                        className="rounded-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto"
                        onClick={(e) => e.stopPropagation()}
                        style={{
                            backgroundColor: 'var(--color-bg-secondary)',
                            border: '1px solid var(--color-border-default)',
                        }}
                    >
                        <div 
                            className="sticky top-0 flex justify-between items-center px-6 py-5 border-b"
                            style={{ 
                                backgroundColor: 'var(--color-bg-secondary)',
                                borderColor: 'var(--color-border-subtle)'
                            }}
                        >
                            <h2
                                className="text-xl font-semibold"
                                style={{
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-text-primary)',
                                }}
                            >
                                Thesis Arena
                            </h2>
                            <button
                                onClick={() => setShowArena(false)}
                                className="w-8 h-8 rounded-lg flex items-center justify-center text-xl transition-colors"
                                style={{ 
                                    color: 'var(--color-text-muted)',
                                    backgroundColor: 'var(--color-bg-surface)'
                                }}
                            >
                                ×
                            </button>
                        </div>
                        <div className="p-6">
                            <ThesisArena onClose={() => setShowArena(false)} />
                        </div>
                    </div>
                </div>
            )}
        </main>
    );
};

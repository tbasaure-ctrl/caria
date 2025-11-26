import React, { useState, useEffect } from 'react';
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
import { LynchValuationTool } from './widgets/LynchValuationTool';
import { AlphaStockPicker } from './widgets/AlphaStockPicker';
import { HiddenGemsScreener } from './widgets/HiddenGemsScreener';
import { WeeklyMedia } from './widgets/WeeklyMedia';
import { RedditSentiment } from './widgets/RedditSentiment';
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
    const [regimeData, setRegimeData] = useState<RegimeData | null>(null);
    const [isLoadingRegime, setIsLoadingRegime] = useState(true);
    const [showArena, setShowArena] = useState(false);
    const [activeTab, setActiveTab] = useState<DashboardTab>('portfolio');

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
                setRegimeData({ regime: 'slowdown', confidence: 0 });
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
                    {/* Top Section */}
                    <div className="flex items-center justify-between py-5">
                        <div>
                            <h1 
                                className="text-2xl md:text-3xl font-bold"
                                style={{
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-text-primary)',
                                    letterSpacing: '-0.02em'
                                }}
                            >
                                Terminal
                            </h1>
                            <p 
                                className="text-sm mt-1"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                Professional investment intelligence
                            </p>
                        </div>
                        
                        {/* Quick Stats */}
                        <div className="hidden md:flex items-center gap-6">
                            <div className="text-right">
                                <div 
                                    className="text-xs font-medium uppercase tracking-wide"
                                    style={{ color: 'var(--color-text-muted)' }}
                                >
                                    Market Status
                                </div>
                                <div className="flex items-center gap-2 mt-1">
                                    <span 
                                        className="w-2 h-2 rounded-full animate-pulse"
                                        style={{ backgroundColor: 'var(--color-positive)' }}
                                    />
                                    <span 
                                        className="text-sm font-medium"
                                        style={{ color: 'var(--color-positive)' }}
                                    >
                                        Markets Open
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Tab Navigation - Bloomberg Style */}
                    <div className="flex gap-1 -mb-px">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
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
            <div className="max-w-[1800px] mx-auto px-6 lg:px-10 py-8 relative z-10">
                
                {/* PORTFOLIO TAB */}
                {activeTab === 'portfolio' && (
                    <div className="space-y-8 animate-fade-in">
                        {/* Market Overview Bar */}
                        <GlobalMarketBar id="market-bar-widget" />

                        {/* Market Indicators */}
                        <div className="grid lg:grid-cols-2 gap-6">
                            <ModelOutlook regimeData={regimeData} isLoading={isLoadingRegime} />
                            <FearGreedIndex />
                        </div>

                        {/* Portfolio Section Header */}
                        <div className="flex items-center gap-3 pt-4">
                            <div 
                                className="w-1 h-6 rounded-full"
                                style={{ backgroundColor: 'var(--color-accent-primary)' }}
                            />
                            <h2 
                                className="text-xl font-semibold"
                                style={{ 
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-text-primary)' 
                                }}
                            >
                                Portfolio Management
                            </h2>
                        </div>

                        {/* Portfolio Grid - Two Column */}
                        <div className="grid lg:grid-cols-2 gap-6">
                            <ProtectedWidget featureName="Portfolio Management">
                                <Portfolio id="portfolio-widget" />
                            </ProtectedWidget>
                            <ProtectedWidget featureName="Portfolio Analytics">
                                <PortfolioAnalytics />
                            </ProtectedWidget>
                        </div>

                        {/* Crisis Simulator - Below Portfolio Analysis */}
                        <div className="pt-8">
                            <ProtectedWidget featureName="Crisis Simulator">
                                <CrisisSimulator />
                            </ProtectedWidget>
                        </div>

                        {/* Scenario Analysis Section */}
                        <div className="pt-4">
                            <div className="flex items-center gap-3 mb-6">
                                <div 
                                    className="w-1 h-6 rounded-full"
                                    style={{ backgroundColor: 'var(--color-warning)' }}
                                />
                                <div>
                                    <h2 
                                        className="text-xl font-semibold"
                                        style={{ 
                                            fontFamily: 'var(--font-display)',
                                            color: 'var(--color-text-primary)' 
                                        }}
                                    >
                                        Scenario Analysis
                                    </h2>
                                    <p 
                                        className="text-sm mt-0.5"
                                        style={{ color: 'var(--color-text-muted)' }}
                                    >
                                        Stress test against historical crises and macro scenarios
                                    </p>
                                </div>
                            </div>
                            
                            <div className="grid lg:grid-cols-2 gap-6">
                                <MacroSimulator />
                                <ProtectedWidget featureName="Regime Test">
                                    <RegimeTestWidget />
                                </ProtectedWidget>
                            </div>
                        </div>
                    </div>
                )}

                {/* ANALYSIS TAB - Bloomberg Terminal Pane Style */}
                {activeTab === 'analysis' && (
                    <div className="space-y-8 animate-fade-in">
                        {/* Stock Screeners - Research Report Cards */}
                        <div className="grid lg:grid-cols-2 gap-6">
                            <ProtectedWidget featureName="Alpha Stock Picker">
                                <AlphaStockPicker />
                            </ProtectedWidget>
                            <ProtectedWidget featureName="Hidden Gems Screener">
                                <HiddenGemsScreener />
                            </ProtectedWidget>
                        </div>

                        {/* Valuation Tools Section */}
                        <div className="pt-4">
                            <div className="flex items-center gap-3 mb-6">
                                <div 
                                    className="w-1 h-6 rounded-full"
                                    style={{ backgroundColor: 'var(--color-positive)' }}
                                />
                                <div>
                                    <h2 
                                        className="text-xl font-semibold"
                                        style={{ 
                                            fontFamily: 'var(--font-display)',
                                            color: 'var(--color-text-primary)' 
                                        }}
                                    >
                                        Valuation Terminal
                                    </h2>
                                    <p 
                                        className="text-sm mt-0.5"
                                        style={{ color: 'var(--color-text-muted)' }}
                                    >
                                        DCF, Monte Carlo, and multi-factor analysis
                                    </p>
                                </div>
                            </div>
                            
                            <ProtectedWidget featureName="Valuation Tool">
                                <ValuationTool />
                            </ProtectedWidget>
                        </div>

                        {/* Thesis Testing */}
                        <div className="pt-4">
                            <div className="flex items-center gap-3 mb-6">
                                <div 
                                    className="w-1 h-6 rounded-full"
                                    style={{ backgroundColor: 'var(--color-accent-primary)' }}
                                />
                                <h2 
                                    className="text-xl font-semibold"
                                    style={{ 
                                        fontFamily: 'var(--font-display)',
                                        color: 'var(--color-text-primary)' 
                                    }}
                                >
                                    Investment Thesis
                                </h2>
                            </div>
                            
                            <div className="grid lg:grid-cols-2 gap-6">
                                <ProtectedWidget 
                                    featureName="Investment Thesis Analysis"
                                    description="Challenge your investment ideas against Caria's AI analysis. Uncover biases and strengthen your conviction."
                                >
                                    <AnalysisCTA
                                        onStartAnalysis={onStartAnalysis}
                                        onEnterArena={() => setShowArena(true)}
                                    />
                                </ProtectedWidget>
                                <ProtectedWidget featureName="Lynch Valuation Tool">
                                    <LynchValuationTool />
                                </ProtectedWidget>
                            </div>
                        </div>
                    </div>
                )}

                {/* RESEARCH TAB - Editorial Newsroom Style */}
                {activeTab === 'research' && (
                    <div className="space-y-8 animate-fade-in">
                        {/* Weekly Content */}
                        <WeeklyMedia compact={false} />

                        {/* Industry Research Section */}
                        <div className="pt-4">
                            <div className="flex items-center gap-3 mb-6">
                                <div 
                                    className="w-1 h-6 rounded-full"
                                    style={{ backgroundColor: 'var(--color-accent-primary)' }}
                                />
                                <div>
                                    <h2 
                                        className="text-xl font-semibold"
                                        style={{ 
                                            fontFamily: 'var(--font-display)',
                                            color: 'var(--color-text-primary)' 
                                        }}
                                    >
                                        Industry Research
                                    </h2>
                                    <p 
                                        className="text-sm mt-0.5"
                                        style={{ color: 'var(--color-text-muted)' }}
                                    >
                                        Sector deep dives and industry analysis
                                    </p>
                                </div>
                            </div>
                            
                            <ProtectedWidget featureName="Industry Research">
                                <IndustryResearch />
                            </ProtectedWidget>
                        </div>

                        {/* Two Column: Sentiment & Resources */}
                        <div className="grid lg:grid-cols-2 gap-6">
                            <RedditSentiment />
                            <Resources />
                        </div>

                        {/* Community Section */}
                        <div className="pt-4">
                            <div className="flex items-center gap-3 mb-6">
                                <div 
                                    className="w-1 h-6 rounded-full"
                                    style={{ backgroundColor: 'var(--color-accent-primary)' }}
                                />
                                <h2 
                                    className="text-xl font-semibold"
                                    style={{ 
                                        fontFamily: 'var(--font-display)',
                                        color: 'var(--color-text-primary)' 
                                    }}
                                >
                                    Community
                                </h2>
                            </div>
                            
                            <div className="grid lg:grid-cols-2 gap-6">
                                <ProtectedWidget featureName="Community">
                                    <CommunityFeed />
                                </ProtectedWidget>
                                <RankingsWidget />
                            </div>
                        </div>
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


import React, { useState, useEffect } from 'react';
import { Portfolio } from './widgets/Portfolio';
import { ThesisIcon } from './Icons';
import { CommunityFeed } from './widgets/CommunityFeed';
import { RankingsWidget } from './widgets/RankingsWidget';
import { PortfolioAnalytics } from './widgets/PortfolioAnalytics';
import { RegimeTestWidget } from './widgets/RegimeTestWidget';
import { ThesisArena } from './widgets/ThesisArena';
import { CrisisSimulator } from './widgets/CrisisSimulator';
import { MacroSimulator } from './widgets/MacroSimulator';
import { ValuationTool } from './widgets/ValuationTool';
import { AlphaStockPicker } from './widgets/AlphaStockPicker';
import { HiddenGemsScreener } from './widgets/HiddenGemsScreener';
import { WeeklyMedia } from './widgets/WeeklyMedia';
import { RedditSentiment } from './widgets/RedditSentiment';
import { Resources } from './widgets/Resources';
import { GlobalMarketBar } from './widgets/GlobalMarketBar';
import { ModelOutlook } from './widgets/ModelOutlook';
import { FearGreedIndex } from './widgets/FearGreedIndex';
import { ModelPortfolioWidget } from './widgets/ModelPortfolioWidget';
import { fetchWithAuth, API_BASE_URL } from '../services/apiService';

const StartAnalysisCTA: React.FC<{ onStartAnalysis: () => void; onEnterArena: () => void; id?: string }> = ({ onStartAnalysis, onEnterArena, id }) => (
    <div id={id}
        className="rounded-lg p-8 flex flex-col items-center justify-center text-center relative overflow-hidden group cursor-pointer transition-all duration-300"
        style={{
            backgroundColor: 'var(--color-bg-secondary)',
            border: '1px solid var(--color-bg-tertiary)',
            boxShadow: '0 4px 20px rgba(0,0,0,0.2)'
        }}
        onClick={onStartAnalysis}
        onMouseEnter={(e) => {
            e.currentTarget.style.borderColor = 'var(--color-primary)';
            e.currentTarget.style.transform = 'translateY(-2px)';
        }}
        onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = 'var(--color-bg-tertiary)';
            e.currentTarget.style.transform = 'translateY(0)';
        }}>
        {/* Background decoration */}
        <div className="absolute top-0 right-0 w-32 h-32 opacity-5"
            style={{ background: 'radial-gradient(circle, var(--color-primary) 0%, transparent 70%)' }}></div>

        <ThesisIcon className="w-14 h-14 mb-4" style={{ color: 'var(--color-secondary)' }} />
        <h3 className="text-2xl font-bold mb-3"
            style={{ fontFamily: 'var(--font-display)', color: 'var(--color-cream)' }}>
            Have an idea?
        </h3>
        <p className="mb-6 max-w-sm leading-relaxed"
            style={{
                fontFamily: 'var(--font-body)',
                color: 'var(--color-text-secondary)',
                fontSize: '0.95rem'
            }}>
            Challenge your investment thesis against Caria. Uncover cognitive biases and strengthen your rationale before you invest.
        </p>
        <div className="flex flex-col gap-3 w-full max-w-sm">
            <button
                onClick={onStartAnalysis}
                className="py-3 px-8 rounded-lg font-semibold transition-all duration-200"
                style={{
                    backgroundColor: 'var(--color-primary)',
                    color: 'var(--color-cream)',
                    fontFamily: 'var(--font-body)'
                }}
                onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = 'var(--color-primary-light)';
                    e.currentTarget.style.transform = 'scale(1.05)';
                }}
                onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'var(--color-primary)';
                    e.currentTarget.style.transform = 'scale(1)';
                }}
            >
                Let's Break It Down
            </button>
            <button
                onClick={(e) => {
                    e.stopPropagation();
                    onEnterArena();
                }}
                className="py-2 px-6 rounded-lg text-sm font-medium transition-all duration-200"
                style={{
                    backgroundColor: 'transparent',
                    color: 'var(--color-text-secondary)',
                    border: '1px solid var(--color-bg-tertiary)',
                    fontFamily: 'var(--font-body)'
                }}
                onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = 'var(--color-primary)';
                    e.currentTarget.style.color = 'var(--color-primary)';
                }}
                onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'var(--color-bg-tertiary)';
                    e.currentTarget.style.color = 'var(--color-text-secondary)';
                }}
            >
                Want deeper analysis? Enter Arena →
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
    const [regimeData, setRegimeData] = useState<RegimeData | null>(null);
    const [isLoadingRegime, setIsLoadingRegime] = useState(true);
    const [showArena, setShowArena] = useState(false);
    const [activeTab, setActiveTab] = useState<DashboardTab>('portfolio');

    useEffect(() => {
        const fetchRegimeData = async () => {
            setIsLoadingRegime(true);
            try {
                // Use centralized API_BASE_URL per audit document
                const response = await fetchWithAuth(`${API_BASE_URL}/api/regime/current`);
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                const data = await response.json();
                setRegimeData({ regime: data.regime, confidence: data.confidence });
            } catch (error) {
                console.error("Failed to fetch regime data:", error);
                setRegimeData({ regime: 'slowdown', confidence: 0 }); // Fallback
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
        <main className="flex-1 overflow-y-auto p-4 md:p-8 max-w-[1600px] mx-auto"
            style={{ backgroundColor: 'var(--color-bg-primary)' }}>

            {/* Dashboard Header */}
            <div className="mb-6 fade-in">
                <h1 className="text-3xl md:text-5xl font-black mb-2"
                    style={{
                        fontFamily: "'Instrument Serif', Georgia, serif",
                        color: 'var(--color-cream)',
                        letterSpacing: '-0.02em'
                    }}>
                    Investment Dashboard
                </h1>
                <p className="text-base md:text-lg"
                    style={{
                        fontFamily: "'Crimson Pro', Georgia, serif",
                        color: 'rgba(232, 230, 227, 0.6)',
                        lineHeight: '1.6'
                    }}>
                    Your comprehensive view of market insights and portfolio analysis
                </p>
            </div>

            {/* Tab Navigation */}
            <div className="mb-8 fade-in delay-100">
                <div className="flex flex-wrap gap-3 md:gap-4 border-b pb-4"
                    style={{ borderColor: 'rgba(74, 144, 226, 0.2)' }}>
                    {tabs.map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className="px-4 md:px-6 py-2 md:py-3 rounded-lg font-semibold text-sm md:text-base transition-all duration-300"
                            style={{
                                backgroundColor: activeTab === tab.id ? 'rgba(74, 144, 226, 0.2)' : 'transparent',
                                color: activeTab === tab.id ? 'var(--color-blue-light)' : 'rgba(232, 230, 227, 0.6)',
                                border: `1px solid ${activeTab === tab.id ? 'rgba(74, 144, 226, 0.4)' : 'transparent'}`,
                                fontFamily: "'Crimson Pro', Georgia, serif",
                            }}
                            onMouseEnter={(e) => {
                                if (activeTab !== tab.id) {
                                    e.currentTarget.style.backgroundColor = 'rgba(74, 144, 226, 0.1)';
                                    e.currentTarget.style.color = 'var(--color-cream)';
                                }
                            }}
                            onMouseLeave={(e) => {
                                if (activeTab !== tab.id) {
                                    e.currentTarget.style.backgroundColor = 'transparent';
                                    e.currentTarget.style.color = 'rgba(232, 230, 227, 0.6)';
                                }
                            }}
                        >
                            {tab.label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Tab Content */}
            <div className="fade-in delay-200" style={{ padding: '2rem 1rem', maxWidth: '1600px', margin: '0 auto' }}>
                {/* PORTFOLIO TAB - Market overview, portfolio management, and stress testing */}
                {activeTab === 'portfolio' && (
                    <div className="space-y-10">
                        {/* Global Market Bar - Full Width */}
                        <GlobalMarketBar id="market-bar-widget" />

                        {/* Market Indicators Row */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                            <ModelOutlook regimeData={regimeData} isLoading={isLoadingRegime} />
                            <FearGreedIndex />
                        </div>

                        {/* Portfolio Management */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                            <Portfolio id="portfolio-widget" />
                            <PortfolioAnalytics />
                        </div>

                        <div className="grid grid-cols-1 gap-10">
                            <ModelPortfolioWidget />
                        </div>

                        {/* Scenario Analysis - Stress test your portfolio */}
                        <div className="mt-6">
                            <h2 className="text-xl font-bold mb-5" style={{ color: 'var(--color-cream)', fontFamily: "'Instrument Serif', serif" }}>
                                Scenario Analysis
                            </h2>
                            <p className="text-sm text-slate-400 mb-6">Stress test your portfolio against historical crises and macroeconomic scenarios</p>
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                                <CrisisSimulator />
                                <MacroSimulator />
                            </div>
                        </div>
                    </div>
                )}

                {/* ANALYSIS TAB - Screeners, Quick Valuation, Investment Thesis */}
                {activeTab === 'analysis' && (
                    <div className="space-y-10">
                        {/* Stock Screeners - Two Column Layout */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                            {/* Alpha Stock Picker - Top picks by CAS score */}
                            <AlphaStockPicker />
                            
                            {/* Hidden Gems Screener - Under the radar mid-caps */}
                            <HiddenGemsScreener />
                        </div>

                        {/* Quick Valuation with Monte Carlo */}
                        <ValuationTool />

                        {/* Investment Thesis with Caria */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                            <StartAnalysisCTA
                                onStartAnalysis={onStartAnalysis}
                                onEnterArena={() => setShowArena(true)}
                                id="analysis-cta-widget"
                            />
                            <RegimeTestWidget />
                        </div>
                    </div>
                )}

                {/* RESEARCH TAB - Weekly Video, Weekly Podcast, Reddit, Lectures, Community */}
                {activeTab === 'research' && (
                    <div className="space-y-10">
                        {/* Weekly Video and Podcast */}
                        <WeeklyMedia compact={false} />

                        {/* Two Column Layout: Reddit Feed and Recommended Lectures */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                            <RedditSentiment />
                            <Resources />
                        </div>

                        {/* Community Section - At Bottom */}
                        <div>
                            <h2 className="text-2xl font-bold mb-5"
                                style={{
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-cream)'
                                }}>
                                Community
                            </h2>
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                                <CommunityFeed />
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
                    style={{ backgroundColor: 'rgba(0, 0, 0, 0.7)' }}
                    onClick={() => setShowArena(false)}
                >
                    <div
                        className="bg-gray-900 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto p-6"
                        onClick={(e) => e.stopPropagation()}
                        style={{
                            backgroundColor: 'var(--color-bg-primary)',
                            border: '1px solid var(--color-bg-tertiary)',
                        }}
                    >
                        <div className="flex justify-between items-center mb-4">
                            <h2
                                className="text-2xl font-bold"
                                style={{
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-cream)',
                                }}
                            >
                                Thesis Arena
                            </h2>
                            <button
                                onClick={() => setShowArena(false)}
                                className="text-2xl font-bold"
                                style={{ color: 'var(--color-text-secondary)' }}
                            >
                                ×
                            </button>
                        </div>
                        <ThesisArena onClose={() => setShowArena(false)} />
                    </div>
                </div>
            )}
        </main>
    );
};

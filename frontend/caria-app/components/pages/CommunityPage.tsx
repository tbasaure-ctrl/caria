import React, { useState } from 'react';
import { IndustryResearch } from '../widgets/IndustryResearch';
import { WeeklyMedia } from '../widgets/WeeklyMedia';
import { OpportunityRadar } from '../widgets/OpportunityRadar';
import { CommunityFeed } from '../widgets/CommunityFeed';
import { Resources } from '../widgets/Resources';
import { ProtectedWidget } from '../ProtectedWidget';
import { ModelOutlook } from '../widgets/ModelOutlook';
import HydraulicStackWidget from '../HydraulicStackWidget';
import TopologicalMRIWidget from '../TopologicalMRIWidget';
import { TSMOMOverviewWidget } from '../widgets/TSMOMOverviewWidget';
import { RankingsWidget } from '../widgets/RankingsWidget';

export const CommunityPage: React.FC = () => {
    const [activeSection, setActiveSection] = useState<'main' | 'models'>('main');

    return (
        <div className="flex gap-8 h-[calc(100vh-100px)] animate-fade-in">
            {/* Sidebar */}
            <div className="w-64 border-r border-white/10 pr-6 hidden md:block">
                <div className="space-y-6">
                    <div>
                        <h3 className="text-lg font-display text-white mb-2">Research Center</h3>
                        <p className="text-xs text-text-secondary leading-relaxed">
                            Curated intelligence, community insights, and proprietary quantitative models.
                        </p>
                    </div>
                    <div className="space-y-1">
                        <button
                            onClick={() => setActiveSection('main')}
                            className={`w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors ${activeSection === 'main' ? 'bg-white/10 text-white' : 'text-text-muted hover:text-white hover:bg-white/5'}`}
                        >
                            Main Research
                        </button>
                        <button
                            onClick={() => setActiveSection('models')}
                            className={`w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors ${activeSection === 'models' ? 'bg-white/10 text-white' : 'text-text-muted hover:text-white hover:bg-white/5'}`}
                        >
                            Our Models
                        </button>
                    </div>
                </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 pb-20">

                {/* MAIN RESEARCH */}
                {activeSection === 'main' && (
                    <div className="space-y-8">
                        {/* Hero Section: Industry & Media */}
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

                        {/* Community & Resources */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <div className="flex flex-col gap-6">
                                <div className="h-[400px]">
                                    <ProtectedWidget featureName="Community">
                                        <CommunityFeed />
                                    </ProtectedWidget>
                                </div>
                            </div>
                            <div className="flex flex-col gap-6">
                                <div className="min-h-[400px] flex-1">
                                    <Resources />
                                </div>
                                <div className="min-h-[400px]">
                                    <RankingsWidget />
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* OUR MODELS */}
                {activeSection === 'models' && (
                    <div className="space-y-12">
                        {/* Intro */}
                        <div className="max-w-3xl">
                            <h2 className="text-2xl font-display text-white mb-4">Quantitative Models</h2>
                            <p className="text-sm text-text-secondary leading-relaxed">
                                This is our "innovation lab." We use these proprietary models to gauge market health, liquidity flows, and structural fragility. They serve as the macro overlay for all individual stock analysis.
                            </p>
                        </div>

                        {/* TSMOM Overview */}
                        <div>
                            <div className="flex items-center justify-between mb-4">
                                <div>
                                    <h3 className="text-lg font-display text-white">TSMOM Global Matrix</h3>
                                    <p className="text-xs text-text-muted">Time Series Momentum across asset classes.</p>
                                </div>
                            </div>
                            <TSMOMOverviewWidget />
                            <div className="mt-4 p-4 bg-bg-secondary border border-white/5 rounded-lg text-xs text-text-secondary">
                                <strong className="text-white block mb-1">How it works:</strong>
                                Based on Moskowitz (2012), we calculate the 12-month excess return adjusted for ex-ante volatility. A "Bullish" signal implies a positive trend, while "High Risk Bullish" warns that while the trend is up, volatility is reaching dangerous levels (indicating a potential crowded trade or bubble).
                            </div>
                        </div>

                        {/* Hydraulic Stack */}
                        <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                            <div>
                                <div className="h-[400px]">
                                    <HydraulicStackWidget />
                                </div>
                                <div className="mt-4 p-4 bg-bg-secondary border border-white/5 rounded-lg text-xs text-text-secondary">
                                    <strong className="text-white block mb-1">The Hydraulic Stack:</strong>
                                    A macro-liquidity engine that tracks Net Liquidity (Fed Balance Sheet - TGA - RRP). It determines if the monetary environment is expansionary (Risk-On) or contractionary (Risk-Off).
                                </div>
                            </div>

                            {/* Caria Cortex */}
                            <div>
                                <div className="h-[400px]">
                                    <TopologicalMRIWidget />
                                </div>
                                <div className="mt-4 p-4 bg-bg-secondary border border-white/5 rounded-lg text-xs text-text-secondary">
                                    <strong className="text-white block mb-1">Caria Cortex (Topological MRI):</strong>
                                    Uses Topological Data Analysis (TDA) to map the geometry of market correlations. It detects "Alien" stocks—assets that have decoupled from the market physics, often signaling imminent mean reversion or a structural break.
                                </div>
                            </div>
                        </div>

                        {/* Model Outlook */}
                        <div className="max-w-md">
                            <h3 className="text-lg font-display text-white mb-4">Regime Outlook</h3>
                            <div className="h-[250px]">
                                <ModelOutlook regimeData={null} isLoading={false} />
                            </div>
                            <div className="mt-4 p-4 bg-bg-secondary border border-white/5 rounded-lg text-xs text-text-secondary">
                                <strong className="text-white block mb-1">HMM Regime Detection:</strong>
                                A Hidden Markov Model that probabilistically determines the current economic state (Expansion, Slowdown, Recession, Stress) based on yield curves and credit spreads.
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

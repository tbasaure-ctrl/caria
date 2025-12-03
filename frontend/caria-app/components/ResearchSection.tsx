import React from 'react';
import { ValuationTool } from './widgets/ValuationTool';
import { Resources } from './widgets/Resources';
import { WeeklyMedia } from './widgets/WeeklyMedia';
import { AlphaStockPicker } from './widgets/AlphaStockPicker';
import { UnderTheRadarScreener } from './widgets/UnderTheRadarScreener';
import { IndustryResearch } from './widgets/IndustryResearch';
import { GlobalEconomicMonitor } from './widgets/GlobalEconomicMonitor';

/**
 * ResearchSection - Professional research hub with Economist-style editorial design
 * 
 * Features:
 * - Clean, text-focused layout (no icons)
 * - Serif typography for professional feel
 * - Educational resources and curated content
 */
export const ResearchSection: React.FC = () => {
    return (
        <div className="fade-in space-y-12">
            {/* Section Header - Editorial Style */}
            <div className="border-b border-white/10 pb-6">
                <h2 className="text-4xl font-serif text-white mb-2">Research & Intelligence</h2>
                <p className="text-white/60 text-lg font-serif italic">
                    Curated insights for the discerning investor
                </p>
            </div>

            {/* Weekly Media - Featured Content */}
            <div>
                <h3 className="text-2xl font-serif text-white mb-4 border-l-4 border-yellow-500 pl-4">
                    Featured This Week
                </h3>
                <WeeklyMedia compact={false} />
            </div>

            {/* Industry Analysis */}
            <div>
                <h3 className="text-2xl font-serif text-white mb-4 border-l-4 border-blue-500 pl-4">
                    Industry Deep Dives
                </h3>
                <IndustryResearch />
            </div>

            {/* Global Economic Monitor */}
            <div>
                <h3 className="text-2xl font-serif text-white mb-4 border-l-4 border-cyan-500 pl-4">
                    Global Economic Monitor
                </h3>
                <GlobalEconomicMonitor />
            </div>

            {/* Stock Screening */}
            <div>
                <h3 className="text-2xl font-serif text-white mb-4 border-l-4 border-green-500 pl-4">
                    Investment Ideas
                </h3>
                <div className="space-y-8">
                    <UnderTheRadarScreener />
                    <AlphaStockPicker />
                </div>
            </div>

            {/* Valuation Tools */}
            <div>
                <h3 className="text-2xl font-serif text-white mb-4 border-l-4 border-purple-500 pl-4">
                    Valuation Analysis
                </h3>
                <ValuationTool />
            </div>

            {/* Resources - Educational Content */}
            <div>
                <h3 className="text-2xl font-serif text-white mb-4 border-l-4 border-orange-500 pl-4">
                    Learning Resources
                </h3>
                <Resources />
            </div>
        </div>
    );
};


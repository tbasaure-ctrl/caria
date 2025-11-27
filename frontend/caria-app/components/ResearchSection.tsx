import React from 'react';
import { ValuationTool } from './widgets/ValuationTool';
import { RedditSentiment } from './widgets/RedditSentiment';
import { Resources } from './widgets/Resources';
import { WeeklyMedia } from './widgets/WeeklyMedia';
import { AlphaStockPicker } from './widgets/AlphaStockPicker';
import { UnderTheRadarScreener } from './widgets/UnderTheRadarScreener';
import { IndustryResearch } from './widgets/IndustryResearch';

/**
 * ResearchSection - Groups related research and analysis widgets together
 * 
 * This component provides a cohesive "Research & Analysis" section that includes:
 * - ValuationTool: Quick multiples valuation and Monte Carlo simulations (2-year horizon)
 * - RedditSentiment: Social sentiment analysis from Reddit
 * - Resources: Recommended articles, books, and educational content
 * - WeeklyContent: Curated podcast and video of the week
 * 
 * All widgets are visually grouped with a section header and improved spacing.
 */
export const ResearchSection: React.FC = () => {
    return (
        <div className="fade-in">
            {/* Research Widgets - Grid Layout */}
            <div className="space-y-8">
                {/* Weekly Media - Video & Podcast at Top of Research */}
                <WeeklyMedia compact={false} />
                
                {/* Industry Research - Sector Deep Dives */}
                <IndustryResearch />
                
                {/* Under-the-Radar Screener - New Stock Screener */}
                <UnderTheRadarScreener />
                
                {/* Monte Carlo Forecast Simulation - Quick multiples valuation and 2-year Monte Carlo simulations */}
                <ValuationTool />

                {/* Alpha Stock Picker - Find undervalued opportunities */}
                <AlphaStockPicker />

                {/* Two Column Layout for Social & Resources */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <RedditSentiment />
                    <Resources />
                </div>
            </div>
        </div>
    );
};


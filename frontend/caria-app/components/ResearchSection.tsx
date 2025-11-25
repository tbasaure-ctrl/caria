import React from 'react';
import { ValuationTool } from './widgets/ValuationTool';
import { RedditSentiment } from './widgets/RedditSentiment';
import { Resources } from './widgets/Resources';
import { WeeklyContent } from './widgets/WeeklyContent';

/**
 * ResearchSection - Groups related research and analysis widgets together
 * 
 * This component provides a cohesive "Research & Analysis" section that includes:
 * - ValuationTool: Quick DCF-based valuation and Monte Carlo simulations
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
                {/* Valuation Tool - Full Width */}
                <ValuationTool />

                {/* Two Column Layout for Community Content */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <RedditSentiment />
                    <Resources />
                </div>

                {/* Weekly Content - Full Width */}
                <WeeklyContent />
            </div>
        </div>
    );
};


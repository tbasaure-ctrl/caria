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
            {/* Section Header */}
            <div className="mb-6">
                <h2 className="text-2xl font-bold mb-2"
                    style={{
                        fontFamily: 'var(--font-display)',
                        color: 'var(--color-cream)'
                    }}>
                    🔬 Research & Analysis
                </h2>
                <p className="text-sm mb-4"
                   style={{
                       fontFamily: 'var(--font-body)',
                       color: 'var(--color-text-secondary)'
                   }}>
                    Tools and insights to help you make informed investment decisions
                </p>
                <div className="h-px mb-6"
                     style={{
                         background: 'linear-gradient(to right, transparent, var(--color-bg-tertiary), transparent)',
                         opacity: 0.5
                     }}></div>
            </div>

            {/* Research Widgets - Grid Layout */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="lg:col-span-2">
                    <ValuationTool />
                </div>
                <div className="lg:col-span-1 space-y-6">
                    <Resources />
                    <RedditSentiment />
                </div>
                <div className="lg:col-span-1">
                    <WeeklyContent />
                </div>
            </div>
        </div>
    );
};


import React from 'react';
import { ValuationTool } from './widgets/ValuationTool';
import { RedditSentiment } from './widgets/RedditSentiment';
import { Resources } from './widgets/Resources';

/**
 * ResearchSection - Groups related research and analysis widgets together
 * 
 * This component provides a cohesive "Research & Analysis" section that includes:
 * - ValuationTool: Quick DCF-based valuation and Monte Carlo simulations
 * - RedditSentiment: Social sentiment analysis from Reddit
 * - Resources: Recommended articles, books, and educational content
 * 
 * All three widgets are visually grouped with a section header and improved spacing.
 */
export const ResearchSection: React.FC = () => {
    return (
        <div className="space-y-6 fade-in">
            {/* Section Header */}
            <div className="mb-6">
                <h2 className="text-2xl font-bold mb-2"
                    style={{
                        fontFamily: 'var(--font-display)',
                        color: 'var(--color-cream)'
                    }}>
                    🔬 Research & Analysis
                </h2>
                <p className="text-sm"
                   style={{
                       fontFamily: 'var(--font-body)',
                       color: 'var(--color-text-secondary)'
                   }}>
                    Tools and insights to help you make informed investment decisions
                </p>
                <div className="mt-3 h-px"
                     style={{
                         background: 'linear-gradient(to right, transparent, var(--color-bg-tertiary), transparent)',
                         opacity: 0.5
                     }}></div>
            </div>

            {/* Research Widgets */}
            <div className="space-y-6">
                <ValuationTool />
                <RedditSentiment />
                <Resources />
            </div>
        </div>
    );
};


import React from 'react';
import { CommunityFeed } from '../widgets/CommunityFeed';
import { RankingsWidget } from '../widgets/RankingsWidget';

export const CommunityPage: React.FC = () => {
    return (
        <main className="flex-1 overflow-y-auto p-6 max-w-[1920px] mx-auto"
            style={{ backgroundColor: 'var(--color-bg-primary)' }}>
            <div className="mb-8 fade-in">
                <h1 className="text-4xl font-bold mb-2"
                    style={{ fontFamily: 'var(--font-display)', color: 'var(--color-cream)' }}>
                    Community Insights
                </h1>
                <p style={{ fontFamily: 'var(--font-body)', color: 'var(--color-text-muted)' }}>
                    Connect, share, and learn from other investors.
                </p>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 space-y-6">
                    <CommunityFeed />
                </div>
                <div className="space-y-6">
                    <RankingsWidget />
                </div>
            </div>
        </main>
    );
};

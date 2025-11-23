import React from 'react';
import { Resources } from '../widgets/Resources';
import { ResearchSection } from '../ResearchSection';

export const ResourcesPage: React.FC = () => {
    return (
        <main className="flex-1 overflow-y-auto p-6 max-w-[1920px] mx-auto"
            style={{ backgroundColor: 'var(--color-bg-primary)' }}>
            <div className="mb-8 fade-in">
                <h1 className="text-4xl font-bold mb-2"
                    style={{ fontFamily: 'var(--font-display)', color: 'var(--color-cream)' }}>
                    Resources & Research
                </h1>
                <p style={{ fontFamily: 'var(--font-body)', color: 'var(--color-text-muted)' }}>
                    Curated readings, podcasts, and deep dives.
                </p>
            </div>
            <div className="space-y-8">
                 <Resources />
                 <ResearchSection />
            </div>
        </main>
    );
};

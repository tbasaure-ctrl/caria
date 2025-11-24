import React from 'react';
import { CommunityFeed } from '../widgets/CommunityFeed';

export const CommunityPage: React.FC = () => {
    return (
        <div className="min-h-screen w-full bg-[var(--color-bg-primary)] p-4 md:p-6 lg:p-8">
            <div className="max-w-[1920px] mx-auto">
                <h1 className="text-3xl md:text-4xl font-bold mb-6 text-[var(--color-text-primary)]">Community</h1>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 md:gap-6">
                    <div className="lg:col-span-2">
                        <CommunityFeed />
                    </div>
                </div>
            </div>
        </div>
    );
};

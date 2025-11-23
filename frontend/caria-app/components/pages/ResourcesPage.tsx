import React from 'react';
import { Resources } from '../widgets/Resources';

export const ResourcesPage: React.FC = () => {
    return (
        <div className="min-h-screen w-full bg-[var(--color-bg-primary)] p-4 md:p-6 lg:p-8">
            <div className="max-w-[1920px] mx-auto">
                <h1 className="text-3xl md:text-4xl font-bold mb-6 text-[var(--color-text-primary)]">Resources</h1>
                <Resources />
            </div>
        </div>
    );
};

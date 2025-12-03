import React from 'react';
import { WorldEconomiesHealth } from '../widgets/WorldEconomiesHealth';

export const EconomicHealthPage: React.FC = () => {
    return (
        <div className="animate-fade-in space-y-6 pb-20">
            <div className="max-w-4xl">
                <h1 className="text-3xl sm:text-4xl font-display text-white mb-2">World Economies Health</h1>
                <p className="text-sm text-text-secondary leading-relaxed max-w-2xl">
                    Global macroeconomic surveillance system. Monitor aggregated health scores, expansionary/contractionary regimes, and key leading indicators across major jurisdictions.
                </p>
            </div>
            
            <div className="min-h-[600px]">
                <WorldEconomiesHealth />
            </div>
        </div>
    );
};



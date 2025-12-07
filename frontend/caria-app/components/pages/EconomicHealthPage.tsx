import React, { useState } from 'react';
import { WorldEconomiesHealth } from '../widgets/WorldEconomiesHealth';
import CariaMultiscaleFragilityIndex from '../widgets/CariaMultiscaleFragilityIndex';
import { Globe2, Activity } from 'lucide-react';

type SubPanel = 'map' | 'msfi';

export const EconomicHealthPage: React.FC = () => {
    const [activePanel, setActivePanel] = useState<SubPanel>('map');

    return (
        <div className="animate-fade-in space-y-6 pb-20">
            <div className="max-w-4xl">
                <h1 className="text-3xl sm:text-4xl font-display text-white mb-2">World Economies Health</h1>
                <p className="text-sm text-text-secondary leading-relaxed max-w-2xl">
                    Global macroeconomic surveillance system. Monitor aggregated health scores, expansionary/contractionary regimes, and systemic fragility indicators across major jurisdictions.
                </p>
            </div>

            {/* Sub-panel Navigation */}
            <div className="flex bg-[#0B1221] border border-white/10 rounded-lg p-1 max-w-md">
                <button
                    onClick={() => setActivePanel('map')}
                    className={`flex-1 py-2.5 px-4 text-xs font-bold uppercase tracking-wider rounded transition-all flex items-center justify-center gap-2 ${activePanel === 'map'
                            ? 'bg-white/10 text-white shadow-glow-sm'
                            : 'text-text-muted hover:text-white'
                        }`}
                >
                    <Globe2 className="w-4 h-4" />
                    Global Economic Map
                </button>
                <button
                    onClick={() => setActivePanel('msfi')}
                    className={`flex-1 py-2.5 px-4 text-xs font-bold uppercase tracking-wider rounded transition-all flex items-center justify-center gap-2 ${activePanel === 'msfi'
                            ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 text-cyan-400 shadow-glow-sm border border-cyan-500/30'
                            : 'text-cyan-500/60 hover:text-cyan-400'
                        }`}
                >
                    <Activity className="w-4 h-4" />
                    <span className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse"></span>
                        Caria Multiscale Fragility Index
                    </span>
                </button>
            </div>

            {/* Panel Content */}
            <div className="min-h-[600px]">
                {activePanel === 'map' && <WorldEconomiesHealth />}
                {activePanel === 'msfi' && <CariaMultiscaleFragilityIndex />}
            </div>
        </div>
    );
};





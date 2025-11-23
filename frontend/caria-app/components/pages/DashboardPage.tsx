import React, { useState } from 'react';
import { Dashboard } from '../Dashboard';
import { AnalysisTool } from '../AnalysisTool';
import { CrisisSimulator } from '../widgets/CrisisSimulator';
import { MindMap } from '../widgets/MindMap';
import { MacroSimulator } from '../widgets/MacroSimulator';

export const DashboardPage: React.FC = () => {
    const [isAnalysisOpen, setAnalysisOpen] = useState(false);

    return (
        <>
            <Dashboard onStartAnalysis={() => setAnalysisOpen(true)} />
            
            {/* Phase 2: Advanced Features Section (Optional, can be integrated into Dashboard.tsx too) */}
            <div className="p-6 max-w-[1920px] mx-auto grid grid-cols-1 lg:grid-cols-2 gap-6">
                 <CrisisSimulator />
                 <MacroSimulator />
                 <MindMap />
            </div>

            {isAnalysisOpen && <AnalysisTool onClose={() => setAnalysisOpen(false)} />}
        </>
    );
};

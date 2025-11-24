import React, { useState } from 'react';
import { Dashboard } from '../Dashboard';
import { AnalysisTool } from '../AnalysisTool';
import { CrisisSimulator } from '../widgets/CrisisSimulator';
import { MindMap } from '../widgets/MindMap';
import { MacroSimulator } from '../widgets/MacroSimulator';

export const DashboardPage: React.FC = () => {
    const [isAnalysisOpen, setAnalysisOpen] = useState(false);

    return (
        <div className="min-h-screen w-full bg-[var(--color-bg-primary)] p-4 md:p-6 lg:p-8">
            <Dashboard onStartAnalysis={() => setAnalysisOpen(true)} />

            {/* Phase 2: Advanced Features Section */}
            <div className="max-w-[1920px] mx-auto mt-6 grid grid-cols-1 xl:grid-cols-2 gap-4 md:gap-6">
                <CrisisSimulator />
                <MacroSimulator />
                <div className="xl:col-span-2">
                    <MindMap />
                </div>
            </div>

            {isAnalysisOpen && <AnalysisTool onClose={() => setAnalysisOpen(false)} />}
        </div>
    );
};

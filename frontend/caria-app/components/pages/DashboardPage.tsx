import React, { useState } from 'react';
import { Dashboard } from '../Dashboard';
import { AnalysisTool } from '../AnalysisTool';
import { CrisisSimulator } from '../widgets/CrisisSimulator';
import { MindMap } from '../widgets/MindMap';
import { MacroSimulator } from '../widgets/MacroSimulator';
import { SafeWidget } from '../SafeWidget';

export const DashboardPage: React.FC = () => {
    const [isAnalysisOpen, setAnalysisOpen] = useState(false);

    return (
        <div className="min-h-screen w-full bg-[var(--color-bg-primary)] p-4 md:p-6 lg:p-8">
            <Dashboard onStartAnalysis={() => setAnalysisOpen(true)} />

            {/* Scenario Analysis Section - Crisis & Macro Simulators */}
            <div className="max-w-[1920px] mx-auto mt-8 mb-8">
                <div className="mb-6">
                    <h2 className="text-2xl font-bold mb-2"
                        style={{
                            fontFamily: 'var(--font-display)',
                            color: 'var(--color-cream)'
                        }}>
                        📊 Scenario Analysis
                    </h2>
                    <p className="text-sm"
                       style={{
                           fontFamily: 'var(--font-body)',
                           color: 'var(--color-text-secondary)'
                       }}>
                        Stress test your portfolio against historical crises and macroeconomic scenarios
                    </p>
                    <div className="h-px mt-4"
                         style={{
                             background: 'linear-gradient(to right, transparent, var(--color-bg-tertiary), transparent)',
                             opacity: 0.5
                         }}></div>
                </div>
                
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 md:gap-6">
                    <SafeWidget>
                        <CrisisSimulator />
                    </SafeWidget>
                    <SafeWidget>
                        <MacroSimulator />
                    </SafeWidget>
                </div>
            </div>

            {/* Mind Map - Full Width */}
            <div className="max-w-[1920px] mx-auto mb-8">
                <SafeWidget>
                    <MindMap />
                </SafeWidget>
            </div>

            {isAnalysisOpen && <AnalysisTool onClose={() => setAnalysisOpen(false)} />}
        </div>
    );
};

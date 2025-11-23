import React, { useState } from 'react';
import { Dashboard } from '../Dashboard';
import { AnalysisTool } from '../AnalysisTool';

export const DashboardPage: React.FC = () => {
    const [isAnalysisOpen, setAnalysisOpen] = useState(false);

    return (
        <>
            <Dashboard onStartAnalysis={() => setAnalysisOpen(true)} />
            {isAnalysisOpen && <AnalysisTool onClose={() => setAnalysisOpen(false)} />}
        </>
    );
};

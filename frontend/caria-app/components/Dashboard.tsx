
import React, { useState, useEffect } from 'react';
import { Portfolio } from './widgets/Portfolio';
import { TopMovers } from './widgets/TopMovers';
import { ModelOutlook } from './widgets/ModelOutlook';
import { IdealPortfolio } from './widgets/IdealPortfolio';
import { ThesisIcon } from './Icons';
import { GlobalMarketBar } from './widgets/GlobalMarketBar';
import { ValuationTool } from './widgets/ValuationTool';
import { HoldingsManager } from './widgets/HoldingsManager';
import { CommunityIdeas } from './widgets/CommunityIdeas';
import { MonteCarloSimulation } from './widgets/MonteCarloSimulation';
import { PortfolioAnalytics } from './widgets/PortfolioAnalytics';
import { OnboardingTour } from './OnboardingTour';
import { Resources } from './widgets/Resources';
import { RedditSentiment } from './widgets/RedditSentiment';
import { fetchWithAuth, API_BASE_URL } from '../services/apiService';

const StartAnalysisCTA: React.FC<{ onStartAnalysis: () => void; id?: string }> = ({ onStartAnalysis, id }) => (
    <div id={id}
         className="rounded-lg p-8 flex flex-col items-center justify-center text-center relative overflow-hidden group cursor-pointer transition-all duration-300"
         style={{
           backgroundColor: 'var(--color-bg-secondary)',
           border: '1px solid var(--color-bg-tertiary)',
           boxShadow: '0 4px 20px rgba(0,0,0,0.2)'
         }}
         onClick={onStartAnalysis}
         onMouseEnter={(e) => {
           e.currentTarget.style.borderColor = 'var(--color-primary)';
           e.currentTarget.style.transform = 'translateY(-2px)';
         }}
         onMouseLeave={(e) => {
           e.currentTarget.style.borderColor = 'var(--color-bg-tertiary)';
           e.currentTarget.style.transform = 'translateY(0)';
         }}>
        {/* Background decoration */}
        <div className="absolute top-0 right-0 w-32 h-32 opacity-5"
             style={{background: 'radial-gradient(circle, var(--color-primary) 0%, transparent 70%)'}}></div>

        <ThesisIcon className="w-14 h-14 mb-4" style={{color: 'var(--color-secondary)'}} />
        <h3 className="text-2xl font-bold mb-3"
            style={{fontFamily: 'var(--font-display)', color: 'var(--color-cream)'}}>
            Have an idea?
        </h3>
        <p className="mb-6 max-w-sm leading-relaxed"
           style={{
             fontFamily: 'var(--font-body)',
             color: 'var(--color-text-secondary)',
             fontSize: '0.95rem'
           }}>
            Challenge your investment thesis against Caria. Uncover cognitive biases and strengthen your rationale before you invest.
        </p>
        <button
            onClick={onStartAnalysis}
            className="py-3 px-8 rounded-lg font-semibold transition-all duration-200"
            style={{
              backgroundColor: 'var(--color-primary)',
              color: 'var(--color-cream)',
              fontFamily: 'var(--font-body)'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'var(--color-primary-light)';
              e.currentTarget.style.transform = 'scale(1.05)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'var(--color-primary)';
              e.currentTarget.style.transform = 'scale(1)';
            }}
        >
            Let's Break It Down
        </button>
    </div>
);


interface DashboardProps {
    onStartAnalysis: () => void;
}

interface RegimeData {
    regime: string;
    confidence: number;
}

export const Dashboard: React.FC<DashboardProps> = ({ onStartAnalysis }) => {
    const [regimeData, setRegimeData] = useState<RegimeData | null>(null);
    const [isLoadingRegime, setIsLoadingRegime] = useState(true);

    useEffect(() => {
        const fetchRegimeData = async () => {
            setIsLoadingRegime(true);
            try {
                // Use centralized API_BASE_URL per audit document
                const response = await fetchWithAuth(`${API_BASE_URL}/api/regime/current`);
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                const data = await response.json();
                setRegimeData({ regime: data.regime, confidence: data.confidence });
            } catch (error) {
                console.error("Failed to fetch regime data:", error);
                setRegimeData({ regime: 'slowdown', confidence: 0 }); // Fallback
            } finally {
                setIsLoadingRegime(false);
            }
        };

        fetchRegimeData();
    }, []);

    return (
        <main className="flex-1 overflow-y-auto p-8"
              style={{backgroundColor: 'var(--color-bg-primary)'}}>
             <OnboardingTour />

             {/* Dashboard Header */}
             <div className="mb-8 fade-in">
                <h1 className="text-4xl font-bold mb-2"
                    style={{fontFamily: 'var(--font-display)', color: 'var(--color-cream)'}}>
                    Investment Dashboard
                </h1>
                <p style={{fontFamily: 'var(--font-body)', color: 'var(--color-text-muted)'}}>
                    Your comprehensive view of market insights and portfolio analysis
                </p>
             </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-7">
                {/* Column 1 - Main Content */}
                <div className="lg:col-span-2 space-y-7 fade-in delay-200">
                    <GlobalMarketBar id="market-bar-widget" />
                    <Portfolio id="portfolio-widget" />
                    <PortfolioAnalytics />
                    <IdealPortfolio regime={regimeData?.regime} isLoading={isLoadingRegime} />
                </div>

                {/* Column 2 - Sidebar Widgets */}
                <div className="space-y-7 fade-in delay-300">
                    <ModelOutlook regimeData={regimeData} isLoading={isLoadingRegime} />
                    <HoldingsManager />
                    <StartAnalysisCTA onStartAnalysis={onStartAnalysis} id="analysis-cta-widget" />
                    <RedditSentiment />
                    <Resources />
                    <ValuationTool />
                    <MonteCarloSimulation />
                    <CommunityIdeas />
                    <TopMovers />
                </div>
            </div>
        </main>
    );
};

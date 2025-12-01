import React, { useState } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';
import { AlertTriangle, ShieldAlert, Info, ScanSearch, FileText } from 'lucide-react';

interface Risk {
    title: string;
    severity: 'High' | 'Medium' | 'Low';
    description: string;
}

interface ReportResponse {
    status: string;
    risks: Risk[];
    raw_text?: string;
    message?: string;
}

export const HiddenRiskReport: React.FC = () => {
    const [loading, setLoading] = useState(false);
    const [report, setReport] = useState<ReportResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    const generateReport = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/analysis/hidden-risk`, {
                method: 'POST',
            });
            
            if (!response.ok) {
                throw new Error('Failed to generate report');
            }
            
            const data = await response.json();
            setReport(data);
        } catch (err: any) {
            console.error(err);
            setError(err.message || 'Failed to analyze portfolio.');
        } finally {
            setLoading(false);
        }
    };

    const getSeverityColor = (severity: string) => {
        switch (severity.toLowerCase()) {
            case 'high': return 'text-negative bg-negative/10 border-negative/20';
            case 'medium': return 'text-warning bg-warning/10 border-warning/20';
            case 'low': return 'text-accent-primary bg-accent-primary/10 border-accent-primary/20';
            default: return 'text-text-muted bg-white/5 border-white/10';
        }
    };

    return (
        <WidgetCard
            title="HIDDEN RISK SCANNER"
            tooltip="AI-driven analysis of structural, sector, and macroeconomic risks hidden in your portfolio."
        >
            <div className="flex flex-col h-full min-h-[300px]">
                {!report && !loading && !error && (
                    <div className="flex-1 flex flex-col items-center justify-center text-center p-6">
                        <div className="w-16 h-16 rounded-full bg-accent-primary/10 flex items-center justify-center mb-4">
                            <ScanSearch className="w-8 h-8 text-accent-primary" />
                        </div>
                        <h3 className="text-lg font-display text-white mb-2">Scan Your Portfolio</h3>
                        <p className="text-sm text-text-secondary max-w-xs mb-6">
                            Caria's AI Risk Engine will analyze your holdings against current liquidity conditions (Hydraulic Stack) and market topology anomalies.
                        </p>
                        <button
                            onClick={generateReport}
                            className="px-6 py-3 bg-accent-primary hover:bg-accent-primary/90 text-black font-bold rounded-lg transition-all shadow-glow-sm flex items-center gap-2"
                        >
                            <FileText className="w-4 h-4" />
                            Generate Risk Report
                        </button>
                    </div>
                )}

                {loading && (
                    <div className="flex-1 flex flex-col items-center justify-center p-6 space-y-4">
                        <div className="relative w-16 h-16">
                            <div className="absolute inset-0 rounded-full border-2 border-white/10"></div>
                            <div className="absolute inset-0 rounded-full border-2 border-t-accent-primary animate-spin"></div>
                            <ScanSearch className="absolute inset-0 m-auto w-6 h-6 text-accent-primary animate-pulse" />
                        </div>
                        <div className="text-center">
                            <h4 className="text-white font-medium">Scanning Market Topology...</h4>
                            <p className="text-xs text-text-muted mt-1">Analyzing correlation matrices & liquidity flows</p>
                        </div>
                    </div>
                )}

                {error && (
                    <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
                        <AlertTriangle className="w-12 h-12 text-negative mb-4" />
                        <p className="text-negative font-medium mb-4">{error}</p>
                        <button
                            onClick={generateReport}
                            className="px-4 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-white transition-colors"
                        >
                            Try Again
                        </button>
                    </div>
                )}

                {report && !loading && (
                    <div className="space-y-4 animate-fade-in">
                        <div className="flex items-center justify-between mb-2">
                            <h4 className="text-sm font-display text-white">Risk Findings</h4>
                            <button 
                                onClick={generateReport}
                                className="text-xs text-accent-primary hover:text-white transition-colors"
                            >
                                Refresh Analysis
                            </button>
                        </div>

                        <div className="space-y-3 overflow-y-auto max-h-[400px] custom-scrollbar pr-2">
                            {report.risks && report.risks.length > 0 ? (
                                report.risks.map((risk, idx) => (
                                    <div 
                                        key={idx} 
                                        className={`p-4 rounded-lg border ${getSeverityColor(risk.severity)} backdrop-blur-sm transition-all duration-300 hover:translate-x-1`}
                                    >
                                        <div className="flex items-start gap-3">
                                            <ShieldAlert className={`w-5 h-5 flex-shrink-0 mt-0.5`} />
                                            <div>
                                                <div className="flex items-center gap-2 mb-1">
                                                    <h5 className="font-bold text-sm">{risk.title}</h5>
                                                    <span className="text-[10px] px-1.5 py-0.5 rounded-full border border-current opacity-80 uppercase tracking-wider font-bold">
                                                        {risk.severity}
                                                    </span>
                                                </div>
                                                <p className="text-xs opacity-90 leading-relaxed">
                                                    {risk.description}
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                ))
                            ) : (
                                <div className="text-center p-6 bg-positive/5 border border-positive/10 rounded-lg">
                                    <p className="text-positive font-medium">No critical hidden risks detected.</p>
                                    <p className="text-xs text-text-muted mt-1">Your portfolio appears aligned with the current regime.</p>
                                </div>
                            )}
                        </div>
                        
                        <div className="text-[10px] text-text-muted text-center mt-4 italic">
                            * AI-generated analysis based on current market data. Not financial advice.
                        </div>
                    </div>
                )}
            </div>
        </WidgetCard>
    );
};



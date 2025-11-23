/**
 * Ideal Portfolio Widget - Tactical Asset Allocation (TAA) per audit document (2.2).
 * Shows macro-conditional allocation based on regime signals (Tabla 4).
 */

import React, { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth } from '../../services/apiService';
import { API_BASE_URL } from '../../services/apiConfig';

interface AllocationData {
    stocks: number;
    bonds: number;
    cash: number;
}

interface ETFRecommendation {
    ticker: string;
    name: string;
    allocation: number;
    category: 'stocks' | 'bonds';
}

interface TacticalAllocation {
    regime: string;
    risk_level: string;
    vix: number | null;
    allocation: AllocationData;
    description: string;
    recommended_etfs: ETFRecommendation[];
    timestamp: string;
}

export const IdealPortfolio: React.FC<{ regime?: string; isLoading: boolean }> = ({ regime, isLoading }) => {
    const [allocation, setAllocation] = useState<TacticalAllocation | null>(null);
    const [isFetching, setIsFetching] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!regime) return;

        const fetchAllocation = async () => {
            setIsFetching(true);
            setError(null);
            try {
                // Use centralized API_BASE_URL per audit document
                const response = await fetchWithAuth(
                    `${API_BASE_URL}/api/portfolio/tactical/allocation?regime=${regime}`
                );
                
                if (!response.ok) {
                    throw new Error('Failed to fetch tactical allocation');
                }
                
                const data: TacticalAllocation = await response.json();
                setAllocation(data);
            } catch (err: any) {
                console.error('Error fetching tactical allocation:', err);
                setError('Coming soon... Tactical allocation recommendations are being enhanced with better strategies.');
            } finally {
                setIsFetching(false);
            }
        };

        fetchAllocation();
    }, [regime]);

    const renderContent = () => {
        if (isLoading || isFetching) {
            return (
                <div className="flex-grow flex items-center justify-center">
                    <p className="text-slate-500">Loading ideal portfolio...</p>
                </div>
            );
        }

        if (error) {
            return (
                <div className="flex-grow flex items-center justify-center">
                    <p className="text-red-400 text-sm">{error}</p>
                </div>
            );
        }

        if (!allocation) {
            return (
                <div className="flex-grow flex items-center justify-center">
                    <p className="text-slate-500">No allocation data available for this regime.</p>
                </div>
            );
        }

        const { allocation: alloc, risk_level, vix, recommended_etfs, description } = allocation;

        return (
            <div className="flex-grow space-y-4">
                {/* Risk Level Badge */}
                <div className="flex items-center justify-between">
                    <div>
                        <span className={`text-xs font-bold px-2 py-1 rounded-md ${
                            risk_level === 'low_risk' ? 'bg-green-900/50 text-green-300' :
                            risk_level === 'moderate_risk' ? 'bg-yellow-900/50 text-yellow-300' :
                            risk_level === 'high_risk' ? 'bg-orange-900/50 text-orange-300' :
                            'bg-red-900/50 text-red-300'
                        }`}>
                            {risk_level.replace('_', ' ').toUpperCase()}
                        </span>
                    </div>
                    {vix !== null && (
                        <div className="text-xs text-slate-400">
                            VIX: <span className="font-mono text-slate-300">{vix.toFixed(1)}</span>
                        </div>
                    )}
                </div>

                {/* Description */}
                <p className="text-xs text-slate-500">{description}</p>

                {/* Allocation Chart */}
                <div className="space-y-2">
                    <h4 className="text-xs font-bold text-slate-400">Asset Allocation</h4>
                    
                    {/* Stocks */}
                    {alloc.stocks > 0 && (
                        <div>
                            <div className="flex justify-between text-xs mb-1">
                                <span className="text-slate-300">Stocks</span>
                                <span className="font-mono text-slate-100">{(alloc.stocks * 100).toFixed(0)}%</span>
                            </div>
                            <div className="w-full bg-gray-800 rounded-full h-2">
                                <div
                                    className="bg-blue-500 h-2 rounded-full transition-all"
                                    style={{ width: `${alloc.stocks * 100}%` }}
                                />
                            </div>
                        </div>
                    )}

                    {/* Bonds */}
                    {alloc.bonds > 0 && (
                        <div>
                            <div className="flex justify-between text-xs mb-1">
                                <span className="text-slate-300">Bonds</span>
                                <span className="font-mono text-slate-100">{(alloc.bonds * 100).toFixed(0)}%</span>
                            </div>
                            <div className="w-full bg-gray-800 rounded-full h-2">
                                <div
                                    className="bg-green-500 h-2 rounded-full transition-all"
                                    style={{ width: `${alloc.bonds * 100}%` }}
                                />
                            </div>
                        </div>
                    )}

                    {/* Cash */}
                    {alloc.cash > 0 && (
                        <div>
                            <div className="flex justify-between text-xs mb-1">
                                <span className="text-slate-300">Cash</span>
                                <span className="font-mono text-slate-100">{(alloc.cash * 100).toFixed(0)}%</span>
                            </div>
                            <div className="w-full bg-gray-800 rounded-full h-2">
                                <div
                                    className="bg-slate-500 h-2 rounded-full transition-all"
                                    style={{ width: `${alloc.cash * 100}%` }}
                                />
                            </div>
                        </div>
                    )}
                </div>

                {/* ETF Recommendations */}
                {recommended_etfs.length > 0 && (
                    <div className="space-y-2">
                        <h4 className="text-xs font-bold text-slate-400">Recommended ETFs</h4>
                        <div className="space-y-1">
                            {recommended_etfs.map(etf => (
                                <div
                                    key={etf.ticker}
                                    className="flex justify-between items-center text-xs p-2 rounded-md bg-gray-900/50"
                                >
                                    <div>
                                        <span className="font-bold text-slate-200">{etf.ticker}</span>
                                        <p className="text-xs text-slate-400 truncate">{etf.name}</p>
                                    </div>
                                    <div className="text-right">
                                        <span className="font-mono text-slate-200">{(etf.allocation * 100).toFixed(1)}%</span>
                                        <p className="text-xs text-slate-500">{etf.category}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        );
    };

    return (
        <WidgetCard title={`IDEAL PORTFOLIO (${regime ? regime.toUpperCase() : '...'})`} className="h-full flex flex-col">
            {renderContent()}
            <p className="text-xs text-slate-600 mt-4 pt-3 border-t border-slate-800/50">
                <strong>Disclaimer:</strong> This is a model portfolio generated for illustrative purposes only. 
                It does not represent real holdings or constitute investment advice. Allocation based on regime signals per audit document (Tabla 4).
            </p>
        </WidgetCard>
    );
};

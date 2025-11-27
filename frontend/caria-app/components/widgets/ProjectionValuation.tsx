import React, { useState } from 'react';
import { ChevronDown, ChevronUp, TrendingUp, TrendingDown } from 'lucide-react';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';

interface Projection {
    year: number;
    revenue: number;
    growth: number;
    op_margin: number;
    net_margin: number;
    fcf: number;
    shares: number;
    fcf_per_share: number;
}

interface ProjectionValuationResponse {
    ticker: string;
    current_price: number;
    fair_value: number;
    upside_percentage: number;
    risk_score: number;
    projections: Projection[];
    projection_data?: {
        [key: string]: {
            [year: number]: number;
        };
    };
}

export const ProjectionValuation: React.FC = () => {
    const [ticker, setTicker] = useState('');
    const [macroRisk, setMacroRisk] = useState(0.2); // Default bajo
    const [industryRisk, setIndustryRisk] = useState(0.2);
    const [data, setData] = useState<ProjectionValuationResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [showDetails, setShowDetails] = useState(false); // Estado para ocultar/mostrar tabla

    const handleCalculate = async () => {
        if (!ticker.trim()) {
            setError('Please enter a ticker symbol');
            return;
        }

        setLoading(true);
        setError(null);
        setShowDetails(false); // Reset al correr nuevo
        try {
            const response = await fetchWithAuth(
                `${API_BASE_URL}/api/valuation/projection/${ticker.toUpperCase()}?macro_risk=${macroRisk}&industry_risk=${industryRisk}`
            );
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch valuation' }));
                throw new Error(errorData.detail || 'Failed to fetch valuation');
            }
            
            const result = await response.json();
            setData(result);
        } catch (err: any) {
            console.error("Failed to fetch valuation", err);
            setError(err.message || 'Failed to calculate valuation');
        } finally {
            setLoading(false);
        }
    };

    const formatMoney = (value: number) => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        }).format(value);
    };

    const formatNumber = (value: number) => {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        }).format(value);
    };

    return (
        <div className="max-w-4xl mx-auto p-6 bg-slate-900 text-white min-h-screen font-sans">
            {/* Header */}
            <div className="mb-8 text-center">
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
                    Caria Valuation Engine
                </h1>
                <p className="text-slate-400 mt-2">AI-Powered DCF Modeling</p>
            </div>

            {/* Inputs Section */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 bg-slate-800 p-6 rounded-2xl shadow-xl mb-8 border border-slate-700">
                <div className="flex flex-col gap-2">
                    <label className="text-sm font-semibold text-slate-300">Ticker Symbol</label>
                    <input 
                        value={ticker}
                        onChange={(e) => setTicker(e.target.value.toUpperCase())}
                        placeholder="e.g. NVDA"
                        className="bg-slate-900 border border-slate-600 rounded-lg p-3 text-white focus:ring-2 focus:ring-blue-500 outline-none"
                        onKeyPress={(e) => {
                            if (e.key === 'Enter') {
                                handleCalculate();
                            }
                        }}
                    />
                </div>
                
                <div className="flex flex-col gap-2">
                    <label className="text-sm font-semibold text-slate-300 flex justify-between">
                        Macro Risk <span>{Math.round(macroRisk * 100)}%</span>
                    </label>
                    <input 
                        type="range" min="0" max="1" step="0.1"
                        value={macroRisk}
                        onChange={(e) => setMacroRisk(parseFloat(e.target.value))}
                        className="accent-blue-500 h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer"
                    />
                </div>

                <div className="flex flex-col gap-2">
                    <label className="text-sm font-semibold text-slate-300 flex justify-between">
                        Industry Risk <span>{Math.round(industryRisk * 100)}%</span>
                    </label>
                    <input 
                        type="range" min="0" max="1" step="0.1"
                        value={industryRisk}
                        onChange={(e) => setIndustryRisk(parseFloat(e.target.value))}
                        className="accent-emerald-500 h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer"
                    />
                </div>
            </div>

            <button 
                onClick={handleCalculate}
                disabled={loading || !ticker.trim()}
                className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-4 rounded-xl transition-all shadow-lg shadow-blue-900/50 mb-10 disabled:opacity-50 disabled:cursor-not-allowed"
            >
                {loading ? "Analyzing Market Data..." : "Run Valuation Model"}
            </button>

            {/* Error Message */}
            {error && (
                <div className="bg-red-900/20 border border-red-500/30 text-red-400 px-4 py-3 rounded-lg text-sm mb-6">
                    {error}
                </div>
            )}

            {/* RESULTS SECTION */}
            {data && (
                <div className="animate-fade-in-up">
                    {/* Main Cards - Lo que querías ver primero */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                        
                        {/* Current Price Card */}
                        <div className="bg-slate-800 p-6 rounded-2xl border border-slate-700 flex flex-col items-center justify-center">
                            <span className="text-slate-400 text-sm uppercase tracking-wider">Current Price</span>
                            <span className="text-3xl font-bold text-white mt-2">{formatMoney(data.current_price)}</span>
                        </div>

                        {/* Fair Value Card (The Hero) */}
                        <div className="bg-slate-800 p-6 rounded-2xl border border-blue-500/30 flex flex-col items-center justify-center relative overflow-hidden">
                            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-purple-500"></div>
                            <span className="text-blue-400 text-sm uppercase tracking-wider font-bold">2029 Target Price</span>
                            <span className="text-4xl font-extrabold text-white mt-2">{formatMoney(data.fair_value)}</span>
                        </div>

                        {/* Upside/Downside Card */}
                        <div className={`p-6 rounded-2xl border flex flex-col items-center justify-center ${
                            data.upside_percentage >= 0 
                                ? 'bg-emerald-900/20 border-emerald-500/30' 
                                : 'bg-red-900/20 border-red-500/30'
                        }`}>
                            <span className={`${
                                data.upside_percentage >= 0 ? 'text-emerald-400' : 'text-red-400'
                            } text-sm uppercase tracking-wider font-bold`}>
                                Potential Upside
                            </span>
                            <div className="flex items-center gap-2 mt-2">
                                {data.upside_percentage >= 0 ? (
                                    <TrendingUp className="text-emerald-400" size={24} />
                                ) : (
                                    <TrendingDown className="text-red-400" size={24} />
                                )}
                                <span className={`text-3xl font-bold ${
                                    data.upside_percentage >= 0 ? 'text-emerald-400' : 'text-red-400'
                                }`}>
                                    {data.upside_percentage >= 0 ? '+' : ''}{formatNumber(data.upside_percentage)}%
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Toggle Details Button */}
                    <div className="flex justify-center mb-6">
                        <button 
                            onClick={() => setShowDetails(!showDetails)}
                            className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors text-sm"
                        >
                            {showDetails ? "Hide Financial Model" : "View Financial Model Details"}
                            {showDetails ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                        </button>
                    </div>

                    {/* Detailed Table (Hidden by default) */}
                    {showDetails && data.projections && (
                        <div className="bg-slate-800 rounded-2xl border border-slate-700 overflow-hidden shadow-2xl transition-all duration-500">
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm text-left text-slate-300">
                                    <thead className="text-xs text-slate-400 uppercase bg-slate-900">
                                        <tr>
                                            <th className="px-6 py-4">Metric</th>
                                            {data.projections.map(p => (
                                                <th key={p.year} className="px-6 py-4">{p.year}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr className="border-b border-slate-700 hover:bg-slate-700/50">
                                            <td className="px-6 py-4 font-medium text-white">Revenue</td>
                                            {data.projections.map(p => (
                                                <td key={p.year} className="px-6 py-4">
                                                    ${(p.revenue / 1e9).toFixed(2)}B
                                                </td>
                                            ))}
                                        </tr>
                                        <tr className="border-b border-slate-700 hover:bg-slate-700/50">
                                            <td className="px-6 py-4 font-medium text-white">Growth %</td>
                                            {data.projections.map(p => (
                                                <td key={p.year} className="px-6 py-4 text-emerald-400">
                                                    {(p.growth * 100).toFixed(1)}%
                                                </td>
                                            ))}
                                        </tr>
                                        <tr className="border-b border-slate-700 hover:bg-slate-700/50">
                                            <td className="px-6 py-4 font-medium text-white">Op Margin</td>
                                            {data.projections.map(p => (
                                                <td key={p.year} className="px-6 py-4 text-blue-400">
                                                    {(p.op_margin * 100).toFixed(1)}%
                                                </td>
                                            ))}
                                        </tr>
                                        <tr className="border-b border-slate-700 hover:bg-slate-700/50">
                                            <td className="px-6 py-4 font-medium text-white">FCF / Share</td>
                                            {data.projections.map(p => (
                                                <td key={p.year} className="px-6 py-4">
                                                    ${p.fcf_per_share.toFixed(2)}
                                                </td>
                                            ))}
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

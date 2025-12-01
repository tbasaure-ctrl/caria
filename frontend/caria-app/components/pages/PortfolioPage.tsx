import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { fetchHoldingsWithPrices, HoldingsWithPrices, HoldingWithPrice, API_BASE_URL, getToken, createHolding, deleteHolding } from '../../services/apiService';
import { getGuestHoldings, createGuestHolding, deleteGuestHolding } from '../../services/guestStorageService';
import { TrendingUp, Plus, Edit2, Trash2, ArrowUpRight } from 'lucide-react';
import { Portfolio } from '../widgets/Portfolio';
import { PortfolioAnalytics } from '../widgets/PortfolioAnalytics';
import { CrisisSimulator } from '../widgets/CrisisSimulator';
import { MacroSimulator } from '../widgets/MacroSimulator';
import { RegimeTestWidget } from '../widgets/RegimeTestWidget';
import { ProtectedWidget } from '../ProtectedWidget';

// TSMOM Status Dot Component
const TrendDot: React.FC<{ ticker: string }> = ({ ticker }) => {
    const [trend, setTrend] = useState<'Bullish' | 'Bearish' | 'Neutral' | 'High Risk Bullish' | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchTrend = async () => {
            try {
                if (!ticker || ticker.length > 10) return;
                
                const token = getToken();
                const headers: HeadersInit = { 'Content-Type': 'application/json' };
                if (token) headers['Authorization'] = `Bearer ${token}`;

                const response = await fetch(`${API_BASE_URL}/api/analysis/tsmom/${ticker}`, { headers });
                if (response.ok) {
                    const data = await response.json();
                    setTrend(data.trend_direction);
                }
            } catch (error) {
                // Silent fail
            } finally {
                setLoading(false);
            }
        };
        fetchTrend();
    }, [ticker]);

    if (loading) return <div className="w-2 h-2 rounded-full bg-white/10 animate-pulse" />;

    let color = 'bg-gray-500';
    let tooltip = 'Neutral';

    if (trend === 'Bullish') { color = 'bg-positive shadow-[0_0_8px_rgba(16,185,129,0.4)]'; tooltip = 'Positive Trend (Bullish)'; }
    else if (trend === 'Bearish') { color = 'bg-negative'; tooltip = 'Negative Trend (Bearish)'; }
    else if (trend === 'High Risk Bullish') { color = 'bg-warning'; tooltip = 'High Risk Trend'; }

    return (
        <div className="group relative flex items-center justify-center w-full h-full cursor-help">
            <div className={`w-2.5 h-2.5 rounded-full ${color}`} />
            <div className="absolute bottom-full mb-2 hidden group-hover:block bg-black border border-white/10 px-2 py-1 text-[10px] rounded whitespace-nowrap z-50 shadow-lg text-white">
                {tooltip}
            </div>
        </div>
    );
};

// Helper to convert guest holdings
const convertGuestHoldings = (guestHoldings: any[]): HoldingsWithPrices => {
    return {
        holdings: guestHoldings.map(h => ({
            ...h,
            current_price: h.average_cost, // Mock price for guest
            current_value: h.quantity * h.average_cost,
            gain_loss: 0,
            gain_loss_pct: 0,
            cost_basis: h.quantity * h.average_cost,
            price_change: 0,
            price_change_pct: 0
        })),
        total_value: 0,
        total_cost: 0,
        total_gain_loss: 0,
        total_gain_loss_pct: 0
    };
};

type SortKey = 'ticker' | 'current_price' | 'current_value' | 'gain_loss_pct';
type Section = 'main' | 'performance' | 'analytics' | 'stress';

export const PortfolioPage: React.FC = () => {
    const navigate = useNavigate();
    const [portfolioData, setPortfolioData] = useState<HoldingsWithPrices | null>(null);
    const [loading, setLoading] = useState(true);
    const [sortKey, setSortKey] = useState<SortKey>('current_value');
    const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
    const [showAddForm, setShowAddForm] = useState(false);
    const [activeSection, setActiveSection] = useState<Section>('main');
    
    // Add Form State
    const [formData, setFormData] = useState({
        ticker: '',
        quantity: '',
        average_cost: '',
        purchase_date: new Date().toISOString().split('T')[0],
        notes: ''
    });
    const [actionLoading, setActionLoading] = useState(false);

    const loadData = async () => {
        setLoading(true);
        try {
            if (getToken()) {
                const data = await fetchHoldingsWithPrices();
                setPortfolioData(data);
            } else {
                const guest = getGuestHoldings();
                setPortfolioData(convertGuestHoldings(guest));
            }
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadData();
    }, []);

    const handleSort = (key: SortKey) => {
        if (sortKey === key) {
            setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
        } else {
            setSortKey(key);
            setSortDir('desc');
        }
    };

    const sortedHoldings = [...(portfolioData?.holdings || [])].sort((a, b) => {
        let valA: any;
        let valB: any;

        if (sortKey === 'ticker') {
            valA = a.ticker;
            valB = b.ticker;
        } else {
            valA = a[sortKey as keyof HoldingWithPrice];
            valB = b[sortKey as keyof HoldingWithPrice];
        }

        if (valA < valB) return sortDir === 'asc' ? -1 : 1;
        if (valA > valB) return sortDir === 'asc' ? 1 : -1;
        return 0;
    });

    const handleAddHolding = async (e: React.FormEvent) => {
        e.preventDefault();
        setActionLoading(true);
        try {
            const holdingData = {
                ticker: formData.ticker.toUpperCase(),
                quantity: parseFloat(formData.quantity),
                average_cost: parseFloat(formData.average_cost),
                purchase_date: formData.purchase_date,
                notes: formData.notes
            };

            if (getToken()) {
                await createHolding(holdingData);
            } else {
                createGuestHolding(holdingData);
            }
            
            setShowAddForm(false);
            setFormData({ ticker: '', quantity: '', average_cost: '', purchase_date: new Date().toISOString().split('T')[0], notes: '' });
            await loadData();
        } catch (err) {
            console.error(err);
            alert('Failed to add holding');
        } finally {
            setActionLoading(false);
        }
    };

    const handleDelete = async (id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!window.confirm('Delete this holding?')) return;
        
        try {
            if (getToken()) {
                await deleteHolding(id);
            } else {
                deleteGuestHolding(id);
            }
            await loadData();
        } catch (err) {
            console.error(err);
        }
    };

    if (loading) return <div className="text-sm text-text-muted animate-pulse">Loading portfolio...</div>;

    return (
        <div className="flex h-[calc(100vh-100px)] animate-fade-in">
            {/* Sidebar Navigation */}
            <div className="w-48 border-r border-white/10 pr-4 hidden md:block">
                <div className="space-y-1 sticky top-0">
                    <button 
                        onClick={() => setActiveSection('main')}
                        className={`w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors ${activeSection === 'main' ? 'bg-white/10 text-white' : 'text-text-muted hover:text-white hover:bg-white/5'}`}
                    >
                        Overview & Holdings
                    </button>
                    <button 
                        onClick={() => setActiveSection('performance')}
                        className={`w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors ${activeSection === 'performance' ? 'bg-white/10 text-white' : 'text-text-muted hover:text-white hover:bg-white/5'}`}
                    >
                        Performance Graph
                    </button>
                    <button 
                        onClick={() => setActiveSection('analytics')}
                        className={`w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors ${activeSection === 'analytics' ? 'bg-white/10 text-white' : 'text-text-muted hover:text-white hover:bg-white/5'}`}
                    >
                        Deep Analytics
                    </button>
                    <button 
                        onClick={() => setActiveSection('stress')}
                        className={`w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors ${activeSection === 'stress' ? 'bg-white/10 text-white' : 'text-text-muted hover:text-white hover:bg-white/5'}`}
                    >
                        Stress Your Portfolio
                    </button>
                </div>
            </div>

            {/* Main Content Area */}
            <div className="flex-1 pl-0 md:pl-8 overflow-y-auto custom-scrollbar pr-2">
                
                {/* Section: Main (Overview + Holdings) */}
                {activeSection === 'main' && (
                    <div className="space-y-8">
                        {/* Header Stats */}
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                            <div className="p-4 rounded-lg border border-white/5 bg-bg-secondary">
                                <div className="text-[10px] text-text-muted uppercase tracking-widest mb-1">Total Equity</div>
                                <div className="text-2xl font-display text-white">
                                    ${portfolioData?.total_value.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                </div>
                            </div>
                            <div className="p-4 rounded-lg border border-white/5 bg-bg-secondary">
                                <div className="text-[10px] text-text-muted uppercase tracking-widest mb-1">Total P&L</div>
                                <div className={`text-2xl font-mono ${portfolioData?.total_gain_loss_pct && portfolioData.total_gain_loss_pct >= 0 ? 'text-positive' : 'text-negative'}`}>
                                    {portfolioData?.total_gain_loss_pct >= 0 ? '+' : ''}{portfolioData?.total_gain_loss_pct.toFixed(2)}%
                                </div>
                            </div>
                            <div className="p-4 rounded-lg border border-white/5 bg-bg-secondary flex items-center justify-between cursor-help group relative">
                                <div>
                                    <div className="text-[10px] text-text-muted uppercase tracking-widest mb-1">TSMOM Status</div>
                                    <div className="text-sm text-text-secondary">Trend Signals Active</div>
                                </div>
                                <TrendingUp className="w-5 h-5 text-accent-primary opacity-50" />
                                <div className="absolute top-full right-0 mt-2 w-64 p-3 bg-black border border-white/10 rounded shadow-xl text-xs text-text-muted z-50 hidden group-hover:block">
                                    Time Series Momentum (TSMOM) analyzes 12-month trends and volatility to signal Bullish/Bearish regimes.
                                </div>
                            </div>
                        </div>

                        {/* Controls */}
                        <div className="flex justify-between items-center">
                            <h3 className="text-lg font-display text-white">Holdings</h3>
                            <button 
                                onClick={() => setShowAddForm(!showAddForm)}
                                className="flex items-center gap-2 px-3 py-1.5 bg-accent-primary/10 text-accent-primary hover:bg-accent-primary/20 border border-accent-primary/20 rounded text-xs font-bold uppercase tracking-wider transition-all"
                            >
                                <Plus className="w-3 h-3" /> Add Holding
                            </button>
                        </div>

                        {/* Add Form */}
                        {showAddForm && (
                            <div className="bg-bg-secondary border border-white/10 rounded-lg p-4 animate-fade-in-up">
                                <form onSubmit={handleAddHolding} className="grid grid-cols-2 md:grid-cols-5 gap-4 items-end">
                                    <div>
                                        <label className="text-[10px] text-text-muted uppercase">Ticker</label>
                                        <input required type="text" value={formData.ticker} onChange={e => setFormData({...formData, ticker: e.target.value})} className="w-full bg-bg-tertiary border border-white/10 rounded px-2 py-1 text-sm text-white" placeholder="AAPL" />
                                    </div>
                                    <div>
                                        <label className="text-[10px] text-text-muted uppercase">Quantity</label>
                                        <input required type="number" step="any" value={formData.quantity} onChange={e => setFormData({...formData, quantity: e.target.value})} className="w-full bg-bg-tertiary border border-white/10 rounded px-2 py-1 text-sm text-white" placeholder="0" />
                                    </div>
                                    <div>
                                        <label className="text-[10px] text-text-muted uppercase">Avg Cost</label>
                                        <input required type="number" step="any" value={formData.average_cost} onChange={e => setFormData({...formData, average_cost: e.target.value})} className="w-full bg-bg-tertiary border border-white/10 rounded px-2 py-1 text-sm text-white" placeholder="0.00" />
                                    </div>
                                    <div>
                                        <label className="text-[10px] text-text-muted uppercase">Date</label>
                                        <input type="date" value={formData.purchase_date} onChange={e => setFormData({...formData, purchase_date: e.target.value})} className="w-full bg-bg-tertiary border border-white/10 rounded px-2 py-1 text-sm text-white" />
                                    </div>
                                    <button disabled={actionLoading} type="submit" className="bg-accent-primary text-black font-bold text-xs py-2 rounded hover:bg-accent-primary/90 transition-colors">
                                        {actionLoading ? 'Saving...' : 'Save Position'}
                                    </button>
                                </form>
                            </div>
                        )}

                        {/* List View */}
                        <div className="border border-white/10 rounded-lg overflow-hidden bg-bg-secondary/50">
                            {/* Header */}
                            <div className="grid grid-cols-12 gap-4 px-4 py-3 bg-bg-tertiary border-b border-white/10 text-[10px] text-text-muted uppercase tracking-wider font-medium">
                                <div className="col-span-3 cursor-pointer hover:text-white" onClick={() => handleSort('ticker')}>Asset {sortKey === 'ticker' && (sortDir === 'asc' ? '↑' : '↓')}</div>
                                <div className="col-span-2 text-right cursor-pointer hover:text-white" onClick={() => handleSort('current_price')}>Price {sortKey === 'current_price' && (sortDir === 'asc' ? '↑' : '↓')}</div>
                                <div className="col-span-2 text-right cursor-pointer hover:text-white" onClick={() => handleSort('current_value')}>Value {sortKey === 'current_value' && (sortDir === 'asc' ? '↑' : '↓')}</div>
                                <div className="col-span-2 text-right cursor-pointer hover:text-white" onClick={() => handleSort('gain_loss_pct')}>P&L % {sortKey === 'gain_loss_pct' && (sortDir === 'asc' ? '↑' : '↓')}</div>
                                <div className="col-span-1 text-center flex items-center justify-center gap-1">
                                    12m Trend <span className="text-[8px] text-text-muted">(TSMOM)</span>
                                </div>
                                <div className="col-span-2 text-right">Analysis</div>
                            </div>

                            {/* Rows */}
                            <div className="divide-y divide-white/5">
                                {sortedHoldings.map((holding) => (
                                    <div 
                                        key={holding.ticker}
                                        onClick={() => navigate(`/analysis?ticker=${holding.ticker}`)}
                                        className="grid grid-cols-12 gap-4 px-4 py-3 hover:bg-white/5 transition-colors cursor-pointer items-center group"
                                    >
                                        <div className="col-span-3 flex items-center gap-3">
                                            <div className="w-8 h-8 rounded bg-white/5 flex items-center justify-center text-[10px] font-bold text-accent-cyan">
                                                {holding.ticker.substring(0, 2)}
                                            </div>
                                            <div>
                                                <div className="text-sm font-bold text-white font-mono">{holding.ticker}</div>
                                                <div className="text-[10px] text-text-muted">{holding.quantity} units</div>
                                            </div>
                                        </div>
                                        <div className="col-span-2 text-right text-sm font-mono text-text-secondary">
                                            ${holding.current_price?.toFixed(2)}
                                        </div>
                                        <div className="col-span-2 text-right">
                                            <div className="text-sm font-mono text-white">
                                                ${holding.current_value?.toLocaleString()}
                                            </div>
                                        </div>
                                        <div className={`col-span-2 text-right text-sm font-mono ${holding.gain_loss_pct >= 0 ? 'text-positive' : 'text-negative'}`}>
                                            {holding.gain_loss_pct >= 0 ? '+' : ''}{holding.gain_loss_pct.toFixed(2)}%
                                        </div>
                                        <div className="col-span-1 flex justify-center">
                                            <TrendDot ticker={holding.ticker} />
                                        </div>
                                        <div className="col-span-2 flex justify-end items-center gap-3">
                                            <button 
                                                onClick={(e) => handleDelete(holding.id, e)}
                                                className="p-1.5 text-text-muted hover:text-negative opacity-0 group-hover:opacity-100 transition-opacity"
                                                title="Sell/Delete"
                                            >
                                                <Trash2 className="w-3 h-3" />
                                            </button>
                                            <button className="flex items-center gap-1 text-[10px] text-accent-cyan hover:text-white transition-colors">
                                                Analyze <ArrowUpRight className="w-3 h-3" />
                                            </button>
                                        </div>
                                    </div>
                                ))}
                                {sortedHoldings.length === 0 && (
                                    <div className="p-8 text-center text-text-muted text-sm">No holdings found.</div>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {/* Section: Performance Graph */}
                {activeSection === 'performance' && (
                    <div className="h-[600px]">
                        {/* Reusing the Portfolio widget which contains the graph logic */}
                        <Portfolio />
                    </div>
                )}

                {/* Section: Deep Analytics */}
                {activeSection === 'analytics' && (
                    <div className="h-[600px]">
                        <PortfolioAnalytics />
                    </div>
                )}

                {/* Section: Stress Your Portfolio */}
                {activeSection === 'stress' && (
                    <div className="space-y-8">
                        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                            <div className="h-[500px]">
                                <ProtectedWidget featureName="Crisis Simulator">
                                    <CrisisSimulator />
                                </ProtectedWidget>
                                <p className="text-xs text-text-muted mt-2 p-2">
                                    Simulate historical crashes (e.g., 2008, Covid-19) to test portfolio resilience.
                                </p>
                            </div>
                            <div className="h-[500px]">
                                <ProtectedWidget featureName="Regime Test">
                                    <RegimeTestWidget />
                                </ProtectedWidget>
                                <p className="text-xs text-text-muted mt-2 p-2">
                                    Test how your assets perform under different economic regimes (Inflation, Recession, Growth).
                                </p>
                            </div>
                        </div>
                        <div className="h-[500px]">
                            <MacroSimulator />
                            <p className="text-xs text-text-muted mt-2 p-2">
                                Adjust macro variables (Rates, GDP, Oil) to see potential impacts on your holdings.
                            </p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

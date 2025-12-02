import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { fetchHoldingsWithPrices, HoldingsWithPrices, HoldingWithPrice, API_BASE_URL, getToken, createHolding, deleteHolding, updateHolding } from '../../services/apiService';
import { getGuestHoldings, createGuestHolding, deleteGuestHolding, updateGuestHolding } from '../../services/guestStorageService';
import { TrendingUp, Plus, Edit2, Trash2, ArrowUpRight, X, Check } from 'lucide-react';
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

    // Edit State
    const [editingId, setEditingId] = useState<string | null>(null);
    const [editData, setEditData] = useState({ quantity: '', average_cost: '' });

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

    const startEdit = (holding: HoldingWithPrice, e: React.MouseEvent) => {
        e.stopPropagation();
        setEditingId(holding.id);
        setEditData({
            quantity: holding.quantity.toString(),
            average_cost: holding.average_cost.toString()
        });
    };

    const cancelEdit = (e: React.MouseEvent) => {
        e.stopPropagation();
        setEditingId(null);
        setEditData({ quantity: '', average_cost: '' });
    };

    const handleUpdate = async (id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        setActionLoading(true);
        try {
            const updates = {
                quantity: parseFloat(editData.quantity),
                average_cost: parseFloat(editData.average_cost)
            };

            if (getToken()) {
                await updateHolding(id, updates);
            } else {
                updateGuestHolding(id, updates);
            }

            setEditingId(null);
            setEditData({ quantity: '', average_cost: '' });
            await loadData();
        } catch (err) {
            console.error(err);
            alert('Failed to update holding');
        } finally {
            setActionLoading(false);
        }
    };

    if (loading) return <div className="text-sm text-text-muted animate-pulse">Loading portfolio...</div>;

    return (
        <div className="flex h-[calc(100vh-100px)] animate-fade-in">
            {/* Sidebar Navigation (Desktop) */}
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

                {/* Mobile Navigation Tabs */}
                <div className="md:hidden flex overflow-x-auto gap-2 mb-6 pb-2 border-b border-white/10 hide-scrollbar">
                    <button
                        onClick={() => setActiveSection('main')}
                        className={`whitespace-nowrap px-4 py-2 rounded-full text-sm font-medium transition-colors ${activeSection === 'main' ? 'bg-white text-black' : 'bg-white/5 text-text-muted'}`}
                    >
                        Overview
                    </button>
                    <button
                        onClick={() => setActiveSection('performance')}
                        className={`whitespace-nowrap px-4 py-2 rounded-full text-sm font-medium transition-colors ${activeSection === 'performance' ? 'bg-white text-black' : 'bg-white/5 text-text-muted'}`}
                    >
                        Performance
                    </button>
                    <button
                        onClick={() => setActiveSection('analytics')}
                        className={`whitespace-nowrap px-4 py-2 rounded-full text-sm font-medium transition-colors ${activeSection === 'analytics' ? 'bg-white text-black' : 'bg-white/5 text-text-muted'}`}
                    >
                        Analytics
                    </button>
                    <button
                        onClick={() => setActiveSection('stress')}
                        className={`whitespace-nowrap px-4 py-2 rounded-full text-sm font-medium transition-colors ${activeSection === 'stress' ? 'bg-white text-black' : 'bg-white/5 text-text-muted'}`}
                    >
                        Stress Test
                    </button>
                </div>

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
                                <form onSubmit={handleAddHolding} className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-4 items-end">
                                    <div>
                                        <label className="text-[10px] text-text-muted uppercase">Ticker</label>
                                        <input required type="text" value={formData.ticker} onChange={e => setFormData({ ...formData, ticker: e.target.value })} className="w-full bg-bg-tertiary border border-white/10 rounded px-2 py-1 text-sm text-white" placeholder="AAPL" />
                                    </div>
                                    <div>
                                        <label className="text-[10px] text-text-muted uppercase">Quantity</label>
                                        <input required type="number" step="any" value={formData.quantity} onChange={e => setFormData({ ...formData, quantity: e.target.value })} className="w-full bg-bg-tertiary border border-white/10 rounded px-2 py-1 text-sm text-white" placeholder="0" />
                                    </div>
                                    <div>
                                        <label className="text-[10px] text-text-muted uppercase">Avg Cost</label>
                                        <input required type="number" step="any" value={formData.average_cost} onChange={e => setFormData({ ...formData, average_cost: e.target.value })} className="w-full bg-bg-tertiary border border-white/10 rounded px-2 py-1 text-sm text-white" placeholder="0.00" />
                                    </div>
                                    <div>
                                        <label className="text-[10px] text-text-muted uppercase">Date</label>
                                        <input type="date" value={formData.purchase_date} onChange={e => setFormData({ ...formData, purchase_date: e.target.value })} className="w-full bg-bg-tertiary border border-white/10 rounded px-2 py-1 text-sm text-white" />
                                    </div>
                                    <button disabled={actionLoading} type="submit" className="bg-accent-primary text-black font-bold text-xs py-2 rounded hover:bg-accent-primary/90 transition-colors">
                                        {actionLoading ? 'Saving...' : 'Save Position'}
                                    </button>
                                </form>
                            </div>
                        )}

                        {/* List View */}
                        <div className="border border-white/10 rounded-lg overflow-hidden bg-bg-secondary/50">
                            {/* Header (Desktop Only) */}
                            <div className="hidden md:grid grid-cols-12 gap-4 px-4 py-3 bg-bg-tertiary border-b border-white/10 text-[10px] text-text-muted uppercase tracking-wider font-medium">
                                <div className="col-span-3 cursor-pointer hover:text-white" onClick={() => handleSort('ticker')}>Asset {sortKey === 'ticker' && (sortDir === 'asc' ? '↑' : '↓')}</div>
                                <div className="col-span-2 text-right cursor-pointer hover:text-white" onClick={() => handleSort('current_price')}>Price {sortKey === 'current_price' && (sortDir === 'asc' ? '↑' : '↓')}</div>
                                <div className="col-span-2 text-right cursor-pointer hover:text-white" onClick={() => handleSort('current_value')}>Value {sortKey === 'current_value' && (sortDir === 'asc' ? '↑' : '↓')}</div>
                                <div className="col-span-2 text-right cursor-pointer hover:text-white" onClick={() => handleSort('gain_loss_pct')}>P&L % {sortKey === 'gain_loss_pct' && (sortDir === 'asc' ? '↑' : '↓')}</div>
                                <div className="col-span-1 text-center flex items-center justify-center gap-1">
                                    12m Trend <span className="text-[8px] text-text-muted">(TSMOM)</span>
                                </div>
                                <div className="col-span-2 text-right">Analysis</div>
                            </div>

                            <div className="divide-y divide-white/5">
                                {sortedHoldings.length === 0 ? (
                                    <div className="p-8 text-center text-text-muted text-xs italic">
                                        No holdings found. Add a position to start tracking.
                                    </div>
                                ) : (
                                    sortedHoldings.map(holding => (
                                        <div key={holding.id} onClick={() => navigate(`/analysis?ticker=${holding.ticker}`)} className="cursor-pointer hover:bg-white/5 transition-colors group">
                                            {/* Desktop Row */}
                                            <div className="hidden md:grid grid-cols-12 gap-4 px-4 py-3 items-center text-sm">
                                                <div className="col-span-3 font-bold text-white flex items-center gap-2">
                                                    <div className="w-8 h-8 rounded bg-bg-tertiary flex items-center justify-center text-[10px] text-text-muted border border-white/10">
                                                        {holding.ticker.substring(0, 2)}
                                                    </div>
                                                    <div>
                                                        <div>{holding.ticker}</div>
                                                        <div className="text-[10px] text-text-muted font-normal">{holding.quantity} shares @ ${holding.average_cost.toFixed(2)}</div>
                                                    </div>
                                                </div>
                                                <div className="col-span-2 text-right text-white font-mono">${holding.current_price?.toFixed(2) || '--'}</div>
                                                <div className="col-span-2 text-right text-white font-mono">${holding.current_value?.toLocaleString(undefined, { minimumFractionDigits: 2 }) || '--'}</div>
                                                <div className={`col-span-2 text-right font-mono ${holding.gain_loss_pct >= 0 ? 'text-positive' : 'text-negative'}`}>
                                                    {holding.gain_loss_pct >= 0 ? '+' : ''}{holding.gain_loss_pct?.toFixed(2)}%
                                                </div>
                                                <div className="col-span-1 flex justify-center h-6">
                                                    <TrendDot ticker={holding.ticker} />
                                                </div>
                                                <div className="col-span-2 text-right flex justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                                    <button onClick={(e) => startEdit(holding, e)} className="p-1 hover:bg-white/10 rounded text-text-muted hover:text-white"><Edit2 className="w-3 h-3" /></button>
                                                    <button onClick={(e) => handleDelete(holding.id, e)} className="p-1 hover:bg-white/10 rounded text-text-muted hover:text-negative"><Trash2 className="w-3 h-3" /></button>
                                                </div>
                                            </div>

                                            {/* Mobile Card */}
                                            <div className="md:hidden p-4 space-y-3">
                                                <div className="flex justify-between items-start">
                                                    <div className="flex items-center gap-3">
                                                        <div className="w-10 h-10 rounded bg-bg-tertiary flex items-center justify-center text-xs font-bold text-white border border-white/10">
                                                            {holding.ticker.substring(0, 2)}
                                                        </div>
                                                        <div>
                                                            <div className="text-lg font-bold text-white">{holding.ticker}</div>
                                                            <div className="text-xs text-text-muted">{holding.quantity} shares</div>
                                                        </div>
                                                    </div>
                                                    <div className="text-right">
                                                        <div className="text-lg font-mono text-white">${holding.current_value?.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
                                                        <div className={`text-xs font-mono ${holding.gain_loss_pct >= 0 ? 'text-positive' : 'text-negative'}`}>
                                                            {holding.gain_loss_pct >= 0 ? '+' : ''}{holding.gain_loss_pct?.toFixed(2)}%
                                                        </div>
                                                    </div>
                                                </div>
                                                <div className="flex items-center justify-between pt-2 border-t border-white/5">
                                                    <div className="flex items-center gap-2">
                                                        <span className="text-[10px] text-text-muted uppercase tracking-wider">Trend</span>
                                                        <div className="h-4 w-4"><TrendDot ticker={holding.ticker} /></div>
                                                    </div>
                                                    <div className="flex gap-3">
                                                        <button onClick={(e) => startEdit(holding, e)} className="text-xs text-text-muted hover:text-white flex items-center gap-1"><Edit2 className="w-3 h-3" /> Edit</button>
                                                        <button onClick={(e) => handleDelete(holding.id, e)} className="text-xs text-text-muted hover:text-negative flex items-center gap-1"><Trash2 className="w-3 h-3" /> Delete</button>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    ))
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

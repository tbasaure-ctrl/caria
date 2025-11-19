import React, { useState, useEffect } from 'react';
import { fetchHoldingsWithPrices, createHolding, deleteHolding, HoldingWithPrice } from '../../services/apiService';
import { WidgetCard } from './WidgetCard';
import { MoreVerticalIcon } from '../Icons';

type SortOption = 'ticker' | 'return' | 'value';

export const HoldingsManager: React.FC = () => {
    const [holdings, setHoldings] = useState<HoldingWithPrice[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [showForm, setShowForm] = useState(false);
    const [openMenuId, setOpenMenuId] = useState<string | null>(null);
    const [sortOption, setSortOption] = useState<SortOption>('value');
    const [formData, setFormData] = useState({
        ticker: '',
        quantity: '',
        average_cost: '',
        purchase_date: new Date().toISOString().split('T')[0], // Default to today
        notes: '',
    });

    useEffect(() => {
        loadHoldings();
    }, []);

    const loadHoldings = async () => {
        try {
            setError(null);
            const data = await fetchHoldingsWithPrices();
            setHoldings(data.holdings);
            setLoading(false);
        } catch (err) {
            console.error('Error loading holdings:', err);
            setError('Could not load holdings with prices.');
            setLoading(false);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            await createHolding({
                ticker: formData.ticker.toUpperCase(),
                quantity: parseFloat(formData.quantity),
                average_cost: parseFloat(formData.average_cost),
                purchase_date: formData.purchase_date,
                notes: formData.notes || undefined,
            });
            setFormData({
                ticker: '',
                quantity: '',
                average_cost: '',
                purchase_date: new Date().toISOString().split('T')[0],
                notes: ''
            });
            setShowForm(false);
            loadHoldings();
        } catch (err) {
            setError('Error creating holding.');
        }
    };

    const handleDelete = async (id: string) => {
        if (!confirm('¿Eliminar esta posición?')) return;
        try {
            await deleteHolding(id);
            loadHoldings();
        } catch (err) {
            setError('Error deleting holding.');
        }
    };

    const getSortedHoldings = () => {
        const sorted = [...holdings];
        switch (sortOption) {
            case 'ticker':
                return sorted.sort((a, b) => a.ticker.localeCompare(b.ticker));
            case 'return':
                return sorted.sort((a, b) => (b.gain_loss_pct || 0) - (a.gain_loss_pct || 0));
            case 'value':
                return sorted.sort((a, b) => (b.current_value || 0) - (a.current_value || 0));
            default:
                return sorted;
        }
    };

    if (loading) {
        return (
            <WidgetCard
                title="GESTIÓN DE HOLDINGS"
                tooltip="Administra tus posiciones de inversión. Agrega, edita o elimina holdings para mantener tu cartera actualizada."
            >
                <div className="text-slate-400 text-sm">Cargando...</div>
            </WidgetCard>
        );
    }

    return (
        <WidgetCard
            title="GESTIÓN DE HOLDINGS"
            tooltip="Administra tus posiciones de inversión. Agrega, edita o elimina holdings para mantener tu cartera actualizada."
        >
            <div className="space-y-4">
                {error && (
                    <div className="text-red-400 text-sm bg-red-900/20 border border-red-800 rounded p-2">
                        {error}
                    </div>
                )}

                <div className="flex justify-between items-center">
                    {!showForm && (
                        <button
                            onClick={() => setShowForm(true)}
                            className="bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold py-1.5 px-3 rounded transition-colors"
                        >
                            + Agregar
                        </button>
                    )}

                    {/* Sort Controls */}
                    <div className="flex bg-slate-800 rounded p-0.5">
                        <button
                            onClick={() => setSortOption('value')}
                            className={`px-2 py-1 text-xs rounded ${sortOption === 'value' ? 'bg-slate-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}
                            title="Sort by Value"
                        >
                            $
                        </button>
                        <button
                            onClick={() => setSortOption('return')}
                            className={`px-2 py-1 text-xs rounded ${sortOption === 'return' ? 'bg-slate-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}
                            title="Sort by Return"
                        >
                            %
                        </button>
                        <button
                            onClick={() => setSortOption('ticker')}
                            className={`px-2 py-1 text-xs rounded ${sortOption === 'ticker' ? 'bg-slate-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}
                            title="Sort by Ticker"
                        >
                            AZ
                        </button>
                    </div>
                </div>

                {showForm && (
                    <form onSubmit={handleSubmit} className="space-y-3 bg-slate-900/50 p-4 rounded border border-slate-800">
                        <div>
                            <label className="block text-xs text-slate-400 mb-1">Ticker</label>
                            <input
                                type="text"
                                value={formData.ticker}
                                onChange={(e) => setFormData({ ...formData, ticker: e.target.value })}
                                className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                                placeholder="AAPL"
                                required
                            />
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                            <div>
                                <label className="block text-xs text-slate-400 mb-1">Cantidad</label>
                                <input
                                    type="number"
                                    step="0.01"
                                    value={formData.quantity}
                                    onChange={(e) => setFormData({ ...formData, quantity: e.target.value })}
                                    className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                                    placeholder="10"
                                    required
                                />
                            </div>
                            <div>
                                <label className="block text-xs text-slate-400 mb-1">Costo Promedio</label>
                                <input
                                    type="number"
                                    step="0.01"
                                    value={formData.average_cost}
                                    onChange={(e) => setFormData({ ...formData, average_cost: e.target.value })}
                                    className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                                    placeholder="150.00"
                                    required
                                />
                            </div>
                        </div>
                        <div>
                            <label className="block text-xs text-slate-400 mb-1">Fecha de Compra</label>
                            <input
                                type="date"
                                value={formData.purchase_date}
                                onChange={(e) => setFormData({ ...formData, purchase_date: e.target.value })}
                                className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                                required
                            />
                        </div>
                        <div>
                            <label className="block text-xs text-slate-400 mb-1">Notas (opcional)</label>
                            <input
                                type="text"
                                value={formData.notes}
                                onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                                className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
                                placeholder="Notas adicionales"
                            />
                        </div>
                        <div className="flex gap-2">
                            <button
                                type="submit"
                                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold py-2 px-4 rounded transition-colors"
                            >
                                Guardar
                            </button>
                            <button
                                type="button"
                                onClick={() => {
                                    setShowForm(false);
                                    setFormData({
                                        ticker: '',
                                        quantity: '',
                                        average_cost: '',
                                        purchase_date: new Date().toISOString().split('T')[0],
                                        notes: ''
                                    });
                                }}
                                className="flex-1 bg-slate-700 hover:bg-slate-600 text-white text-sm font-semibold py-2 px-4 rounded transition-colors"
                            >
                                Cancelar
                            </button>
                        </div>
                    </form>
                )}

                <div>
                    <h4 className="text-xs text-slate-400 mb-2">Mis Posiciones ({holdings.length})</h4>
                    {holdings.length === 0 ? (
                        <div className="text-slate-500 text-sm">No hay posiciones. Agrega una para comenzar.</div>
                    ) : (
                        <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1 custom-scrollbar">
                            {getSortedHoldings().map((holding) => (
                                <div
                                    key={holding.id}
                                    className="flex justify-between items-center p-2 bg-slate-900/50 rounded border border-slate-800 hover:border-slate-700 transition-colors"
                                >
                                    <div>
                                        <div className="flex items-baseline gap-2">
                                            <span className="text-slate-200 font-semibold">{holding.ticker}</span>
                                            {holding.gain_loss_pct !== undefined && (
                                                <span className={`text-xs ${holding.gain_loss_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                    {holding.gain_loss_pct >= 0 ? '+' : ''}{holding.gain_loss_pct.toFixed(2)}%
                                                </span>
                                            )}
                                        </div>
                                        <div className="text-xs text-slate-400">
                                            {holding.quantity} @ ${holding.average_cost.toFixed(2)}
                                        </div>
                                        {holding.current_value !== undefined && (
                                            <div className="text-xs text-slate-500 mt-0.5">
                                                Val: ${holding.current_value.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                                            </div>
                                        )}
                                    </div>
                                    <div className="relative">
                                        <button
                                            onClick={() => setOpenMenuId(openMenuId === holding.id ? null : holding.id)}
                                            className="text-slate-400 hover:text-slate-200 p-1 rounded hover:bg-slate-800 transition-colors"
                                        >
                                            <MoreVerticalIcon className="w-4 h-4" />
                                        </button>

                                        {openMenuId === holding.id && (
                                            <div className="absolute right-0 mt-1 w-32 bg-slate-800 border border-slate-700 rounded shadow-lg z-10 py-1">
                                                <button
                                                    onClick={() => {
                                                        // TODO: Implement edit
                                                        setOpenMenuId(null);
                                                        alert('Edit coming soon');
                                                    }}
                                                    className="w-full text-left px-3 py-2 text-xs text-slate-300 hover:bg-slate-700 hover:text-white transition-colors"
                                                >
                                                    Editar
                                                </button>
                                                <div className="h-px bg-slate-700 my-1"></div>
                                                <button
                                                    onClick={() => {
                                                        handleDelete(holding.id);
                                                        setOpenMenuId(null);
                                                    }}
                                                    className="w-full text-left px-3 py-2 text-xs text-red-400 hover:bg-slate-700 hover:text-red-300 transition-colors"
                                                >
                                                    Eliminar
                                                </button>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </WidgetCard>
    );
};


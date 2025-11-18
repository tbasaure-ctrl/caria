import React, { useState, useEffect } from 'react';
import { fetchHoldings, createHolding, deleteHolding, Holding } from '../../services/apiService';
import { WidgetCard } from './WidgetCard';

export const HoldingsManager: React.FC = () => {
    const [holdings, setHoldings] = useState<Holding[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [showForm, setShowForm] = useState(false);
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
            const data = await fetchHoldings();
            setHoldings(data);
            setLoading(false);
        } catch (err) {
            console.error('Error loading holdings:', err);
            setError('Coming soon... Holdings management is being enhanced for a better experience.');
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
            setError('Coming soon... Holdings creation is being enhanced for better reliability.');
        }
    };

    const handleDelete = async (id: string) => {
        if (!confirm('¿Eliminar esta posición?')) return;
        try {
            await deleteHolding(id);
            loadHoldings();
        } catch (err) {
            setError('Coming soon... Holdings deletion is being enhanced for better reliability.');
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

                {!showForm ? (
                    <button
                        onClick={() => setShowForm(true)}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold py-2 px-4 rounded transition-colors"
                    >
                        + Agregar Posición
                    </button>
                ) : (
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
                    <h4 className="text-xs text-slate-400 mb-2">Mis Posiciones</h4>
                    {holdings.length === 0 ? (
                        <div className="text-slate-500 text-sm">No hay posiciones. Agrega una para comenzar.</div>
                    ) : (
                        <div className="space-y-2">
                            {holdings.map((holding) => (
                                <div
                                    key={holding.id}
                                    className="flex justify-between items-center p-2 bg-slate-900/50 rounded border border-slate-800"
                                >
                                    <div>
                                        <div className="text-slate-200 font-semibold">{holding.ticker}</div>
                                        <div className="text-xs text-slate-400">
                                            {holding.quantity} @ ${holding.average_cost.toFixed(2)}
                                        </div>
                                        {(holding as any).purchase_date && (
                                            <div className="text-xs text-slate-500 mt-0.5">
                                                Bought: {new Date((holding as any).purchase_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                                            </div>
                                        )}
                                    </div>
                                    <button
                                        onClick={() => handleDelete(holding.id)}
                                        className="text-red-400 hover:text-red-300 text-xs px-2 py-1"
                                    >
                                        Eliminar
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </WidgetCard>
    );
};


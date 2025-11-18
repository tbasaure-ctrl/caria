import React, { useState } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';
import { CommunityCard } from './CommunityCard';
import { CommunityTooltip } from './CommunityTooltip';
import { ArenaThreadModal } from './ArenaThreadModal';
import { ThesisEditorModal } from './ThesisEditorModal';

interface CommunityResponse {
    community: string;
    response: string;
    impact_score: number;
}

interface ThesisArenaResponse {
    thesis: string;
    ticker: string | null;
    initial_conviction: number;
    community_responses: CommunityResponse[];
    conviction_impact: {
        conviction_change: number;
        new_conviction: number;
        initial_conviction: number;
        community_impacts: Record<string, any>;
    };
    arena_id?: string | null;
    round_number?: number;
}

const COMMUNITY_INFO = {
    value_investor: {
        name: 'Value Investor',
        icon: '💰',
        description: 'Enfocado en comprar acciones infravaloradas con fundamentos sólidos y margen de seguridad.',
        color: '#10b981',
    },
    crypto_bro: {
        name: 'Crypto Bro',
        icon: '🚀',
        description: 'Busca inversiones de alto riesgo y alto retorno, tecnologías disruptivas y potencial exponencial.',
        color: '#f59e0b',
    },
    growth_investor: {
        name: 'Growth Investor',
        icon: '📈',
        description: 'Invierte en empresas con altas tasas de crecimiento, enfocado en potencial futuro.',
        color: '#3b82f6',
    },
    contrarian: {
        name: 'Contrarian',
        icon: '🔄',
        description: 'Va contra la multitud, busca oportunidades cuando otros venden, enfocado en ciclos de mercado.',
        color: '#8b5cf6',
    },
};

export const ThesisArena: React.FC<{ onClose?: () => void }> = ({ onClose }) => {
    const [thesis, setThesis] = useState('');
    const [ticker, setTicker] = useState('');
    const [conviction, setConviction] = useState(50);
    const [isLoading, setIsLoading] = useState(false);
    const [results, setResults] = useState<ThesisArenaResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [hoveredCommunity, setHoveredCommunity] = useState<string | null>(null);
    const [showThreadModal, setShowThreadModal] = useState(false);
    const [showEditorModal, setShowEditorModal] = useState(false);

    const handleChallenge = async () => {
        if (!thesis.trim() || thesis.length < 10) {
            setError('Por favor ingresa una tesis de al menos 10 caracteres');
            return;
        }

        setIsLoading(true);
        setError(null);
        setResults(null);

        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/thesis/arena/challenge`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    thesis: thesis.trim(),
                    ticker: ticker.trim() || null,
                    initial_conviction: conviction,
                }),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Error desconocido' }));
                throw new Error(errorData.detail || `Error ${response.status}`);
            }

            const data = await response.json();
            setResults(data);
        } catch (err: any) {
            console.error('Error challenging thesis:', err);
            setError('Coming soon... Thesis Arena is being enhanced to provide even better analysis from our investment communities.');
        } finally{
            setIsLoading(false);
        }
    };

    // If thread modal is shown, render it instead
    if (showThreadModal && results?.arena_id) {
        return (
            <ArenaThreadModal
                threadId={results.arena_id}
                initialThesis={results.thesis}
                initialTicker={results.ticker}
                initialConviction={results.initial_conviction}
                onClose={() => setShowThreadModal(false)}
            />
        );
    }

    return (
        <div className="space-y-6">
            <WidgetCard
                title="Thesis Arena"
                className="fade-in"
                tooltip="Desafía tus tesis de inversión con 4 comunidades diferentes. Recibe feedback crítico para refinar tu convicción y análisis."
            >
                <div className="space-y-4">
                    {/* Ticker Input */}
                    <div>
                        <label 
                            htmlFor="ticker-input"
                            className="block text-sm font-medium mb-2"
                            style={{ color: 'var(--color-text-secondary)' }}
                        >
                            Ticker (Opcional)
                        </label>
                        <input
                            id="ticker-input"
                            type="text"
                            value={ticker}
                            onChange={(e) => setTicker(e.target.value.toUpperCase())}
                            placeholder="AAPL"
                            maxLength={10}
                            className="w-full px-4 py-2 rounded-lg border"
                            style={{
                                backgroundColor: 'var(--color-bg-secondary)',
                                borderColor: 'var(--color-bg-tertiary)',
                                color: 'var(--color-text-primary)',
                                fontFamily: 'var(--font-body)',
                            }}
                        />
                    </div>

                    {/* Thesis Textarea */}
                    <div>
                        <label 
                            htmlFor="thesis-textarea"
                            className="block text-sm font-medium mb-2"
                            style={{ color: 'var(--color-text-secondary)' }}
                        >
                            Tesis de Inversión
                        </label>
                        <textarea
                            id="thesis-textarea"
                            value={thesis}
                            onChange={(e) => setThesis(e.target.value)}
                            placeholder="Describe tu tesis de inversión aquí..."
                            rows={6}
                            maxLength={2000}
                            className="w-full px-4 py-2 rounded-lg border resize-none"
                            style={{
                                backgroundColor: 'var(--color-bg-secondary)',
                                borderColor: 'var(--color-bg-tertiary)',
                                color: 'var(--color-text-primary)',
                                fontFamily: 'var(--font-body)',
                            }}
                        />
                        <div 
                            className="text-xs mt-1 text-right"
                            style={{ color: 'var(--color-text-secondary)' }}
                        >
                            {thesis.length} / 2000
                        </div>
                    </div>

                    {/* Conviction Slider */}
                    <div>
                        <label 
                            htmlFor="conviction-slider"
                            className="block text-sm font-medium mb-2"
                            style={{ color: 'var(--color-text-secondary)' }}
                        >
                            Nivel de Conviction Inicial: {conviction}%
                        </label>
                        <input
                            id="conviction-slider"
                            type="range"
                            min="0"
                            max="100"
                            value={conviction}
                            onChange={(e) => setConviction(parseInt(e.target.value))}
                            className="w-full"
                            style={{
                                accentColor: 'var(--color-primary)',
                            }}
                        />
                        <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--color-text-secondary)' }}>
                            <span>Baja</span>
                            <span>Media</span>
                            <span>Alta</span>
                        </div>
                    </div>

                    {/* Challenge Button */}
                    <button
                        onClick={handleChallenge}
                        disabled={isLoading || !thesis.trim() || thesis.length < 10}
                        className="w-full px-6 py-3 rounded-lg font-medium transition-all"
                        style={{
                            backgroundColor: (isLoading || !thesis.trim() || thesis.length < 10)
                                ? 'var(--color-bg-tertiary)' 
                                : 'var(--color-primary)',
                            color: 'var(--color-cream)',
                            fontFamily: 'var(--font-display)',
                            cursor: (isLoading || !thesis.trim() || thesis.length < 10) ? 'not-allowed' : 'pointer',
                            opacity: (isLoading || !thesis.trim() || thesis.length < 10) ? 0.6 : 1,
                        }}
                    >
                        {isLoading ? 'Desafiando...' : 'Desafiar con Comunidades'}
                    </button>

                    {/* Error Display */}
                    {error && (
                        <div 
                            className="p-4 rounded-lg"
                            style={{
                                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                                border: '1px solid rgba(239, 68, 68, 0.3)',
                                color: '#ef4444',
                            }}
                        >
                            <strong>Error:</strong> {error}
                        </div>
                    )}
                </div>
            </WidgetCard>

            {/* Results */}
            {results && (
                <div className="space-y-4">
                    {/* Conviction Impact */}
                    <WidgetCard
                        title="Impacto en Conviction"
                        className="fade-in"
                        tooltip="Muestra cómo cambió tu nivel de convicción después del análisis de las comunidades."
                    >
                        <div className="text-center">
                            <div className="text-4xl font-bold mb-2" style={{ color: 'var(--color-primary)' }}>
                                {results.conviction_impact.new_conviction.toFixed(1)}%
                            </div>
                            <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                                {results.conviction_impact.conviction_change >= 0 ? '+' : ''}
                                {results.conviction_impact.conviction_change.toFixed(1)}% desde {results.conviction_impact.initial_conviction}%
                            </div>
                        </div>
                    </WidgetCard>

                    {/* Community Responses Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {results.community_responses.map((response) => {
                            const info = COMMUNITY_INFO[response.community as keyof typeof COMMUNITY_INFO];
                            if (!info) return null;

                            return (
                                <div
                                    key={response.community}
                                    onMouseEnter={() => setHoveredCommunity(response.community)}
                                    onMouseLeave={() => setHoveredCommunity(null)}
                                    className="relative"
                                >
                                    <CommunityCard
                                        community={info}
                                        response={response.response}
                                        impactScore={response.impact_score}
                                    />
                                    {hoveredCommunity === response.community && (
                                        <CommunityTooltip
                                            description={info.description}
                                            community={response.community}
                                        />
                                    )}
                                </div>
                            );
                        })}
                    </div>

                    {/* Action Buttons */}
                    <div className="flex gap-3 mt-4">
                        {results.arena_id && (
                            <button
                                onClick={() => setShowThreadModal(true)}
                                className="flex-1 px-6 py-3 rounded-lg font-medium transition-all"
                                style={{
                                    backgroundColor: 'var(--color-bg-secondary)',
                                    color: 'var(--color-text-primary)',
                                    border: '1px solid var(--color-bg-tertiary)',
                                    fontFamily: 'var(--font-display)',
                                }}
                            >
                                Continuar Conversación →
                            </button>
                        )}
                        <button
                            onClick={() => setShowEditorModal(true)}
                            className="flex-1 px-6 py-3 rounded-lg font-medium transition-all"
                            style={{
                                backgroundColor: 'var(--color-primary)',
                                color: 'var(--color-cream)',
                                fontFamily: 'var(--font-display)',
                            }}
                        >
                            Publicar en Feed →
                        </button>
                    </div>
                </div>
            )}

            {/* Thesis Editor Modal */}
            {showEditorModal && results && (
                <ThesisEditorModal
                    isOpen={showEditorModal}
                    onClose={() => setShowEditorModal(false)}
                    onSuccess={() => {
                        setShowEditorModal(false);
                        // Optionally reload community feed or show success message
                    }}
                    prefillData={{
                        title: `${results.ticker ? `${results.ticker}: ` : ''}Investment Thesis`,
                        thesis_preview: results.thesis.substring(0, 500),
                        full_thesis: results.thesis,
                        ticker: results.ticker,
                        arena_thread_id: results.arena_id || null,
                        arena_round_id: results.round_number ? String(results.round_number) : null,
                        arena_community: results.community_responses[0]?.community || null,
                    }}
                />
            )}
        </div>
    );
};


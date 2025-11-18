import React, { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';
import { CommunityCard } from './CommunityCard';
import { CommunityTooltip } from './CommunityTooltip';

interface CommunityResponse {
    community: string;
    response: string;
    impact_score: number;
}

interface Round {
    round_number: number;
    user_message?: string;
    community_responses: CommunityResponse[];
    conviction_before: number;
    conviction_after: number;
    conviction_change: number;
}

interface ArenaThread {
    thread_id: string;
    thesis: string;
    ticker: string | null;
    initial_conviction: number;
    current_conviction: number;
    rounds: Round[];
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

interface ArenaThreadModalProps {
    threadId: string;
    initialThesis: string;
    initialTicker: string | null;
    initialConviction: number;
    onClose: () => void;
}

export const ArenaThreadModal: React.FC<ArenaThreadModalProps> = ({
    threadId,
    initialThesis,
    initialTicker,
    initialConviction,
    onClose,
}) => {
    const [userMessage, setUserMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [rounds, setRounds] = useState<Round[]>([]);
    const [currentConviction, setCurrentConviction] = useState(initialConviction);
    const [error, setError] = useState<string | null>(null);
    const [hoveredCommunity, setHoveredCommunity] = useState<string | null>(null);

    // Load existing rounds on mount
    useEffect(() => {
        // For now, we'll start with empty rounds
        // In a full implementation, we'd fetch existing rounds from the API
        setRounds([]);
    }, [threadId]);

    const handleRespond = async () => {
        if (!userMessage.trim()) {
            setError('Por favor ingresa un mensaje');
            return;
        }

        setIsLoading(true);
        setError(null);

        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/thesis/arena/respond`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    thread_id: threadId,
                    user_message: userMessage.trim(),
                }),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Error desconocido' }));
                throw new Error(errorData.detail || `Error ${response.status}`);
            }

            const data = await response.json();
            
            // Add new round to rounds list
            const newRound: Round = {
                round_number: data.round_number,
                user_message: data.user_message,
                community_responses: data.community_responses,
                conviction_before: currentConviction,
                conviction_after: data.current_conviction,
                conviction_change: data.conviction_impact.conviction_change,
            };
            
            setRounds([...rounds, newRound]);
            setCurrentConviction(data.current_conviction);
            setUserMessage(''); // Clear input
        } catch (err: any) {
            console.error('Error responding in arena:', err);
            setError(err.message || 'Error al responder en arena');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex justify-between items-start mb-4">
                <div>
                    <h2 
                        className="text-2xl font-bold mb-2"
                        style={{ 
                            fontFamily: 'var(--font-display)',
                            color: 'var(--color-cream)',
                        }}
                    >
                        Arena Thread
                    </h2>
                    <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                        <div>Tesis: {initialThesis}</div>
                        {initialTicker && <div>Ticker: {initialTicker}</div>}
                    </div>
                </div>
                <button
                    onClick={onClose}
                    className="text-2xl font-bold"
                    style={{ color: 'var(--color-text-secondary)' }}
                >
                    ×
                </button>
            </div>

            {/* Conviction Tracker */}
            <WidgetCard title="Conviction Tracker" className="fade-in">
                <div className="text-center">
                    <div className="text-4xl font-bold mb-2" style={{ color: 'var(--color-primary)' }}>
                        {currentConviction.toFixed(1)}%
                    </div>
                    <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                        Inicial: {initialConviction}% | Cambio: {currentConviction >= initialConviction ? '+' : ''}
                        {(currentConviction - initialConviction).toFixed(1)}%
                    </div>
                </div>
            </WidgetCard>

            {/* Rounds History */}
            {rounds.length > 0 && (
                <div className="space-y-6">
                    {rounds.map((round, index) => (
                        <div key={round.round_number} className="space-y-4">
                            <div className="flex items-center gap-2">
                                <div 
                                    className="px-3 py-1 rounded-full text-sm font-medium"
                                    style={{
                                        backgroundColor: 'var(--color-bg-tertiary)',
                                        color: 'var(--color-text-secondary)',
                                    }}
                                >
                                    Round {round.round_number}
                                </div>
                                <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                                    Conviction: {round.conviction_before.toFixed(1)}% → {round.conviction_after.toFixed(1)}% 
                                    ({round.conviction_change >= 0 ? '+' : ''}{round.conviction_change.toFixed(1)}%)
                                </div>
                            </div>

                            {/* User Message */}
                            {round.user_message && (
                                <div 
                                    className="p-4 rounded-lg"
                                    style={{ backgroundColor: 'var(--color-bg-secondary)' }}
                                >
                                    <div className="text-xs font-semibold mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                                        Tu mensaje:
                                    </div>
                                    <div style={{ color: 'var(--color-text-primary)' }}>
                                        {round.user_message}
                                    </div>
                                </div>
                            )}

                            {/* Community Responses */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {round.community_responses.map((response) => {
                                    const info = COMMUNITY_INFO[response.community as keyof typeof COMMUNITY_INFO];
                                    if (!info) return null;

                                    return (
                                        <div
                                            key={`${round.round_number}-${response.community}`}
                                            onMouseEnter={() => setHoveredCommunity(`${round.round_number}-${response.community}`)}
                                            onMouseLeave={() => setHoveredCommunity(null)}
                                            className="relative"
                                        >
                                            <CommunityCard
                                                community={info}
                                                response={response.response}
                                                impactScore={response.impact_score}
                                            />
                                            {hoveredCommunity === `${round.round_number}-${response.community}` && (
                                                <CommunityTooltip
                                                    description={info.description}
                                                    community={response.community}
                                                />
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Respond Input */}
            <WidgetCard title="Continuar Conversación" className="fade-in">
                <div className="space-y-4">
                    <textarea
                        value={userMessage}
                        onChange={(e) => setUserMessage(e.target.value)}
                        placeholder="Escribe tu pregunta o comentario para las comunidades..."
                        rows={4}
                        maxLength={1000}
                        className="w-full px-4 py-2 rounded-lg border resize-none"
                        style={{
                            backgroundColor: 'var(--color-bg-secondary)',
                            borderColor: 'var(--color-bg-tertiary)',
                            color: 'var(--color-text-primary)',
                            fontFamily: 'var(--font-body)',
                        }}
                    />
                    <div className="flex justify-between items-center">
                        <div 
                            className="text-xs"
                            style={{ color: 'var(--color-text-secondary)' }}
                        >
                            {userMessage.length} / 1000
                        </div>
                        <button
                            onClick={handleRespond}
                            disabled={isLoading || !userMessage.trim()}
                            className="px-6 py-2 rounded-lg font-medium transition-all"
                            style={{
                                backgroundColor: (isLoading || !userMessage.trim())
                                    ? 'var(--color-bg-tertiary)' 
                                    : 'var(--color-primary)',
                                color: 'var(--color-cream)',
                                fontFamily: 'var(--font-display)',
                                cursor: (isLoading || !userMessage.trim()) ? 'not-allowed' : 'pointer',
                                opacity: (isLoading || !userMessage.trim()) ? 0.6 : 1,
                            }}
                        >
                            {isLoading ? 'Enviando...' : 'Responder'}
                        </button>
                    </div>

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
        </div>
    );
};


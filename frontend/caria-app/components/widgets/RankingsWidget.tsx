/**
 * RankingsWidget - Displays top communities, hot theses, and survivors.
 * Shows community rankings from the Arena system.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchCommunityRankings, CommunityRankings } from '../../services/communityService';

const COMMUNITY_LABELS: Record<string, string> = {
    value_investor: 'Value Investor',
    crypto_bro: 'Crypto Bro',
    growth_investor: 'Growth Investor',
    contrarian: 'Contrarian',
};

export const RankingsWidget: React.FC = () => {
    const [rankings, setRankings] = useState<CommunityRankings | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<'communities' | 'hot' | 'survivors'>('communities');
    const [statusMessage, setStatusMessage] = useState<string | null>(null);
    const cacheHydrated = useRef(false);
    const rankingsRef = useRef<CommunityRankings | null>(null);
    const CACHE_KEY = 'caria.community.rankings.cache.v1';

    useEffect(() => {
        rankingsRef.current = rankings;
    }, [rankings]);

    useEffect(() => {
        if (!rankings) return;
        sessionStorage.setItem(
            CACHE_KEY,
            JSON.stringify({ data: rankings, savedAt: Date.now() })
        );
    }, [rankings]);

    const hydrateFromCache = useCallback(() => {
        if (cacheHydrated.current) return;
        try {
            const cachedRaw = sessionStorage.getItem(CACHE_KEY);
            if (!cachedRaw) return;
            const cached = JSON.parse(cachedRaw);
            if (cached?.data) {
                setRankings(cached.data);
                setStatusMessage('Mostrando rankings guardados mientras se sincroniza.');
            }
        } catch (cacheError) {
            console.warn('No se pudo hidratar cache de rankings', cacheError);
        } finally {
            cacheHydrated.current = true;
        }
    }, []);

    const loadRankings = useCallback(
        async (signal?: AbortSignal) => {
            setLoading(true);
            setError(null);
            try {
                const data = await fetchCommunityRankings(signal);
                setRankings(data);
                setStatusMessage(null);
            } catch (err: any) {
                if (err?.name === 'AbortError') return;
                console.error('Error loading rankings:', err);
                hydrateFromCache();
                const message = err?.message || '';
                if (message.includes('401') || message.includes('403')) {
                    setError('Please log in to view community rankings');
                } else if (message.includes('connect')) {
                    setError('Unable to connect to rankings service');
                } else if (!rankingsRef.current) {
                    setError('Unable to load rankings. Please try again later.');
                } else {
                    setStatusMessage('Mostrando rankings guardados. Último refresh falló.');
                }
            } finally {
                setLoading(false);
            }
        },
        [hydrateFromCache]
    );

    useEffect(() => {
        hydrateFromCache();
        const controller = new AbortController();
        loadRankings(controller.signal);
        return () => controller.abort();
    }, [hydrateFromCache, loadRankings]);

    useEffect(() => {
        const handleRefresh = () => loadRankings();
        if (typeof window !== 'undefined') {
            window.addEventListener('caria-community-refresh', handleRefresh);
        }
        return () => {
            if (typeof window !== 'undefined') {
                window.removeEventListener('caria-community-refresh', handleRefresh);
            }
        };
    }, [loadRankings]);

    if (loading) {
        return (
            <WidgetCard
                title="Community Rankings"
                id="rankings-widget"
                tooltip="Rankings de la comunidad: comunidades más activas, tesis más populares, y tesis con mayor convicción sostenida."
            >
                <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                    Loading rankings...
                </div>
            </WidgetCard>
        );
    }

    if (error) {
        return (
            <WidgetCard
                title="Community Rankings"
                id="rankings-widget"
                tooltip="Rankings de la comunidad: comunidades más activas, tesis más populares, y tesis con mayor convicción sostenida."
            >
                <div className="text-sm" style={{ color: '#ef4444' }}>{error}</div>
                <button
                    onClick={() => loadRankings()}
                    className="mt-2 text-xs underline"
                    style={{ color: 'var(--color-primary)' }}
                >
                    Retry
                </button>
            </WidgetCard>
        );
    }

    if (!rankings) {
        return (
            <WidgetCard
                title="Community Rankings"
                id="rankings-widget"
                tooltip="Rankings de la comunidad: comunidades más activas, tesis más populares, y tesis con mayor convicción sostenida."
            >
                <div className="text-center h-[124px] flex items-center justify-center">
                    <p className="text-slate-500">No rankings available</p>
                </div>
            </WidgetCard>
        );
    }

    return (
        <WidgetCard
            title="Community Rankings"
            id="rankings-widget"
            tooltip="Rankings de la comunidad: comunidades más activas, tesis más populares, y tesis con mayor convicción sostenida."
        >
            <div className="space-y-4">
                {/* Tabs */}
                <div className="flex flex-col gap-2 border-b" style={{ borderColor: 'var(--color-bg-tertiary)' }}>
                    {statusMessage && (
                        <div className="text-xs" style={{ color: '#fbbf24' }}>
                            {statusMessage}
                        </div>
                    )}
                    <div className="flex gap-2">
                    <button
                        onClick={() => setActiveTab('communities')}
                        className={`px-3 py-2 text-sm font-medium transition-colors ${
                            activeTab === 'communities'
                                ? 'border-b-2'
                                : 'opacity-60 hover:opacity-100'
                        }`}
                        style={{
                            color: activeTab === 'communities' ? 'var(--color-primary)' : 'var(--color-text-secondary)',
                            borderColor: activeTab === 'communities' ? 'var(--color-primary)' : 'transparent',
                        }}
                    >
                        Top Communities
                    </button>
                    <button
                        onClick={() => setActiveTab('hot')}
                        className={`px-3 py-2 text-sm font-medium transition-colors ${
                            activeTab === 'hot'
                                ? 'border-b-2'
                                : 'opacity-60 hover:opacity-100'
                        }`}
                        style={{
                            color: activeTab === 'hot' ? 'var(--color-primary)' : 'var(--color-text-secondary)',
                            borderColor: activeTab === 'hot' ? 'var(--color-primary)' : 'transparent',
                        }}
                    >
                        Hot Theses
                    </button>
                    <button
                        onClick={() => setActiveTab('survivors')}
                        className={`px-3 py-2 text-sm font-medium transition-colors ${
                            activeTab === 'survivors'
                                ? 'border-b-2'
                                : 'opacity-60 hover:opacity-100'
                        }`}
                        style={{
                            color: activeTab === 'survivors' ? 'var(--color-primary)' : 'var(--color-text-secondary)',
                            borderColor: activeTab === 'survivors' ? 'var(--color-primary)' : 'transparent',
                        }}
                    >
                        Survivors
                    </button>
                    </div>
                </div>

                {/* Content */}
                <div className="space-y-3 max-h-96 overflow-y-auto">
                    {activeTab === 'communities' && (
                        <>
                            {rankings.top_communities.length === 0 ? (
                                <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                                    No community activity yet.
                                </div>
                            ) : (
                                rankings.top_communities.map((community, index) => (
                                    <div
                                        key={community.community}
                                        className="flex items-center justify-between p-3 rounded-lg"
                                        style={{
                                            backgroundColor: 'var(--color-bg-secondary)',
                                            border: '1px solid var(--color-bg-tertiary)',
                                        }}
                                    >
                                        <div className="flex items-center gap-3">
                                            <div
                                                className="w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm"
                                                style={{
                                                    backgroundColor: 'var(--color-primary)',
                                                    color: 'var(--color-cream)',
                                                }}
                                            >
                                                {index + 1}
                                            </div>
                                            <div>
                                                <div className="font-semibold" style={{ color: 'var(--color-cream)' }}>
                                                    {COMMUNITY_LABELS[community.community] || community.community}
                                                </div>
                                                <div className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                                                    {community.post_count} posts • {community.unique_users} users
                                                </div>
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <div className="text-sm font-semibold" style={{ color: 'var(--color-primary)' }}>
                                                {community.total_upvotes}
                                            </div>
                                            <div className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                                                upvotes
                                            </div>
                                        </div>
                                    </div>
                                ))
                            )}
                        </>
                    )}

                    {activeTab === 'hot' && (
                        <>
                            {rankings.hot_theses.length === 0 ? (
                                <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                                    No hot theses yet.
                                </div>
                            ) : (
                                rankings.hot_theses.map((thesis) => (
                                    <div
                                        key={thesis.id}
                                        className="p-3 rounded-lg"
                                        style={{
                                            backgroundColor: 'var(--color-bg-secondary)',
                                            border: '1px solid var(--color-bg-tertiary)',
                                        }}
                                    >
                                        <div className="flex items-start justify-between gap-2 mb-1">
                                            <h4 className="font-semibold flex-1" style={{ color: 'var(--color-cream)' }}>
                                                {thesis.title}
                                            </h4>
                                            <div className="flex items-center gap-1 text-sm" style={{ color: 'var(--color-primary)' }}>
                                                <span>▲</span>
                                                <span className="font-mono">{thesis.upvotes}</span>
                                            </div>
                                        </div>
                                        <p className="text-xs mb-2 line-clamp-2" style={{ color: 'var(--color-text-secondary)' }}>
                                            {thesis.thesis_preview}
                                        </p>
                                        <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                                            {thesis.ticker && (
                                                <>
                                                    <span className="font-mono">{thesis.ticker}</span>
                                                    <span>•</span>
                                                </>
                                            )}
                                            {thesis.username && <span>by {thesis.username}</span>}
                                            <span>•</span>
                                            <span>
                                                {new Date(thesis.created_at).toLocaleDateString('en-US', {
                                                    month: 'short',
                                                    day: 'numeric',
                                                })}
                                            </span>
                                        </div>
                                    </div>
                                ))
                            )}
                        </>
                    )}

                    {activeTab === 'survivors' && (
                        <>
                            {rankings.survivors.length === 0 ? (
                                <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                                    No survivors yet. Share your thesis in the Arena!
                                </div>
                            ) : (
                                rankings.survivors.map((survivor) => (
                                    <div
                                        key={survivor.thread_id}
                                        className="p-3 rounded-lg"
                                        style={{
                                            backgroundColor: 'var(--color-bg-secondary)',
                                            border: '1px solid var(--color-bg-tertiary)',
                                        }}
                                    >
                                        <div className="flex items-start justify-between gap-2 mb-2">
                                            <div className="flex-1">
                                                <p className="text-sm mb-1 line-clamp-2" style={{ color: 'var(--color-cream)' }}>
                                                    {survivor.thesis}
                                                </p>
                                                {survivor.ticker && (
                                                    <span className="text-xs font-mono" style={{ color: 'var(--color-text-secondary)' }}>
                                                        {survivor.ticker}
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                        <div className="flex items-center justify-between mt-2">
                                            <div className="flex items-center gap-3">
                                                <div>
                                                    <div className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                                                        Initial
                                                    </div>
                                                    <div className="text-sm font-semibold" style={{ color: 'var(--color-text-secondary)' }}>
                                                        {survivor.initial_conviction.toFixed(0)}%
                                                    </div>
                                                </div>
                                                <div className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                                                    →
                                                </div>
                                                <div>
                                                    <div className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                                                        Current
                                                    </div>
                                                    <div
                                                        className="text-sm font-semibold"
                                                        style={{
                                                            color:
                                                                survivor.current_conviction >= survivor.initial_conviction
                                                                    ? '#10b981'
                                                                    : '#ef4444',
                                                        }}
                                                    >
                                                        {survivor.current_conviction.toFixed(0)}%
                                                    </div>
                                                </div>
                                            </div>
                                            <div className="text-right">
                                                <div className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                                                    {survivor.round_count} rounds
                                                </div>
                                                {survivor.username && (
                                                    <div className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                                                        by {survivor.username}
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                ))
                            )}
                        </>
                    )}
                </div>
            </div>
        </WidgetCard>
    );
};


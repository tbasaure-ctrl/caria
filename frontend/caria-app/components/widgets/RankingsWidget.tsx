/**
 * RankingsWidget - Displays top communities, hot theses, and survivors.
 * Shows community rankings from the Arena system.
 */

import React, { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';

interface TopCommunity {
    community: string;
    post_count: number;
    total_upvotes: number;
    unique_users: number;
}

interface HotThesis {
    id: string;
    title: string;
    thesis_preview: string;
    ticker: string | null;
    upvotes: number;
    created_at: string;
    username: string | null;
}

interface Survivor {
    thread_id: string;
    thesis: string;
    ticker: string | null;
    initial_conviction: number;
    current_conviction: number;
    round_count: number;
    created_at: string;
    username: string | null;
}

interface RankingsData {
    top_communities: TopCommunity[];
    hot_theses: HotThesis[];
    survivors: Survivor[];
}

const COMMUNITY_LABELS: Record<string, string> = {
    value_investor: 'Value Investor',
    crypto_bro: 'Crypto Bro',
    growth_investor: 'Growth Investor',
    contrarian: 'Contrarian',
};

export const RankingsWidget: React.FC = () => {
    const [rankings, setRankings] = useState<RankingsData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<'communities' | 'hot' | 'survivors'>('communities');

    useEffect(() => {
        loadRankings();
    }, []);

    const loadRankings = async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await fetchWithAuth(`${API_BASE_URL}/api/community/rankings`);

            if (!response.ok) {
                throw new Error('Failed to load rankings');
            }

            const data: RankingsData = await response.json();
            setRankings(data);
        } catch (err: any) {
            console.error('Error loading rankings:', err);
            setError(err.message || 'Error loading rankings');
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <WidgetCard title="Community Rankings" id="rankings-widget">
                <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                    Loading rankings...
                </div>
            </WidgetCard>
        );
    }

    if (error) {
        return (
            <WidgetCard title="Community Rankings" id="rankings-widget">
                <div className="text-sm" style={{ color: '#ef4444' }}>{error}</div>
                <button
                    onClick={loadRankings}
                    className="mt-2 text-xs underline"
                    style={{ color: 'var(--color-primary)' }}
                >
                    Retry
                </button>
            </WidgetCard>
        );
    }

    if (!rankings) {
        return null;
    }

    return (
        <WidgetCard title="Community Rankings" id="rankings-widget">
            <div className="space-y-4">
                {/* Tabs */}
                <div className="flex gap-2 border-b" style={{ borderColor: 'var(--color-bg-tertiary)' }}>
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


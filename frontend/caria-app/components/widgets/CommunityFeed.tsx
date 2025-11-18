/**
 * Community Feed - Enhanced version of CommunityIdeas with search, Arena badges, and Arena integration.
 * Replaces CommunityIdeas widget with improved functionality.
 */

import React, { useState, useEffect, useMemo } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';

interface CommunityPost {
    id: string;
    user_id: string;
    username: string | null;
    title: string;
    thesis_preview: string;
    full_thesis: string | null;
    ticker: string | null;
    analysis_merit_score: number;
    upvotes: number;
    user_has_voted: boolean;
    created_at: string;
    updated_at: string;
    is_arena_post: boolean;
    arena_thread_id: string | null;
    arena_round_id: string | null;
    arena_community: string | null;
}

const COMMUNITY_LABELS: Record<string, string> = {
    value_investor: 'Value Investor',
    crypto_bro: 'Crypto Bro',
    growth_investor: 'Growth Investor',
    contrarian: 'Contrarian',
};

export const CommunityFeed: React.FC = () => {
    const [posts, setPosts] = useState<CommunityPost[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [expandedPost, setExpandedPost] = useState<string | null>(null);
    const [voting, setVoting] = useState<Set<string>>(new Set());
    const [searchQuery, setSearchQuery] = useState('');
    const [sortBy, setSortBy] = useState<'upvotes' | 'created_at' | 'analysis_merit_score'>('upvotes');

    useEffect(() => {
        loadPosts();
    }, [sortBy]);

    const loadPosts = async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await fetchWithAuth(
                `${API_BASE_URL}/api/community/posts?limit=50&sort_by=${sortBy}`
            );

            if (!response.ok) {
                throw new Error('Failed to load community posts');
            }

            const data = await response.json();
            setPosts(data);
        } catch (err: any) {
            console.error('Error loading community posts:', err);
            setError(err.message || 'Error loading posts');
        } finally {
            setLoading(false);
        }
    };

    const filteredPosts = useMemo(() => {
        if (!searchQuery.trim()) {
            return posts;
        }

        const query = searchQuery.toLowerCase();
        return posts.filter(
            (post) =>
                post.title.toLowerCase().includes(query) ||
                post.thesis_preview.toLowerCase().includes(query) ||
                (post.ticker && post.ticker.toLowerCase().includes(query)) ||
                (post.username && post.username.toLowerCase().includes(query))
        );
    }, [posts, searchQuery]);

    const handleVote = async (postId: string, hasVoted: boolean) => {
        if (voting.has(postId)) return;

        setVoting((prev) => new Set(prev).add(postId));

        try {
            const response = await fetchWithAuth(
                `${API_BASE_URL}/api/community/posts/${postId}/vote`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ vote_type: 'up' }),
                }
            );

            if (!response.ok) {
                throw new Error('Failed to vote');
            }

            const result = await response.json();

            setPosts((prevPosts) =>
                prevPosts.map((post) =>
                    post.id === postId
                        ? {
                              ...post,
                              upvotes: result.upvotes,
                              user_has_voted: result.user_has_voted,
                          }
                        : post
                )
            );
        } catch (err: any) {
            console.error('Error voting:', err);
        } finally {
            setVoting((prev) => {
                const next = new Set(prev);
                next.delete(postId);
                return next;
            });
        }
    };

    const handleExpand = async (postId: string) => {
        if (expandedPost === postId) {
            setExpandedPost(null);
            return;
        }

        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/community/posts/${postId}`);

            if (response.ok) {
                const fullPost: CommunityPost = await response.json();
                setPosts((prevPosts) =>
                    prevPosts.map((post) => (post.id === postId ? { ...post, ...fullPost } : post))
                );
                setExpandedPost(postId);
            }
        } catch (err) {
            console.error('Error loading full post:', err);
        }
    };

    const handleViewArenaThread = (threadId: string) => {
        // TODO: Open Arena thread modal or navigate to arena thread view
        console.log('View arena thread:', threadId);
        // This could open a modal or navigate to a dedicated arena thread view
    };

    if (loading) {
        return (
            <WidgetCard
                title="Community Feed"
                id="community-feed-widget"
                tooltip="Feed de tesis e ideas de inversión de la comunidad. Busca, vota y descubre insights compartidos por otros usuarios."
            >
                <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                    Loading community posts...
                </div>
            </WidgetCard>
        );
    }

    if (error) {
        return (
            <WidgetCard
                title="Community Feed"
                id="community-feed-widget"
                tooltip="Feed de tesis e ideas de inversión de la comunidad. Busca, vota y descubre insights compartidos por otros usuarios."
            >
                <div className="text-sm" style={{ color: '#ef4444' }}>{error}</div>
                <button
                    onClick={loadPosts}
                    className="mt-2 text-xs underline"
                    style={{ color: 'var(--color-primary)' }}
                >
                    Retry
                </button>
            </WidgetCard>
        );
    }

    return (
        <WidgetCard
            title="Community Feed"
            id="community-feed-widget"
            tooltip="Feed de tesis e ideas de inversión de la comunidad. Busca, vota y descubre insights compartidos por otros usuarios."
        >
            <div className="space-y-4">
                {/* Search and Sort Controls */}
                <div className="space-y-2">
                    <div className="flex gap-2">
                        <input
                            type="text"
                            placeholder="Search posts, tickers, users..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="flex-1 px-3 py-2 rounded-lg text-sm"
                            style={{
                                backgroundColor: 'var(--color-bg-secondary)',
                                border: '1px solid var(--color-bg-tertiary)',
                                color: 'var(--color-text-primary)',
                            }}
                        />
                        <select
                            value={sortBy}
                            onChange={(e) =>
                                setSortBy(e.target.value as 'upvotes' | 'created_at' | 'analysis_merit_score')
                            }
                            className="px-3 py-2 rounded-lg text-sm"
                            style={{
                                backgroundColor: 'var(--color-bg-secondary)',
                                border: '1px solid var(--color-bg-tertiary)',
                                color: 'var(--color-text-primary)',
                            }}
                        >
                            <option value="upvotes">Most Upvoted</option>
                            <option value="created_at">Newest</option>
                            <option value="analysis_merit_score">Best Analysis</option>
                        </select>
                    </div>
                    {searchQuery && (
                        <div className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                            Showing {filteredPosts.length} of {posts.length} posts
                        </div>
                    )}
                </div>

                {/* Posts List */}
                {filteredPosts.length === 0 ? (
                    <div className="text-sm text-center py-8" style={{ color: 'var(--color-text-secondary)' }}>
                        {searchQuery ? 'No posts match your search.' : 'No community posts yet. Be the first to share your investment thesis!'}
                    </div>
                ) : (
                    <div className="space-y-3 max-h-96 overflow-y-auto">
                        {filteredPosts.map((post) => (
                            <div
                                key={post.id}
                                className="p-3 rounded-lg border-b last:border-b-0"
                                style={{
                                    backgroundColor: 'var(--color-bg-secondary)',
                                    borderColor: 'var(--color-bg-tertiary)',
                                }}
                            >
                                {/* Header with title, Arena badge, and vote button */}
                                <div className="flex items-start justify-between gap-2 mb-2">
                                    <div className="flex-1">
                                        <div className="flex items-center gap-2 mb-1">
                                            <h4
                                                className="font-bold text-sm hover:underline cursor-pointer transition-colors flex-1"
                                                style={{ color: 'var(--color-cream)' }}
                                                onClick={() => handleExpand(post.id)}
                                            >
                                                {post.title}
                                            </h4>
                                            {post.is_arena_post && (
                                                <span
                                                    className="px-2 py-0.5 rounded text-xs font-semibold"
                                                    style={{
                                                        backgroundColor: 'var(--color-primary)',
                                                        color: 'var(--color-cream)',
                                                    }}
                                                    title="This post came from the Thesis Arena"
                                                >
                                                    🏛️ Arena
                                                </span>
                                            )}
                                            {post.arena_community && (
                                                <span
                                                    className="px-2 py-0.5 rounded text-xs"
                                                    style={{
                                                        backgroundColor: 'var(--color-bg-tertiary)',
                                                        color: 'var(--color-text-secondary)',
                                                    }}
                                                    title={`From ${COMMUNITY_LABELS[post.arena_community] || post.arena_community} community`}
                                                >
                                                    {COMMUNITY_LABELS[post.arena_community] || post.arena_community}
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                    <button
                                        onClick={() => handleVote(post.id, post.user_has_voted)}
                                        disabled={voting.has(post.id)}
                                        className={`flex flex-col items-center px-2 py-1 rounded-md transition-colors text-xs ${
                                            post.user_has_voted
                                                ? 'bg-blue-900/50'
                                                : 'bg-gray-800 hover:bg-gray-700'
                                        } ${voting.has(post.id) ? 'opacity-50 cursor-not-allowed' : ''}`}
                                        style={{
                                            color: post.user_has_voted ? '#93c5fd' : 'var(--color-text-secondary)',
                                        }}
                                        title={post.user_has_voted ? 'Remove vote' : 'Upvote'}
                                    >
                                        <span className="text-lg leading-none">▲</span>
                                        <span className="font-mono">{post.upvotes}</span>
                                    </button>
                                </div>

                                {/* Metadata */}
                                <div className="flex items-center gap-2 text-xs mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                                    {post.username && <span>by {post.username}</span>}
                                    {post.ticker && (
                                        <>
                                            <span>•</span>
                                            <span className="font-mono">{post.ticker}</span>
                                        </>
                                    )}
                                    {post.analysis_merit_score > 0 && (
                                        <>
                                            <span>•</span>
                                            <span>Merit: {(post.analysis_merit_score * 100).toFixed(0)}%</span>
                                        </>
                                    )}
                                    {post.is_arena_post && post.arena_thread_id && (
                                        <>
                                            <span>•</span>
                                            <button
                                                onClick={() => handleViewArenaThread(post.arena_thread_id!)}
                                                className="underline hover:no-underline"
                                                style={{ color: 'var(--color-primary)' }}
                                            >
                                                View Arena Thread
                                            </button>
                                        </>
                                    )}
                                </div>

                                {/* Preview or Full Thesis */}
                                <div className="text-xs mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                                    {expandedPost === post.id && post.full_thesis ? (
                                        <div>
                                            <p className="whitespace-pre-wrap">{post.full_thesis}</p>
                                            <button
                                                onClick={() => setExpandedPost(null)}
                                                className="mt-2 underline"
                                                style={{ color: 'var(--color-primary)' }}
                                            >
                                                Show less
                                            </button>
                                        </div>
                                    ) : (
                                        <div>
                                            <p>{post.thesis_preview}</p>
                                            {post.full_thesis && (
                                                <button
                                                    onClick={() => handleExpand(post.id)}
                                                    className="mt-1 underline"
                                                    style={{ color: 'var(--color-primary)' }}
                                                >
                                                    Read more...
                                                </button>
                                            )}
                                        </div>
                                    )}
                                </div>

                                {/* Timestamp */}
                                <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                                    {new Date(post.created_at).toLocaleDateString('en-US', {
                                        month: 'short',
                                        day: 'numeric',
                                        year: 'numeric',
                                    })}
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </WidgetCard>
    );
};


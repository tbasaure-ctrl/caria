/**
 * Community Ideas Widget - Top investment thesis shared by users.
 * Per user requirements: Shows title/preview, Reddit-style UP voting.
 */

import React, { useState, useEffect } from 'react';
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
}

export const CommunityIdeas: React.FC = () => {
    const [posts, setPosts] = useState<CommunityPost[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [expandedPost, setExpandedPost] = useState<string | null>(null);
    const [voting, setVoting] = useState<Set<string>>(new Set());

    useEffect(() => {
        loadPosts();
    }, []);

    const loadPosts = async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await fetchWithAuth(
                `${API_BASE_URL}/api/community/posts?limit=10&sort_by=upvotes`
            );
            
            if (!response.ok) {
                throw new Error('Failed to load community posts');
            }

            const data = await response.json();
            setPosts(data);
        } catch (err: any) {
            console.error('Error loading community posts:', err);
            setError('Coming soon... Community ideas are being enhanced with better curation.');
        } finally {
            setLoading(false);
        }
    };

    const handleVote = async (postId: string, hasVoted: boolean) => {
        if (voting.has(postId)) return; // Prevent double-click

        setVoting(prev => new Set(prev).add(postId));

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

            // Update local state
            setPosts(prevPosts =>
                prevPosts.map(post =>
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
            // Could show toast notification here
        } finally {
            setVoting(prev => {
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

        // Fetch full post details
        try {
            const response = await fetchWithAuth(
                `${API_BASE_URL}/api/community/posts/${postId}`
            );

            if (response.ok) {
                const fullPost: CommunityPost = await response.json();
                setPosts(prevPosts =>
                    prevPosts.map(post =>
                        post.id === postId ? { ...post, ...fullPost } : post
                    )
                );
                setExpandedPost(postId);
            }
        } catch (err) {
            console.error('Error loading full post:', err);
        }
    };

    if (loading) {
        return (
            <WidgetCard title="COMMUNITY TOP IDEAS">
                <div className="text-slate-400 text-sm">Loading community ideas...</div>
            </WidgetCard>
        );
    }

    if (error) {
        return (
            <WidgetCard title="COMMUNITY TOP IDEAS">
                <div className="text-red-400 text-sm">{error}</div>
                <button
                    onClick={loadPosts}
                    className="mt-2 text-xs text-slate-400 hover:text-slate-300 underline"
                >
                    Retry
                </button>
            </WidgetCard>
        );
    }

    if (posts.length === 0) {
        return (
            <WidgetCard title="COMMUNITY TOP IDEAS">
                <div className="text-slate-400 text-sm">
                    No community posts yet. Be the first to share your investment thesis!
                </div>
            </WidgetCard>
        );
    }

    return (
        <WidgetCard title="COMMUNITY TOP IDEAS">
            <div className="space-y-3 max-h-96 overflow-y-auto">
                {posts.map(post => (
                    <div
                        key={post.id}
                        className="text-sm border-b border-slate-800/50 pb-3 last:border-b-0 last:pb-0"
                    >
                        {/* Header with title and vote button */}
                        <div className="flex items-start justify-between gap-2 mb-1">
                            <h4
                                className="font-bold text-slate-200 hover:text-white cursor-pointer transition-colors flex-1"
                                onClick={() => handleExpand(post.id)}
                            >
                                {post.title}
                            </h4>
                            <button
                                onClick={() => handleVote(post.id, post.user_has_voted)}
                                disabled={voting.has(post.id)}
                                className={`flex flex-col items-center px-2 py-1 rounded-md transition-colors text-xs ${
                                    post.user_has_voted
                                        ? 'bg-blue-900/50 text-blue-300'
                                        : 'bg-gray-800 text-slate-400 hover:bg-gray-700'
                                } ${voting.has(post.id) ? 'opacity-50 cursor-not-allowed' : ''}`}
                                title={post.user_has_voted ? 'Remove vote' : 'Upvote'}
                            >
                                <span className="text-lg leading-none">▲</span>
                                <span className="font-mono">{post.upvotes}</span>
                            </button>
                        </div>

                        {/* Metadata */}
                        <div className="flex items-center gap-2 text-xs text-slate-500 mb-2">
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
                        </div>

                        {/* Preview or Full Thesis */}
                        <div className="text-slate-400 text-xs">
                            {expandedPost === post.id && post.full_thesis ? (
                                <div>
                                    <p className="whitespace-pre-wrap">{post.full_thesis}</p>
                                    <button
                                        onClick={() => setExpandedPost(null)}
                                        className="mt-2 text-blue-400 hover:text-blue-300 underline"
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
                                            className="mt-1 text-blue-400 hover:text-blue-300 underline"
                                        >
                                            Read more...
                                        </button>
                                    )}
                                </div>
                            )}
                        </div>

                        {/* Timestamp */}
                        <div className="text-xs text-slate-600 mt-1">
                            {new Date(post.created_at).toLocaleDateString('en-US', {
                                month: 'short',
                                day: 'numeric',
                            })}
                        </div>
                    </div>
                ))}
            </div>
        </WidgetCard>
    );
};

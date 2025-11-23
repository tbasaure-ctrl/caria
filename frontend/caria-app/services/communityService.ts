import { fetchWithAuth } from './apiService';
import { API_BASE_URL } from './apiConfig';

export type CommunityPostSort = 'upvotes' | 'created_at' | 'analysis_merit_score';

export interface CommunityPost {
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

export interface FetchCommunityPostsOptions {
    limit?: number;
    offset?: number;
    sortBy?: CommunityPostSort;
    ticker?: string;
    search?: string;
    signal?: AbortSignal;
}

export const fetchCommunityPosts = async (
    options: FetchCommunityPostsOptions = {}
): Promise<CommunityPost[]> => {
    const { limit = 50, offset = 0, sortBy = 'upvotes', ticker, search, signal } = options;

    const params = new URLSearchParams({
        limit: String(limit),
        offset: String(offset),
        sort_by: sortBy,
    });

    if (ticker) params.append('ticker', ticker.toUpperCase());
    if (search) params.append('search', search);

    const response = await fetchWithAuth(`${API_BASE_URL}/api/community/posts?${params.toString()}`, {
        signal,
    });
    const data = await response.json();
    if (!Array.isArray(data)) {
        throw new Error('Unexpected community posts response shape');
    }
    return data as CommunityPost[];
};

export interface TopCommunity {
    community: string;
    post_count: number;
    total_upvotes: number;
    unique_users: number;
}

export interface HotThesis {
    id: string;
    title: string;
    thesis_preview: string;
    ticker: string | null;
    upvotes: number;
    created_at: string;
    username: string | null;
}

export interface Survivor {
    thread_id: string;
    thesis: string;
    ticker: string | null;
    initial_conviction: number;
    current_conviction: number;
    round_count: number;
    created_at: string;
    username: string | null;
}

export interface CommunityRankings {
    top_communities: TopCommunity[];
    hot_theses: HotThesis[];
    survivors: Survivor[];
}

export const fetchCommunityRankings = async (signal?: AbortSignal): Promise<CommunityRankings> => {
    const response = await fetchWithAuth(`${API_BASE_URL}/api/community/rankings`, { signal });
    const data = await response.json();

    if (!data || !Array.isArray(data.top_communities) || !Array.isArray(data.hot_theses) || !Array.isArray(data.survivors)) {
        throw new Error('Unexpected community rankings response shape');
    }

    return data as CommunityRankings;
};


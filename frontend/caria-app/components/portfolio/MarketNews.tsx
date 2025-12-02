import React, { useState, useEffect } from 'react';
import { Newspaper, TrendingUp, TrendingDown, ExternalLink } from 'lucide-react';
import { API_BASE_URL, getToken } from '../../services/apiService';

interface NewsArticle {
    id: number;
    title: string;
    source_domain: string;
    url: string;
    published_at: string;
    tone: number | null;
    related_tickers?: string[];
}

interface MarketNewsProps {
    tickers?: string[];
}

export const MarketNews: React.FC<MarketNewsProps> = ({ tickers }) => {
    const [news, setNews] = useState<NewsArticle[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchNews();
    }, [tickers]);

    const fetchNews = async () => {
        setLoading(true);
        try {
            const token = getToken();
            const headers: HeadersInit = { 'Content-Type': 'application/json' };
            if (token) headers['Authorization'] = `Bearer ${token}`;

            const tickerParam = tickers && tickers.length > 0 ? `?tickers=${tickers.join(',')}` : '';
            const response = await fetch(`${API_BASE_URL}/api/news/market${tickerParam}`, { headers });

            if (response.ok) {
                const data = await response.json();
                setNews(data);
            } else {
                console.error('Failed to fetch news');
                setNews([]);
            }
        } catch (error) {
            console.error('Error fetching news:', error);
            setNews([]);
        } finally {
            setLoading(false);
        }
    };

    const getToneIcon = (tone: number | null) => {
        if (!tone) return null;
        if (tone > 2) return <TrendingUp className="w-4 h-4 text-green-400" />;
        if (tone < -2) return <TrendingDown className="w-4 h-4 text-red-400" />;
        return null;
    };

    const getToneColor = (tone: number | null) => {
        if (!tone) return 'text-white/30';
        if (tone > 2) return 'text-green-400';
        if (tone < -2) return 'text-red-400';
        return 'text-white/50';
    };

    const formatDate = (dateStr: string) => {
        const date = new Date(dateStr);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffHrs = Math.floor(diffMs / (1000 * 60 * 60));

        if (diffHrs < 1) return 'Just now';
        if (diffHrs < 24) return `${diffHrs}h ago`;
        const diffDays = Math.floor(diffHrs / 24);
        if (diffDays === 1) return 'Yesterday';
        return `${diffDays}d ago`;
    };

    return (
        <div className="bg-white/5 border border-white/10 rounded-lg p-6">
            <div className="flex items-center gap-2 mb-4">
                <Newspaper className="w-5 h-5 text-blue-400" />
                <h3 className="text-lg font-semibold text-white">What's Happening</h3>
            </div>

            {loading ? (
                <div className="space-y-3">
                    {[...Array(3)].map((_, i) => (
                        <div key={i} className="animate-pulse">
                            <div className="h-4 bg-white/10 rounded w-3/4 mb-2"></div>
                            <div className="h-3 bg-white/5 rounded w-1/2"></div>
                        </div>
                    ))}
                </div>
            ) : news.length === 0 ? (
                <p className="text-white/30 text-sm">No recent news available. Check back later.</p>
            ) : (
                <div className="space-y-4">
                    {news.map((article) => (
                        <a
                            key={article.id}
                            href={article.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block group"
                        >
                            <div className="border-l-2 border-white/10 pl-3 hover:border-blue-400 transition-colors">
                                <div className="flex items-start gap-2">
                                    <div className="flex-1">
                                        <h4 className="text-sm text-white group-hover:text-blue-400 transition-colors line-clamp-2">
                                            {article.title}
                                        </h4>
                                        <div className="flex items-center gap-2 mt-1 text-xs text-white/40">
                                            <span>{article.source_domain}</span>
                                            <span>•</span>
                                            <span>{formatDate(article.published_at)}</span>
                                            {article.tone !== null && (
                                                <>
                                                    <span>•</span>
                                                    <span className={getToneColor(article.tone)}>
                                                        Tone: {article.tone.toFixed(1)}
                                                    </span>
                                                </>
                                            )}
                                        </div>
                                        {article.related_tickers && article.related_tickers.length > 0 && (
                                            <div className="flex gap-1 mt-2">
                                                {article.related_tickers.map((ticker) => (
                                                    <span
                                                        key={ticker}
                                                        className="text-xs px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded"
                                                    >
                                                        {ticker}
                                                    </span>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-2 flex-shrink-0">
                                        {getToneIcon(article.tone)}
                                        <ExternalLink className="w-3 h-3 text-white/20 group-hover:text-blue-400 transition-colors" />
                                    </div>
                                </div>
                            </div>
                        </a>
                    ))}
                </div>
            )}
        </div>
    );
};

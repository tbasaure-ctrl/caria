import React, { useState, useEffect } from 'react';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';
import { getErrorMessage } from '../../src/utils/errorHandling';

interface RedditStock {
    ticker: string;
    mentions: number;
    sentiment: 'bullish' | 'bearish' | 'neutral';
    trending_score: number;
    top_post_title?: string;
    subreddit?: string;
}

export const RedditSentiment: React.FC = () => {
    const [stocks, setStocks] = useState<RedditStock[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [timeframe, setTimeframe] = useState<'hour' | 'day' | 'week'>('day');

    useEffect(() => {
        fetchRedditData();
    }, [timeframe]);

    const fetchRedditData = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/social/reddit?timeframe=${timeframe}`);
            if (!response.ok) throw new Error('Failed to fetch Reddit data');
            const data = await response.json();
            const stocks = data.stocks || [];
            // Remove duplicates and improve variety by filtering similar tickers
            const uniqueStocks = stocks.filter((stock: RedditStock, index: number, self: RedditStock[]) => 
                index === self.findIndex((s: RedditStock) => s.ticker === stock.ticker)
            );
            // Limit to top 5 for better variety
            setStocks(uniqueStocks.slice(0, 5));
        } catch (err: unknown) {
            setError('Coming soon... Reddit sentiment analysis is being enhanced to provide even better social media insights.');
            // Fallback mock data for development
            setStocks([
                { ticker: 'NVDA', mentions: 1247, sentiment: 'bullish', trending_score: 92, top_post_title: 'NVDA earnings beat expectations', subreddit: 'wallstreetbets' },
                { ticker: 'TSLA', mentions: 856, sentiment: 'neutral', trending_score: 78, top_post_title: 'Tesla production numbers released', subreddit: 'stocks' },
                { ticker: 'AAPL', mentions: 634, sentiment: 'bullish', trending_score: 71, top_post_title: 'Apple Vision Pro sales surging', subreddit: 'investing' },
                { ticker: 'SPY', mentions: 521, sentiment: 'bearish', trending_score: 65, top_post_title: 'Market correction incoming?', subreddit: 'wallstreetbets' },
                { ticker: 'AMD', mentions: 412, sentiment: 'bullish', trending_score: 58, top_post_title: 'AMD new chip announcement', subreddit: 'stocks' }
            ]);
        } finally {
            setIsLoading(false);
        }
    };

    const getSentimentColor = (sentiment: string) => {
        switch(sentiment) {
            case 'bullish': return '#10b981'; // green
            case 'bearish': return '#ef4444'; // red
            case 'neutral': return '#6b7280'; // gray
            default: return 'var(--color-text-secondary)';
        }
    };

    const getSentimentEmoji = (sentiment: string) => {
        switch(sentiment) {
            case 'bullish': return '🚀';
            case 'bearish': return '📉';
            case 'neutral': return '➡️';
            default: return '•';
        }
    };

    return (
        <div className="rounded-lg p-6" style={{
            backgroundColor: 'var(--color-bg-secondary)',
            border: '1px solid var(--color-bg-tertiary)'
        }}>
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold" style={{
                    fontFamily: 'var(--font-display)',
                    color: 'var(--color-cream)'
                }}>
                    Trending on Reddit
                </h2>
                <button
                    onClick={fetchRedditData}
                    className="p-2 rounded transition-all hover:bg-slate-700"
                    title="Refresh"
                >
                    🔄
                </button>
            </div>

            <p className="text-sm mb-4" style={{color: 'var(--color-text-secondary)'}}>
                Most mentioned stocks on r/wallstreetbets, r/stocks, and r/investing
            </p>

            {/* Timeframe selector */}
            <div className="flex gap-2 mb-4">
                {(['hour', 'day', 'week'] as const).map(tf => (
                    <button
                        key={tf}
                        onClick={() => setTimeframe(tf)}
                        className="px-3 py-1 rounded text-sm transition-all"
                        style={{
                            backgroundColor: timeframe === tf ? 'var(--color-primary)' : 'var(--color-bg-tertiary)',
                            color: timeframe === tf ? 'var(--color-cream)' : 'var(--color-text-secondary)',
                        }}
                    >
                        {tf.charAt(0).toUpperCase() + tf.slice(1)}
                    </button>
                ))}
            </div>

            {error && (
                <div className="mb-4 p-3 rounded bg-yellow-900/30 text-yellow-200 text-sm">
                    ⚠️ Using demo data (API not connected yet)
                </div>
            )}

            {isLoading ? (
                <div className="text-center py-8" style={{color: 'var(--color-text-secondary)'}}>
                    Loading...
                </div>
            ) : stocks.length === 0 ? (
                <div className="text-center py-8" style={{color: 'var(--color-text-secondary)'}}>
                    No data available
                </div>
            ) : (
                <div className="space-y-3">
                    {stocks.map((stock, idx) => (
                        <div
                            key={idx}
                            className="p-4 rounded-lg transition-all hover:transform hover:-translate-y-1 cursor-pointer"
                            style={{
                                backgroundColor: 'var(--color-bg-primary)',
                                border: '1px solid var(--color-bg-tertiary)',
                            }}
                        >
                            <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-3">
                                    <span className="text-lg font-bold" style={{color: 'var(--color-cream)'}}>
                                        ${stock.ticker}
                                    </span>
                                    <span
                                        className="text-sm px-2 py-1 rounded"
                                        style={{
                                            backgroundColor: getSentimentColor(stock.sentiment) + '20',
                                            color: getSentimentColor(stock.sentiment)
                                        }}
                                    >
                                        {getSentimentEmoji(stock.sentiment)} {stock.sentiment}
                                    </span>
                                </div>
                                <div className="text-right">
                                    <div className="text-sm font-bold" style={{color: 'var(--color-cream)'}}>
                                        {stock.mentions.toLocaleString()} mentions
                                    </div>
                                    <div className="text-xs" style={{color: 'var(--color-text-secondary)'}}>
                                        Trending: {stock.trending_score}/100
                                    </div>
                                </div>
                            </div>

                            {stock.top_post_title && (
                                <div className="mt-2 pt-2 border-t" style={{borderColor: 'var(--color-bg-tertiary)'}}>
                                    <p className="text-sm" style={{color: 'var(--color-text-secondary)'}}>
                                        💬 "{stock.top_post_title}"
                                    </p>
                                    {stock.subreddit && (
                                        <p className="text-xs mt-1" style={{color: 'var(--color-text-secondary)'}}>
                                            r/{stock.subreddit}
                                        </p>
                                    )}
                                </div>
                            )}

                            {/* Trending bar */}
                            <div className="mt-3 h-2 bg-gray-700 rounded-full overflow-hidden">
                                <div
                                    className="h-full rounded-full transition-all duration-500"
                                    style={{
                                        width: `${stock.trending_score}%`,
                                        backgroundColor: getSentimentColor(stock.sentiment)
                                    }}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            )}

            <div className="mt-4 p-3 rounded text-xs" style={{
                backgroundColor: 'var(--color-bg-tertiary)',
                color: 'var(--color-text-secondary)'
            }}>
                ⚠️ Social sentiment is not investment advice. Always do your own research.
            </div>
        </div>
    );
};

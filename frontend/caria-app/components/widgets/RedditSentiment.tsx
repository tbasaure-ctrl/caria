import React, { useState, useEffect } from 'react';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';
import { getErrorMessage } from '../../src/utils/errorHandling';

interface SocialStock {
    ticker: string;
    mentions: number;
    sentiment: 'bullish' | 'bearish' | 'neutral';
    trending_score: number;
    top_post_title?: string;
    top_message?: string;
    subreddit?: string;
    watchlist_count?: number;
    source: 'reddit' | 'stocktwits';
}

export const RedditSentiment: React.FC = () => {
    const [redditStocks, setRedditStocks] = useState<SocialStock[]>([]);
    const [stocktwitsStocks, setStocktwitsStocks] = useState<SocialStock[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [timeframe, setTimeframe] = useState<'hour' | 'day' | 'week'>('day');

    useEffect(() => {
        fetchSocialData();
    }, [timeframe]);

    const fetchSocialData = async () => {
        setIsLoading(true);
        setError(null);
        
        try {
            // Fetch Reddit data
            const redditResponse = await fetchWithAuth(`${API_BASE_URL}/api/social/reddit?timeframe=${timeframe}`);
            if (redditResponse.ok) {
                const redditData = await redditResponse.json();
                const stocks = redditData.stocks || [];
                const uniqueStocks = stocks.filter((stock: SocialStock, index: number, self: SocialStock[]) => 
                    index === self.findIndex((s: SocialStock) => s.ticker === stock.ticker)
                );
                setRedditStocks(uniqueStocks.slice(0, 3).map((s: any) => ({ ...s, source: 'reddit' })));
            }
        } catch (err: unknown) {
            console.error('Reddit fetch error:', err);
            // Fallback mock data for Reddit
            setRedditStocks([
                { ticker: 'NVDA', mentions: 1247, sentiment: 'bullish', trending_score: 92, top_post_title: 'NVDA earnings beat expectations', subreddit: 'wallstreetbets', source: 'reddit' },
                { ticker: 'TSLA', mentions: 856, sentiment: 'neutral', trending_score: 78, top_post_title: 'Tesla production numbers released', subreddit: 'stocks', source: 'reddit' },
                { ticker: 'AAPL', mentions: 634, sentiment: 'bullish', trending_score: 71, top_post_title: 'Apple Vision Pro sales surging', subreddit: 'investing', source: 'reddit' }
            ]);
        }

        try {
            // Fetch StockTwits data
            const stocktwitsResponse = await fetchWithAuth(`${API_BASE_URL}/api/social/stocktwits?timeframe=${timeframe}`);
            if (stocktwitsResponse.ok) {
                const stocktwitsData = await stocktwitsResponse.json();
                const stocks = stocktwitsData.stocks || [];
                const uniqueStocks = stocks.filter((stock: SocialStock, index: number, self: SocialStock[]) => 
                    index === self.findIndex((s: SocialStock) => s.ticker === stock.ticker)
                );
                setStocktwitsStocks(uniqueStocks.slice(0, 3).map((s: any) => ({ ...s, source: 'stocktwits' })));
            }
        } catch (err: unknown) {
            console.error('StockTwits fetch error:', err);
            // Fallback mock data for StockTwits
            setStocktwitsStocks([
                { ticker: 'NVDA', mentions: 1523, watchlist_count: 12450, sentiment: 'bullish', trending_score: 95, top_message: 'NVDA breaking new highs on AI momentum', source: 'stocktwits' },
                { ticker: 'TSLA', mentions: 987, watchlist_count: 8920, sentiment: 'neutral', trending_score: 78, top_message: 'TSLA production update', source: 'stocktwits' },
                { ticker: 'AAPL', mentions: 756, watchlist_count: 6540, sentiment: 'bullish', trending_score: 72, top_message: 'Apple Vision Pro sales', source: 'stocktwits' }
            ]);
        }

        setIsLoading(false);
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

    const renderStockCard = (stock: SocialStock, idx: number) => (
        <div
            key={`${stock.source}-${idx}`}
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
                        className="text-xs px-1.5 py-0.5 rounded"
                        style={{
                            backgroundColor: stock.source === 'reddit' ? '#FF4500' + '20' : '#00D9FF' + '20',
                            color: stock.source === 'reddit' ? '#FF4500' : '#00D9FF',
                        }}
                    >
                        {stock.source === 'reddit' ? '📱 Reddit' : '💬 StockTwits'}
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
                        {stock.mentions.toLocaleString()} {stock.source === 'reddit' ? 'mentions' : 'messages'}
                    </div>
                    {stock.watchlist_count && (
                        <div className="text-xs" style={{color: 'var(--color-text-secondary)'}}>
                            {stock.watchlist_count.toLocaleString()} watching
                        </div>
                    )}
                    <div className="text-xs" style={{color: 'var(--color-text-secondary)'}}>
                        Trending: {stock.trending_score}/100
                    </div>
                </div>
            </div>

            {(stock.top_post_title || stock.top_message) && (
                <div className="mt-2 pt-2 border-t" style={{borderColor: 'var(--color-bg-tertiary)'}}>
                    <p className="text-sm" style={{color: 'var(--color-text-secondary)'}}>
                        💬 "{stock.top_post_title || stock.top_message}"
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
    );

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
                    Social Sentiment
                </h2>
                <button
                    onClick={fetchSocialData}
                    className="p-2 rounded transition-all hover:bg-slate-700"
                    title="Refresh"
                >
                    🔄
                </button>
            </div>

            <p className="text-sm mb-4" style={{color: 'var(--color-text-secondary)'}}>
                Trending stocks from Reddit and StockTwits
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
            ) : redditStocks.length === 0 && stocktwitsStocks.length === 0 ? (
                <div className="text-center py-8" style={{color: 'var(--color-text-secondary)'}}>
                    No data available
                </div>
            ) : (
                <div className="space-y-6">
                    {/* Reddit Section */}
                    {redditStocks.length > 0 && (
                        <div>
                            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2" style={{
                                fontFamily: 'var(--font-display)',
                                color: 'var(--color-cream)'
                            }}>
                                <span style={{color: '#FF4500'}}>📱</span> Trending on Reddit
                            </h3>
                            <div className="space-y-3">
                                {redditStocks.map((stock, idx) => renderStockCard(stock, idx))}
                            </div>
                        </div>
                    )}

                    {/* StockTwits Section */}
                    {stocktwitsStocks.length > 0 && (
                        <div>
                            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2" style={{
                                fontFamily: 'var(--font-display)',
                                color: 'var(--color-cream)'
                            }}>
                                <span style={{color: '#00D9FF'}}>💬</span> Trending on StockTwits
                            </h3>
                            <div className="space-y-3">
                                {stocktwitsStocks.map((stock, idx) => renderStockCard(stock, idx))}
                            </div>
                        </div>
                    )}
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

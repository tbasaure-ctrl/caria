/**
 * WeeklyMedia - Displays podcast and YouTube video of the week
 * Easy to update weekly by modifying the mediaItems array
 * YouTube titles are fetched automatically if not provided
 */

import React, { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';

interface MediaItem {
    type: 'podcast' | 'youtube';
    title?: string; // Optional - will be fetched from YouTube if not provided
    description: string;
    url: string;
    thumbnail?: string;
    host?: string;
}

// Update this array weekly with new podcast and YouTube video
const mediaItems: MediaItem[] = [
    {
        type: 'youtube',
        description: 'Weekly investment video recommendation',
        url: 'https://www.youtube.com/watch?v=BCY6MycxtIo',
        thumbnail: 'https://img.youtube.com/vi/BCY6MycxtIo/hqdefault.jpg'
    },
    {
        type: 'podcast',
        title: 'Podcast of the Week',
        description: 'Weekly curated investment podcast',
        url: 'https://open.spotify.com/episode/4RyYkVkiFSZUdxWTiod8QG?si=4fe4c86534d540a5'
    }
];

// Function to extract video ID from YouTube URL
const getYouTubeVideoId = (url: string): string | null => {
    const patterns = [
        /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/,
        /youtube\.com\/watch\?.*v=([^&\n?#]+)/
    ];
    
    for (const pattern of patterns) {
        const match = url.match(pattern);
        if (match && match[1]) {
            return match[1];
        }
    }
    return null;
};

// Function to fetch YouTube video title using oEmbed API
const fetchYouTubeTitle = async (url: string): Promise<string | null> => {
    try {
        const videoId = getYouTubeVideoId(url);
        if (!videoId) return null;
        
        const oEmbedUrl = `https://www.youtube.com/oembed?url=${encodeURIComponent(url)}&format=json`;
        const response = await fetch(oEmbedUrl);
        
        if (!response.ok) return null;
        
        const data = await response.json();
        return data.title || null;
    } catch (error) {
        console.error('Error fetching YouTube title:', error);
        return null;
    }
};

interface WeeklyMediaProps {
    compact?: boolean;
}

export const WeeklyMedia: React.FC<WeeklyMediaProps> = ({ compact = false }) => {
    const [itemsWithTitles, setItemsWithTitles] = useState<MediaItem[]>(mediaItems);

    useEffect(() => {
        // Fetch YouTube titles for items that don't have titles
        const fetchTitles = async () => {
            const updatedItems = await Promise.all(
                mediaItems.map(async (item) => {
                    if (item.type === 'youtube' && !item.title) {
                        const title = await fetchYouTubeTitle(item.url);
                        return { ...item, title: title || 'Investment Video' };
                    }
                    return item;
                })
            );
            setItemsWithTitles(updatedItems);
        };

        fetchTitles();
    }, []);

    // If no items, show placeholder
    if (itemsWithTitles.length === 0) {
        return (
            <WidgetCard
                title="📺 Media of the Week"
                tooltip="Weekly curated podcast and YouTube video recommendations"
            >
                <div className="text-center py-4">
                    <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
                        Weekly media recommendations coming soon...
                    </p>
                </div>
            </WidgetCard>
        );
    }

    if (compact) {
        // Compact version: Two cards side by side with smaller size
        return (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {itemsWithTitles.map((item, index) => (
                    <a
                        key={index}
                        href={item.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block group"
                    >
                        <div
                            className="rounded-lg p-3 transition-all duration-200 border h-full"
                            style={{
                                backgroundColor: 'var(--color-bg-secondary)',
                                borderColor: 'var(--color-bg-tertiary)',
                            }}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.borderColor = 'var(--color-primary)';
                                e.currentTarget.style.transform = 'translateY(-2px)';
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.borderColor = 'var(--color-bg-tertiary)';
                                e.currentTarget.style.transform = 'translateY(0)';
                            }}
                        >
                            <div className="flex items-center gap-3">
                                {/* Small icon */}
                                <div className="flex-shrink-0">
                                    {item.thumbnail ? (
                                        <img
                                            src={item.thumbnail}
                                            alt={item.title}
                                            className="w-12 h-12 rounded object-cover"
                                        />
                                    ) : (
                                        <div
                                            className="w-12 h-12 rounded flex items-center justify-center"
                                            style={{
                                                backgroundColor: 'var(--color-blue)',
                                            }}
                                        >
                                            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24" style={{ color: 'var(--color-cream)' }}>
                                                <path d="M10 16.5l6-4.5-6-4.5v9zM12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
                                            </svg>
                                        </div>
                                    )}
                                </div>

                                {/* Content */}
                                <div className="flex-1 min-w-0">
                                    {item.type === 'youtube' && (
                                        <div className="flex items-center gap-2 mb-0.5">
                                            <span
                                                className="text-[10px] font-semibold px-1.5 py-0.5 rounded"
                                                style={{
                                                    backgroundColor: 'var(--color-blue)',
                                                    color: 'var(--color-cream)',
                                                }}
                                            >
                                                ▶️ Video
                                            </span>
                                        </div>
                                    )}
                                    {item.type === 'podcast' && (
                                        <div className="flex items-center gap-2 mb-0.5">
                                            <span
                                                className="text-[10px] font-semibold px-1.5 py-0.5 rounded"
                                                style={{
                                                    backgroundColor: 'var(--color-primary)',
                                                    color: 'var(--color-cream)',
                                                }}
                                            >
                                                🎙️ Podcast
                                            </span>
                                        </div>
                                    )}
                                    <h3
                                        className="text-sm font-bold truncate group-hover:underline"
                                        style={{
                                            fontFamily: 'var(--font-display)',
                                            color: 'var(--color-cream)',
                                        }}
                                    >
                                        {item.title || 'Media of the Week'}
                                    </h3>
                                    <p
                                        className="text-xs truncate"
                                        style={{
                                            fontFamily: 'var(--font-body)',
                                            color: 'var(--color-text-secondary)',
                                        }}
                                    >
                                        {item.description}
                                    </p>
                                </div>

                                {/* Arrow */}
                                <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" style={{ color: 'var(--color-primary)' }}>
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                </svg>
                            </div>
                        </div>
                    </a>
                ))}
            </div>
        );
    }

    // Full version (original)
    return (
        <WidgetCard
            title="📺 Media of the Week"
            tooltip="Weekly curated podcast and YouTube video recommendations"
        >
            <div className="space-y-4">
                {itemsWithTitles.map((item, index) => (
                    <a
                        key={index}
                        href={item.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block group"
                    >
                        <div
                            className="rounded-lg p-4 transition-all duration-200 border"
                            style={{
                                backgroundColor: 'var(--color-bg-secondary)',
                                borderColor: 'var(--color-bg-tertiary)',
                            }}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.borderColor = 'var(--color-primary)';
                                e.currentTarget.style.transform = 'translateY(-2px)';
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.borderColor = 'var(--color-bg-tertiary)';
                                e.currentTarget.style.transform = 'translateY(0)';
                            }}
                        >
                            <div className="flex items-start gap-4">
                                {/* Icon/Thumbnail */}
                                <div className="flex-shrink-0">
                                    {item.thumbnail ? (
                                        <img
                                            src={item.thumbnail}
                                            alt={item.title}
                                            className="w-16 h-16 rounded object-cover"
                                        />
                                    ) : (
                                        <div
                                            className="w-16 h-16 rounded flex items-center justify-center"
                                            style={{
                                                backgroundColor: 'var(--color-blue)',
                                            }}
                                        >
                                            <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24" style={{ color: 'var(--color-cream)' }}>
                                                <path d="M10 16.5l6-4.5-6-4.5v9zM12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
                                            </svg>
                                        </div>
                                    )}
                                </div>

                                {/* Content */}
                                <div className="flex-1 min-w-0">
                                    {item.type === 'youtube' && (
                                        <div className="flex items-center gap-2 mb-1">
                                            <span
                                                className="text-xs font-semibold px-2 py-1 rounded"
                                                style={{
                                                    backgroundColor: 'var(--color-blue)',
                                                    color: 'var(--color-cream)',
                                                }}
                                            >
                                                ▶️ YouTube
                                            </span>
                                        </div>
                                    )}
                                    {item.type === 'podcast' && (
                                        <div className="flex items-center gap-2 mb-1">
                                            <span
                                                className="text-xs font-semibold px-2 py-1 rounded"
                                                style={{
                                                    backgroundColor: 'var(--color-primary)',
                                                    color: 'var(--color-cream)',
                                                }}
                                            >
                                                🎙️ Spotify
                                            </span>
                                        </div>
                                    )}
                                    <h3
                                        className="text-base font-bold mb-1 group-hover:underline"
                                        style={{
                                            fontFamily: 'var(--font-display)',
                                            color: 'var(--color-cream)',
                                        }}
                                    >
                                        {item.title || 'Media of the Week'}
                                    </h3>
                                    <p
                                        className="text-sm line-clamp-2"
                                        style={{
                                            fontFamily: 'var(--font-body)',
                                            color: 'var(--color-text-secondary)',
                                        }}
                                    >
                                        {item.description}
                                    </p>
                                    <div className="mt-2 text-xs flex items-center gap-1" style={{ color: 'var(--color-primary)' }}>
                                        <span>Watch now</span>
                                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                        </svg>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </a>
                ))}
            </div>
        </WidgetCard>
    );
};

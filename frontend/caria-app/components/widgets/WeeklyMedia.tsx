/**
 * WeeklyMedia - Displays podcast and YouTube video of the week
 * Easy to update weekly by modifying the mediaItems array
 */

import React from 'react';
import { WidgetCard } from './WidgetCard';

interface MediaItem {
    type: 'podcast' | 'youtube';
    title: string;
    description: string;
    url: string;
    thumbnail?: string;
    host?: string;
}

// Update this array weekly with new podcast and YouTube video
const mediaItems: MediaItem[] = [
    {
        type: 'youtube',
        title: 'Charlie Munger - Taking Drawdowns with Philosophy and Equanimity',
        description: 'Wisdom on handling market downturns with philosophical perspective and calm composure.',
        url: 'https://youtu.be/7sNUg2kPJwI?si=wZfqg2mKkVD8HN4w',
        thumbnail: 'https://img.youtube.com/vi/7sNUg2kPJwI/maxresdefault.jpg'
    },
    {
        type: 'podcast',
        title: 'The Knowledge Project - Morgan Housel',
        description: 'Insights on investing, decision-making, and understanding human behavior in markets.',
        url: 'https://open.spotify.com/episode/00ll36IR6rEoigiqFRwBzW?si=Qa78MY88QjWilhw4lJjQCA',
        host: 'The Knowledge Project'
    }
];

export const WeeklyMedia: React.FC = () => {
    // If no items, show placeholder
    if (mediaItems.length === 0) {
        return (
            <WidgetCard
                title="📺 Media of the Week"
                tooltip="Weekly curated podcast and YouTube video recommendations"
            >
                <div className="text-center py-8">
                    <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
                        Weekly media recommendations coming soon...
                    </p>
                </div>
            </WidgetCard>
        );
    }

    return (
        <WidgetCard
            title="📺 Media of the Week"
            tooltip="Weekly curated podcast and YouTube video recommendations"
        >
            <div className="space-y-4">
                {mediaItems.map((item, index) => (
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
                                                backgroundColor: item.type === 'podcast' 
                                                    ? 'var(--color-primary)' 
                                                    : 'var(--color-blue)',
                                            }}
                                        >
                                            {item.type === 'podcast' ? (
                                                <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24" style={{ color: 'var(--color-cream)' }}>
                                                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z"/>
                                                </svg>
                                            ) : (
                                                <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24" style={{ color: 'var(--color-cream)' }}>
                                                    <path d="M10 16.5l6-4.5-6-4.5v9zM12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
                                                </svg>
                                            )}
                                        </div>
                                    )}
                                </div>

                                {/* Content */}
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 mb-1">
                                        <span
                                            className="text-xs font-semibold px-2 py-1 rounded"
                                            style={{
                                                backgroundColor: item.type === 'podcast' 
                                                    ? 'var(--color-primary)' 
                                                    : 'var(--color-blue)',
                                                color: 'var(--color-cream)',
                                            }}
                                        >
                                            {item.type === 'podcast' ? '🎙️ Podcast' : '▶️ YouTube'}
                                        </span>
                                        {item.host && (
                                            <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                                                {item.host}
                                            </span>
                                        )}
                                    </div>
                                    <h3
                                        className="text-base font-bold mb-1 group-hover:underline"
                                        style={{
                                            fontFamily: 'var(--font-display)',
                                            color: 'var(--color-cream)',
                                        }}
                                    >
                                        {item.title}
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

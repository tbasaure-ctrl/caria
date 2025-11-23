import React from 'react';
import { WidgetCard } from './WidgetCard';

interface ContentItem {
    id: string;
    title: string;
    description: string;
    type: 'podcast' | 'video';
    youtubeUrl?: string;
    spotifyUrl?: string;
    thumbnailUrl?: string;
}

// Configuration object - easy to update weekly
const WEEKLY_CONTENT: ContentItem[] = [
    {
        id: 'pod-1',
        title: 'The Memo by Howard Marks: Sea Change',
        description: 'Howard Marks discusses the fundamental shifts in the investment landscape and why the strategies of the last decade may not work in the next.',
        type: 'podcast',
        youtubeUrl: 'https://www.youtube.com/watch?v=example',
        spotifyUrl: 'https://open.spotify.com/episode/example',
    },
    {
        id: 'vid-1',
        title: 'Aswath Damodaran: Valuing the Magnificent Seven',
        description: 'The "Dean of Valuation" breaks down the numbers behind the biggest tech stocks and whether they are still a buy.',
        type: 'video',
        youtubeUrl: 'https://www.youtube.com/watch?v=example',
        thumbnailUrl: 'https://img.youtube.com/vi/uWj6is1ae_0/maxresdefault.jpg' // Example thumbnail
    }
];

export const WeeklyContent: React.FC = () => {
    const podcast = WEEKLY_CONTENT.find(c => c.type === 'podcast');
    const video = WEEKLY_CONTENT.find(c => c.type === 'video');

    return (
        <div className="space-y-6">
            {/* Weekly Podcast Section */}
            {podcast && (
                <WidgetCard title="🎧 Weekly Podcast" tooltip="Curated investment podcast of the week">
                    <div className="flex flex-col gap-4">
                        <div>
                            <h4 className="font-bold text-lg text-[var(--color-cream)] leading-tight mb-2">
                                {podcast.title}
                            </h4>
                            <p className="text-sm text-[var(--color-text-secondary)]">
                                {podcast.description}
                            </p>
                        </div>
                        <div className="flex gap-3">
                            {podcast.spotifyUrl && (
                                <a 
                                    href={podcast.spotifyUrl} 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[#1DB954]/10 hover:bg-[#1DB954]/20 text-[#1DB954] transition-colors text-sm font-bold"
                                >
                                    <span>Listen on Spotify</span>
                                </a>
                            )}
                            {podcast.youtubeUrl && (
                                <a 
                                    href={podcast.youtubeUrl} 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[#FF0000]/10 hover:bg-[#FF0000]/20 text-[#FF0000] transition-colors text-sm font-bold"
                                >
                                    <span>Watch on YouTube</span>
                                </a>
                            )}
                        </div>
                    </div>
                </WidgetCard>
            )}

            {/* Weekly Video Section */}
            {video && (
                <WidgetCard title="📺 Weekly Video" tooltip="Must-watch investment video of the week">
                    <div className="flex flex-col gap-4">
                        {video.thumbnailUrl && (
                            <div className="relative aspect-video w-full rounded-lg overflow-hidden group">
                                <img 
                                    src={video.thumbnailUrl} 
                                    alt={video.title} 
                                    className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105 opacity-80 group-hover:opacity-100"
                                />
                                <a 
                                    href={video.youtubeUrl}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="absolute inset-0 flex items-center justify-center"
                                >
                                    <div className="w-12 h-12 rounded-full bg-red-600 flex items-center justify-center shadow-lg transform transition-transform group-hover:scale-110">
                                        <svg className="w-6 h-6 text-white ml-1" fill="currentColor" viewBox="0 0 24 24">
                                            <path d="M8 5v14l11-7z" />
                                        </svg>
                                    </div>
                                </a>
                            </div>
                        )}
                        <div>
                            <h4 className="font-bold text-lg text-[var(--color-cream)] leading-tight mb-2">
                                {video.title}
                            </h4>
                            <p className="text-sm text-[var(--color-text-secondary)]">
                                {video.description}
                            </p>
                        </div>
                    </div>
                </WidgetCard>
            )}
        </div>
    );
};

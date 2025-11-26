import React, { useState, useEffect } from 'react';
import { fetchRecommendedLectures, LectureRecommendation } from '../../services/apiService';

interface Resource {
    title: string;
    description: string;
    type: 'article' | 'lecture' | 'book';
    url?: string; // Optional - if no URL, show content inline
    content?: string; // For articles without external links
    tags: string[];
}

const STATIC_RESOURCES: Resource[] = [
    {
        title: "The Psychology of Money",
        description: "Understanding behavioral finance and cognitive biases in investing",
        type: "book",
        url: "https://www.amazon.com/Psychology-Money-Timeless-lessons-happiness/dp/0857197681",
        tags: ["behavioral", "psychology"]
    },
    {
        title: "Thinking, Fast and Slow",
        description: "Daniel Kahneman's masterwork on cognitive biases and decision-making",
        type: "book",
        url: "https://www.amazon.com/Thinking-Fast-Slow-Daniel-Kahneman/dp/0374533555",
        tags: ["behavioral", "decision"]
    },
    {
        title: "Common Investment Biases",
        description: "Learn about anchoring bias, confirmation bias, and how they affect your portfolio decisions",
        type: "article",
        content: `
# Common Investment Biases

## 1. Confirmation Bias
You tend to seek information that confirms your existing beliefs while ignoring contradicting evidence.

**How it affects investing**: You might only read bullish articles about a stock you own.

## 2. Anchoring Bias
You rely too heavily on the first piece of information (the "anchor") when making decisions.

**How it affects investing**: Fixating on your purchase price instead of current valuation.

## 3. Loss Aversion
The pain of losing is psychologically twice as powerful as the pleasure of gaining.

**How it affects investing**: Holding losing positions too long, hoping they'll recover.

## 4. Herd Mentality
Following what everyone else is doing without independent analysis.

**How it affects investing**: Buying hot stocks because "everyone is talking about them."
        `,
        tags: ["behavioral", "bias"]
    },
    {
        title: "Warren Buffett's Letters to Shareholders",
        description: "Decades of wisdom from the Oracle of Omaha",
        type: "article",
        url: "https://www.berkshirehathaway.com/letters/letters.html",
        tags: ["valuation", "value investing"]
    },
    {
        title: "Valuation Methods Explained",
        description: "DCF, Comparables, and other valuation techniques",
        type: "lecture",
        content: `
# Valuation Methods

## Discounted Cash Flow (DCF)
Value a company based on projected future cash flows, discounted to present value.

**When to use**: Companies with predictable cash flows

## Comparable Company Analysis
Value a company relative to similar public companies (P/E, EV/EBITDA multiples).

**When to use**: When good comparables exist

## Precedent Transactions
Value based on M&A deals of similar companies.

**When to use**: For M&A scenarios

**Remember**: No valuation method is perfect. Use multiple approaches.
        `,
        tags: ["valuation", "education"]
    },
    {
        title: "Market Cycles and Regime Detection",
        description: "Understanding bull markets, bear markets, and everything in between",
        type: "article",
        content: `
# Market Cycles

Markets move in cycles. Understanding where we are in the cycle helps with positioning.

## Four Market Regimes

1. **Bull Market (Expansion)**: Rising prices, economic growth, investor confidence
2. **Peak (Late Cycle)**: High valuations, euphoria, warning signs emerge
3. **Bear Market (Contraction)**: Falling prices, recession fears, panic selling
4. **Trough (Early Cycle)**: Low valuations, pessimism, but recovery beginning

## How Caria Helps
Our regime detection model analyzes multiple indicators to identify the current market environment.
        `,
        tags: ["macro", "cycles"]
    }
];

export const Resources: React.FC = () => {
    const [selectedResource, setSelectedResource] = useState<Resource | null>(null);
    const [filter, setFilter] = useState<string>('all');
    const [resources, setResources] = useState<Resource[]>(STATIC_RESOURCES);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const loadLectures = async () => {
            setLoading(true);
            try {
                const lectures = await fetchRecommendedLectures();
                const mappedLectures: Resource[] = lectures.map(l => ({
                    title: l.title,
                    description: `${l.source} • ${new Date(l.date).toLocaleDateString()}`,
                    type: 'article',
                    url: l.url,
                    tags: ['finance', 'curated', l.source.toLowerCase()]
                }));

                // Limit to top 3-4 lectures to reduce clutter
                const limitedLectures = mappedLectures.slice(0, 4);
                // Prepend dynamic lectures to static resources
                setResources([...limitedLectures, ...STATIC_RESOURCES]);
            } catch (err) {
                console.error("Failed to load lectures:", err);
                // Keep static resources on error
            } finally {
                setLoading(false);
            }
        };

        loadLectures();
    }, []);

    const filteredResources = filter === 'all'
        ? resources
        : resources.filter(r => r.tags.some(tag => tag.toLowerCase().includes(filter.toLowerCase())));

    const getTypeColor = (type: string) => {
        switch (type) {
            case 'book': return 'var(--color-primary)';
            case 'lecture': return 'var(--color-secondary)';
            case 'article': return 'var(--color-accent)';
            default: return 'var(--color-text-secondary)';
        }
    };

    return (

        <div className="rounded-lg p-6" style={{
            backgroundColor: 'var(--color-bg-secondary)',
            border: '1px solid var(--color-bg-tertiary)'
        }}>
            <h2 className="text-2xl font-bold mb-2" style={{
                fontFamily: 'var(--font-display)',
                color: 'var(--color-cream)'
            }}>
                Recommended Research & Lectures
            </h2>
            <p className="text-xs mb-4" style={{ color: 'var(--color-text-secondary)' }}>
                Curated daily from Abnormal Returns, The Motley Fool, CFA Institute, Farnam Street, and other trusted sources. Sponsored and low-quality pieces are filtered out.
            </p>

            {/* Filter buttons */}
            <div className="flex gap-2 mb-4 flex-wrap">
                {[
                    { label: 'All', value: 'all' },
                    { label: 'Finance', value: 'finance' },
                    { label: 'Valuation', value: 'valuation' },
                    { label: 'Behavioral', value: 'behavioral' },
                    { label: 'Macro', value: 'macro' },
                ].map(({ label, value }) => (
                    <button
                        key={value}
                        onClick={() => setFilter(value)}
                        className="px-3 py-1 rounded text-sm transition-all"
                        style={{
                            backgroundColor: filter === value ? 'var(--color-primary)' : 'var(--color-bg-tertiary)',
                            color: filter === value ? 'var(--color-cream)' : 'var(--color-text-secondary)',
                        }}
                    >
                        {label}
                    </button>
                ))}
            </div>

            {/* Modal for inline content */}
            {selectedResource && selectedResource.content && (
                <div
                    className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4"
                    onClick={() => setSelectedResource(null)}
                >
                    <div
                        className="relative w-full max-w-3xl bg-gray-950 rounded-lg p-6 overflow-y-auto max-h-[80vh]"
                        onClick={(e) => e.stopPropagation()}
                        style={{ border: '1px solid var(--color-bg-tertiary)' }}
                    >
                        <button
                            onClick={() => setSelectedResource(null)}
                            className="absolute top-4 right-4 text-slate-400 hover:text-white"
                        >
                            ✕
                        </button>
                        <h3 className="text-2xl font-bold mb-2" style={{ color: 'var(--color-cream)' }}>
                            {selectedResource.title}
                        </h3>
                        <p className="text-sm mb-4" style={{ color: 'var(--color-text-secondary)' }}>
                            {selectedResource.description}
                        </p>
                        <div className="prose prose-invert max-w-none" style={{ color: 'var(--color-text-primary)' }}>
                            {selectedResource.content.split('\n').map((line, i) => {
                                if (line.startsWith('# ')) return <h1 key={i} className="text-2xl font-bold mt-6 mb-3">{line.slice(2)}</h1>;
                                if (line.startsWith('## ')) return <h2 key={i} className="text-xl font-bold mt-4 mb-2">{line.slice(3)}</h2>;
                                if (line.startsWith('**') && line.endsWith('**')) return <p key={i} className="font-bold mt-2">{line.slice(2, -2)}</p>;
                                if (line.trim() === '') return <br key={i} />;
                                return <p key={i} className="mb-2">{line}</p>;
                            })}
                        </div>
                    </div>
                </div>
            )}

            {/* Resources grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {filteredResources.map((resource, idx) => (
                    <div
                        key={idx}
                        className="p-4 rounded-lg cursor-pointer transition-all hover:transform hover:-translate-y-1"
                        style={{
                            backgroundColor: 'var(--color-bg-primary)',
                            border: '1px solid var(--color-bg-tertiary)',
                        }}
                        onClick={() => {
                            if (resource.content) {
                                setSelectedResource(resource);
                            } else if (resource.url) {
                                window.open(resource.url, '_blank');
                            }
                        }}
                    >
                        <div className="flex items-start justify-between mb-2">
                            <span
                                className="text-xs px-2 py-1 rounded"
                                style={{
                                    backgroundColor: getTypeColor(resource.type) + '20',
                                    color: getTypeColor(resource.type)
                                }}
                            >
                                {resource.type.toUpperCase()}
                            </span>
                        </div>
                        <h3 className="font-bold mb-1" style={{ color: 'var(--color-cream)' }}>
                            {resource.title}
                        </h3>
                        <p className="text-sm mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                            {resource.description}
                        </p>
                        <div className="flex gap-2 flex-wrap">
                            {resource.tags.map((tag, i) => (
                                <span
                                    key={i}
                                    className="text-xs px-2 py-1 rounded"
                                    style={{
                                        backgroundColor: 'var(--color-bg-tertiary)',
                                        color: 'var(--color-text-secondary)'
                                    }}
                                >
                                    {tag}
                                </span>
                            ))}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

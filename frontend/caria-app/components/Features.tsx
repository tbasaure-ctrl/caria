import React, { useRef, useEffect, useState } from 'react';

export const Features: React.FC = () => {
    const [isVisible, setIsVisible] = useState(false);
    const sectionRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    setIsVisible(true);
                    observer.disconnect();
                }
            },
            { threshold: 0.1 }
        );

        if (sectionRef.current) {
            observer.observe(sectionRef.current);
        }

        return () => observer.disconnect();
    }, []);

    return (
        <section className="py-48 min-h-screen flex items-center relative" ref={sectionRef}>
            <div className={`container mx-auto px-6 transition-all duration-1000 transform ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-24'}`}>
                {/* Section Header */}
                <div className="text-center mb-16">
                    <h2 className="text-5xl font-bold mb-4" style={{ fontFamily: 'var(--font-display)', color: 'var(--color-cream)' }}>
                        Everything You Need to Succeed
                    </h2>
                    <p className="text-lg" style={{ fontFamily: 'var(--font-body)', color: 'var(--color-text-secondary)' }}>
                        Powerful tools and a vibrant community to elevate your investing journey
                    </p>
                </div>

                {/* Features Grid */}
                <div className="grid md:grid-cols-3 gap-8">
                    {/* Feature 1: AI-Powered Portfolio Intelligence */}
                    <div className="p-8 rounded-xl transition-transform hover:scale-105"
                        style={{
                            backgroundColor: 'var(--color-bg-tertiary)',
                            border: '1px solid var(--color-primary-dark)'
                        }}>
                        <div className="mb-4">
                            <div className="w-16 h-16 rounded-lg flex items-center justify-center"
                                style={{ backgroundColor: 'var(--color-blue-dark)' }}>
                                <svg className="w-8 h-8" style={{ color: 'var(--color-blue-light)' }} width="32" height="32" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                </svg>
                            </div>
                        </div>
                        <h3 className="text-2xl font-bold mb-3" style={{ fontFamily: 'var(--font-display)', color: 'var(--color-cream)' }}>
                            AI-Powered Portfolio Intelligence
                        </h3>
                        <p style={{ fontFamily: 'var(--font-body)', color: 'var(--color-text-secondary)', lineHeight: 1.7 }}>
                            Track your investments with real-time analytics, advanced metrics (Sharpe, Alpha, Beta), and stress test your portfolio
                            against different economic scenarios with Monte Carlo simulations.
                        </p>
                    </div>

                    {/* Feature 2: Market Intelligence & Regime Detection */}
                    <div className="p-8 rounded-xl transition-transform hover:scale-105"
                        style={{
                            backgroundColor: 'var(--color-bg-tertiary)',
                            border: '1px solid var(--color-primary-dark)'
                        }}>
                        <div className="mb-4">
                            <div className="w-16 h-16 rounded-lg flex items-center justify-center"
                                style={{ backgroundColor: 'var(--color-blue-dark)' }}>
                                <svg className="w-8 h-8" style={{ color: 'var(--color-blue-light)' }} width="32" height="32" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                                </svg>
                            </div>
                        </div>
                        <h3 className="text-2xl font-bold mb-3" style={{ fontFamily: 'var(--font-display)', color: 'var(--color-cream)' }}>
                            Market Intelligence & Regime Detection
                        </h3>
                        <p style={{ fontFamily: 'var(--font-body)', color: 'var(--color-text-secondary)', lineHeight: 1.7 }}>
                            Stay ahead with economic data and our proprietary macroeconomic regime detection that adapts your
                            strategy to evolving market conditions.
                        </p>
                    </div>

                    {/* Feature 3: Collaborative Insights */}
                    <div className="p-8 rounded-xl transition-transform hover:scale-105"
                        style={{
                            backgroundColor: 'var(--color-bg-tertiary)',
                            border: '1px solid var(--color-primary-dark)'
                        }}>
                        <div className="mb-4">
                            <div className="w-16 h-16 rounded-lg flex items-center justify-center"
                                style={{ backgroundColor: 'var(--color-blue-dark)' }}>
                                <svg className="w-8 h-8" style={{ color: 'var(--color-blue-light)' }} width="32" height="32" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                                </svg>
                            </div>
                        </div>
                        <h3 className="text-2xl font-bold mb-3" style={{ fontFamily: 'var(--font-display)', color: 'var(--color-cream)' }}>
                            Collaborative Insights
                        </h3>
                        <p style={{ fontFamily: 'var(--font-body)', color: 'var(--color-text-secondary)', lineHeight: 1.7 }}>
                            Publish investment ideas, share your analysis, and learn from a community of driven investors in
                            our collaborative thesis arena.
                        </p>
                    </div>
                </div>
            </div>
        </section>
    );
};

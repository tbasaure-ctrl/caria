import React from 'react';

interface HeroProps {
    onLogin: () => void;
}

// Market data for the right panel
const marketIndices = [
    { symbol: 'SPX', name: 'S&P 500', value: '5,234.18', change: '+0.82%', isPositive: true },
    { symbol: 'NDX', name: 'Nasdaq 100', value: '18,547.63', change: '+1.14%', isPositive: true },
    { symbol: 'GOLD', name: 'Gold Spot', value: '$2,341.20', change: '+0.45%', isPositive: true },
    { symbol: 'BTC', name: 'Bitcoin', value: '$67,824', change: '+2.31%', isPositive: true },
    { symbol: 'IPSA', name: 'IPSA Chile', value: '6,842.15', change: '+0.67%', isPositive: true },
    { symbol: 'MSCI', name: 'MSCI EM', value: '1,082.34', change: '-0.23%', isPositive: false },
];

const capabilityBullets = [
    'AI-powered stock screening & alpha generation',
    'Monte Carlo simulations for risk analysis',
    'Real-time regime detection & market intelligence',
    'Portfolio stress testing against historical crises',
    'Deep fundamental valuation (DCF, multiples)',
];

export const Hero: React.FC<HeroProps> = ({ onLogin }) => {
    return (
        <section 
            className="relative min-h-[90vh] overflow-hidden"
            style={{ backgroundColor: 'var(--color-bg-primary)' }}
        >
            {/* Subtle Background Grid */}
            <div 
                className="absolute inset-0 opacity-[0.03]"
                style={{
                    backgroundImage: `
                        linear-gradient(rgba(46, 124, 246, 0.5) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(46, 124, 246, 0.5) 1px, transparent 1px)
                    `,
                    backgroundSize: '60px 60px',
                }}
            />
            
            {/* Gradient Overlay */}
            <div 
                className="absolute inset-0 pointer-events-none"
                style={{
                    background: `
                        radial-gradient(ellipse 80% 50% at 20% 40%, rgba(46, 124, 246, 0.08) 0%, transparent 60%),
                        radial-gradient(ellipse 60% 40% at 80% 60%, rgba(46, 124, 246, 0.05) 0%, transparent 50%)
                    `,
                }}
            />

            <div className="container mx-auto px-6 lg:px-10 pt-16 pb-20 lg:pt-24 lg:pb-28 relative z-10">
                {/* Main Hero Grid: 70/30 Split */}
                <div className="grid lg:grid-cols-12 gap-10 lg:gap-16 items-start">
                    
                    {/* LEFT COLUMN - 70% - Main Content */}
                    <div className="lg:col-span-8 space-y-8">
                        {/* Eyebrow */}
                        <div className="animate-fade-in">
                            <span 
                                className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-xs font-semibold tracking-wide uppercase"
                                style={{
                                    backgroundColor: 'rgba(46, 124, 246, 0.12)',
                                    color: 'var(--color-accent-primary)',
                                    border: '1px solid rgba(46, 124, 246, 0.25)',
                                }}
                            >
                                <span className="w-2 h-2 rounded-full bg-positive animate-pulse-subtle" />
                                Professional Investment Intelligence
                            </span>
                        </div>

                        {/* Main Headline */}
                        <h1 
                            className="animate-fade-in-up"
                            style={{ animationDelay: '100ms' }}
                        >
                            <span 
                                className="block text-4xl md:text-5xl lg:text-6xl font-bold leading-[1.1] tracking-tight"
                                style={{
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-text-primary)',
                                }}
                            >
                                Institutional-Grade
                            </span>
                            <span 
                                className="block text-4xl md:text-5xl lg:text-6xl font-bold leading-[1.1] tracking-tight mt-2"
                                style={{
                                    fontFamily: 'var(--font-display)',
                                    background: 'linear-gradient(135deg, var(--color-accent-primary) 0%, #60A5FA 100%)',
                                    WebkitBackgroundClip: 'text',
                                    WebkitTextFillColor: 'transparent',
                                    backgroundClip: 'text',
                                }}
                            >
                                Research Platform
                            </span>
                            <span 
                                className="block text-4xl md:text-5xl lg:text-6xl font-bold leading-[1.1] tracking-tight mt-2"
                                style={{
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-text-primary)',
                                }}
                            >
                                For Active Investors
                            </span>
                        </h1>

                        {/* Subheadline */}
                        <p 
                            className="text-lg md:text-xl max-w-2xl animate-fade-in-up leading-relaxed"
                            style={{ 
                                color: 'var(--color-text-secondary)',
                                fontFamily: 'var(--font-body)',
                                animationDelay: '200ms'
                            }}
                        >
                            Caria combines quantitative screening, fundamental analysis, and AI-driven insights 
                            to help you find high-conviction investment opportunities—before the crowd.
                        </p>

                        {/* CTA Buttons */}
                        <div 
                            className="flex flex-wrap gap-4 pt-4 animate-fade-in-up"
                            style={{ animationDelay: '300ms' }}
                        >
                            <button
                                onClick={onLogin}
                                className="group relative px-8 py-4 rounded-lg font-semibold text-base transition-all duration-300 overflow-hidden"
                                style={{
                                    backgroundColor: 'var(--color-accent-primary)',
                                    color: '#FFFFFF',
                                    fontFamily: 'var(--font-body)',
                                }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.transform = 'translateY(-2px)';
                                    e.currentTarget.style.boxShadow = '0 8px 24px rgba(46, 124, 246, 0.4)';
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.transform = 'translateY(0)';
                                    e.currentTarget.style.boxShadow = 'none';
                                }}
                            >
                                <span className="relative z-10 flex items-center gap-2">
                                    Enter Caria
                                    <svg className="w-4 h-4 transition-transform group-hover:translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                                    </svg>
                                </span>
                            </button>
                            
                            <button
                                className="px-8 py-4 rounded-lg font-semibold text-base transition-all duration-300"
                                style={{
                                    backgroundColor: 'transparent',
                                    color: 'var(--color-text-secondary)',
                                    border: '1px solid var(--color-border-default)',
                                    fontFamily: 'var(--font-body)',
                                }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.borderColor = 'var(--color-accent-primary)';
                                    e.currentTarget.style.color = 'var(--color-text-primary)';
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.borderColor = 'var(--color-border-default)';
                                    e.currentTarget.style.color = 'var(--color-text-secondary)';
                                }}
                            >
                                Explore Features
                            </button>
                        </div>

                        {/* Trust Indicators */}
                        <div 
                            className="flex items-center gap-6 pt-6 animate-fade-in-up"
                            style={{ animationDelay: '400ms' }}
                        >
                            <div className="flex items-center gap-2">
                                <svg className="w-4 h-4 text-positive" fill="currentColor" viewBox="0 0 20 20">
                                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                </svg>
                                <span className="text-sm" style={{ color: 'var(--color-text-muted)' }}>Real-time data</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <svg className="w-4 h-4 text-positive" fill="currentColor" viewBox="0 0 20 20">
                                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                </svg>
                                <span className="text-sm" style={{ color: 'var(--color-text-muted)' }}>Quantitative models</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <svg className="w-4 h-4 text-positive" fill="currentColor" viewBox="0 0 20 20">
                                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                </svg>
                                <span className="text-sm" style={{ color: 'var(--color-text-muted)' }}>AI-enhanced</span>
                            </div>
                        </div>
                    </div>

                    {/* RIGHT COLUMN - 30% - Market Data + Benefits */}
                    <div className="lg:col-span-4 space-y-6">
                        {/* Live Market Indices Panel */}
                        <div 
                            className="rounded-xl p-5 animate-fade-in-up"
                            style={{ 
                                backgroundColor: 'var(--color-bg-secondary)',
                                border: '1px solid var(--color-border-subtle)',
                                animationDelay: '200ms'
                            }}
                        >
                            <div className="flex items-center justify-between mb-4">
                                <h3 
                                    className="text-xs font-semibold tracking-widest uppercase"
                                    style={{ color: 'var(--color-text-muted)' }}
                                >
                                    Live Markets
                                </h3>
                                <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--color-text-muted)' }}>
                                    <span className="w-1.5 h-1.5 rounded-full bg-positive animate-pulse" />
                                    Live
                                </span>
                            </div>
                            
                            <div className="space-y-3">
                                {marketIndices.map((index, idx) => (
                                    <div 
                                        key={index.symbol}
                                        className="flex items-center justify-between py-2 border-b last:border-b-0"
                                        style={{ 
                                            borderColor: 'var(--color-border-subtle)',
                                            animationDelay: `${300 + idx * 50}ms`
                                        }}
                                    >
                                        <div className="flex items-center gap-3">
                                            <span 
                                                className="text-xs font-mono font-semibold px-2 py-0.5 rounded"
                                                style={{ 
                                                    backgroundColor: 'var(--color-bg-surface)',
                                                    color: 'var(--color-text-primary)'
                                                }}
                                            >
                                                {index.symbol}
                                            </span>
                                            <span className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                                                {index.name}
                                            </span>
                                        </div>
                                        <div className="text-right">
                                            <div 
                                                className="text-sm font-mono font-medium"
                                                style={{ color: 'var(--color-text-primary)' }}
                                            >
                                                {index.value}
                                            </div>
                                            <div 
                                                className="text-xs font-mono"
                                                style={{ color: index.isPositive ? 'var(--color-positive)' : 'var(--color-negative)' }}
                                            >
                                                {index.change}
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* What Caria Does Panel */}
                        <div 
                            className="rounded-xl p-5 animate-fade-in-up"
                            style={{ 
                                backgroundColor: 'var(--color-bg-secondary)',
                                border: '1px solid var(--color-border-subtle)',
                                animationDelay: '400ms'
                            }}
                        >
                            <h3 
                                className="text-xs font-semibold tracking-widest uppercase mb-4"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                What Caria Does
                            </h3>
                            
                            <ul className="space-y-3">
                                {capabilityBullets.map((bullet, idx) => (
                                    <li 
                                        key={idx}
                                        className="flex items-start gap-3 text-sm"
                                        style={{ color: 'var(--color-text-secondary)' }}
                                    >
                                        <svg className="w-4 h-4 mt-0.5 flex-shrink-0 text-accent-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                        </svg>
                                        {bullet}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                </div>

                {/* Feature Blocks Row - Below Hero */}
                <div 
                    className="grid md:grid-cols-3 gap-6 mt-20 animate-fade-in-up"
                    style={{ animationDelay: '500ms' }}
                >
                    {/* Feature 1 - Alpha Stock Picker */}
                    <div 
                        className="group p-6 rounded-xl transition-all duration-300"
                        style={{ 
                            backgroundColor: 'var(--color-bg-secondary)',
                            border: '1px solid var(--color-border-subtle)'
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.borderColor = 'var(--color-border-emphasis)';
                            e.currentTarget.style.transform = 'translateY(-4px)';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.borderColor = 'var(--color-border-subtle)';
                            e.currentTarget.style.transform = 'translateY(0)';
                        }}
                    >
                        <div 
                            className="w-12 h-12 rounded-lg flex items-center justify-center mb-4"
                            style={{ backgroundColor: 'rgba(46, 124, 246, 0.15)' }}
                        >
                            <svg className="w-6 h-6 text-accent-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                            </svg>
                        </div>
                        <h3 
                            className="text-lg font-semibold mb-2"
                            style={{ 
                                fontFamily: 'var(--font-display)',
                                color: 'var(--color-text-primary)' 
                            }}
                        >
                            Alpha Stock Picker
                        </h3>
                        <p className="text-sm mb-3" style={{ color: 'var(--color-text-muted)' }}>
                            3 high-conviction ideas weekly
                        </p>
                        <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                            Composite scoring across momentum, quality, valuation, and catalyst factors.
                        </p>
                    </div>

                    {/* Feature 2 - Hidden Gems Screener */}
                    <div 
                        className="group p-6 rounded-xl transition-all duration-300"
                        style={{ 
                            backgroundColor: 'var(--color-bg-secondary)',
                            border: '1px solid var(--color-border-subtle)'
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.borderColor = 'var(--color-border-emphasis)';
                            e.currentTarget.style.transform = 'translateY(-4px)';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.borderColor = 'var(--color-border-subtle)';
                            e.currentTarget.style.transform = 'translateY(0)';
                        }}
                    >
                        <div 
                            className="w-12 h-12 rounded-lg flex items-center justify-center mb-4"
                            style={{ backgroundColor: 'rgba(0, 200, 83, 0.15)' }}
                        >
                            <svg className="w-6 h-6 text-positive" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                            </svg>
                        </div>
                        <h3 
                            className="text-lg font-semibold mb-2"
                            style={{ 
                                fontFamily: 'var(--font-display)',
                                color: 'var(--color-text-primary)' 
                            }}
                        >
                            Under-the-Radar Screener
                        </h3>
                        <p className="text-sm mb-3" style={{ color: 'var(--color-text-muted)' }}>
                            Signals before Wall Street reacts
                        </p>
                        <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                            Undervalued mid-caps with strong quality, valuation, and momentum scores.
                        </p>
                    </div>

                    {/* Feature 3 - Portfolio War-Game Engine */}
                    <div 
                        className="group p-6 rounded-xl transition-all duration-300"
                        style={{ 
                            backgroundColor: 'var(--color-bg-secondary)',
                            border: '1px solid var(--color-border-subtle)'
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.borderColor = 'var(--color-border-emphasis)';
                            e.currentTarget.style.transform = 'translateY(-4px)';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.borderColor = 'var(--color-border-subtle)';
                            e.currentTarget.style.transform = 'translateY(0)';
                        }}
                    >
                        <div 
                            className="w-12 h-12 rounded-lg flex items-center justify-center mb-4"
                            style={{ backgroundColor: 'rgba(255, 152, 0, 0.15)' }}
                        >
                            <svg className="w-6 h-6 text-warning" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                            </svg>
                        </div>
                        <h3 
                            className="text-lg font-semibold mb-2"
                            style={{ 
                                fontFamily: 'var(--font-display)',
                                color: 'var(--color-text-primary)' 
                            }}
                        >
                            Portfolio War-Game Engine
                        </h3>
                        <p className="text-sm mb-3" style={{ color: 'var(--color-text-muted)' }}>
                            Simulate crashes, wars, black swans
                        </p>
                        <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                            Stress test against historical crises and macroeconomic scenarios.
                        </p>
                    </div>
                </div>

                {/* Social Proof Line */}
                <div 
                    className="text-center mt-16 animate-fade-in-up"
                    style={{ animationDelay: '600ms' }}
                >
                    <div 
                        className="inline-flex items-center gap-3 px-6 py-3 rounded-full"
                        style={{ 
                            backgroundColor: 'var(--color-bg-secondary)',
                            border: '1px solid var(--color-border-subtle)'
                        }}
                    >
                        <span className="text-sm font-medium" style={{ color: 'var(--color-text-muted)' }}>
                            "Trust built on fundamentals, not hype."
                        </span>
                    </div>
                </div>
            </div>
        </section>
    );
};

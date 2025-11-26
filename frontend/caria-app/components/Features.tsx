import React, { useRef, useState, useEffect } from 'react';

/**
 * Custom hook to detect when an element is visible on the screen.
 */
const useOnScreen = (ref: React.RefObject<HTMLElement>, threshold: number = 0.1): boolean => {
    const [isIntersecting, setIntersecting] = useState(false);

    useEffect(() => {
        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    setIntersecting(true);
                    observer.unobserve(entry.target);
                }
            },
            { threshold }
        );

        const currentRef = ref.current;
        if (currentRef) {
            observer.observe(currentRef);
        }

        return () => {
            if (currentRef) {
                observer.unobserve(currentRef);
            }
        };
    }, [ref, threshold]);

    return isIntersecting;
};

interface Feature {
    icon: string;
    title: string;
    subtitle: string;
    description: string;
    metrics?: { label: string; value: string }[];
}

const featuresData: Feature[] = [
    {
        icon: '📊',
        title: 'Portfolio Intelligence',
        subtitle: 'Professional Analytics',
        description: 'Track investments with real-time pricing, Sharpe Ratio, Alpha, Beta, and comprehensive risk metrics. Visualize allocation and performance with institutional-grade charts.',
        metrics: [
            { label: 'Metrics', value: '15+' },
            { label: 'Update', value: 'Real-time' },
        ]
    },
    {
        icon: '🔬',
        title: 'Deep Valuation Engine',
        subtitle: 'DCF & Monte Carlo',
        description: 'Multi-method valuation combining discounted cash flow, historical multiples, reverse DCF, and Monte Carlo simulations for comprehensive price targets.',
        metrics: [
            { label: 'Simulations', value: '10,000+' },
            { label: 'Models', value: '4' },
        ]
    },
    {
        icon: '🎯',
        title: 'Alpha Generation',
        subtitle: 'Weekly Stock Picks',
        description: 'Composite Alpha Score (CAS) model screens the entire market for momentum, quality, valuation, and catalyst signals to surface high-conviction opportunities.',
        metrics: [
            { label: 'Picks/Week', value: '3' },
            { label: 'Factors', value: '4' },
        ]
    },
    {
        icon: '⚔️',
        title: 'Thesis Arena',
        subtitle: 'Stress-Test Your Ideas',
        description: 'Challenge your investment thesis against AI-powered contrarian analysis. Identify cognitive biases and strengthen your conviction before committing capital.',
        metrics: [
            { label: 'Personas', value: '5' },
            { label: 'Analysis', value: 'Deep' },
        ]
    },
    {
        icon: '🌍',
        title: 'Market Regime Detection',
        subtitle: 'Macro Intelligence',
        description: 'Proprietary HMM-based regime detection identifies market conditions in real-time—expansion, slowdown, recession—to inform tactical allocation decisions.',
        metrics: [
            { label: 'Regimes', value: '4' },
            { label: 'Accuracy', value: 'High' },
        ]
    },
    {
        icon: '🛡️',
        title: 'Crisis Simulation',
        subtitle: 'Historical Stress Tests',
        description: 'Simulate portfolio performance during major historical crises—from 1929 to COVID-19. Understand drawdown risk and recovery patterns.',
        metrics: [
            { label: 'Crises', value: '12' },
            { label: 'Timeframes', value: '3' },
        ]
    },
];

const FeatureCard: React.FC<{ feature: Feature; index: number }> = ({ feature, index }) => {
    const cardRef = useRef<HTMLDivElement>(null);
    const isVisible = useOnScreen(cardRef, 0.2);

    return (
        <div
            ref={cardRef}
            className={`
                p-6 rounded-xl transition-all duration-500
                ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'}
            `}
            style={{
                backgroundColor: 'var(--color-bg-secondary)',
                border: '1px solid var(--color-border-subtle)',
                transitionDelay: `${index * 100}ms`,
            }}
            onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.borderColor = 'var(--color-border-emphasis)';
                (e.currentTarget as HTMLElement).style.transform = 'translateY(-4px)';
                (e.currentTarget as HTMLElement).style.boxShadow = '0 8px 24px rgba(0, 0, 0, 0.4)';
            }}
            onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.borderColor = 'var(--color-border-subtle)';
                (e.currentTarget as HTMLElement).style.transform = 'translateY(0)';
                (e.currentTarget as HTMLElement).style.boxShadow = 'none';
            }}
        >
            {/* Icon & Subtitle Row */}
            <div className="flex items-center justify-between mb-4">
                <span className="text-3xl">{feature.icon}</span>
                <span 
                    className="text-xs font-medium tracking-wide uppercase px-3 py-1 rounded"
                    style={{ 
                        backgroundColor: 'var(--color-bg-surface)',
                        color: 'var(--color-text-muted)'
                    }}
                >
                    {feature.subtitle}
                </span>
            </div>

            {/* Title */}
            <h3 
                className="text-xl font-semibold mb-3"
                style={{
                    fontFamily: 'var(--font-display)',
                    color: 'var(--color-text-primary)',
                }}
            >
                {feature.title}
            </h3>

            {/* Description */}
            <p 
                className="text-sm leading-relaxed mb-5"
                style={{ color: 'var(--color-text-secondary)' }}
            >
                {feature.description}
            </p>

            {/* Metrics */}
            {feature.metrics && (
                <div 
                    className="flex gap-4 pt-4 border-t"
                    style={{ borderColor: 'var(--color-border-subtle)' }}
                >
                    {feature.metrics.map((metric, idx) => (
                        <div key={idx}>
                            <div 
                                className="text-lg font-semibold font-mono"
                                style={{ color: 'var(--color-accent-primary)' }}
                            >
                                {metric.value}
                            </div>
                            <div 
                                className="text-xs uppercase tracking-wide"
                                style={{ color: 'var(--color-text-muted)' }}
                            >
                                {metric.label}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};


export const Features: React.FC = () => {
    const sectionRef = useRef<HTMLElement>(null);
    const isVisible = useOnScreen(sectionRef, 0.1);

    return (
        <section 
            ref={sectionRef}
            className="py-20 lg:py-28 relative"
            style={{ backgroundColor: 'var(--color-bg-primary)' }}
        >
            {/* Subtle Background */}
            <div 
                className="absolute inset-0 opacity-[0.02]"
                style={{
                    backgroundImage: `
                        linear-gradient(rgba(46, 124, 246, 0.5) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(46, 124, 246, 0.5) 1px, transparent 1px)
                    `,
                    backgroundSize: '80px 80px',
                }}
            />

            <div className="container mx-auto px-6 lg:px-10 relative z-10">
                {/* Section Header */}
                <div 
                    className={`max-w-3xl mb-16 transition-all duration-700 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}
                >
                    <span 
                        className="text-xs font-semibold tracking-widest uppercase mb-4 block"
                        style={{ color: 'var(--color-accent-primary)' }}
                    >
                        Platform Capabilities
                    </span>
                    <h2 
                        className="text-3xl md:text-4xl lg:text-5xl font-bold mb-6"
                        style={{
                            fontFamily: 'var(--font-display)',
                            color: 'var(--color-text-primary)',
                            lineHeight: 1.15,
                            letterSpacing: '-0.02em',
                        }}
                    >
                        Everything You Need for 
                        <span 
                            className="block"
                            style={{
                                background: 'linear-gradient(135deg, var(--color-accent-primary) 0%, #60A5FA 100%)',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent',
                                backgroundClip: 'text',
                            }}
                        >
                            Informed Decisions
                        </span>
                    </h2>
                    <p 
                        className="text-lg"
                        style={{ 
                            color: 'var(--color-text-secondary)',
                            lineHeight: 1.7 
                        }}
                    >
                        Professional-grade tools and insights, designed for active investors 
                        who demand more than surface-level analysis.
                    </p>
                </div>

                {/* Feature Grid */}
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {featuresData.map((feature, index) => (
                        <FeatureCard key={feature.title} feature={feature} index={index} />
                    ))}
                </div>

                {/* Bottom CTA */}
                <div 
                    className={`text-center mt-16 transition-all duration-700 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}
                    style={{ transitionDelay: '600ms' }}
                >
                    <p 
                        className="text-sm mb-6"
                        style={{ color: 'var(--color-text-muted)' }}
                    >
                        Ready to elevate your research?
                    </p>
                    <button
                        className="px-8 py-4 rounded-lg font-semibold text-base transition-all duration-300"
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
                        Start Free Trial
                    </button>
                </div>
            </div>
        </section>
    );
};

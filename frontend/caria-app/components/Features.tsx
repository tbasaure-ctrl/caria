import React, { useState, useEffect, useRef } from 'react';
import type { Feature } from '../types';
import { PortfolioIcon, ChartIcon, CommunityIcon } from './Icons';

/**
 * Custom hook to detect when an element is visible on the screen.
 * @param ref - A React ref attached to the element to observe.
 * @param threshold - The percentage of the element that must be visible to trigger the hook.
 * @returns {boolean} - True if the element is on screen, false otherwise.
 */
const useOnScreen = (ref: React.RefObject<HTMLElement>, threshold: number = 0.1): boolean => {
  const [isIntersecting, setIntersecting] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        // Update state when the element's intersection status changes.
        if (entry.isIntersecting) {
          setIntersecting(true);
          // Stop observing the element once it has become visible.
          observer.unobserve(entry.target);
        }
      },
      {
        threshold,
      }
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


const MarketVisual: React.FC = () => (
    <div className="space-y-2 mt-4 text-sm font-mono">
        <div className="flex justify-between items-center bg-gray-800/60 p-2 rounded">
            <span>S&P 500</span>
            <span className="font-bold text-blue-300">+0.78% ▲</span>
        </div>
        <div className="flex justify-between items-center bg-gray-800/60 p-2 rounded">
            <span>NASDAQ</span>
            <span className="font-bold text-slate-400">-0.21% ▼</span>
        </div>
        <div className="flex justify-between items-center bg-gray-800/60 p-2 rounded">
            <span>BTC/USD</span>
            <span className="font-bold text-blue-300">+2.50% ▲</span>
        </div>
    </div>
);


const featuresData: Feature[] = [
  {
    icon: PortfolioIcon,
    title: 'AI-Powered Portfolio Intelligence',
    description: 'Track your investments with real-time analytics, advanced metrics (Sharpe, Alpha, Beta), and stress test your portfolio against different economic scenarios with Monte Carlo simulations.',
  },
  {
    icon: ChartIcon,
    title: 'Market Intelligence & Regime Detection',
    description: 'Stay ahead with live market data and our proprietary macroeconomic regime detection that adapts your strategy to market conditions in real-time.',
    visual: <MarketVisual />,
  },
  {
    icon: CommunityIcon,
    title: 'Collaborative Insights',
    description: 'Explore investment ideas, share your analysis, and learn from a community of driven investors in our exclusive forum.',
  },
];

const FeatureCard: React.FC<{ feature: Feature, index: number }> = ({ feature, index }) => {
    const cardRef = useRef<HTMLDivElement>(null);
    const isVisible = useOnScreen(cardRef, 0.2);

    return (
        <div
            ref={cardRef}
            className={`
                rounded-2xl p-8 flex flex-col relative overflow-hidden group
                transition-all ease-out duration-700
                ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}
            `}
            style={{
                backgroundColor: 'rgba(28, 33, 39, 0.6)',
                border: '1px solid rgba(74, 144, 226, 0.2)',
                transitionDelay: `${index * 150}ms`,
                backdropFilter: 'blur(10px)',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
            }}
            onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'rgba(74, 144, 226, 0.6)';
                e.currentTarget.style.transform = 'translateY(-8px) scale(1.02)';
                e.currentTarget.style.boxShadow = '0 20px 60px rgba(74, 144, 226, 0.3)';
            }}
            onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'rgba(74, 144, 226, 0.2)';
                e.currentTarget.style.transform = 'translateY(0) scale(1)';
                e.currentTarget.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.3)';
            }}
        >
            {/* Animated gradient background on hover */}
            <div
                className="absolute inset-0 opacity-0 group-hover:opacity-10 transition-opacity duration-500"
                style={{
                    background: 'radial-gradient(circle at top left, var(--color-blue-light), transparent)',
                }}
            />

            <div className="relative z-10">
                <feature.icon
                    className="w-14 h-14 mb-6 transition-transform duration-500 group-hover:scale-110 group-hover:rotate-3"
                    style={{color: 'var(--color-blue-light)'}}
                />
                <h3
                    className="text-2xl md:text-3xl font-bold mb-4"
                    style={{
                        fontFamily: "'Instrument Serif', Georgia, serif",
                        color: 'var(--color-cream)',
                        letterSpacing: '-0.01em',
                    }}>
                    {feature.title}
                </h3>
                <p
                    className="flex-grow leading-relaxed"
                    style={{
                        fontFamily: "'Crimson Pro', Georgia, serif",
                        color: 'rgba(232, 230, 227, 0.8)',
                        fontSize: '1.05rem',
                        lineHeight: '1.7',
                    }}>
                    {feature.description}
                </p>
                {feature.visual && <div className="mt-6 pt-6 border-t border-blue-900/30">{feature.visual}</div>}
            </div>
        </div>
    );
};


export const Features: React.FC = () => {
  return (
    <section className="py-24 md:py-32 relative overflow-hidden" style={{backgroundColor: 'var(--color-bg-primary)'}}>
      {/* Background decorative elements */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute top-0 left-1/4 w-96 h-96 rounded-full blur-3xl"
             style={{background: 'radial-gradient(circle, var(--color-blue-light) 0%, transparent 70%)'}}></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 rounded-full blur-3xl"
             style={{background: 'radial-gradient(circle, var(--color-blue-dark) 0%, transparent 70%)'}}></div>
      </div>

      <div className="container mx-auto px-6 md:px-12 relative z-10">
        <div className="text-center mb-20">
            <h2
              className="text-4xl md:text-6xl font-black mb-6 fade-in"
              style={{
                fontFamily: "'Instrument Serif', Georgia, serif",
                color: 'var(--color-cream)',
                letterSpacing: '-0.02em',
              }}>
              Everything You Need to Succeed
            </h2>
            <p
              className="mt-6 max-w-2xl mx-auto text-lg md:text-xl fade-in delay-200"
              style={{
                fontFamily: "'Crimson Pro', Georgia, serif",
                color: 'rgba(232, 230, 227, 0.7)',
                lineHeight: '1.8',
              }}>
                Powerful tools and a vibrant community to elevate your investing journey.
            </p>
        </div>
        <div className="grid md:grid-cols-3 gap-8 md:gap-10">
          {featuresData.map((feature, index) => (
            <FeatureCard key={feature.title} feature={feature} index={index} />
          ))}
        </div>
      </div>
    </section>
  );
};

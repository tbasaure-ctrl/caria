import React, { useState, useEffect, useRef } from 'react';
import type { Feature } from '../types';

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


const featuresData: Feature[] = [
  {
    title: 'AI-Powered Portfolio Intelligence',
    description: 'Real-time analytics, advanced metrics, and Monte Carlo simulations. Track performance, analyze risk, and optimize allocation with institutional-grade tools.',
  },
  {
    title: 'Market Intelligence & Regime Detection',
    description: 'Proprietary algorithms detect market regimes in real-time. Adapt your strategy to changing conditions with actionable insights from economic indicators and sentiment data.',
  },
  {
    title: 'Collaborative Insights',
    description: 'Share ideas, challenge theses, and learn from a community of driven investors. Thoughtful discussions and peer-to-peer learning to elevate your analysis.',
  },
];

const FeatureCard: React.FC<{ feature: Feature, index: number }> = ({ feature, index }) => {
    const cardRef = useRef<HTMLDivElement>(null);
    const isVisible = useOnScreen(cardRef, 0.2);

    return (
        <div
            ref={cardRef}
            className={`
                rounded-xl p-6 sm:p-8 flex flex-col relative overflow-hidden group
                transition-all ease-out duration-500
                ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'}
            `}
            style={{
                backgroundColor: 'rgba(25, 30, 38, 0.8)',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                transitionDelay: `${index * 100}ms`,
            }}
            onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.15)';
                e.currentTarget.style.transform = 'translateY(-4px)';
            }}
            onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.08)';
                e.currentTarget.style.transform = 'translateY(0)';
            }}
        >
            <h3
                className="text-xl sm:text-2xl font-semibold mb-3"
                style={{
                    fontFamily: "'Instrument Serif', Georgia, serif",
                    color: 'var(--color-cream)',
                    letterSpacing: '-0.01em',
                }}>
                {feature.title}
            </h3>
            <p
                className="flex-grow leading-relaxed text-sm sm:text-base"
                style={{
                    fontFamily: "'Crimson Pro', Georgia, serif",
                    color: 'rgba(232, 230, 227, 0.7)',
                    lineHeight: '1.7',
                }}>
                {feature.description}
            </p>
        </div>
    );
};


export const Features: React.FC = () => {
  return (
    <section id="features" className="py-20 md:py-32 relative overflow-hidden" style={{backgroundColor: 'var(--color-bg-primary)'}}>
      <div className="container mx-auto px-4 sm:px-6 md:px-12 relative z-10">
        {/* Section Header */}
        <div className="text-center mb-16 md:mb-20">
            <h2
              className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold mb-4 md:mb-6"
              style={{
                fontFamily: "'Instrument Serif', Georgia, serif",
                color: 'var(--color-cream)',
                letterSpacing: '-0.02em',
              }}>
              Everything you need to navigate the markets
            </h2>
            <p
              className="max-w-xl mx-auto text-base md:text-lg px-4"
              style={{
                fontFamily: "'Crimson Pro', Georgia, serif",
                color: 'rgba(232, 230, 227, 0.6)',
                lineHeight: '1.7',
              }}>
                Powerful tools and a vibrant community to elevate your investing journey.
            </p>
        </div>

        {/* Feature Cards Grid */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8 max-w-6xl mx-auto">
          {featuresData.map((feature, index) => (
            <FeatureCard key={feature.title} feature={feature} index={index} />
          ))}
        </div>
      </div>
    </section>
  );
};

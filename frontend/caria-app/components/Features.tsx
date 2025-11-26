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
                rounded-2xl p-10 md:p-12 flex flex-col relative overflow-hidden group
                transition-all ease-out duration-700
                ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}
            `}
            style={{
                backgroundColor: 'rgba(28, 33, 39, 0.6)',
                border: '1px solid rgba(74, 144, 226, 0.2)',
                transitionDelay: `${index * 150}ms`,
                backdropFilter: 'blur(10px)',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
                minHeight: '280px',
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
                <h3
                    className="text-2xl md:text-3xl font-bold mb-6"
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
                        fontSize: '1.1rem',
                        lineHeight: '1.8',
                    }}>
                    {feature.description}
                </p>
            </div>
        </div>
    );
};


export const Features: React.FC = () => {
  return (
    <section className="py-32 md:py-48 relative overflow-hidden" style={{backgroundColor: 'var(--color-bg-primary)'}}>
      {/* Background decorative elements */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute top-0 left-1/4 w-96 h-96 rounded-full blur-3xl"
             style={{background: 'radial-gradient(circle, var(--color-blue-light) 0%, transparent 70%)'}}></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 rounded-full blur-3xl"
             style={{background: 'radial-gradient(circle, var(--color-blue-dark) 0%, transparent 70%)'}}></div>
      </div>

      <div className="container mx-auto px-6 md:px-12 relative z-10">
        <div className="text-center mb-32">
            <h2
              className="text-4xl md:text-6xl font-black mb-6 fade-in"
              style={{
                fontFamily: "'Instrument Serif', Georgia, serif",
                color: 'var(--color-cream)',
                letterSpacing: '-0.02em',
              }}>
              Everything you need to navigate the markets
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
        <div className="grid md:grid-cols-3 gap-12 md:gap-16">
          {featuresData.map((feature, index) => (
            <FeatureCard key={feature.title} feature={feature} index={index} />
          ))}
        </div>
      </div>
    </section>
  );
};

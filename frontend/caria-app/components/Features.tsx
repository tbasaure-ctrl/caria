import React, { useRef } from 'react';
import type { Feature } from '../types';

const useOnScreen = (ref: React.RefObject<HTMLElement>, threshold: number = 0.1): boolean => {
  const [isIntersecting, setIntersecting] = React.useState(false);

  React.useEffect(() => {
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
    if (currentRef) observer.observe(currentRef);
    return () => { if (currentRef) observer.unobserve(currentRef); };
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
                flex flex-col items-start p-6 md:p-8
                transition-all duration-700 ease-out
                ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}
            `}
            style={{ transitionDelay: `${index * 150}ms` }}
        >
            {/* Icon Placeholder - Minimalist Circle */}
            <div className="w-12 h-12 rounded-full border border-white/20 flex items-center justify-center mb-6 text-accent-cyan">
                {index === 0 && <span className="text-xl">⚡</span>}
                {index === 1 && <span className="text-xl">⚖️</span>}
                {index === 2 && <span className="text-xl">∞</span>}
            </div>

            <h3 className="text-2xl font-display text-white mb-4 tracking-wide">
                {feature.title}
            </h3>
            
            <p className="text-text-secondary leading-relaxed font-light text-base max-w-sm">
                {feature.description}
            </p>
        </div>
    );
};

export const Features: React.FC = () => {
  return (
    <section id="features" className="py-32 bg-bg-primary border-t border-white/5">
      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid md:grid-cols-3 gap-12 lg:gap-16">
          {featuresData.map((feature, index) => (
            <FeatureCard key={feature.title} feature={feature} index={index} />
          ))}
        </div>
      </div>
    </section>
  );
};

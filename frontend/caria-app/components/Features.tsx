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
    title: 'Cognitive Core',
    description: 'Proprietary algorithms that process global sentiment and macroeconomic shifts instantly, predicting market movements before they materialize.',
  },
  {
    title: 'Risk Shield',
    description: 'Dynamic calibration of exposure. Our system acts as an automated hedge, protecting capital during volatility while capturing upside.',
  },
  {
    title: 'Compound Growth',
    description: 'Self-improving neural networks that learn from every transaction, constantly refining the strategy to outperform the benchmark.',
  },
];

const FeatureCard: React.FC<{ feature: Feature, index: number }> = ({ feature, index }) => {
    const cardRef = useRef<HTMLDivElement>(null);
    const isVisible = useOnScreen(cardRef, 0.2);

    return (
        <div
            ref={cardRef}
            className={`
                flex flex-col items-start p-8 rounded-sm
                transition-all duration-700 ease-out
                ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-12'}
            `}
            style={{ 
                transitionDelay: `${index * 200}ms`,
                backgroundColor: '#000000' // Pure black as seen in image
            }}
        >
            {/* Gold Icon */}
            <div className="mb-6 text-accent-gold">
                {index === 0 && (
                    <svg className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                    </svg>
                )}
                {index === 1 && (
                    <svg className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.956 11.956 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                    </svg>
                )}
                {index === 2 && (
                    <svg className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                    </svg>
                )}
            </div>

            <h3 className="text-2xl font-display text-white mb-6 tracking-wide">
                {feature.title}
            </h3>
            
            <p className="text-text-secondary leading-relaxed font-light text-base">
                {feature.description}
            </p>
        </div>
    );
};

export const Features: React.FC = () => {
  return (
    <section id="features" className="py-32 bg-bg-primary border-t border-white/5">
      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid md:grid-cols-3 gap-8">
          {featuresData.map((feature, index) => (
            <FeatureCard key={feature.title} feature={feature} index={index} />
          ))}
        </div>
      </div>
    </section>
  );
};

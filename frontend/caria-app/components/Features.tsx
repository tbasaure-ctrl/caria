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
                flex flex-col items-start p-8 rounded-lg border border-white/5 bg-[#050A14]
                transition-all duration-1000 ease-out transform
                ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-12'}
                hover:border-accent-cyan/20 hover:shadow-glow-sm
            `}
            style={{ transitionDelay: `${index * 200}ms` }}
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

            <h3 className="text-2xl font-display text-white mb-4 tracking-wide">
                {feature.title}
            </h3>
            
            <p className="text-text-secondary leading-relaxed font-light text-base">
                {feature.description}
            </p>
        </div>
    );
};

const PlanetHorizon: React.FC = () => (
    <div className="relative w-full h-[600px] mt-32 overflow-hidden flex items-end justify-center">
        {/* Stars */}
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20"></div>
        
        {/* Planet Curve */}
        <div className="absolute -bottom-[80%] left-1/2 -translate-x-1/2 w-[150%] aspect-square rounded-full bg-black shadow-[0_-50px_150px_rgba(34,211,238,0.15)] overflow-hidden border-t border-accent-cyan/20">
            {/* Atmosphere Glow */}
            <div className="absolute inset-0 bg-gradient-to-b from-accent-cyan/10 via-transparent to-transparent"></div>
            {/* City Lights (Simulated) */}
            <div className="absolute inset-0 opacity-60 mix-blend-overlay" 
                 style={{ 
                     backgroundImage: 'radial-gradient(circle at 50% 0%, rgba(255,255,255,0.8) 0%, transparent 1%)',
                     backgroundSize: '4px 4px'
                 }} 
            />
        </div>
        
        {/* Horizon Line */}
        <div className="absolute bottom-[10%] left-0 right-0 h-px bg-gradient-to-r from-transparent via-accent-cyan/50 to-transparent box-shadow-[0_0_20px_#22D3EE]"></div>
        
        {/* Text Overlay */}
        <div className="absolute bottom-[30%] text-center z-10">
            <h2 className="text-4xl md:text-6xl font-display text-transparent bg-clip-text bg-gradient-to-b from-white to-white/50 tracking-tight mb-4">
                The Future is Clear
            </h2>
            <p className="text-text-muted text-lg font-light tracking-widest uppercase">
                Caria Intelligence
            </p>
        </div>
    </div>
);

export const Features: React.FC = () => {
  return (
    <section id="features" className="py-32 bg-bg-primary relative">
      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        
        {/* Slogan Header - Moved Here */}
        <div className="text-center mb-24 animate-fade-in-up">
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-display text-white mb-6">
                We help you visualize the future you want
            </h2>
            <div className="h-px w-24 mx-auto bg-gradient-to-r from-transparent via-accent-gold to-transparent opacity-50" />
        </div>

        {/* Feature Grid */}
        <div className="grid md:grid-cols-3 gap-8 mb-32">
          {featuresData.map((feature, index) => (
            <FeatureCard key={feature.title} feature={feature} index={index} />
          ))}
        </div>
      </div>

      {/* Planet Footer Graphic */}
      <PlanetHorizon />
    </section>
  );
};

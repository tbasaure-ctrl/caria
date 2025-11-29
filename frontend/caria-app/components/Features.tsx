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
    title: 'Portfolio Analytics',
    description: 'Track your holdings, visualize asset allocation, and monitor real-time performance across all your investments in one unified dashboard.',
  },
  {
    title: 'AI Investment Partner',
    description: 'Chat with Caria to analyze investment theses, stress-test your ideas, and get data-driven insights powered by advanced AI models.',
  },
  {
    title: 'Valuation Tools',
    description: 'Professional-grade DCF models, Monte Carlo simulations, and business valuation workshops to make informed investment decisions.',
  },
];

const FeatureCard: React.FC<{ feature: Feature, index: number }> = ({ feature, index }) => {
    const cardRef = useRef<HTMLDivElement>(null);
    const isVisible = useOnScreen(cardRef, 0.2);

    const tabLinks = ['portfolio', 'analysis', 'analysis'];

    return (
        <div
            ref={cardRef}
            className={`
                flex flex-col items-start p-8 rounded-lg border border-white/5 bg-[#050A14]
                transition-all duration-1000 ease-out transform
                ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-12'}
                hover:border-accent-cyan/20 hover:shadow-glow-sm group
            `}
            style={{ transitionDelay: `${index * 200}ms` }}
        >
            {/* Icon */}
            <div className="mb-6 text-accent-cyan">
                {index === 0 && (
                    <svg className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
                    </svg>
                )}
                {index === 1 && (
                    <svg className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z" />
                    </svg>
                )}
                {index === 2 && (
                    <svg className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 15.75V18m-7.5-6.75h.008v.008H8.25v-.008zm0 2.25h.008v.008H8.25V13.5zm0 2.25h.008v.008H8.25v-.008zm0 2.25h.008v.008H8.25V18zm2.498-6.75h.007v.008h-.007v-.008zm0 2.25h.007v.008h-.007V13.5zm0 2.25h.007v.008h-.007v-.008zm0 2.25h.007v.008h-.007V18zm2.504-6.75h.008v.008h-.008v-.008zm0 2.25h.008v.008h-.008V13.5zm0 2.25h.008v.008h-.008v-.008zm0 2.25h.008v.008h-.008V18zm2.498-6.75h.008v.008h-.008v-.008zm0 2.25h.008v.008h-.008V13.5zM8.25 6h7.5v2.25h-7.5V6zM12 2.25c-1.892 0-3.758.11-5.593.322C5.307 2.7 4.5 3.65 4.5 4.757V19.5a2.25 2.25 0 002.25 2.25h10.5a2.25 2.25 0 002.25-2.25V4.757c0-1.108-.806-2.057-1.907-2.185A48.507 48.507 0 0012 2.25z" />
                    </svg>
                )}
            </div>

            <h3 className="text-xl font-display text-white mb-3 tracking-wide">
                {feature.title}
            </h3>

            <p className="text-text-secondary leading-relaxed font-light text-sm mb-4 flex-1">
                {feature.description}
            </p>

            <button
                onClick={() => window.location.href = `/dashboard?tab=${tabLinks[index]}`}
                className="text-accent-cyan text-xs font-medium uppercase tracking-wider flex items-center gap-1.5 hover:gap-2.5 transition-all group-hover:text-white"
            >
                Learn more
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
            </button>
        </div>
    );
};

const PlanetHorizon: React.FC = () => (
    <div className="relative w-full h-[400px] mt-24 overflow-hidden flex items-end justify-center">
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
    </div>
);

export const Features: React.FC = () => {
  return (
    <section id="features" className="py-24 bg-bg-primary relative">
      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">

        {/* Section Header */}
        <div className="text-center mb-16">
            <p className="text-accent-cyan text-sm font-mono uppercase tracking-widest mb-4">What You Can Do</p>
            <h2 className="text-3xl md:text-4xl font-display text-white mb-4">
                Professional Investment Tools
            </h2>
            <p className="text-text-muted text-base max-w-2xl mx-auto">
                Everything you need to research, analyze, and manage your investments like a professional.
            </p>
        </div>

        {/* Feature Grid */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          {featuresData.map((feature, index) => (
            <FeatureCard key={feature.title} feature={feature} index={index} />
          ))}
        </div>

        {/* Learn More CTA */}
        <div className="text-center">
            <button
                onClick={() => window.location.href = '/dashboard'}
                className="inline-flex items-center gap-2 px-8 py-3 bg-transparent border border-accent-cyan/40 rounded-full text-accent-cyan text-sm font-medium hover:bg-accent-cyan/10 hover:border-accent-cyan transition-all duration-300"
            >
                Explore All Features
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                </svg>
            </button>
        </div>
      </div>

      {/* Planet Footer Graphic */}
      <PlanetHorizon />
    </section>
  );
};

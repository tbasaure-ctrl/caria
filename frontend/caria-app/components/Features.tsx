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

// Bento grid card configurations
const bentoFeatures = [
  {
    title: 'Portfolio Analytics',
    description: 'Track your holdings, visualize asset allocation, and monitor real-time performance across all your investments.',
    icon: 'chart',
    tab: 'portfolio',
    size: 'large', // spans 2 columns
    preview: 'portfolio',
  },
  {
    title: 'AI Investment Partner',
    description: 'Analyze investment theses and get data-driven insights powered by advanced AI.',
    icon: 'brain',
    tab: 'analysis',
    size: 'medium',
    preview: 'chat',
  },
  {
    title: 'Valuation Tools',
    description: 'DCF models, Monte Carlo simulations, and business valuation workshops.',
    icon: 'calculator',
    tab: 'analysis',
    size: 'medium',
    preview: 'valuation',
  },
  {
    title: 'Market Regime Detection',
    description: 'AI-powered analysis of market conditions to optimize your positioning.',
    icon: 'radar',
    tab: 'portfolio',
    size: 'small',
    preview: 'regime',
  },
  {
    title: 'Hidden Gems Screener',
    description: 'Discover undervalued opportunities with quality metrics.',
    icon: 'gem',
    tab: 'analysis',
    size: 'small',
    preview: 'screener',
  },
];

// Mini preview components for bento cards
const PortfolioPreview: React.FC = () => (
  <div className="mt-4 p-3 rounded-lg bg-black/30 border border-white/5">
    <div className="flex items-end gap-1 h-16">
      {[40, 65, 45, 80, 60, 75, 55].map((h, i) => (
        <div
          key={i}
          className="flex-1 bg-gradient-to-t from-accent-cyan/60 to-accent-cyan/20 rounded-t"
          style={{ height: `${h}%` }}
        />
      ))}
    </div>
    <div className="flex justify-between mt-2 text-[10px] text-text-muted">
      <span>Portfolio Value</span>
      <span className="text-positive">+12.4%</span>
    </div>
  </div>
);

const ChatPreview: React.FC = () => (
  <div className="mt-4 space-y-2">
    <div className="flex gap-2">
      <div className="w-6 h-6 rounded-full bg-accent-cyan/20 flex items-center justify-center text-[8px] text-accent-cyan">AI</div>
      <div className="flex-1 p-2 rounded-lg bg-black/30 border border-white/5 text-[10px] text-text-secondary">
        Based on the DCF analysis, the intrinsic value appears to be...
      </div>
    </div>
  </div>
);

const ValuationPreview: React.FC = () => (
  <div className="mt-4 p-3 rounded-lg bg-black/30 border border-white/5">
    <div className="grid grid-cols-2 gap-2 text-[10px]">
      <div>
        <div className="text-text-muted">Fair Value</div>
        <div className="text-white font-mono">$142.50</div>
      </div>
      <div>
        <div className="text-text-muted">Upside</div>
        <div className="text-positive font-mono">+23%</div>
      </div>
    </div>
  </div>
);

const RegimePreview: React.FC = () => (
  <div className="mt-3 flex items-center gap-2">
    <div className="w-8 h-8 rounded-full bg-positive/20 flex items-center justify-center">
      <div className="w-3 h-3 rounded-full bg-positive animate-pulse" />
    </div>
    <div className="text-[10px]">
      <div className="text-positive font-medium">Expansion</div>
      <div className="text-text-muted">87% confidence</div>
    </div>
  </div>
);

const ScreenerPreview: React.FC = () => (
  <div className="mt-3 space-y-1">
    {['NVDA', 'MSFT', 'AAPL'].map((ticker, i) => (
      <div key={ticker} className="flex justify-between text-[10px] p-1 rounded bg-black/20">
        <span className="text-white font-mono">{ticker}</span>
        <span className="text-positive">A+</span>
      </div>
    ))}
  </div>
);

const previewComponents: Record<string, React.FC> = {
  portfolio: PortfolioPreview,
  chat: ChatPreview,
  valuation: ValuationPreview,
  regime: RegimePreview,
  screener: ScreenerPreview,
};

const iconComponents: Record<string, React.ReactNode> = {
  chart: (
    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
    </svg>
  ),
  brain: (
    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z" />
    </svg>
  ),
  calculator: (
    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 15.75V18m-7.5-6.75h.008v.008H8.25v-.008zm0 2.25h.008v.008H8.25V13.5zm0 2.25h.008v.008H8.25v-.008zm0 2.25h.008v.008H8.25V18zm2.498-6.75h.007v.008h-.007v-.008zm0 2.25h.007v.008h-.007V13.5zm0 2.25h.007v.008h-.007v-.008zm0 2.25h.007v.008h-.007V18zm2.504-6.75h.008v.008h-.008v-.008zm0 2.25h.008v.008h-.008V13.5zm0 2.25h.008v.008h-.008v-.008zm0 2.25h.008v.008h-.008V18zm2.498-6.75h.008v.008h-.008v-.008zm0 2.25h.008v.008h-.008V13.5zM8.25 6h7.5v2.25h-7.5V6zM12 2.25c-1.892 0-3.758.11-5.593.322C5.307 2.7 4.5 3.65 4.5 4.757V19.5a2.25 2.25 0 002.25 2.25h10.5a2.25 2.25 0 002.25-2.25V4.757c0-1.108-.806-2.057-1.907-2.185A48.507 48.507 0 0012 2.25z" />
    </svg>
  ),
  radar: (
    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 14.25v2.25m3-4.5v4.5m3-6.75v6.75m3-9v9M6 20.25h12A2.25 2.25 0 0020.25 18V6A2.25 2.25 0 0018 3.75H6A2.25 2.25 0 003.75 6v12A2.25 2.25 0 006 20.25z" />
    </svg>
  ),
  gem: (
    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z" />
    </svg>
  ),
};

interface BentoCardProps {
  feature: typeof bentoFeatures[0];
  index: number;
}

const BentoCard: React.FC<BentoCardProps> = ({ feature, index }) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const isVisible = useOnScreen(cardRef, 0.15);
  const PreviewComponent = previewComponents[feature.preview];

  const sizeClasses = {
    large: 'md:col-span-2 md:row-span-1',
    medium: 'md:col-span-1 md:row-span-1',
    small: 'md:col-span-1 md:row-span-1',
  };

  return (
    <div
      ref={cardRef}
      onClick={() => window.location.href = `/dashboard?tab=${feature.tab}`}
      className={`
        glass-card rounded-2xl p-6 cursor-pointer group
        transition-all duration-700 ease-out transform
        ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}
        ${sizeClasses[feature.size as keyof typeof sizeClasses]}
        hover:scale-[1.02] hover:shadow-[0_8px_40px_rgba(34,211,238,0.15)]
      `}
      style={{ transitionDelay: `${index * 100}ms` }}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-2">
        <div className="p-2 rounded-xl bg-accent-cyan/10 text-accent-cyan group-hover:bg-accent-cyan/20 transition-colors">
          {iconComponents[feature.icon]}
        </div>
        <svg className="w-4 h-4 text-text-muted group-hover:text-accent-cyan group-hover:translate-x-1 transition-all" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </div>

      {/* Content */}
      <h3 className="text-lg font-display text-white mb-2 group-hover:text-accent-cyan transition-colors">
        {feature.title}
      </h3>
      <p className="text-text-secondary text-sm leading-relaxed">
        {feature.description}
      </p>

      {/* Preview */}
      {PreviewComponent && <PreviewComponent />}
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
    <section id="features" className="py-24 bg-gradient-to-b from-transparent via-bg-primary to-black relative">
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

        {/* Bento Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6 mb-16">
          {bentoFeatures.map((feature, index) => (
            <BentoCard key={feature.title} feature={feature} index={index} />
          ))}
        </div>

        {/* Learn More CTA */}
        <div className="text-center">
            <button
                onClick={() => window.location.href = '/dashboard'}
                className="inline-flex items-center gap-2 px-8 py-3 glass-card rounded-full text-accent-cyan text-sm font-medium hover:bg-accent-cyan/10 hover:border-accent-cyan transition-all duration-300"
            >
                Explore All Features
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                </svg>
            </button>
        </div>
      </div>

      {/* Planet Footer Graphic - Preserved */}
      <PlanetHorizon />
    </section>
  );
};

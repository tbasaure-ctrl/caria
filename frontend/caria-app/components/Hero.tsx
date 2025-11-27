import React from 'react';
import { useNavigate } from 'react-router-dom';

interface HeroProps {
    onLogin?: () => void;
}

export const Hero: React.FC<HeroProps> = ({ onLogin }) => {
  const navigate = useNavigate();

  const handleDiscoverCaria = () => {
    navigate('/dashboard');
  };

  return (
    <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
      {/* Background Effects - Subtle and Elegant */}
      <div className="absolute inset-0 bg-hero-glow pointer-events-none mix-blend-screen opacity-50" />
      
      {/* Central thin line - Visual anchor */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-px h-40 bg-gradient-to-b from-transparent via-accent-cyan/20 to-transparent" />

      {/* Subtle Waves */}
      <div 
        className="absolute inset-0 opacity-30 pointer-events-none"
        style={{
            background: 'radial-gradient(circle at 50% 40%, rgba(34, 211, 238, 0.03) 0%, transparent 60%)'
        }}
      />

      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 flex flex-col items-center text-center">
        
        {/* Main Title - Two Lines, High Hierarchy */}
        <h1 className="font-display font-medium tracking-tight mb-10 animate-fade-in">
          <span className="block text-7xl sm:text-8xl md:text-9xl text-white leading-[0.9] tracking-tight">
            Reason First,
          </span>
          <span className="block text-4xl sm:text-5xl md:text-6xl italic text-text-secondary mt-4 font-light">
            returns will follow
          </span>
        </h1>

        {/* Subtitle */}
        <p 
          className="text-lg md:text-xl text-text-muted max-w-2xl mx-auto mb-12 font-light leading-relaxed animate-slide-up delay-100"
        >
          Merge timeless investment wisdom with the analytical power of deep learning. 
          An invaluable partner for your financial journey in the digital age.
        </p>

        {/* CTA Button - Elegant Outline */}
        <button
            onClick={handleDiscoverCaria}
            className="group relative px-12 py-4 bg-transparent border border-white/10 rounded-full overflow-hidden transition-all duration-500 hover:border-accent-cyan/40 animate-slide-up delay-200"
        >
            <div className="absolute inset-0 bg-accent-cyan/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <span className="relative text-xs font-bold tracking-[0.25em] uppercase text-white group-hover:text-accent-cyan transition-colors">
                Access Caria
            </span>
        </button>

        {/* Bottom Vertical Line */}
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-px h-32 bg-gradient-to-t from-transparent via-accent-cyan/10 to-transparent delay-300 animate-fade-in" />
      </div>
    </section>
  );
};

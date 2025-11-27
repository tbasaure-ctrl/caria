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
    <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden bg-bg-primary">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-hero-glow pointer-events-none" />
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-px h-32 bg-gradient-to-b from-transparent via-accent-cyan/30 to-transparent" />

      {/* Subtle Waves/Gradients from image */}
      <div 
        className="absolute inset-0 opacity-20 pointer-events-none"
        style={{
            background: 'radial-gradient(circle at 50% 50%, rgba(34, 211, 238, 0.05) 0%, transparent 50%)'
        }}
      />

      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 flex flex-col items-center text-center">
        
        {/* Main Title */}
        <h1 className="text-6xl sm:text-7xl md:text-8xl font-display font-normal tracking-tight mb-8 animate-fade-in">
          <span className="text-gradient-hero block">
            Reason First,
          </span>
          <span className="text-gradient-hero block italic mt-2">
            returns will follow
          </span>
        </h1>

        {/* Subtitle (Existing Text) */}
        <p 
          className="text-lg md:text-xl text-text-secondary max-w-2xl mx-auto mb-12 font-light leading-relaxed animate-slide-up delay-100"
        >
          Precision without distraction. Navigating the complexities of modern markets
          with absolute clarity and electric resolve.
        </p>

        {/* CTA Button */}
        <button
            onClick={handleDiscoverCaria}
            className="group relative px-10 py-4 bg-transparent border border-white/10 rounded-full overflow-hidden transition-all duration-300 hover:border-accent-cyan/50 animate-slide-up delay-200"
        >
            <div className="absolute inset-0 bg-white/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <span className="relative text-sm font-medium tracking-[0.2em] uppercase text-white group-hover:text-accent-cyan transition-colors">
                Access Caria
            </span>
        </button>

        {/* Bottom Vertical Line */}
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-px h-24 bg-gradient-to-t from-transparent via-accent-cyan/20 to-transparent delay-300 animate-fade-in" />
      </div>
    </section>
  );
};

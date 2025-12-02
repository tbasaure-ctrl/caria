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
    <section className="relative min-h-[100vh] flex items-center justify-center overflow-hidden dark-wave-bg">
      {/* Wave overlay layers for depth */}
      <div className="wave-overlay" />
      <div className="grain-texture" />

      {/* Subtle top accent line */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-px h-40 bg-gradient-to-b from-accent-cyan/30 via-accent-cyan/10 to-transparent z-10" />

      {/* Main content */}
      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 flex flex-col items-center text-center mt-[-5vh]">

        {/* Main Title - Logo as C in CARIA */}
        <h1 className="font-display font-medium tracking-tight mb-10 animate-fade-in relative">
          <div className="flex items-center justify-center gap-2 sm:gap-3">
            {/* Logo replaces "C" in CARIA */}
            <div className="w-16 h-16 sm:w-20 sm:h-20 md:w-28 md:h-28 flex items-center justify-center flex-shrink-0">
              <svg viewBox="0 0 100 100" className="w-full h-full">
                <circle
                  cx="50"
                  cy="50"
                  r="40"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeDasharray="200"
                  strokeDashoffset="50"
                  className="text-accent-cyan"
                  style={{ transform: 'rotate(-45deg)', transformOrigin: 'center' }}
                />
              </svg>
            </div>
            <span className="text-7xl sm:text-8xl md:text-9xl text-white leading-[0.9] tracking-tight">
              ARIA
            </span>
          </div>
          <span className="block text-4xl sm:text-5xl md:text-6xl italic text-accent-cyan mt-4 font-light">
            Cognitive Analysis and Risk Investment Assistant
          </span>
        </h1>

        {/* Subtitle - Preserved */}
        <p
          className="text-lg md:text-xl text-text-muted max-w-2xl mx-auto mb-12 font-light leading-relaxed animate-slide-up delay-100 relative"
        >
          Merge timeless investment wisdom with the analytical power of deep learning.
          An invaluable partner for your financial journey in the digital age.
        </p>

        {/* CTA Button - Preserved */}
        <button
          onClick={handleDiscoverCaria}
          className="group relative px-12 py-4 bg-accent-cyan/10 border border-accent-cyan/50 rounded-full overflow-hidden transition-all duration-500 hover:bg-accent-cyan hover:border-accent-cyan hover:shadow-[0_0_30px_rgba(34,211,238,0.6)] animate-slide-up delay-200 backdrop-blur-sm"
        >
          <div className="absolute inset-0 bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
          <span className="relative text-sm font-bold tracking-[0.25em] uppercase text-accent-cyan group-hover:text-bg-primary transition-colors">
            Access Caria
          </span>
        </button>

        {/* Bottom Vertical Line */}
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-px h-32 bg-gradient-to-t from-transparent via-accent-cyan/10 to-transparent delay-300 animate-fade-in" />
      </div>
    </section>
  );
};

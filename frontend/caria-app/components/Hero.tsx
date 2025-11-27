import React from 'react';
import { useNavigate } from 'react-router-dom';

interface HeroProps {
    onLogin?: () => void;
}

// SVG Component mimicking the "Digital Wave" image provided
const DigitalWave: React.FC = () => (
    <svg viewBox="0 0 1440 400" className="w-full h-full opacity-40" preserveAspectRatio="none">
        <defs>
            <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="transparent" />
                <stop offset="20%" stopColor="#22D3EE" stopOpacity="0.2" />
                <stop offset="50%" stopColor="#38BDF8" stopOpacity="0.5" />
                <stop offset="80%" stopColor="#22D3EE" stopOpacity="0.2" />
                <stop offset="100%" stopColor="transparent" />
            </linearGradient>
            <filter id="glow">
                <feGaussianBlur stdDeviation="4" result="coloredBlur" />
                <feMerge>
                    <feMergeNode in="coloredBlur" />
                    <feMergeNode in="SourceGraphic" />
                </feMerge>
            </filter>
        </defs>
        
        {/* Sine Waves */}
        <path d="M0,200 Q180,100 360,200 T720,200 T1080,200 T1440,200" fill="none" stroke="url(#waveGradient)" strokeWidth="1" filter="url(#glow)" />
        <path d="M0,200 Q180,300 360,200 T720,200 T1080,200 T1440,200" fill="none" stroke="url(#waveGradient)" strokeWidth="1" opacity="0.5" />
        <path d="M0,200 Q360,50 720,200 T1440,200" fill="none" stroke="url(#waveGradient)" strokeWidth="0.5" opacity="0.3" />
        
        {/* Tech HUD Circle (Right side) */}
        <g transform="translate(1100, 200)" filter="url(#glow)">
            <circle r="80" fill="none" stroke="#22D3EE" strokeWidth="1" opacity="0.8" />
            <circle r="60" fill="none" stroke="#22D3EE" strokeWidth="2" strokeDasharray="20 10" opacity="0.6" />
            <circle r="40" fill="none" stroke="#38BDF8" strokeWidth="1" opacity="0.9" />
            <path d="M-90,0 L-40,0" stroke="#22D3EE" strokeWidth="1" />
            <path d="M40,0 L90,0" stroke="#22D3EE" strokeWidth="1" />
            <path d="M0,-90 L0,-40" stroke="#22D3EE" strokeWidth="1" />
            <path d="M0,40 L0,90" stroke="#22D3EE" strokeWidth="1" />
        </g>
        
        {/* Connecting Line */}
        <line x1="0" y1="200" x2="1440" y2="200" stroke="url(#waveGradient)" strokeWidth="0.5" opacity="0.2" />
    </svg>
);

export const Hero: React.FC<HeroProps> = ({ onLogin }) => {
  const navigate = useNavigate();

  const handleDiscoverCaria = () => {
    navigate('/dashboard');
  };

  return (
    <section className="relative min-h-[100vh] flex items-center justify-center overflow-hidden bg-bg-primary">
      {/* Digital Wave Background - Visible from entry */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-0">
          <div className="w-full h-full max-w-[1920px] opacity-60 animate-pulse-subtle scale-110">
              <DigitalWave />
          </div>
      </div>

      {/* Background Effects */}
      <div className="absolute inset-0 bg-hero-glow pointer-events-none mix-blend-screen opacity-30" />
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-px h-40 bg-gradient-to-b from-transparent via-accent-cyan/20 to-transparent" />

      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 flex flex-col items-center text-center mt-[-5vh]">
        
        {/* Main Title */}
        <h1 className="font-display font-medium tracking-tight mb-10 animate-fade-in relative">
          <span className="block text-7xl sm:text-8xl md:text-9xl text-white leading-[0.9] tracking-tight">
            Reason First,
          </span>
          <span className="block text-4xl sm:text-5xl md:text-6xl italic text-text-secondary mt-4 font-light">
            returns will follow
          </span>
        </h1>

        {/* Subtitle */}
        <p 
          className="text-lg md:text-xl text-text-muted max-w-2xl mx-auto mb-12 font-light leading-relaxed animate-slide-up delay-100 relative"
        >
          Merge timeless investment wisdom with the analytical power of deep learning. 
          An invaluable partner for your financial journey in the digital age.
        </p>

        {/* CTA Button */}
        <button
            onClick={handleDiscoverCaria}
            className="group relative px-12 py-4 bg-transparent border border-white/10 rounded-full overflow-hidden transition-all duration-500 hover:border-accent-cyan/40 animate-slide-up delay-200 backdrop-blur-sm"
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

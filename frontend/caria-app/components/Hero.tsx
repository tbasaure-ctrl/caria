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
    <section className="relative py-20 md:py-32 lg:py-40 overflow-hidden min-h-[70vh] md:min-h-[80vh] flex items-center">
      {/* Subtle gradient background */}
      <div
        className="absolute inset-0 opacity-20 transition-all duration-1000"
        style={{
          background: `radial-gradient(ellipse at 50% 50%,
            rgba(74, 144, 226, 0.3) 0%,
            rgba(58, 122, 194, 0.1) 40%,
            transparent 70%)`,
        }}
      />

      <div className="container mx-auto px-4 sm:px-6 md:px-12 relative z-10">
        <div className="max-w-5xl mx-auto text-center">
          {/* Eyebrow */}
          <div className="mb-6 md:mb-8">
            <span
              className="inline-block px-5 py-2 rounded-full text-xs font-semibold tracking-[0.2em] uppercase"
              style={{
                backgroundColor: 'rgba(74, 144, 226, 0.1)',
                color: 'var(--color-blue-light)',
                border: '1px solid rgba(74, 144, 226, 0.2)',
              }}>
              Cognitive Analysis & Risk Investment Assistant
            </span>
          </div>

          {/* Main headline */}
          <h1
            className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold leading-[1.1] mb-8 md:mb-10"
            style={{
              fontFamily: "'Instrument Serif', Georgia, serif",
              color: 'var(--color-cream)',
              letterSpacing: '-0.02em',
            }}>
            Own Your Investments
          </h1>

          {/* Subtitle */}
          <p
            className="text-base sm:text-lg md:text-xl leading-relaxed max-w-3xl mx-auto mb-10 md:mb-14 px-2"
            style={{
              fontFamily: "'Crimson Pro', Georgia, serif",
              color: 'rgba(232, 230, 227, 0.8)',
              fontWeight: 400,
              lineHeight: '1.8',
            }}>
            We merge timeless investment wisdom with the analytical power of deep learning,
            creating an invaluable partner for your financial journey.
          </p>

          {/* CTA Button */}
          <div className="flex justify-center">
            <button
              onClick={handleDiscoverCaria}
              className="group relative px-8 py-3.5 text-base font-medium rounded-lg transition-all duration-300"
              style={{
                fontFamily: 'var(--font-display)',
                backgroundColor: 'rgba(30, 35, 45, 0.9)',
                color: 'var(--color-cream)',
                border: '1px solid rgba(232, 230, 227, 0.15)',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'rgba(232, 230, 227, 0.3)';
                e.currentTarget.style.backgroundColor = 'rgba(40, 45, 55, 0.95)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'rgba(232, 230, 227, 0.15)';
                e.currentTarget.style.backgroundColor = 'rgba(30, 35, 45, 0.9)';
              }}
            >
              Discover Caria
            </button>
          </div>

        </div>
      </div>
    </section>
  );
};

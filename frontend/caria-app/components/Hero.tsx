import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

interface HeroProps {
    onLogin?: () => void;
}

export const Hero: React.FC<HeroProps> = ({ onLogin }) => {
  const navigate = useNavigate();
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth) * 100,
        y: (e.clientY / window.innerHeight) * 100,
      });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const handleDiscoverCaria = () => {
    navigate('/dashboard');
  };

  return (
    <section className="relative py-24 md:py-40 overflow-hidden min-h-[90vh] flex items-center">
      {/* Dynamic gradient background with parallax effect */}
      <div
        className="absolute inset-0 opacity-30 transition-all duration-1000"
        style={{
          background: `radial-gradient(circle at ${mousePosition.x}% ${mousePosition.y}%,
            rgba(74, 144, 226, 0.4) 0%,
            rgba(58, 122, 194, 0.2) 30%,
            rgba(15, 20, 25, 0.1) 70%,
            transparent 100%)`,
        }}
      />


      <div className="container mx-auto px-6 md:px-12 relative z-10">
        <div className="max-w-6xl mx-auto">
          {/* Eyebrow with animation */}
          <div className="text-center mb-8 fade-in">
            <span
              className="inline-block px-6 py-3 rounded-full text-xs font-bold tracking-[0.3em] uppercase"
              style={{
                backgroundColor: 'rgba(74, 144, 226, 0.15)',
                color: 'var(--color-blue-light)',
                border: '1px solid rgba(74, 144, 226, 0.3)',
                fontFamily: 'var(--font-mono)',
                backdropFilter: 'blur(10px)',
                boxShadow: '0 4px 20px rgba(74, 144, 226, 0.2)',
              }}>
              Cognitive Analysis & Risk Investment Assistant
            </span>
          </div>

          {/* Main headline - bold and dynamic, centered */}
          <h1
            className="text-5xl md:text-7xl lg:text-8xl font-black leading-[1.1] mb-10 fade-in delay-200"
            style={{
              fontFamily: "'Instrument Serif', Georgia, serif",
              color: 'var(--color-cream)',
              textAlign: 'center',
              letterSpacing: '-0.03em',
            }}>
            <span
              className="block gradient-text"
              style={{
                opacity: 0,
                background: 'linear-gradient(135deg, #4A90E2 0%, #E8E6E3 50%, #5B9FE5 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                backgroundSize: '200% auto',
                animation: 'fadeIn 0.8s ease-out 0.5s forwards, shimmer 3s linear infinite',
              }}>
              Own Your Investments
            </span>
          </h1>

          {/* Decorative divider with animation */}
          <div className="flex items-center justify-center gap-6 mb-12 fade-in delay-300">
            <div className="h-[2px] w-20 bg-gradient-to-r from-transparent via-blue-400 to-transparent"></div>
            <div className="flex gap-2">
              <div className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></div>
              <div className="w-2 h-2 rounded-full bg-blue-300 animate-pulse" style={{ animationDelay: '0.2s' }}></div>
              <div className="w-2 h-2 rounded-full bg-blue-200 animate-pulse" style={{ animationDelay: '0.4s' }}></div>
            </div>
            <div className="h-[2px] w-20 bg-gradient-to-r from-transparent via-blue-400 to-transparent"></div>
          </div>

          {/* Subtitle - elegant and spacious */}
          <p
            className="text-lg md:text-xl lg:text-2xl leading-relaxed max-w-6xl mx-auto text-center mb-14 fade-in delay-400 px-4"
            style={{
              fontFamily: "'Crimson Pro', Georgia, serif",
              color: 'rgba(232, 230, 227, 0.85)',
              fontWeight: 400,
              lineHeight: '1.8',
            }}>
            We merge timeless investment wisdom with the analytical power of deep learning,
            <br className="hidden md:block" />
            creating an invaluable partner for your financial journey.
          </p>

          {/* CTA Button - Discover Caria */}
          <div className="flex justify-center fade-in delay-500">
            <button
              onClick={handleDiscoverCaria}
              className="group relative px-8 py-4 text-lg font-semibold rounded-lg transition-all duration-300 transform hover:scale-105 active:scale-95"
              style={{
                fontFamily: 'var(--font-display)',
                background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
                color: '#E8E6E3',
                boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(232, 230, 227, 0.1)',
                border: '1px solid rgba(232, 230, 227, 0.2)',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.boxShadow = '0 12px 32px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(232, 230, 227, 0.2)';
                e.currentTarget.style.background = 'linear-gradient(135deg, #2d2d2d 0%, #3a3a3a 100%)';
                e.currentTarget.style.borderColor = 'rgba(232, 230, 227, 0.3)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.boxShadow = '0 8px 24px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(232, 230, 227, 0.1)';
                e.currentTarget.style.background = 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)';
                e.currentTarget.style.borderColor = 'rgba(232, 230, 227, 0.2)';
              }}
            >
              <span className="relative z-10 flex items-center gap-2">
                Discover Caria
                <svg 
                  className="w-5 h-5 transition-transform duration-300 group-hover:translate-x-1" 
                  fill="none" 
                  stroke="currentColor" 
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </span>
            </button>
          </div>

        </div>
      </div>

      <style>{`
        @keyframes gridSlide {
          from {
            transform: translate(0, 0);
          }
          to {
            transform: translate(60px, 60px);
          }
        }

        @keyframes shimmer {
          0% {
            background-position: 0% center;
          }
          100% {
            background-position: 200% center;
          }
        }
      `}</style>
    </section>
  );
};

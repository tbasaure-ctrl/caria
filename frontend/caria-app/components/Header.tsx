import React from 'react';
import { CariaLogoIcon } from './Icons';

interface HeaderProps {
    onLogin: () => void;
    onRegister?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onLogin, onRegister }) => {
  return (
    <header className="sticky top-0 backdrop-blur-md z-50 fade-in"
            style={{
              backgroundColor: 'rgba(10, 13, 18, 0.8)',
              borderBottom: '1px solid var(--color-bg-tertiary)'
            }}>
      <div className="container mx-auto px-6 py-5 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <CariaLogoIcon className="w-9 h-9" style={{color: 'var(--color-secondary)'}}/>
          <h1 className="text-3xl font-bold tracking-tight"
              style={{
                fontFamily: 'var(--font-display)',
                color: 'var(--color-cream)',
                letterSpacing: '-0.02em'
              }}>
            Caria
          </h1>
        </div>
        <nav className="hidden md:flex items-center gap-10"
             style={{
               fontFamily: 'var(--font-body)',
               fontSize: '0.95rem',
               fontWeight: 500
             }}>
          <a href="#"
             className="transition-all duration-200"
             style={{color: 'var(--color-text-secondary)'}}
             onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-cream)'}
             onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-secondary)'}>
            Features
          </a>
          <a href="#"
             className="transition-all duration-200"
             style={{color: 'var(--color-text-secondary)'}}
             onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-cream)'}
             onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-secondary)'}>
            Community
          </a>
          <a href="#"
             className="transition-all duration-200"
             style={{color: 'var(--color-text-secondary)'}}
             onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-cream)'}
             onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-secondary)'}>
            Pricing
          </a>
        </nav>
        <div className="flex items-center gap-5">
            <button
              onClick={onLogin}
              className="transition-all duration-200 font-medium"
              style={{
                color: 'var(--color-text-secondary)',
                fontFamily: 'var(--font-body)'
              }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-cream)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-secondary)'}>
              Login
            </button>
            <button
              onClick={onRegister || onLogin}
              className="py-2.5 px-6 rounded-lg font-semibold transition-all duration-200"
              style={{
                backgroundColor: 'var(--color-primary)',
                color: 'var(--color-cream)',
                fontFamily: 'var(--font-body)',
                border: '1px solid var(--color-primary)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = 'var(--color-primary-light)';
                e.currentTarget.style.transform = 'translateY(-1px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'var(--color-primary)';
                e.currentTarget.style.transform = 'translateY(0)';
              }}>
            Sign Up
            </button>
        </div>
      </div>
    </header>
  );
};

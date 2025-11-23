import React, { useState } from 'react';
import { CariaLogoIcon } from './Icons';

interface HeaderProps {
    onLogin: () => void;
    onRegister?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onLogin, onRegister }) => {
  const [showFeaturesTooltip, setShowFeaturesTooltip] = useState(false);
  const [showFeaturesModal, setShowFeaturesModal] = useState(false);

  return (
    <>
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
            <div className="relative">
              <a href="#features"
                 className="transition-all duration-200 cursor-pointer"
                 style={{color: 'var(--color-text-secondary)'}}
                 onMouseEnter={(e) => {
                   e.currentTarget.style.color = 'var(--color-cream)';
                   setShowFeaturesTooltip(true);
                 }}
                 onMouseLeave={(e) => {
                   e.currentTarget.style.color = 'var(--color-text-secondary)';
                   setShowFeaturesTooltip(false);
                 }}
                 onClick={(e) => {
                   e.preventDefault();
                   setShowFeaturesModal(true);
                 }}>
                Features
              </a>
              {showFeaturesTooltip && (
                <div
                  className="absolute left-0 top-8 w-64 p-3 rounded-lg shadow-lg z-50"
                  style={{
                    backgroundColor: 'var(--color-bg-primary)',
                    border: '1px solid var(--color-primary)',
                    color: 'var(--color-text-primary)',
                    fontSize: '0.875rem',
                    lineHeight: '1.4'
                  }}
                >
                  Explore the features and what Caria offers to help you succeed
                </div>
              )}
            </div>
            <a href="#contact"
               className="transition-all duration-200"
               style={{color: 'var(--color-text-secondary)'}}
               onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-cream)'}
               onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-secondary)'}>
              Contact Us
            </a>
            <a href="#pricing"
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

    {/* Features Modal */}
    {showFeaturesModal && (
      <div
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        style={{ backgroundColor: 'rgba(0, 0, 0, 0.8)' }}
        onClick={() => setShowFeaturesModal(false)}
      >
        <div
          className="rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto p-8"
          onClick={(e) => e.stopPropagation()}
          style={{
            backgroundColor: 'var(--color-bg-primary)',
            border: '1px solid var(--color-bg-tertiary)',
          }}
        >
          <div className="flex justify-between items-center mb-6">
            <h2
              className="text-3xl font-bold"
              style={{
                fontFamily: 'var(--font-display)',
                color: 'var(--color-cream)',
              }}
            >
              Caria Features
            </h2>
            <button
              onClick={() => setShowFeaturesModal(false)}
              className="text-2xl font-bold"
              style={{ color: 'var(--color-text-secondary)' }}
            >
              ×
            </button>
          </div>

          <div className="space-y-6">
            {/* Portfolio Management */}
            <div className="p-4 rounded-lg" style={{ backgroundColor: 'var(--color-bg-secondary)', border: '1px solid var(--color-bg-tertiary)' }}>
              <h3 className="text-xl font-bold mb-2" style={{ fontFamily: 'var(--font-display)', color: 'var(--color-cream)' }}>
                📊 Portfolio Management
              </h3>
              <p style={{ color: 'var(--color-text-secondary)' }}>
                Track your investments with real-time pricing, performance analytics, and advanced metrics including Sharpe Ratio, Alpha, Beta, and drawdown analysis.
              </p>
            </div>

            {/* Market Intelligence */}
            <div className="p-4 rounded-lg" style={{ backgroundColor: 'var(--color-bg-secondary)', border: '1px solid var(--color-bg-tertiary)' }}>
              <h3 className="text-xl font-bold mb-2" style={{ fontFamily: 'var(--font-display)', color: 'var(--color-cream)' }}>
                🌍 Market Intelligence
              </h3>
              <p style={{ color: 'var(--color-text-secondary)' }}>
                Stay informed with global market indicators, Fear & Greed Index, macroeconomic regime detection, and live market data from major exchanges.
              </p>
            </div>

            {/* Analysis Tools */}
            <div className="p-4 rounded-lg" style={{ backgroundColor: 'var(--color-bg-secondary)', border: '1px solid var(--color-bg-tertiary)' }}>
              <h3 className="text-xl font-bold mb-2" style={{ fontFamily: 'var(--font-display)', color: 'var(--color-cream)' }}>
                🔬 Advanced Analysis
              </h3>
              <p style={{ color: 'var(--color-text-secondary)' }}>
                Deep-dive with DCF valuation, Monte Carlo simulations, regime stress testing, and AI-powered portfolio recommendations tailored to market conditions.
              </p>
            </div>

            {/* Thesis Arena */}
            <div className="p-4 rounded-lg" style={{ backgroundColor: 'var(--color-bg-secondary)', border: '1px solid var(--color-bg-tertiary)' }}>
              <h3 className="text-xl font-bold mb-2" style={{ fontFamily: 'var(--font-display)', color: 'var(--color-cream)' }}>
                ⚔️ Thesis Arena
              </h3>
              <p style={{ color: 'var(--color-text-secondary)' }}>
                Challenge your investment ideas against diverse perspectives: value investors, growth investors, contrarians, and crypto enthusiasts. Refine your conviction before you invest.
              </p>
            </div>

            {/* Community Insights */}
            <div className="p-4 rounded-lg" style={{ backgroundColor: 'var(--color-bg-secondary)', border: '1px solid var(--color-bg-tertiary)' }}>
              <h3 className="text-xl font-bold mb-2" style={{ fontFamily: 'var(--font-display)', color: 'var(--color-cream)' }}>
                👥 Community Insights
              </h3>
              <p style={{ color: 'var(--color-text-secondary)' }}>
                Share investment theses, discover trending ideas, and learn from a community of analytical investors. Vote, discuss, and collaborate on research.
              </p>
            </div>
          </div>
        </div>
      </div>
    )}
    </>
  );
};

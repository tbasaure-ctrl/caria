import React, { useState } from 'react';
import { CariaLogoIcon } from './Icons';

interface HeaderProps {
    onLogin?: () => void;
    onRegister?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onLogin, onRegister }) => {
    const [showFeaturesModal, setShowFeaturesModal] = useState(false);

    return (
        <>
            <header 
                className="sticky top-0 z-50 border-b transition-all duration-300"
                style={{
                    backgroundColor: 'rgba(10, 14, 20, 0.95)',
                    backdropFilter: 'blur(16px)',
                    WebkitBackdropFilter: 'blur(16px)',
                    borderColor: 'var(--color-border-subtle)',
                }}
            >
                {/* Main Header */}
                <div className="container mx-auto px-4 sm:px-6 lg:px-10">
                    <div className="flex items-center justify-between h-16 md:h-20">
                        {/* Logo - Left */}
                        <div className="flex items-center gap-2 flex-shrink-0">
                            <CariaLogoIcon 
                                className="w-7 h-7 md:w-8 md:h-8" 
                                style={{ color: 'var(--color-cream)' }}
                            />
                            <span 
                                className="text-lg md:text-xl font-semibold hidden sm:block"
                                style={{
                                    fontFamily: 'var(--font-display)',
                                    color: 'var(--color-cream)',
                                    letterSpacing: '-0.01em'
                                }}
                            >
                                Caria
                            </span>
                        </div>

                        {/* Navigation - Center (hidden on mobile) */}
                        <nav 
                            className="hidden md:flex items-center justify-center gap-8"
                            style={{
                                fontFamily: 'var(--font-body)',
                                fontSize: '14px',
                                fontWeight: 500
                            }}
                        >
                            <a 
                                href="#features"
                                className="transition-colors duration-200 py-2"
                                style={{ color: 'var(--color-text-secondary)' }}
                                onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-text-primary)'}
                                onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-secondary)'}
                            >
                                Features
                            </a>
                            <a 
                                href="#community"
                                className="transition-colors duration-200 py-2"
                                style={{ color: 'var(--color-text-secondary)' }}
                                onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-text-primary)'}
                                onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-secondary)'}
                            >
                                Community
                            </a>
                            <a 
                                href="#pricing"
                                className="transition-colors duration-200 py-2"
                                style={{ color: 'var(--color-text-secondary)' }}
                                onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-text-primary)'}
                                onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-secondary)'}
                            >
                                Pricing
                            </a>
                        </nav>

                        {/* Auth Buttons - Right */}
                        <div className="flex items-center gap-2 sm:gap-3 flex-shrink-0">
                            <button
                                onClick={onLogin}
                                className="px-3 sm:px-4 py-2 text-sm font-medium transition-all duration-200"
                                style={{
                                    color: 'var(--color-text-secondary)',
                                }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.color = 'var(--color-text-primary)';
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.color = 'var(--color-text-secondary)';
                                }}
                            >
                                Login
                            </button>
                            <button
                                onClick={onRegister}
                                className="px-4 sm:px-5 py-2 rounded-lg text-sm font-semibold transition-all duration-200"
                                style={{
                                    backgroundColor: 'var(--color-bg-tertiary)',
                                    color: 'var(--color-text-primary)',
                                    border: '1px solid var(--color-border-subtle)',
                                }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.borderColor = 'var(--color-accent-primary)';
                                    e.currentTarget.style.backgroundColor = 'var(--color-bg-surface)';
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.borderColor = 'var(--color-border-subtle)';
                                    e.currentTarget.style.backgroundColor = 'var(--color-bg-tertiary)';
                                }}
                            >
                                Sign Up
                            </button>
                        </div>
                    </div>
                </div>
            </header>

            {/* Features Modal */}
            {showFeaturesModal && (
                <div
                    className="fixed inset-0 z-50 flex items-center justify-center p-4"
                    style={{ backgroundColor: 'rgba(0, 0, 0, 0.85)' }}
                    onClick={() => setShowFeaturesModal(false)}
                >
                    <div
                        className="rounded-xl max-w-4xl w-full max-h-[85vh] overflow-y-auto"
                        onClick={(e) => e.stopPropagation()}
                        style={{
                            backgroundColor: 'var(--color-bg-secondary)',
                            border: '1px solid var(--color-border-default)',
                        }}
                    >
                        {/* Modal Header */}
                        <div 
                            className="sticky top-0 flex justify-between items-center px-8 py-5 border-b"
                            style={{ 
                                backgroundColor: 'var(--color-bg-secondary)',
                                borderColor: 'var(--color-border-subtle)'
                            }}
                        >
                            <div>
                                <h2 
                                    className="text-2xl font-bold"
                                    style={{
                                        fontFamily: 'var(--font-display)',
                                        color: 'var(--color-text-primary)',
                                    }}
                                >
                                    Platform Capabilities
                                </h2>
                                <p 
                                    className="text-sm mt-1"
                                    style={{ color: 'var(--color-text-muted)' }}
                                >
                                    Professional-grade investment intelligence
                                </p>
                            </div>
                            <button
                                onClick={() => setShowFeaturesModal(false)}
                                className="w-10 h-10 rounded-full flex items-center justify-center text-xl transition-colors"
                                style={{ 
                                    color: 'var(--color-text-muted)',
                                    backgroundColor: 'var(--color-bg-tertiary)'
                                }}
                            >
                                ×
                            </button>
                        </div>

                        {/* Modal Content */}
                        <div className="p-8 space-y-6">
                            {/* Feature Grid */}
                            <div className="grid md:grid-cols-2 gap-4">
                                {[
                                    {
                                        icon: '📊',
                                        title: 'Portfolio Analytics',
                                        desc: 'Real-time performance tracking with Sharpe, Alpha, Beta, and advanced risk metrics.'
                                    },
                                    {
                                        icon: '🌍',
                                        title: 'Market Intelligence',
                                        desc: 'Global indices, Fear & Greed, and proprietary regime detection for market timing.'
                                    },
                                    {
                                        icon: '🔬',
                                        title: 'Deep Valuation',
                                        desc: 'DCF, Monte Carlo, and multi-factor scoring to find undervalued opportunities.'
                                    },
                                    {
                                        icon: '⚔️',
                                        title: 'Thesis Arena',
                                        desc: 'Stress-test your investment ideas against contrarian perspectives.'
                                    },
                                    {
                                        icon: '🎯',
                                        title: 'Alpha Stock Picker',
                                        desc: 'Weekly top picks based on momentum, quality, valuation & catalysts.'
                                    },
                                    {
                                        icon: '💎',
                                        title: 'Hidden Gems Screener',
                                        desc: 'Discover undervalued mid-caps before Wall Street notices.'
                                    },
                                ].map((feature, idx) => (
                                    <div 
                                        key={idx}
                                        className="p-5 rounded-lg transition-all duration-200"
                                        style={{ 
                                            backgroundColor: 'var(--color-bg-tertiary)',
                                            border: '1px solid var(--color-border-subtle)'
                                        }}
                                    >
                                        <div className="flex items-start gap-4">
                                            <span className="text-2xl">{feature.icon}</span>
                                            <div>
                                                <h3 
                                                    className="font-semibold text-base mb-1"
                                                    style={{ 
                                                        fontFamily: 'var(--font-display)',
                                                        color: 'var(--color-text-primary)' 
                                                    }}
                                                >
                                                    {feature.title}
                                                </h3>
                                                <p 
                                                    className="text-sm leading-relaxed"
                                                    style={{ color: 'var(--color-text-secondary)' }}
                                                >
                                                    {feature.desc}
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
};

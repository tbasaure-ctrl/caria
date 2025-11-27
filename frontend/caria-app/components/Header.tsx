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
                    backgroundColor: 'rgba(10, 14, 20, 0.92)',
                    backdropFilter: 'blur(16px)',
                    WebkitBackdropFilter: 'blur(16px)',
                    borderColor: 'var(--color-border-subtle)',
                }}
            >
                {/* Main Header */}
                <div className="container mx-auto px-6 lg:px-10">
                    <div className="flex items-center justify-between h-20 relative">
                        {/* Auth Buttons - Left Aligned */}
                        <div className="flex items-center gap-3">
                            <button
                                onClick={onLogin}
                                className="px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200"
                                style={{
                                    color: 'var(--color-text-secondary)',
                                    backgroundColor: 'transparent',
                                    border: '1px solid var(--color-border-subtle)'
                                }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.color = 'var(--color-text-primary)';
                                    e.currentTarget.style.borderColor = 'var(--color-accent-primary)';
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.color = 'var(--color-text-secondary)';
                                    e.currentTarget.style.borderColor = 'var(--color-border-subtle)';
                                }}
                            >
                                Sign In
                            </button>
                            <button
                                onClick={onRegister}
                                className="px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-200"
                                style={{
                                    backgroundColor: 'var(--color-accent-primary)',
                                    color: '#FFFFFF',
                                }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.opacity = '0.9';
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.opacity = '1';
                                }}
                            >
                                Get Started
                            </button>
                        </div>

                        {/* Centered Content */}
                        <div className="flex-1 flex flex-col items-center justify-center absolute left-0 right-0">
                            {/* Logo and Title */}
                            <div className="flex items-center gap-3 mb-4">
                                <CariaLogoIcon 
                                    className="w-8 h-8" 
                                    style={{ color: 'var(--color-accent-primary)' }}
                                />
                                <h1 
                                    className="text-lg font-bold tracking-tight leading-tight text-center max-w-4xl"
                                    style={{
                                        fontFamily: 'var(--font-display)',
                                        color: 'var(--color-text-primary)',
                                        letterSpacing: '-0.02em'
                                    }}
                                >
                                    CARIA: We do not intend to make financial advisement, we want to join your journey to financial freedom.
                                </h1>
                            </div>

                            {/* Navigation - Centered and Spaced */}
                            <nav 
                                className="flex items-center justify-center gap-12"
                                style={{
                                    fontFamily: 'var(--font-body)',
                                    fontSize: '14px',
                                    fontWeight: 500
                                }}
                            >
                                <a 
                                    href="#portfolio"
                                    className="transition-colors duration-200 hover:text-text-primary px-4 py-2"
                                    style={{ color: 'var(--color-text-secondary)' }}
                                >
                                    Portfolio
                                </a>
                                <a 
                                    href="#analysis"
                                    className="transition-colors duration-200 hover:text-text-primary px-4 py-2"
                                    style={{ color: 'var(--color-text-secondary)' }}
                                >
                                    Analysis
                                </a>
                                <a 
                                    href="#research"
                                    className="transition-colors duration-200 hover:text-text-primary px-4 py-2"
                                    style={{ color: 'var(--color-text-secondary)' }}
                                >
                                    Research
                                </a>
                            </nav>
                        </div>

                        {/* Spacer for right alignment */}
                        <div className="w-40"></div>
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

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
                {/* Top Ticker Bar - Bloomberg Style */}
                <div 
                    className="hidden md:block overflow-hidden py-1.5"
                    style={{ 
                        backgroundColor: 'var(--color-bg-secondary)',
                        borderBottom: '1px solid var(--color-border-subtle)'
                    }}
                >
                    <div className="flex items-center gap-6 text-xs font-mono animate-ticker whitespace-nowrap">
                        <span className="flex items-center gap-2">
                            <span className="text-text-muted">S&P 500</span>
                            <span className="text-text-primary font-medium">5,234.18</span>
                            <span className="text-positive">+0.82%</span>
                        </span>
                        <span className="text-border">|</span>
                        <span className="flex items-center gap-2">
                            <span className="text-text-muted">NASDAQ</span>
                            <span className="text-text-primary font-medium">16,742.39</span>
                            <span className="text-positive">+1.14%</span>
                        </span>
                        <span className="text-border">|</span>
                        <span className="flex items-center gap-2">
                            <span className="text-text-muted">DOW</span>
                            <span className="text-text-primary font-medium">39,512.84</span>
                            <span className="text-negative">-0.21%</span>
                        </span>
                        <span className="text-border">|</span>
                        <span className="flex items-center gap-2">
                            <span className="text-text-muted">GOLD</span>
                            <span className="text-text-primary font-medium">2,341.20</span>
                            <span className="text-positive">+0.45%</span>
                        </span>
                        <span className="text-border">|</span>
                        <span className="flex items-center gap-2">
                            <span className="text-text-muted">BTC</span>
                            <span className="text-text-primary font-medium">67,824.50</span>
                            <span className="text-positive">+2.31%</span>
                        </span>
                        <span className="text-border">|</span>
                        <span className="flex items-center gap-2">
                            <span className="text-text-muted">IPSA</span>
                            <span className="text-text-primary font-medium">6,842.15</span>
                            <span className="text-positive">+0.67%</span>
                        </span>
                        {/* Repeat for seamless loop */}
                        <span className="text-border">|</span>
                        <span className="flex items-center gap-2">
                            <span className="text-text-muted">S&P 500</span>
                            <span className="text-text-primary font-medium">5,234.18</span>
                            <span className="text-positive">+0.82%</span>
                        </span>
                        <span className="text-border">|</span>
                        <span className="flex items-center gap-2">
                            <span className="text-text-muted">NASDAQ</span>
                            <span className="text-text-primary font-medium">16,742.39</span>
                            <span className="text-positive">+1.14%</span>
                        </span>
                    </div>
                </div>

                {/* Main Header */}
                <div className="container mx-auto px-6 lg:px-10">
                    <div className="flex justify-between items-center h-16">
                        {/* Logo */}
                        <div className="flex items-center gap-3">
                            <CariaLogoIcon 
                                className="w-8 h-8" 
                                style={{ color: 'var(--color-accent-primary)' }}
                            />
                            <div className="flex flex-col">
                                <h1 
                                    className="text-xl font-bold tracking-tight leading-none"
                                    style={{
                                        fontFamily: 'var(--font-display)',
                                        color: 'var(--color-text-primary)',
                                        letterSpacing: '-0.02em'
                                    }}
                                >
                                    CARIA
                                </h1>
                                <span 
                                    className="text-[9px] font-medium tracking-[0.2em] uppercase"
                                    style={{ color: 'var(--color-text-muted)' }}
                                >
                                    Financial Intelligence
                                </span>
                            </div>
                        </div>

                        {/* Navigation */}
                        <nav 
                            className="hidden md:flex items-center gap-8"
                            style={{
                                fontFamily: 'var(--font-body)',
                                fontSize: '14px',
                                fontWeight: 500
                            }}
                        >
                            <button
                                onClick={() => setShowFeaturesModal(true)}
                                className="transition-colors duration-200 hover:text-text-primary"
                                style={{ color: 'var(--color-text-secondary)' }}
                            >
                                Platform
                            </button>
                            <a 
                                href="#research"
                                className="transition-colors duration-200 hover:text-text-primary"
                                style={{ color: 'var(--color-text-secondary)' }}
                            >
                                Research
                            </a>
                            <a 
                                href="#pricing"
                                className="transition-colors duration-200 hover:text-text-primary"
                                style={{ color: 'var(--color-text-secondary)' }}
                            >
                                Pricing
                            </a>
                            <div className="flex items-center gap-3 ml-4 pl-4 border-l" style={{ borderColor: 'var(--color-border-subtle)' }}>
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
                        </nav>

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

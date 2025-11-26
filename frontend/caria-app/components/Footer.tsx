import React from 'react';
import { CariaLogoIcon } from './Icons';

export const Footer: React.FC = () => {
    const currentYear = new Date().getFullYear();

    return (
        <footer 
            style={{
                backgroundColor: 'var(--color-bg-secondary)',
                borderTop: '1px solid var(--color-border-subtle)'
            }}
        >
            {/* Main Footer Content */}
            <div className="container mx-auto px-6 lg:px-10 py-16">
                <div className="grid lg:grid-cols-12 gap-12">
                    {/* Brand Column */}
                    <div className="lg:col-span-4">
                        <div className="flex items-center gap-3 mb-5">
                            <CariaLogoIcon 
                                className="w-8 h-8" 
                                style={{ color: 'var(--color-accent-primary)' }}
                            />
                            <div>
                                <h3 
                                    className="text-xl font-bold tracking-tight"
                                    style={{
                                        fontFamily: 'var(--font-display)',
                                        color: 'var(--color-text-primary)',
                                    }}
                                >
                                    CARIA
                                </h3>
                                <span 
                                    className="text-[9px] font-medium tracking-[0.15em] uppercase"
                                    style={{ color: 'var(--color-text-muted)' }}
                                >
                                    Financial Intelligence
                                </span>
                            </div>
                        </div>
                        <p 
                            className="text-sm leading-relaxed mb-6 max-w-sm"
                            style={{ color: 'var(--color-text-secondary)' }}
                        >
                            Institutional-grade investment research platform combining 
                            quantitative analysis, AI insights, and deep fundamentals.
                        </p>
                        <div className="flex gap-4">
                            {/* Social Links */}
                            {['twitter', 'linkedin', 'github'].map((social) => (
                                <a
                                    key={social}
                                    href={`#${social}`}
                                    className="w-9 h-9 rounded-lg flex items-center justify-center transition-colors"
                                    style={{ 
                                        backgroundColor: 'var(--color-bg-surface)',
                                        color: 'var(--color-text-muted)'
                                    }}
                                    onMouseEnter={(e) => {
                                        e.currentTarget.style.backgroundColor = 'var(--color-accent-primary)';
                                        e.currentTarget.style.color = '#FFFFFF';
                                    }}
                                    onMouseLeave={(e) => {
                                        e.currentTarget.style.backgroundColor = 'var(--color-bg-surface)';
                                        e.currentTarget.style.color = 'var(--color-text-muted)';
                                    }}
                                >
                                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                                        {social === 'twitter' && (
                                            <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                                        )}
                                        {social === 'linkedin' && (
                                            <path d="M20.5 2h-17A1.5 1.5 0 002 3.5v17A1.5 1.5 0 003.5 22h17a1.5 1.5 0 001.5-1.5v-17A1.5 1.5 0 0020.5 2zM8 19H5v-9h3zM6.5 8.25A1.75 1.75 0 118.3 6.5a1.78 1.78 0 01-1.8 1.75zM19 19h-3v-4.74c0-1.42-.6-1.93-1.38-1.93A1.74 1.74 0 0013 14.19a.66.66 0 000 .14V19h-3v-9h2.9v1.3a3.11 3.11 0 012.7-1.4c1.55 0 3.36.86 3.36 3.66z" />
                                        )}
                                        {social === 'github' && (
                                            <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
                                        )}
                                    </svg>
                                </a>
                            ))}
                        </div>
                    </div>

                    {/* Links Columns */}
                    <div className="lg:col-span-8">
                        <div className="grid sm:grid-cols-3 gap-8">
                            {/* Platform */}
                            <div>
                                <h4 
                                    className="text-xs font-semibold tracking-widest uppercase mb-5"
                                    style={{ color: 'var(--color-text-muted)' }}
                                >
                                    Platform
                                </h4>
                                <ul className="space-y-3">
                                    {['Portfolio Analytics', 'Stock Screener', 'Valuation Tools', 'Thesis Arena', 'Market Intelligence'].map((item) => (
                                        <li key={item}>
                                            <a 
                                                href="#"
                                                className="text-sm transition-colors"
                                                style={{ color: 'var(--color-text-secondary)' }}
                                                onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-text-primary)'}
                                                onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-secondary)'}
                                            >
                                                {item}
                                            </a>
                                        </li>
                                    ))}
                                </ul>
                            </div>

                            {/* Resources */}
                            <div>
                                <h4 
                                    className="text-xs font-semibold tracking-widest uppercase mb-5"
                                    style={{ color: 'var(--color-text-muted)' }}
                                >
                                    Resources
                                </h4>
                                <ul className="space-y-3">
                                    {['Documentation', 'API Reference', 'Research Library', 'Community Forum', 'Weekly Insights'].map((item) => (
                                        <li key={item}>
                                            <a 
                                                href="#"
                                                className="text-sm transition-colors"
                                                style={{ color: 'var(--color-text-secondary)' }}
                                                onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-text-primary)'}
                                                onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-secondary)'}
                                            >
                                                {item}
                                            </a>
                                        </li>
                                    ))}
                                </ul>
                            </div>

                            {/* Company */}
                            <div>
                                <h4 
                                    className="text-xs font-semibold tracking-widest uppercase mb-5"
                                    style={{ color: 'var(--color-text-muted)' }}
                                >
                                    Company
                                </h4>
                                <ul className="space-y-3">
                                    {['About', 'Careers', 'Contact', 'Privacy Policy', 'Terms of Service'].map((item) => (
                                        <li key={item}>
                                            <a 
                                                href="#"
                                                className="text-sm transition-colors"
                                                style={{ color: 'var(--color-text-secondary)' }}
                                                onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-text-primary)'}
                                                onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-secondary)'}
                                            >
                                                {item}
                                            </a>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Bottom Bar */}
            <div 
                className="border-t"
                style={{ borderColor: 'var(--color-border-subtle)' }}
            >
                <div className="container mx-auto px-6 lg:px-10 py-5">
                    <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
                        <p 
                            className="text-xs"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            © {currentYear} Caria. All rights reserved.
                        </p>
                        <div className="flex items-center gap-6">
                            <span 
                                className="text-xs"
                                style={{ color: 'var(--color-text-subtle)' }}
                            >
                                Market data provided for informational purposes only. Not financial advice.
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </footer>
    );
};

import React from 'react';

const ChartCard: React.FC<{ children: React.ReactNode; title: string; className?: string }> = ({ children, title, className = '' }) => (
    <div className={`bg-[#0B1221] border border-white/5 rounded-lg p-4 flex flex-col relative overflow-hidden ${className}`}>
        {/* Subtle grid background */}
        <div className="absolute inset-0 opacity-10" 
             style={{ backgroundImage: 'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)', backgroundSize: '20px 20px' }} 
        />
        <h4 className="text-[10px] text-text-muted uppercase tracking-widest mb-2 relative z-10 font-mono">{title}</h4>
        <div className="flex-1 relative z-10 flex items-end">
            {children}
        </div>
    </div>
);

export const ArtisticDashboard: React.FC = () => {
    return (
        <section className="py-32 bg-bg-primary relative overflow-hidden">
            <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                {/* Section Header */}
                <div className="text-center mb-20">
                    <h2 className="text-4xl md:text-6xl font-display text-white mb-6">
                        We help you visualize the future you want
                    </h2>
                    {/* Optional image from prompt (Earth horizon) - simulated with gradient */}
                    <div className="h-px w-32 mx-auto bg-gradient-to-r from-transparent via-accent-gold to-transparent opacity-50" />
                </div>

                {/* Dashboard Visual */}
                <div className="relative mx-auto max-w-6xl p-1 rounded-2xl bg-gradient-to-b from-white/10 to-white/0">
                    <div className="bg-[#050A14] rounded-2xl p-2 md:p-8 shadow-2xl border border-white/5">
                        {/* Header Bar */}
                        <div className="flex justify-between items-center mb-8 px-2">
                            <span className="text-sm text-accent-gold font-display tracking-wide">Fintech Dashboard</span>
                            <div className="flex gap-2">
                                <div className="w-3 h-3 rounded-full border border-white/20" />
                                <div className="w-3 h-3 rounded-full border border-white/20" />
                            </div>
                        </div>

                        {/* Grid Layout */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6 h-64">
                            {/* Chart 1: Momentum (Area) */}
                            <ChartCard title="Momentum">
                                <svg viewBox="0 0 100 50" className="w-full h-full overflow-visible" preserveAspectRatio="none">
                                    <defs>
                                        <linearGradient id="grad1" x1="0%" y1="0%" x2="0%" y2="100%">
                                            <stop offset="0%" stopColor="#D4AF37" stopOpacity="0.2" />
                                            <stop offset="100%" stopColor="#D4AF37" stopOpacity="0" />
                                        </linearGradient>
                                    </defs>
                                    <path d="M0 40 Q 20 35, 40 10 T 80 25 T 100 20 V 50 H 0 Z" fill="url(#grad1)" />
                                    <path d="M0 40 Q 20 35, 40 10 T 80 25 T 100 20" fill="none" stroke="#D4AF37" strokeWidth="0.5" />
                                    {/* Second line */}
                                    <path d="M0 45 Q 25 40, 50 25 T 100 30" fill="none" stroke="#64748B" strokeWidth="0.5" opacity="0.5" />
                                </svg>
                            </ChartCard>

                            {/* Chart 2: Opyn (Multi-wave) */}
                            <ChartCard title="Option">
                                <svg viewBox="0 0 100 50" className="w-full h-full overflow-visible" preserveAspectRatio="none">
                                    <path d="M0 30 C 20 30, 30 10, 50 25 S 80 40, 100 15" fill="none" stroke="#F1F5F9" strokeWidth="0.5" />
                                    <path d="M0 40 C 20 45, 40 20, 60 30 S 90 10, 100 35" fill="none" stroke="#D4AF37" strokeWidth="0.5" />
                                    <path d="M0 25 C 30 10, 50 40, 70 20 S 90 30, 100 25" fill="none" stroke="#64748B" strokeWidth="0.5" opacity="0.5" />
                                </svg>
                            </ChartCard>

                            {/* Chart 3: Earning (Line) */}
                            <ChartCard title="Earning">
                                <svg viewBox="0 0 100 50" className="w-full h-full overflow-visible" preserveAspectRatio="none">
                                    <path d="M0 35 Q 25 45, 50 15 T 100 25" fill="none" stroke="#D4AF37" strokeWidth="0.5" />
                                    <path d="M0 40 Q 25 30, 50 40 T 100 35" fill="none" stroke="#64748B" strokeWidth="0.5" opacity="0.5" />
                                </svg>
                            </ChartCard>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 h-48">
                            {/* Gauge */}
                            <ChartCard title="" className="flex items-center justify-center">
                                <div className="relative w-32 h-32">
                                    <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                                        <circle cx="50" cy="50" r="40" fill="none" stroke="#1E293B" strokeWidth="4" />
                                        <circle cx="50" cy="50" r="40" fill="none" stroke="#D4AF37" strokeWidth="4" strokeDasharray="251.2" strokeDashoffset="180" />
                                        <circle cx="50" cy="50" r="30" fill="none" stroke="#1E293B" strokeWidth="15" opacity="0.5" />
                                        <circle cx="50" cy="50" r="30" fill="none" stroke="#D4AF37" strokeWidth="15" strokeDasharray="188.4" strokeDashoffset="140" opacity="0.2" />
                                    </svg>
                                </div>
                            </ChartCard>

                            {/* Projection */}
                            <ChartCard title="Projection">
                                <svg viewBox="0 0 100 50" className="w-full h-full overflow-visible" preserveAspectRatio="none">
                                    <path d="M0 30 Q 25 20, 50 30 T 100 20" fill="none" stroke="#D4AF37" strokeWidth="0.5" />
                                    <path d="M0 35 Q 25 25, 50 35 T 100 25" fill="none" stroke="#64748B" strokeWidth="0.5" opacity="0.5" />
                                </svg>
                            </ChartCard>

                            {/* Gap */}
                            <ChartCard title="Gap">
                                <svg viewBox="0 0 100 50" className="w-full h-full overflow-visible" preserveAspectRatio="none">
                                    <path d="M0 40 Q 25 35, 50 20 T 100 30" fill="none" stroke="#D4AF37" strokeWidth="0.5" />
                                    <path d="M0 25 Q 25 30, 50 15 T 100 25" fill="none" stroke="#64748B" strokeWidth="0.5" opacity="0.5" />
                                </svg>
                            </ChartCard>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};


import React from 'react';

export const Footer: React.FC = () => {
    const currentYear = new Date().getFullYear();

    return (
        <footer className="relative overflow-hidden">
            {/* Hero Image Section */}
            <div className="relative h-[400px] sm:h-[500px] lg:h-[600px]">
                {/* Background Image - Earth from space */}
                <div
                    className="absolute inset-0 bg-cover bg-center bg-no-repeat"
                    style={{
                        backgroundImage: `url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop')`,
                    }}
                >
                    {/* Gradient overlay */}
                    <div className="absolute inset-0 bg-gradient-to-b from-[#020408] via-transparent to-[#020408]/90" />
                </div>

                {/* Content */}
                <div className="relative z-10 h-full flex flex-col items-center justify-center px-4">
                    <h2
                        className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-serif text-white text-center leading-tight mb-6 max-w-4xl"
                        style={{
                            fontFamily: 'Georgia, "Times New Roman", serif',
                            fontStyle: 'italic',
                            fontWeight: 400,
                            textShadow: '0 2px 20px rgba(0,0,0,0.5)'
                        }}
                    >
                        We help you visualize the future you want
                    </h2>

                    {/* Optional subtitle */}
                    <p className="text-sm sm:text-base text-white/60 text-center max-w-2xl mb-8">
                        Professional-grade investment tools. AI-powered analysis. Your financial future, clarified.
                    </p>

                    {/* CTA Button */}
                    <button
                        onClick={() => window.location.href = '/?login=true'}
                        className="px-8 sm:px-10 py-3 sm:py-4 rounded-lg bg-white/10 backdrop-blur-sm border border-white/20 text-white text-sm sm:text-base font-medium hover:bg-white/20 hover:border-white/30 transition-all duration-300"
                    >
                        Start Your Journey
                    </button>
                </div>
            </div>

            {/* Bottom Bar */}
            <div
                className="py-6 sm:py-8"
                style={{
                    backgroundColor: '#020408',
                    borderTop: '1px solid rgba(255,255,255,0.05)'
                }}
            >
                <div className="container mx-auto px-4 sm:px-6 lg:px-10">
                    <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                        <p className="text-xs sm:text-sm text-white/40">
                            © {currentYear} Caria. All rights reserved.
                        </p>
                        <div className="flex items-center gap-6 text-xs sm:text-sm text-white/40">
                            <a href="#" className="hover:text-white/70 transition-colors">Privacy</a>
                            <a href="#" className="hover:text-white/70 transition-colors">Terms</a>
                            <a href="#" className="hover:text-white/70 transition-colors">Contact</a>
                        </div>
                    </div>
                </div>
            </div>
        </footer>
    );
};

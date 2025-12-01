import React from 'react';
import { Header } from './Header';
import { Hero } from './Hero';
import { Features } from './Features';
import { Footer } from './Footer';

interface LandingPageProps {
    onLogin?: () => void;
    onRegister?: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onLogin, onRegister }) => {
    return (
        <>
            <Header onLogin={onLogin} onRegister={onRegister} />
            <main className="flex-1 bg-bg-primary">
                <Hero onLogin={onLogin} />
                
                {/* Our Philosophy Section */}
                <div className="py-24 px-6 sm:px-8 lg:px-16 bg-bg-primary border-b border-white/5">
                    <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-24 items-start">
                        {/* Left / Top: Title */}
                        <div className="lg:sticky lg:top-24">
                            <h2 className="text-4xl sm:text-5xl lg:text-6xl font-display text-white leading-tight">
                                Our<br className="hidden lg:block"/>Philosophy
                            </h2>
                            <div className="mt-6 h-1 w-24 bg-accent-primary rounded-full opacity-50"></div>
                        </div>

                        {/* Right / Bottom: Text */}
                        <div className="prose prose-invert prose-lg text-text-secondary font-serif leading-relaxed opacity-90">
                            <p className="mb-6 text-base sm:text-lg">
                                Caria is not here to tell you what to buy — it is here to help you think. 
                                We challenge ideas, test assumptions, weigh risk against reward, and look uncertainty in the eyes without demanding that it disappear.
                            </p>
                            <p className="text-base sm:text-lg">
                                Retail by birth and proud of it, we stand inside the market with a lantern, not a crystal ball — turning history into guidance, data into direction, and doubt into discipline. 
                                No prophecies, no promises, only tools you can use to see clearer, judge steadier, and invest with a mind anchored in reason rather than noise. 
                                If you walk with us, you won’t find guarantees — only the quiet confidence that comes from understanding what is at stake, and the courage to act anyway.
                            </p>
                        </div>
                    </div>
                </div>

                <Features />
            </main>
            <Footer />
        </>
    );
};

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
                <div className="py-20 px-4 sm:px-6 lg:px-8 bg-bg-primary border-b border-white/5">
                    <div className="max-w-4xl mx-auto text-center">
                        <h2 className="text-2xl sm:text-3xl font-display text-white mb-8">The Caria Philosophy</h2>
                        <div className="prose prose-invert prose-lg mx-auto">
                            <p className="text-text-secondary leading-relaxed font-serif italic text-xl sm:text-2xl">
                                "Caria is not here to tell you what to buy — it is here to help you think. 
                                We challenge ideas, test assumptions, weigh risk against reward, and look uncertainty in the eyes without demanding that it disappear."
                            </p>
                            <p className="text-text-muted mt-6 text-base leading-relaxed max-w-2xl mx-auto">
                                Retail by birth and proud of it, we stand inside the market with a lantern, not a crystal ball — turning history into guidance, data into direction, and doubt into discipline. 
                                No prophecies, no promises, only tools you can use to see clearer, judge steadier, and invest with a mind anchored in reason rather than noise. 
                                If you walk with us, you won’t find guarantees — only the quiet confidence that comes from understanding what is at stake, and the courage to act anyway.
                            </p>
                        </div>
                        <div className="mt-10 flex justify-center">
                            <div className="h-1 w-20 bg-accent-primary rounded-full opacity-50"></div>
                        </div>
                    </div>
                </div>

                <Features />
            </main>
            <Footer />
        </>
    );
};

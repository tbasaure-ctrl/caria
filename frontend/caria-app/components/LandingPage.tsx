import React from 'react';
import { Header } from './Header';
import { Hero } from './Hero';
import { Features } from './Features';
import { ArtisticDashboard } from './ArtisticDashboard';
import { Footer } from './Footer';

interface LandingPageProps {
    onLogin?: () => void;
    onRegister?: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onLogin, onRegister }) => {
    return (
        <>
            <Header onLogin={onLogin} onRegister={onRegister} />
            <main className="flex-1">
                <Hero onLogin={onLogin} />
                <Features />
                <ArtisticDashboard />
            </main>
            <Footer />
        </>
    );
};

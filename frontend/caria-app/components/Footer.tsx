import React from 'react';

export const Footer: React.FC = () => {
    return (
        <footer className="border-t" style={{ borderColor: 'var(--color-primary-dark)', backgroundColor: 'var(--color-bg-secondary)' }}>
            <div className="container mx-auto px-6 py-12">
                <div className="grid md:grid-cols-4 gap-8">
                    <div>
                        <h3 className="font-bold mb-4" style={{ fontFamily: 'var(--font-display)', color: 'var(--color-cream)' }}>
                            Caria
                        </h3>
                        <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                            A considered approach to investing.
                        </p>
                    </div>
                    <div>
                        <h4 className="font-semibold mb-3" style={{ color: 'var(--color-cream)' }}>Product</h4>
                        <ul className="space-y-2 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                            <li><a href="#" className="hover:text-white">Features</a></li>
                            <li><a href="#" className="hover:text-white">Pricing</a></li>
                        </ul>
                    </div>
                    <div>
                        <h4 className="font-semibold mb-3" style={{ color: 'var(--color-cream)' }}>Company</h4>
                        <ul className="space-y-2 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                            <li><a href="#" className="hover:text-white">About</a></li>
                            <li><a href="#" className="hover:text-white">Contact</a></li>
                        </ul>
                    </div>
                    <div>
                        <h4 className="font-semibold mb-3" style={{ color: 'var(--color-cream)' }}>Legal</h4>
                        <ul className="space-y-2 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                            <li><a href="#" className="hover:text-white">Privacy</a></li>
                            <li><a href="#" className="hover:text-white">Terms</a></li>
                        </ul>
                    </div>
                </div>
                <div className="mt-8 pt-8 border-t text-center text-sm" style={{ borderColor: 'var(--color-primary-dark)', color: 'var(--color-text-muted)' }}>
                    © {new Date().getFullYear()} Caria. All rights reserved.
                </div>
            </div>
        </footer>
    );
};

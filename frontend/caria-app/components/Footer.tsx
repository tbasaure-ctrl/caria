import React from 'react';

export const Footer: React.FC = () => {
    const currentYear = new Date().getFullYear();

    return (
        <footer 
            className="py-8 md:py-12"
            style={{
                backgroundColor: 'var(--color-bg-primary)',
                borderTop: '1px solid var(--color-border-subtle)'
            }}
        >
            <div className="container mx-auto px-4 sm:px-6 lg:px-10">
                <div className="text-center">
                    <p 
                        className="text-sm"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            © {currentYear} Caria. All rights reserved.
                        </p>
                </div>
            </div>
        </footer>
    );
};

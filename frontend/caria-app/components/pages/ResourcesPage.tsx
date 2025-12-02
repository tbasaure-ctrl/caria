import React from 'react';
import { Resources } from '../widgets/Resources';

export const ResourcesPage: React.FC = () => {
    return (
        <div className="animate-fade-in space-y-12 pb-20 max-w-4xl mx-auto">
            {/* Manifesto Section */}
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl sm:text-5xl font-display text-white mb-2">CARIA</h1>
                    <p className="text-lg text-accent-cyan font-medium tracking-wide">Cognitive Analysis and Risk Investment Assistant</p>
                </div>

                <div className="space-y-8">
                    <h2 className="text-3xl sm:text-4xl font-display text-white">About Caria</h2>

                    <div className="prose prose-invert prose-sm sm:prose-base max-w-none text-text-secondary space-y-6 leading-relaxed font-serif text-lg">
                        <p>
                            Turning history into guidance, data into direction, and doubt into discipline. No prophecies, no promises, only tools you can use to see clearer, judge steadier, and invest with a mind anchored in reason rather than noise. If you walk with us, you won't find guarantees — only the quiet confidence that comes from understanding what is at stake, and the courage to act anyway.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

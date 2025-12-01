import React from 'react';
import { Resources } from '../widgets/Resources';

export const ResourcesPage: React.FC = () => {
    return (
        <div className="animate-fade-in space-y-12 pb-20 max-w-4xl mx-auto">
            {/* Manifesto Section */}
            <div className="space-y-8">
                <h1 className="text-4xl sm:text-5xl font-display text-white mb-8">Our Philosophy</h1>
                
                <div className="prose prose-invert prose-sm sm:prose-base max-w-none text-text-secondary space-y-6 leading-relaxed font-serif text-lg">
                    <p>
                        First—and with a cute little punch to the table—nothing you encounter here is financial advice.
                        <br/>
                        No whispered tips, no promises of fortunes, no secret prophecy encoded in candlesticks.
                    </p>

                    <p>
                        Caria does not tell you what to buy; it listens to your idea and usually tries to talk you out of it—unless it is so unexpectedly brilliant that we may end up buying next to you. The screeners we offer act as a tool to expand you universe selection, nothing else.
                    </p>

                    <p>
                        Our main goal is, in the best of our imperfect capacity, to help you think clearer.
                        <br/>
                        Clear thinking is an overused concept, so here is our meaning of it: less bias, more awareness of risk, sharper recognition of uncertainty, and a deep respect for the limits of what we can know about markets, economies, and the world.
                    </p>

                    <p>
                        We build tools not to hide from the unknown, but to become sharper within it.
                        <br/>
                        We are proud retail investors—amateurs, “dumb money” by Wall Street taxonomy—kin to the majority who work, save, hope, err, learn, and try again.
                    </p>

                    <p>
                        We do not stand outside the system peering in with envy; we stand inside it with a lantern, mapping the cave one careful decision at a time.
                        <br/>
                        Improvement is our one sure and constant quality. We iterate, revise, rewrite, second-guess, test, second-guess ourself for a while, refactor, sharpen, and rebuild—like students who refuse graduation because curiosity tastes better than certainty.
                    </p>

                    <div className="border-l-2 border-accent-gold pl-4 italic text-text-muted my-8">
                        "Write to us. Challenge us. Throw stones into our pond: ripples are always welcome."
                    </div>

                    <p>
                        Now, about uncertainty—we do embrace it.
                        <br/>
                        The market is less a machine and more a living myth: unpredictable, wild, sometimes cruel, sometimes generous to those who meet it with humility instead of bravado.
                        <br/>
                        Here we take inspiration from antifragility: the art of growing stronger under stress, learning from volatility, benefitting from disorder instead of fearing it.
                    </p>

                    <p>
                        Caria—Cognitive Analysis and Risk Investment Assistant, if you were wondering—is not a new alpha Messiah. We are not here claiming discovery of a financial North Star, and truthfully we prefer proper sleep hygiene to frantic, caffeinated, micro-second trading algorithms dancing like neurons in distress. We are archivists, librarians of wisdom others paid tuition for—sometimes painfully. We weave together lessons from Buffett, Munger, Lynch, Taleb, Marks, Graham, and from countless anonymous investors who left fragments of triumph and ruin in their wake, so that we may lose less and think more.
                    </p>

                    <p>
                        Annoyingly enough, this makes us patient. Long-term oriented. Sickeningly rational on most days, stubbornly curious on the rest.
                        <br/>
                        Our contribution is synthesis, clarity, and translation—turning the messy brilliance of history into a compass for the future.
                    </p>

                    <hr className="border-white/10 my-12" />

                    <h3 className="text-2xl font-display text-white mb-4">So this is Caria:</h3>
                    <ul className="list-none space-y-2 pl-0">
                        <li>A place built by retail minds for retail minds.</li>
                        <li>A workshop where we turn data into direction, uncertainty into structure, risk into reason.</li>
                        <li>Not prophecy, but practice.</li>
                        <li>Not advice, but companionship.</li>
                        <li>Not something new—only newly usable: simulations, scenario analysis, risk-reward engines, valuation frameworks, and the gentle (but firm) voice asking “Have you considered the downside?”</li>
                    </ul>

                    <p className="mt-8">
                        If you find something here that sharpens you—broadens your sight, tempers your impulse, or nudges you toward wiser decisions—then we are satisfied. Walk with us. Build with us. Discover with us what is possible when reason replaces noise.
                    </p>

                    <p className="font-bold text-white mt-8">
                        Because the future favors not those who demand certainty, but those who walk forward despite uncertainty—who understand that to fear suffering is already to suffer it, and that true valor lies not in strength of hand, but in strength of spirit, and that this is less of a quality than a decision in everyone life´s.
                    </p>
                </div>
            </div>

            {/* Resources Widget (Legacy) */}
            <div className="pt-12 border-t border-white/5">
                <h2 className="text-2xl font-display text-white mb-6">Library</h2>
                <Resources />
            </div>
        </div>
    );
};

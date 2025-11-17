import React from 'react';

interface HeroProps {
    onLogin: () => void;
}

export const Hero: React.FC<HeroProps> = ({ onLogin }) => {
  return (
    <section className="relative py-32 md:py-48 overflow-hidden">
      {/* Investing Legends Background */}
      <div className="legends-background">
        {/* Warren Buffett */}
      <div className="legend-figure legend-buffett"
        style={{backgroundImage: "url('/images/legends/warren-buffett.jpg')"}}></div>

        {/* Charlie Munger */}
        <div className="legend-figure legend-munger"
          style={{backgroundImage: "url('/images/legends/charlie-munger.jpg')"}}></div>

        {/* Stan Druckenmiller */}
        <div className="legend-figure legend-druckenmiller"
          style={{backgroundImage: "url('/images/legends/stan-druckenmiller.jpg')"}}></div>

        {/* Ben Graham */}
        <div className="legend-figure legend-graham"
          style={{backgroundImage: "url('/images/legends/ben-graham.jpg')"}}></div>

        {/* Peter Lynch */}
        <div className="legend-figure legend-lynch"
          style={{backgroundImage: "url('/images/legends/peter-lynch.jpg')"}}></div>

        {/* John Maynard Keynes */}
        <div className="legend-figure legend-keynes"
          style={{backgroundImage: "url('/images/legends/john-keynes.jpg')"}}></div>

        {/* David Tepper */}
        <div className="legend-figure legend-tepper"
          style={{backgroundImage: "url('/images/legends/david-tepper.jpg')"}}></div>

        {/* Terry Smith */}
        <div className="legend-figure legend-smith"
          style={{backgroundImage: "url('/images/legends/terry-smith.jpg')"}}></div>
      </div>

      {/* Decorative background elements with blue accent */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-20 left-10 w-96 h-96 rounded-full"
             style={{background: 'radial-gradient(circle, var(--color-blue) 0%, transparent 70%)'}}></div>
        <div className="absolute bottom-20 right-10 w-80 h-80 rounded-full"
             style={{background: 'radial-gradient(circle, var(--color-primary) 0%, transparent 70%)'}}></div>
        <div className="absolute top-1/2 left-1/2 w-[600px] h-[600px] rounded-full"
             style={{background: 'radial-gradient(circle, var(--color-blue-dark) 0%, transparent 60%)', transform: 'translate(-50%, -50%)'}}></div>
      </div>

      <div className="container mx-auto px-6 relative z-10">
        <div className="max-w-5xl mx-auto">
          {/* Small eyebrow text */}
          <div className="text-center mb-6 fade-in">
            <span className="inline-block px-4 py-2 rounded-full text-sm font-medium tracking-wide"
                  style={{
                    backgroundColor: 'var(--color-bg-tertiary)',
                    color: 'var(--color-blue-light)',
                    border: '1px solid var(--color-blue-dark)',
                    fontFamily: 'var(--font-mono)'
                  }}>
              COGNITIVE ANALYSIS & RISK INVESTMENT ASSISTANT
            </span>
          </div>

          {/* Main headline with editorial styling */}
          <h1 className="text-6xl md:text-8xl font-bold leading-[0.95] mb-8 fade-in delay-200"
              style={{
                fontFamily: 'var(--font-display)',
                color: 'var(--color-cream)',
                textAlign: 'center'
              }}>
            <span className="block">Enduring</span>
            <span className="block text-gradient">Principles,</span>
            <span className="block">Modern Insight</span>
          </h1>

          {/* Decorative divider */}
          <div className="flex items-center justify-center gap-4 mb-10 fade-in delay-300">
            <div className="h-px w-16" style={{backgroundColor: 'var(--color-blue)'}}></div>
            <div className="w-2 h-2 rounded-full" style={{backgroundColor: 'var(--color-blue-light)'}}></div>
            <div className="h-px w-16" style={{backgroundColor: 'var(--color-blue)'}}></div>
          </div>

          {/* Subtitle with refined typography */}
          <p className="text-xl md:text-2xl leading-relaxed max-w-3xl mx-auto text-center mb-12 fade-in delay-400"
             style={{
               fontFamily: 'var(--font-body)',
               color: 'var(--color-text-secondary)',
               fontWeight: 400
             }}>
            We merge timeless investment wisdom with the analytical power of deep learning,
            creating an invaluable partner for your financial journey.
          </p>

          {/* CTA with sophisticated styling */}
          <div className="flex justify-center gap-4 fade-in delay-500">
            <button
              onClick={onLogin}
              className="group relative px-10 py-4 rounded-lg font-semibold text-lg transition-all duration-300 overflow-hidden"
              style={{
                backgroundColor: 'var(--color-blue)',
                color: 'var(--color-cream)',
                fontFamily: 'var(--font-body)',
                boxShadow: '0 10px 30px -10px var(--color-blue)',
                border: '1px solid var(--color-blue-light)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 15px 40px -10px var(--color-blue-light)';
                e.currentTarget.style.backgroundColor = 'var(--color-blue-light)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 10px 30px -10px var(--color-blue)';
                e.currentTarget.style.backgroundColor = 'var(--color-blue)';
              }}
            >
              <span className="relative z-10">Begin Your Journey</span>
              <div className="absolute inset-0 opacity-0 group-hover:opacity-20 transition-opacity duration-300"
                   style={{background: 'linear-gradient(135deg, var(--color-blue-light), var(--color-blue-dark))'}}></div>
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};








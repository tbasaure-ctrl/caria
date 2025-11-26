import React, { useState } from 'react';
import { WidgetCard } from './WidgetCard';

interface CategoryData {
    text: string;
    pe: number;
}

const CATEGORY_DATA: Record<string, CategoryData> = {
    slow: { 
        text: "Slow Grower: Usually large, aging companies (e.g., Electric Utility). Expected to grow slightly faster than the economy. Generous dividends are the main attraction here. Typical P/E: 8-12x", 
        pe: 10 
    },
    stalwart: { 
        text: "Stalwart: Large, established companies. Good protection during recessions. Expect 10-12% earnings growth. Don't overpay. Typical P/E: 12-15x", 
        pe: 15 
    },
    fast: { 
        text: "Fast Grower: Aggressive new enterprises. Growing 20-25% a year. This is where the huge returns are found, but they are riskier. Typical P/E: 20-30x", 
        pe: 25 
    },
    cyclical: { 
        text: "Cyclical: Earnings rise and fall with the economy (e.g., Autos, Airlines). Warning: A low P/E here often means the good times are ending! Typical P/E: Varies Wildly", 
        pe: 8 
    },
};

const formatMoney = (n: number): string => {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(n);
};

const formatNum = (n: number): string => {
    return new Intl.NumberFormat('en-US').format(n);
};

type Mode = 'landing' | 'beginner' | 'advanced';

export const ValuationWorkshop: React.FC = () => {
    const [mode, setMode] = useState<Mode>('landing');

    return (
        <WidgetCard
            title="Business Valuation Workshop"
            tooltip="Learn the art and science of valuing a company. Choose your path based on your experience level."
        >
            {mode === 'landing' && <LandingScreen onSelectMode={setMode} />}
            {mode === 'beginner' && <BeginnerWorkspace onBack={() => setMode('landing')} />}
            {mode === 'advanced' && <AdvancedWorkspace onBack={() => setMode('landing')} />}
        </WidgetCard>
    );
};

const LandingScreen: React.FC<{ onSelectMode: (mode: Mode) => void }> = ({ onSelectMode }) => {
    return (
        <div className="space-y-6">
            <div className="text-center mb-8">
                <h2 
                    className="text-2xl font-semibold mb-2"
                    style={{ 
                        fontFamily: 'var(--font-display)',
                        color: 'var(--color-text-primary)' 
                    }}
                >
                    Choose Your Path
                </h2>
                <p style={{ color: 'var(--color-text-secondary)' }}>
                    Learn the art and science of valuing a company.
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
                {/* Start from Scratch Card */}
                <div
                    onClick={() => onSelectMode('beginner')}
                    className="p-6 rounded-lg cursor-pointer transition-all duration-300"
                    style={{
                        backgroundColor: 'var(--color-bg-tertiary)',
                        border: '1px solid var(--color-border-subtle)',
                    }}
                    onMouseEnter={(e) => {
                        e.currentTarget.style.borderColor = 'var(--color-accent-primary)';
                        e.currentTarget.style.transform = 'translateY(-4px)';
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.borderColor = 'var(--color-border-subtle)';
                        e.currentTarget.style.transform = 'translateY(0)';
                    }}
                >
                    <div className="text-center">
                        <div className="text-5xl mb-4">🌱</div>
                        <h3 
                            className="text-xl font-semibold mb-3"
                            style={{ color: 'var(--color-accent-primary)' }}
                        >
                            Start From Scratch
                        </h3>
                        <p style={{ color: 'var(--color-text-secondary)' }}>
                            I am new to this. I don't know what P/E means or where to find Cash Flow. Walk me through it simply.
                        </p>
                    </div>
                </div>

                {/* Advanced Valuation Card */}
                <div
                    onClick={() => onSelectMode('advanced')}
                    className="p-6 rounded-lg cursor-pointer transition-all duration-300"
                    style={{
                        backgroundColor: 'var(--color-bg-tertiary)',
                        border: '1px solid var(--color-border-subtle)',
                    }}
                    onMouseEnter={(e) => {
                        e.currentTarget.style.borderColor = 'var(--color-accent-primary)';
                        e.currentTarget.style.transform = 'translateY(-4px)';
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.borderColor = 'var(--color-border-subtle)';
                        e.currentTarget.style.transform = 'translateY(0)';
                    }}
                >
                    <div className="text-center">
                        <div className="text-5xl mb-4">🚀</div>
                        <h3 
                            className="text-xl font-semibold mb-3"
                            style={{ color: 'var(--color-accent-primary)' }}
                        >
                            Advanced Valuation
                        </h3>
                        <p style={{ color: 'var(--color-text-secondary)' }}>
                            I know the basics. Give me the strategic tools to evaluate growth rates, dividends, and fair value.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

const BeginnerWorkspace: React.FC<{ onBack: () => void }> = ({ onBack }) => {
    const [netIncome, setNetIncome] = useState(2500000);
    const [shares, setShares] = useState(1000000);
    const [peRatio, setPeRatio] = useState(15);
    const [cashFlow, setCashFlow] = useState(2500000);

    const eps = netIncome / shares;
    const price = eps * peRatio;
    const qualityRatio = cashFlow / netIncome;

    let qualityStatus = '';
    let qualityColor = '';
    let qualityMessage = '';

    if (qualityRatio >= 1.0) {
        qualityStatus = 'High Quality';
        qualityColor = 'var(--color-positive)';
        qualityMessage = 'Excellent. The business is generating real cash equal to or greater than its reported profit. The valuation is credible.';
    } else if (qualityRatio >= 0.8) {
        qualityStatus = 'Caution';
        qualityColor = 'var(--color-warning)';
        qualityMessage = 'Cash flow is slightly lower than profit. This might be normal (buying inventory), but check the financial notes.';
    } else {
        qualityStatus = 'Red Flag';
        qualityColor = 'var(--color-negative)';
        qualityMessage = 'Warning! Profit is high, but cash is low. Management might be manipulating numbers or customers aren\'t paying bills. Be very careful.';
    }

    let priceAnalysis = '';
    if (peRatio < 12) {
        priceAnalysis = 'This is a conservative valuation. The market thinks the company is growing slowly or faces risks.';
    } else if (peRatio > 25) {
        priceAnalysis = 'This is a "Growth" valuation. Investors are betting the company will make much more money in the future.';
    } else {
        priceAnalysis = 'At a 15x multiple, the market is saying: "We expect this company to grow at an average pace."';
    }

    return (
        <div className="space-y-8">
            <button
                onClick={onBack}
                className="text-sm font-medium mb-4"
                style={{ color: 'var(--color-text-muted)' }}
            >
                ← Back to Menu
            </button>

            {/* Step 1: The Profit Pie */}
            <div className="space-y-4 pb-6 border-b" style={{ borderColor: 'var(--color-border-subtle)' }}>
                <h2 
                    className="text-xl font-semibold"
                    style={{ color: 'var(--color-text-primary)' }}
                >
                    Step 1: The Profit Pie
                </h2>
                <p style={{ color: 'var(--color-text-secondary)' }}>
                    Before we value the whole business, we need to know how much profit belongs to a single share of stock.
                </p>

                <div className="space-y-4">
                    <div>
                        <label className="flex items-center gap-2 mb-2">
                            <span style={{ color: 'var(--color-text-secondary)', fontWeight: 700 }}>
                                Total Net Income
                            </span>
                            <HelpIcon tooltip="Net Income is the 'bottom line' profit found on the Income Statement. It's what's left after paying all expenses and taxes." />
                            <span className="ml-auto font-mono font-bold" style={{ color: 'var(--color-accent-primary)' }}>
                                {formatMoney(netIncome)}
                            </span>
                        </label>
                        <input
                            type="range"
                            min="100000"
                            max="10000000"
                            step="100000"
                            value={netIncome}
                            onChange={(e) => setNetIncome(parseInt(e.target.value))}
                            className="w-full"
                            style={{ accentColor: 'var(--color-accent-primary)' }}
                        />
                    </div>

                    <div>
                        <label className="flex items-center gap-2 mb-2">
                            <span style={{ color: 'var(--color-text-secondary)', fontWeight: 700 }}>
                                Total Shares Outstanding
                            </span>
                            <HelpIcon tooltip="The total number of 'slices' the company is cut into. You can find this on the front of the Balance Sheet or Income Statement." />
                            <span className="ml-auto font-mono font-bold" style={{ color: 'var(--color-accent-primary)' }}>
                                {formatNum(shares)}
                            </span>
                        </label>
                        <input
                            type="range"
                            min="100000"
                            max="5000000"
                            step="50000"
                            value={shares}
                            onChange={(e) => setShares(parseInt(e.target.value))}
                            className="w-full"
                            style={{ accentColor: 'var(--color-accent-primary)' }}
                        />
                    </div>
                </div>

                <div 
                    className="p-5 rounded-lg border-l-4"
                    style={{
                        backgroundColor: 'rgba(46, 124, 246, 0.1)',
                        borderLeftColor: 'var(--color-accent-primary)',
                    }}
                >
                    <h3 className="text-sm font-semibold mb-2 uppercase tracking-wide" style={{ color: 'var(--color-text-primary)' }}>
                        Earnings Per Share (EPS)
                    </h3>
                    <div className="text-3xl font-bold font-mono mb-2" style={{ color: 'var(--color-accent-primary)' }}>
                        {formatMoney(eps)}
                    </div>
                    <p style={{ color: 'var(--color-text-secondary)' }}>
                        This means for every share you buy, the company earns this much profit.
                    </p>
                </div>
            </div>

            {/* Step 2: The Price Tag */}
            <div className="space-y-4 pb-6 border-b" style={{ borderColor: 'var(--color-border-subtle)' }}>
                <h2 
                    className="text-xl font-semibold"
                    style={{ color: 'var(--color-text-primary)' }}
                >
                    Step 2: The Price Tag
                </h2>
                <p style={{ color: 'var(--color-text-secondary)' }}>
                    The stock price isn't random. It's the EPS multiplied by a "Sentiment Factor."
                </p>

                <div>
                    <label className="flex items-center gap-2 mb-2">
                        <span style={{ color: 'var(--color-text-secondary)', fontWeight: 700 }}>
                            The Multiplier (P/E Ratio)
                        </span>
                        <HelpIcon tooltip="P/E (Price to Earnings) tells you how many years it would take to earn back your investment if profits stayed the same. 10x = Cheap/Slow Growth, 15x = Average, 25x+ = Expensive/High Growth" />
                        <span className="ml-auto font-mono font-bold" style={{ color: 'var(--color-accent-primary)' }}>
                            {peRatio}x
                        </span>
                    </label>
                    <input
                        type="range"
                        min="5"
                        max="50"
                        step="1"
                        value={peRatio}
                        onChange={(e) => setPeRatio(parseInt(e.target.value))}
                        className="w-full"
                        style={{ accentColor: 'var(--color-accent-primary)' }}
                    />
                </div>

                <div 
                    className="p-5 rounded-lg border-l-4"
                    style={{
                        backgroundColor: 'rgba(46, 124, 246, 0.1)',
                        borderLeftColor: 'var(--color-accent-primary)',
                    }}
                >
                    <h3 className="text-sm font-semibold mb-2 uppercase tracking-wide" style={{ color: 'var(--color-text-primary)' }}>
                        Estimated Stock Price
                    </h3>
                    <div className="text-3xl font-bold font-mono mb-2" style={{ color: 'var(--color-accent-primary)' }}>
                        {formatMoney(price)}
                    </div>
                    <div 
                        className="p-3 rounded mt-3 text-sm italic"
                        style={{
                            backgroundColor: 'var(--color-bg-tertiary)',
                            color: 'var(--color-text-secondary)',
                        }}
                    >
                        {priceAnalysis}
                    </div>
                </div>
            </div>

            {/* Step 3: Is the Money Real? */}
            <div className="space-y-4">
                <h2 
                    className="text-xl font-semibold"
                    style={{ color: 'var(--color-text-primary)' }}
                >
                    Step 3: Is the Money Real?
                </h2>
                <p style={{ color: 'var(--color-text-secondary)' }}>
                    Accounting profit (Net Income) is an opinion. Cash is a fact. We must verify if the business actually collects the cash it claims to earn.
                </p>

                <div>
                    <label className="flex items-center gap-2 mb-2">
                        <span style={{ color: 'var(--color-text-secondary)', fontWeight: 700 }}>
                            Cash From Operations
                        </span>
                        <HelpIcon tooltip="Found on the Statement of Cash Flows. It shows the actual cash deposited in the bank from selling goods or services." />
                        <span className="ml-auto font-mono font-bold" style={{ color: 'var(--color-accent-primary)' }}>
                            {formatMoney(cashFlow)}
                        </span>
                    </label>
                    <input
                        type="range"
                        min="0"
                        max="5000000"
                        step="100000"
                        value={cashFlow}
                        onChange={(e) => setCashFlow(parseInt(e.target.value))}
                        className="w-full"
                        style={{ accentColor: 'var(--color-accent-primary)' }}
                    />
                </div>

                <div 
                    className="p-5 rounded-lg border-l-4"
                    style={{
                        backgroundColor: 'var(--color-bg-tertiary)',
                        borderLeftColor: qualityColor,
                    }}
                >
                    <h3 className="text-sm font-semibold mb-2 uppercase tracking-wide" style={{ color: 'var(--color-text-primary)' }}>
                        Quality Check
                    </h3>
                    <div 
                        className="inline-block px-3 py-1 rounded-full text-sm font-bold mb-3"
                        style={{ backgroundColor: qualityColor, color: '#FFFFFF' }}
                    >
                        {qualityStatus}
                    </div>
                    <p style={{ color: 'var(--color-text-secondary)' }}>
                        {qualityMessage}
                    </p>
                </div>
            </div>
        </div>
    );
};

const AdvancedWorkspace: React.FC<{ onBack: () => void }> = ({ onBack }) => {
    const [category, setCategory] = useState<string>('stalwart');
    const [peRatio, setPeRatio] = useState(15);
    const [growthRate, setGrowthRate] = useState(12);
    const [dividendYield, setDividendYield] = useState(3);

    const updateCategory = (newCategory: string) => {
        setCategory(newCategory);
        setPeRatio(CATEGORY_DATA[newCategory].pe);
    };

    const score = (growthRate + dividendYield) / peRatio;

    let verdict = '';
    let verdictColor = '';
    let summary = '';

    if (score < 1.0) {
        verdict = 'Poor / Overpriced';
        verdictColor = 'var(--color-negative)';
        summary = 'You are paying a high price for very little growth.';
    } else if (score < 1.5) {
        verdict = 'Fair Value';
        verdictColor = 'var(--color-warning)';
        summary = 'The price is reasonable for the growth you are getting.';
    } else if (score < 2.0) {
        verdict = 'Good Buy';
        verdictColor = 'var(--color-positive)';
        summary = 'This stock is undervalued relative to its growth potential.';
    } else {
        verdict = 'Strong Bargain';
        verdictColor = 'var(--color-positive)';
        summary = 'Rare opportunity! High growth and dividends for a low price.';
    }

    return (
        <div className="space-y-8">
            <button
                onClick={onBack}
                className="text-sm font-medium mb-4"
                style={{ color: 'var(--color-text-muted)' }}
            >
                ← Back to Menu
            </button>

            {/* Company Profile */}
            <div className="space-y-4 pb-6 border-b" style={{ borderColor: 'var(--color-border-subtle)' }}>
                <h2 
                    className="text-xl font-semibold"
                    style={{ color: 'var(--color-text-primary)' }}
                >
                    1. Company Profile
                </h2>
                <p style={{ color: 'var(--color-text-secondary)' }}>
                    Valuation depends on the company's lifecycle stage. A fast grower deserves a higher multiple than a slow grower.
                </p>

                <div>
                    <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                        Category Strategy
                    </label>
                    <select
                        value={category}
                        onChange={(e) => updateCategory(e.target.value)}
                        className="w-full px-4 py-2 rounded-lg"
                        style={{
                            backgroundColor: 'var(--color-bg-tertiary)',
                            border: '1px solid var(--color-border-subtle)',
                            color: 'var(--color-text-primary)',
                        }}
                    >
                        <option value="slow">Slow Grower (e.g., Electric Utility)</option>
                        <option value="stalwart">Stalwart (e.g., Coca-Cola, P&G)</option>
                        <option value="fast">Fast Grower (Aggressive Expansion)</option>
                        <option value="cyclical">Cyclical (e.g., Auto Manufacturer)</option>
                    </select>
                </div>

                <div 
                    className="p-4 rounded-lg text-sm"
                    style={{
                        backgroundColor: 'var(--color-bg-tertiary)',
                        color: 'var(--color-text-secondary)',
                    }}
                >
                    <strong>{CATEGORY_DATA[category].text.split(':')[0]}:</strong>{' '}
                    {CATEGORY_DATA[category].text.split(':').slice(1).join(':')}
                </div>
            </div>

            {/* Financial Inputs */}
            <div className="space-y-4 pb-6 border-b" style={{ borderColor: 'var(--color-border-subtle)' }}>
                <h2 
                    className="text-xl font-semibold"
                    style={{ color: 'var(--color-text-primary)' }}
                >
                    2. Financial Inputs
                </h2>

                <div>
                    <label className="flex items-center justify-between mb-2">
                        <span style={{ color: 'var(--color-text-secondary)', fontWeight: 700 }}>
                            P/E Ratio
                        </span>
                        <span className="font-mono font-bold" style={{ color: 'var(--color-accent-primary)' }}>
                            {peRatio}x
                        </span>
                    </label>
                    <input
                        type="range"
                        min="5"
                        max="60"
                        step="1"
                        value={peRatio}
                        onChange={(e) => setPeRatio(parseInt(e.target.value))}
                        className="w-full"
                        style={{ accentColor: 'var(--color-accent-primary)' }}
                    />
                </div>

                <div>
                    <label className="flex items-center justify-between mb-2">
                        <span style={{ color: 'var(--color-text-secondary)', fontWeight: 700 }}>
                            Long-Term Growth Rate (%)
                        </span>
                        <span className="font-mono font-bold" style={{ color: 'var(--color-accent-primary)' }}>
                            {growthRate}%
                        </span>
                    </label>
                    <input
                        type="range"
                        min="0"
                        max="40"
                        step="1"
                        value={growthRate}
                        onChange={(e) => setGrowthRate(parseInt(e.target.value))}
                        className="w-full"
                        style={{ accentColor: 'var(--color-accent-primary)' }}
                    />
                </div>

                <div>
                    <label className="flex items-center justify-between mb-2">
                        <span style={{ color: 'var(--color-text-secondary)', fontWeight: 700 }}>
                            Dividend Yield (%)
                        </span>
                        <span className="font-mono font-bold" style={{ color: 'var(--color-accent-primary)' }}>
                            {dividendYield}%
                        </span>
                    </label>
                    <input
                        type="range"
                        min="0"
                        max="10"
                        step="0.5"
                        value={dividendYield}
                        onChange={(e) => setDividendYield(parseFloat(e.target.value))}
                        className="w-full"
                        style={{ accentColor: 'var(--color-accent-primary)' }}
                    />
                </div>
            </div>

            {/* The Strategic Score */}
            <div className="space-y-4">
                <h2 
                    className="text-xl font-semibold"
                    style={{ color: 'var(--color-text-primary)' }}
                >
                    3. The Strategic Score
                </h2>
                <p style={{ color: 'var(--color-text-secondary)' }}>
                    The "Fair Value" Formula: (Growth Rate + Dividend Yield) / P/E Ratio.
                </p>

                <div 
                    className="p-6 rounded-lg text-center border-l-4"
                    style={{
                        backgroundColor: 'rgba(46, 124, 246, 0.1)',
                        borderLeftColor: 'var(--color-accent-primary)',
                    }}
                >
                    <h3 className="text-sm font-semibold mb-2 uppercase tracking-wide" style={{ color: 'var(--color-text-primary)' }}>
                        Valuation Score
                    </h3>
                    <div className="text-4xl font-bold font-mono mb-3" style={{ color: 'var(--color-accent-primary)' }}>
                        {score.toFixed(2)}
                    </div>
                    <div 
                        className="inline-block px-4 py-2 rounded-full text-sm font-bold mb-3"
                        style={{ backgroundColor: verdictColor, color: '#FFFFFF' }}
                    >
                        {verdict}
                    </div>
                    <p style={{ color: 'var(--color-text-secondary)' }}>
                        {summary}
                    </p>
                </div>

                <div 
                    className="p-4 rounded-lg text-sm"
                    style={{
                        backgroundColor: 'var(--color-bg-tertiary)',
                        color: 'var(--color-text-secondary)',
                    }}
                >
                    <strong>The Rule of Thumb:</strong><br />
                    • Score &lt; 1.0: <strong>Poor / Overpriced</strong> (You are paying too much for too little growth).<br />
                    • Score 1.5: <strong>Okay</strong>.<br />
                    • Score 2.0+: <strong>Bargain</strong> (High growth/dividend for a low price).
                </div>
            </div>
        </div>
    );
};

const HelpIcon: React.FC<{ tooltip: string }> = ({ tooltip }) => {
    const [showTooltip, setShowTooltip] = useState(false);

    return (
        <div className="relative inline-block">
            <button
                onMouseEnter={() => setShowTooltip(true)}
                onMouseLeave={() => setShowTooltip(false)}
                className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold transition-colors"
                style={{
                    backgroundColor: 'var(--color-accent-primary)',
                    color: '#FFFFFF',
                }}
            >
                ?
            </button>
            {showTooltip && (
                <div
                    className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-72 p-3 rounded-lg text-xs z-50"
                    style={{
                        backgroundColor: 'var(--color-bg-elevated)',
                        color: 'var(--color-text-primary)',
                        border: '1px solid var(--color-border-default)',
                        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
                    }}
                >
                    <div dangerouslySetInnerHTML={{ __html: tooltip.replace(/\n/g, '<br />') }} />
                    <div
                        className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1 w-0 h-0 border-l-4 border-r-4 border-t-4"
                        style={{
                            borderLeftColor: 'transparent',
                            borderRightColor: 'transparent',
                            borderTopColor: 'var(--color-bg-elevated)',
                        }}
                    />
                </div>
            )}
        </div>
    );
};

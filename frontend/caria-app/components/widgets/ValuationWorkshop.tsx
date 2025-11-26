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
    const [currentStep, setCurrentStep] = useState(1);
    const totalSteps = 6;
    
    const [netIncome, setNetIncome] = useState(2500000);
    const [shares, setShares] = useState(1000000);
    const [category, setCategory] = useState('stalwart');
    const [peRatio, setPeRatio] = useState(15);
    const [cashFlow, setCashFlow] = useState(2500000);
    const [growthRate, setGrowthRate] = useState(12);
    const [dividendYield, setDividendYield] = useState(3);
    const [equity, setEquity] = useState(10000000);
    const [debt, setDebt] = useState(5000000);

    const categoryPEs: Record<string, number> = {
        slow: 10,
        stalwart: 15,
        fast: 25,
        cyclical: 8,
    };

    const changeStep = (n: number) => {
        const nextStep = currentStep + n;
        if (nextStep >= 1 && nextStep <= totalSteps) {
            setCurrentStep(nextStep);
        }
    };

    // Calculations
    const eps = netIncome / shares;
    const price = eps * peRatio;
    const qualityRatio = cashFlow / netIncome;
    const score = (growthRate + dividendYield) / peRatio;
    const deRatio = debt / equity;

    // Quality status
    let qualityStatus = '';
    let qualityColor = '';
    let qualityMessage = '';
    if (qualityRatio >= 1.0) {
        qualityStatus = 'High Quality Earnings';
        qualityColor = '#27ae60';
        qualityMessage = 'High';
    } else if (qualityRatio >= 0.8) {
        qualityStatus = 'Acceptable Quality';
        qualityColor = '#f39c12';
        qualityMessage = 'Medium';
    } else {
        qualityStatus = 'Low Quality (Warning)';
        qualityColor = '#c0392b';
        qualityMessage = 'Low';
    }

    // Valuation score status
    let scoreStatus = '';
    let scoreColor = '';
    let scoreMessage = '';
    if (score < 1.0) {
        scoreStatus = 'Overpriced (Poor Value)';
        scoreColor = '#c0392b';
        scoreMessage = 'Poor';
    } else if (score < 1.5) {
        scoreStatus = 'Fairly Priced';
        scoreColor = '#f39c12';
        scoreMessage = 'Fair';
    } else {
        scoreStatus = 'Undervalued (Bargain)';
        scoreColor = '#27ae60';
        scoreMessage = 'Bargain';
    }

    // Debt status
    let deStatus = '';
    let deColor = '';
    let deMessage = '';
    if (deRatio < 0.5) {
        deStatus = 'Very Safe (Low Debt)';
        deColor = '#27ae60';
        deMessage = 'Safe';
    } else if (deRatio < 1.0) {
        deStatus = 'Normal / Moderate';
        deColor = '#f39c12';
        deMessage = 'Moderate';
    } else {
        deStatus = 'Risky (High Leverage)';
        deColor = '#c0392b';
        deMessage = 'Risky';
    }

    // Final verdict
    let finalVerdict = '';
    if (qualityMessage === 'High' && scoreMessage === 'Bargain' && deMessage === 'Safe') {
        finalVerdict = 'STRONG BUY';
    } else if (qualityMessage === 'Low' || deMessage === 'Risky') {
        finalVerdict = 'AVOID / SELL';
    } else {
        finalVerdict = 'HOLD / WATCH';
    }

    const progressPercent = ((currentStep - 1) / (totalSteps - 1)) * 100;

    return (
        <div className="space-y-6" style={{ minHeight: '600px' }}>
            {/* Header */}
            <div className="text-center">
                <h1 
                    className="text-2xl font-bold mb-2"
                    style={{ 
                        fontFamily: 'var(--font-display)',
                        color: 'var(--color-text-primary)' 
                    }}
                >
                    Valuation Engine
                </h1>
            </div>

            {/* Progress Bar */}
            <div className="relative px-10 py-5">
                <div 
                    className="absolute top-1/2 left-10 right-10 h-1 -translate-y-1/2"
                    style={{ backgroundColor: '#eee' }}
                />
                <div 
                    className="absolute top-1/2 left-10 h-1 -translate-y-1/2 transition-all duration-400"
                    style={{ 
                        backgroundColor: '#3498db',
                        width: `${progressPercent}%`
                    }}
                />
                <div className="relative flex justify-between">
                    {[1, 2, 3, 4, 5, 6].map((step) => (
                        <div
                            key={step}
                            className="w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm transition-all duration-400"
                            style={{
                                backgroundColor: step <= currentStep ? '#3498db' : '#fff',
                                border: `3px solid ${step <= currentStep ? '#3498db' : '#eee'}`,
                                color: step <= currentStep ? '#fff' : '#7f8c8d',
                                transform: step === currentStep ? 'scale(1.2)' : 'scale(1)',
                            }}
                        >
                            {step === 6 ? '🏁' : step}
                        </div>
                    ))}
                </div>
            </div>

            {/* Content Area */}
            <div className="relative" style={{ minHeight: '400px' }}>
                {/* Card 1: Profitability */}
                <div
                    className={`absolute inset-0 transition-all duration-400 ${currentStep === 1 ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-12 pointer-events-none'}`}
                    style={{ padding: '20px 40px' }}
                >
                    <h2 className="text-xl font-semibold mb-2" style={{ color: '#3498db' }}>
                        Layer 1: Profitability
                    </h2>
                    <p className="text-sm mb-6" style={{ color: '#7f8c8d', lineHeight: '1.5' }}>
                        Before valuing the whole business, we must determine how much profit belongs to a single share of stock. This is the "Unit Price" of the business.
                    </p>

                    <div className="space-y-6 mb-6">
                        <div>
                            <label className="flex justify-between items-center mb-2 font-semibold">
                                <span>Net Income</span>
                                <span style={{ color: '#3498db' }}>{formatMoney(netIncome)}</span>
                            </label>
                            <input
                                type="range"
                                min="500000"
                                max="10000000"
                                step="100000"
                                value={netIncome}
                                onChange={(e) => setNetIncome(parseInt(e.target.value))}
                                className="w-full"
                                style={{ accentColor: '#2c3e50' }}
                            />
                        </div>

                        <div>
                            <label className="flex justify-between items-center mb-2 font-semibold">
                                <span>Shares Outstanding</span>
                                <span style={{ color: '#3498db' }}>{formatNum(shares)}</span>
                            </label>
                            <input
                                type="range"
                                min="100000"
                                max="5000000"
                                step="50000"
                                value={shares}
                                onChange={(e) => setShares(parseInt(e.target.value))}
                                className="w-full"
                                style={{ accentColor: '#2c3e50' }}
                            />
                        </div>
                    </div>

                    <div className="p-5 rounded-lg text-center border" style={{ backgroundColor: '#f8f9fa', borderColor: '#e1e1e1' }}>
                        <div className="text-xs uppercase tracking-wide mb-2" style={{ color: '#7f8c8d' }}>
                            Earnings Per Share (EPS)
                        </div>
                        <div className="text-3xl font-bold" style={{ color: '#2c3e50' }}>
                            ${eps.toFixed(2)}
                        </div>
                    </div>
                </div>

                {/* Card 2: Market Sentiment */}
                <div
                    className={`absolute inset-0 transition-all duration-400 ${currentStep === 2 ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-12 pointer-events-none'}`}
                    style={{ padding: '20px 40px' }}
                >
                    <h2 className="text-xl font-semibold mb-2" style={{ color: '#3498db' }}>
                        Layer 2: Market Sentiment
                    </h2>
                    <p className="text-sm mb-6" style={{ color: '#7f8c8d', lineHeight: '1.5' }}>
                        The stock price is rarely just the EPS. It is the EPS multiplied by investor expectations (the P/E Ratio). How much is the market willing to pay for $1 of earnings?
                    </p>

                    <div className="space-y-6 mb-6">
                        <div>
                            <label className="block mb-2 font-semibold">Category Profile</label>
                            <select
                                value={category}
                                onChange={(e) => {
                                    setCategory(e.target.value);
                                    setPeRatio(categoryPEs[e.target.value]);
                                }}
                                className="w-full p-2 rounded border"
                                style={{ backgroundColor: 'var(--color-bg-tertiary)', borderColor: '#ccc' }}
                            >
                                <option value="slow">Slow Grower (Utilities)</option>
                                <option value="stalwart">Stalwart (Steady Brands)</option>
                                <option value="fast">Fast Grower (Tech/New Retail)</option>
                                <option value="cyclical">Cyclical (Auto/Heavy Industry)</option>
                            </select>
                        </div>

                        <div>
                            <label className="flex justify-between items-center mb-2 font-semibold">
                                <span>P/E Multiplier</span>
                                <span style={{ color: '#3498db' }}>{peRatio}x</span>
                            </label>
                            <input
                                type="range"
                                min="5"
                                max="50"
                                step="1"
                                value={peRatio}
                                onChange={(e) => setPeRatio(parseInt(e.target.value))}
                                className="w-full"
                                style={{ accentColor: '#2c3e50' }}
                            />
                        </div>
                    </div>

                    <div className="p-5 rounded-lg text-center border" style={{ backgroundColor: '#f8f9fa', borderColor: '#e1e1e1' }}>
                        <div className="text-xs uppercase tracking-wide mb-2" style={{ color: '#7f8c8d' }}>
                            Implied Share Price
                        </div>
                        <div className="text-3xl font-bold" style={{ color: '#2c3e50' }}>
                            ${price.toFixed(2)}
                        </div>
                    </div>
                </div>

                {/* Card 3: Cash Reality Check */}
                <div
                    className={`absolute inset-0 transition-all duration-400 ${currentStep === 3 ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-12 pointer-events-none'}`}
                    style={{ padding: '20px 40px' }}
                >
                    <h2 className="text-xl font-semibold mb-2" style={{ color: '#3498db' }}>
                        Layer 3: The Cash Reality Check
                    </h2>
                    <p className="text-sm mb-6" style={{ color: '#7f8c8d', lineHeight: '1.5' }}>
                        Accounting profit is an opinion; Cash is a fact. Does the company actually collect the money it claims to earn? If Cash Flow is lower than Profit, be suspicious.
                    </p>

                    <div className="space-y-6 mb-6">
                        <div>
                            <label className="flex justify-between items-center mb-2 font-semibold">
                                <span>Operating Cash Flow</span>
                                <span style={{ color: '#3498db' }}>{formatMoney(cashFlow)}</span>
                            </label>
                            <input
                                type="range"
                                min="0"
                                max="5000000"
                                step="100000"
                                value={cashFlow}
                                onChange={(e) => setCashFlow(parseInt(e.target.value))}
                                className="w-full"
                                style={{ accentColor: '#2c3e50' }}
                            />
                        </div>
                    </div>

                    <div className="p-5 rounded-lg text-center border" style={{ backgroundColor: '#f8f9fa', borderColor: '#e1e1e1' }}>
                        <div className="text-xs uppercase tracking-wide mb-2" style={{ color: '#7f8c8d' }}>
                            Quality Ratio
                        </div>
                        <div className="text-3xl font-bold mb-3" style={{ color: '#2c3e50' }}>
                            {qualityRatio.toFixed(2)}
                        </div>
                        <div className="p-2 rounded text-sm font-bold text-white" style={{ backgroundColor: qualityColor }}>
                            {qualityStatus}
                        </div>
                    </div>
                </div>

                {/* Card 4: Growth Valuation */}
                <div
                    className={`absolute inset-0 transition-all duration-400 ${currentStep === 4 ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-12 pointer-events-none'}`}
                    style={{ padding: '20px 40px' }}
                >
                    <h2 className="text-xl font-semibold mb-2" style={{ color: '#3498db' }}>
                        Layer 4: The Growth Valuation
                    </h2>
                    <p className="text-sm mb-6" style={{ color: '#7f8c8d', lineHeight: '1.5' }}>
                        Is the stock cheap or expensive relative to its growth? We use a Fair Value formula: (Growth Rate + Dividend Yield) / P/E Ratio.
                    </p>

                    <div className="space-y-6 mb-6">
                        <div>
                            <label className="flex justify-between items-center mb-2 font-semibold">
                                <span>Growth Rate</span>
                                <span style={{ color: '#3498db' }}>{growthRate}%</span>
                            </label>
                            <input
                                type="range"
                                min="0"
                                max="40"
                                step="1"
                                value={growthRate}
                                onChange={(e) => setGrowthRate(parseInt(e.target.value))}
                                className="w-full"
                                style={{ accentColor: '#2c3e50' }}
                            />
                        </div>

                        <div>
                            <label className="flex justify-between items-center mb-2 font-semibold">
                                <span>Dividend Yield</span>
                                <span style={{ color: '#3498db' }}>{dividendYield}%</span>
                            </label>
                            <input
                                type="range"
                                min="0"
                                max="10"
                                step="0.5"
                                value={dividendYield}
                                onChange={(e) => setDividendYield(parseFloat(e.target.value))}
                                className="w-full"
                                style={{ accentColor: '#2c3e50' }}
                            />
                        </div>
                    </div>

                    <div className="p-5 rounded-lg text-center border" style={{ backgroundColor: '#f8f9fa', borderColor: '#e1e1e1' }}>
                        <div className="text-xs uppercase tracking-wide mb-2" style={{ color: '#7f8c8d' }}>
                            Valuation Score
                        </div>
                        <div className="text-3xl font-bold mb-3" style={{ color: '#2c3e50' }}>
                            {score.toFixed(2)}
                        </div>
                        <div className="p-2 rounded text-sm font-bold text-white" style={{ backgroundColor: scoreColor }}>
                            {scoreStatus}
                        </div>
                    </div>
                </div>

                {/* Card 5: Financial Health */}
                <div
                    className={`absolute inset-0 transition-all duration-400 ${currentStep === 5 ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-12 pointer-events-none'}`}
                    style={{ padding: '20px 40px' }}
                >
                    <h2 className="text-xl font-semibold mb-2" style={{ color: '#3498db' }}>
                        Layer 5: Financial Health
                    </h2>
                    <p className="text-sm mb-6" style={{ color: '#7f8c8d', lineHeight: '1.5' }}>
                        Can the company survive a downturn? We check the <strong>Debt-to-Equity Ratio</strong>. If a company has too much debt, earnings don't matter—it could go bankrupt.
                    </p>

                    <div className="space-y-6 mb-6">
                        <div>
                            <label className="flex justify-between items-center mb-2 font-semibold">
                                <span>Total Equity</span>
                                <span style={{ color: '#3498db' }}>{formatNum(equity)}</span>
                            </label>
                            <input
                                type="range"
                                min="1000000"
                                max="20000000"
                                step="500000"
                                value={equity}
                                onChange={(e) => setEquity(parseInt(e.target.value))}
                                className="w-full"
                                style={{ accentColor: '#2c3e50' }}
                            />
                        </div>

                        <div>
                            <label className="flex justify-between items-center mb-2 font-semibold">
                                <span>Total Debt</span>
                                <span style={{ color: '#3498db' }}>{formatNum(debt)}</span>
                            </label>
                            <input
                                type="range"
                                min="0"
                                max="20000000"
                                step="500000"
                                value={debt}
                                onChange={(e) => setDebt(parseInt(e.target.value))}
                                className="w-full"
                                style={{ accentColor: '#2c3e50' }}
                            />
                        </div>
                    </div>

                    <div className="p-5 rounded-lg text-center border" style={{ backgroundColor: '#f8f9fa', borderColor: '#e1e1e1' }}>
                        <div className="text-xs uppercase tracking-wide mb-2" style={{ color: '#7f8c8d' }}>
                            Debt-to-Equity Ratio
                        </div>
                        <div className="text-3xl font-bold mb-3" style={{ color: '#2c3e50' }}>
                            {deRatio.toFixed(2)}
                        </div>
                        <div className="p-2 rounded text-sm font-bold text-white" style={{ backgroundColor: deColor }}>
                            {deStatus}
                        </div>
                    </div>
                </div>

                {/* Card 6: Summary Dashboard */}
                <div
                    className={`absolute inset-0 transition-all duration-400 ${currentStep === 6 ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-12 pointer-events-none'}`}
                    style={{ padding: '20px 40px' }}
                >
                    <h2 className="text-xl font-semibold mb-2" style={{ color: '#3498db' }}>
                        Valuation Dashboard
                    </h2>
                    <p className="text-sm mb-6" style={{ color: '#7f8c8d', lineHeight: '1.5' }}>
                        Here is the complete picture of the business based on your inputs.
                    </p>

                    <div className="grid grid-cols-2 gap-4 mb-4">
                        <div className="p-4 rounded-lg border" style={{ backgroundColor: '#fff', borderColor: '#eee' }}>
                            <div className="text-xs uppercase tracking-wide mb-2" style={{ color: '#7f8c8d' }}>
                                Target Price
                            </div>
                            <div className="text-2xl font-bold" style={{ color: '#2c3e50' }}>
                                ${price.toFixed(2)}
                            </div>
                        </div>
                        <div className="p-4 rounded-lg border" style={{ backgroundColor: '#fff', borderColor: '#eee' }}>
                            <div className="text-xs uppercase tracking-wide mb-2" style={{ color: '#7f8c8d' }}>
                                Earnings Quality
                            </div>
                            <div className="text-lg font-bold" style={{ color: qualityColor }}>
                                {qualityMessage}
                            </div>
                        </div>
                        <div className="p-4 rounded-lg border" style={{ backgroundColor: '#fff', borderColor: '#eee' }}>
                            <div className="text-xs uppercase tracking-wide mb-2" style={{ color: '#7f8c8d' }}>
                                Valuation
                            </div>
                            <div className="text-lg font-bold" style={{ color: scoreColor }}>
                                {scoreMessage}
                            </div>
                        </div>
                        <div className="p-4 rounded-lg border" style={{ backgroundColor: '#fff', borderColor: '#eee' }}>
                            <div className="text-xs uppercase tracking-wide mb-2" style={{ color: '#7f8c8d' }}>
                                Solvency
                            </div>
                            <div className="text-lg font-bold" style={{ color: deColor }}>
                                {deMessage}
                            </div>
                        </div>
                    </div>

                    <div className="p-5 rounded-lg text-center" style={{ backgroundColor: '#2c3e50', color: '#fff' }}>
                        <div className="text-xs uppercase tracking-wide mb-2" style={{ color: '#eee' }}>
                            Final Verdict
                        </div>
                        <div className="text-2xl font-bold">
                            {finalVerdict}
                        </div>
                    </div>
                </div>
            </div>

            {/* Footer Controls */}
            <div className="flex justify-between pt-4 border-t" style={{ borderColor: '#eee' }}>
                <button
                    onClick={() => changeStep(-1)}
                    className={`px-6 py-3 rounded-lg font-semibold transition-all ${currentStep === 1 ? 'invisible' : ''}`}
                    style={{
                        backgroundColor: 'transparent',
                        color: '#7f8c8d',
                    }}
                >
                    Back
                </button>
                {currentStep < totalSteps ? (
                    <button
                        onClick={() => changeStep(1)}
                        className="px-6 py-3 rounded-lg font-semibold text-white transition-all"
                        style={{ backgroundColor: '#2c3e50' }}
                    >
                        Next Step
                    </button>
                ) : (
                    <button
                        onClick={onBack}
                        className="px-6 py-3 rounded-lg font-semibold text-white transition-all"
                        style={{ backgroundColor: '#2c3e50' }}
                    >
                        Finish
                    </button>
                )}
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

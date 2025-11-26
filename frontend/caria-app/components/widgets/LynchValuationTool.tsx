import React, { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';

interface CategoryData {
    text: string;
    pe: number;
}

const CATEGORY_DATA: Record<string, CategoryData> = {
    slow: { 
        text: "Slow Grower: Usually large and aging companies. Expected to grow slightly faster than GNP. Generous dividends are key here.", 
        pe: 10 
    },
    stalwart: { 
        text: "Stalwart: The Coca-Colas and P&Gs. Good protection during recessions. Lynch says: 'I always keep some stalwarts in my portfolio.'", 
        pe: 15 
    },
    fast: { 
        text: "Fast Grower: Aggressive new enterprises. Growing 20-25% a year. This is where the '10-baggers' are. Riskier, but huge upside.", 
        pe: 25 
    },
    cyclical: { 
        text: "Cyclical: Autos, airlines, steel. Earnings rise and fall with the economy. P/E is tricky here (often low P/E means the end of the cycle!).", 
        pe: 8 
    },
};

const formatMoney = (n: number): string => {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(n);
};

const formatNum = (n: number): string => {
    return new Intl.NumberFormat('en-US').format(n);
};

export const LynchValuationTool: React.FC = () => {
    const [netIncome, setNetIncome] = useState(2642000);
    const [shares, setShares] = useState(800000);
    const [peRatio, setPeRatio] = useState(15);
    const [category, setCategory] = useState<string>('stalwart');
    const [cashFlow, setCashFlow] = useState(3105000);
    const [growthRate, setGrowthRate] = useState(12);
    const [dividendYield, setDividendYield] = useState(3);
    const [activeStep, setActiveStep] = useState<string>('step1');

    // Calculate EPS
    const eps = netIncome / shares;

    // Calculate Fair Market Price
    const fairPrice = eps * peRatio;

    // Calculate Quality Ratio (Cash Flow / Net Income)
    const qualityRatio = cashFlow / netIncome;
    let qualityStatus = '';
    let qualityColor = '';
    let qualityMessage = '';

    if (qualityRatio >= 1.0) {
        qualityStatus = 'High Quality';
        qualityColor = 'var(--color-positive)';
        qualityMessage = "Tracy Approved: Cash flow exceeds Net Income. The company is actually collecting the money it claims to earn.";
    } else if (qualityRatio > 0.8) {
        qualityStatus = 'Acceptable';
        qualityColor = 'var(--color-warning)';
        qualityMessage = "Caution: Cash flow is slightly lagging. This might be due to increasing inventory (Chapter 5) or unpaid bills. Check the Balance Sheet.";
    } else {
        qualityStatus = 'Low Quality';
        qualityColor = 'var(--color-negative)';
        qualityMessage = "Red Flag: Significant gap between Profit and Cash. Management might be 'massaging the numbers' (Chapter 20) or facing a liquidity crisis.";
    }

    // Calculate Lynch Score: (Growth + Dividend) / PE
    const lynchScore = (growthRate + dividendYield) / peRatio;
    let lynchVerdict = '';
    let lynchVerdictColor = '';

    if (lynchScore < 1) {
        lynchVerdict = 'Poor (Overpriced)';
        lynchVerdictColor = 'var(--color-negative)';
    } else if (lynchScore < 1.5) {
        lynchVerdict = 'Okay / Fair';
        lynchVerdictColor = 'var(--color-warning)';
    } else if (lynchScore < 2) {
        lynchVerdict = 'Good Buy';
        lynchVerdictColor = 'var(--color-positive)';
    } else {
        lynchVerdict = 'Strong Buy (Bargain)';
        lynchVerdictColor = '#8e44ad'; // Purple for Lynch
    }

    const updateCategory = (newCategory: string) => {
        setCategory(newCategory);
        setPeRatio(CATEGORY_DATA[newCategory].pe);
    };

    const revealStep = (stepId: string) => {
        setActiveStep(stepId);
        // Scroll to step
        setTimeout(() => {
            const element = document.getElementById(stepId);
            if (element) {
                element.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }, 100);
    };

    // Determine if a step should be active (current step or any previous step)
    const isStepActive = (stepId: string): boolean => {
        const stepOrder = ['step1', 'step2', 'step3', 'step4'];
        const currentIndex = stepOrder.indexOf(activeStep);
        const stepIndex = stepOrder.indexOf(stepId);
        return stepIndex <= currentIndex;
    };

    return (
        <WidgetCard
            title="Don't know how to value a stock? We got you: Lynch's Categories"
            tooltip="Interactive valuation workshop based on John A. Tracy & Peter Lynch. Use Tracy's accounting logic to find the numbers, and Lynch's strategy to value the story."
        >
            <div className="space-y-6">
                <div className="mb-6">
                    <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                        <strong>Philosophy:</strong> Use Tracy's accounting logic to find the numbers, and Lynch's strategy to value the story.
                    </p>
                </div>

                {/* Step 1: The Foundation (EPS) */}
                <div 
                    id="step1"
                    className="transition-all duration-300 rounded-lg p-6"
                    style={{
                        borderLeft: isStepActive('step1') ? '5px solid var(--color-accent-primary)' : '5px solid var(--color-bg-tertiary)',
                        backgroundColor: isStepActive('step1') ? 'var(--color-bg-secondary)' : 'transparent',
                        opacity: isStepActive('step1') ? 1 : 0.6,
                    }}
                >
                    <h2 
                        className="text-lg font-semibold mb-3"
                        style={{ color: 'var(--color-text-primary)' }}
                    >
                        1. The Foundation (EPS)
                    </h2>
                    <p className="text-sm mb-4" style={{ color: 'var(--color-text-secondary)' }}>
                        Before you invest, you must know how much profit belongs to you. John A. Tracy insists: "The starting point is to calculate earnings per share."
                    </p>

                    <div className="space-y-4 mb-4">
                        <div>
                            <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                                Net Income (Annual Profit): {formatNum(netIncome)}
                            </label>
                            <input
                                type="range"
                                min="500000"
                                max="10000000"
                                step="100000"
                                value={netIncome}
                                onChange={(e) => setNetIncome(parseInt(e.target.value))}
                                className="w-full"
                                style={{ accentColor: 'var(--color-accent-primary)' }}
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                                Shares Outstanding: {formatNum(shares)}
                            </label>
                            <input
                                type="range"
                                min="100000"
                                max="2000000"
                                step="50000"
                                value={shares}
                                onChange={(e) => setShares(parseInt(e.target.value))}
                                className="w-full"
                                style={{ accentColor: 'var(--color-accent-primary)' }}
                            />
                        </div>
                    </div>

                    <div 
                        className="flex justify-between items-center p-4 rounded-lg mb-4"
                        style={{ backgroundColor: 'var(--color-bg-tertiary)' }}
                    >
                        <span className="text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>
                            Calculated EPS:
                        </span>
                        <span 
                            className="text-xl font-bold font-mono"
                            style={{ color: 'var(--color-accent-primary)' }}
                        >
                            {formatMoney(eps)}
                        </span>
                    </div>

                    <button
                        onClick={() => revealStep('step2')}
                        className="w-full py-3 px-6 rounded-lg font-medium transition-all"
                        style={{
                            backgroundColor: 'var(--color-accent-primary)',
                            color: '#FFFFFF',
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.opacity = '0.9';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.opacity = '1';
                        }}
                    >
                        Next: Define the Story (Lynch) ↓
                    </button>
                </div>

                {/* Step 2: The "One Up" Story Check */}
                <div 
                    id="step2"
                    className="transition-all duration-300 rounded-lg p-6"
                    style={{
                        borderLeft: isStepActive('step2') ? '5px solid #8e44ad' : '5px solid var(--color-bg-tertiary)',
                        backgroundColor: isStepActive('step2') ? '#fcf4ff' : 'transparent',
                        opacity: isStepActive('step2') ? 1 : 0.6,
                    }}
                >
                    <h2 
                        className="text-lg font-semibold mb-3"
                        style={{ color: 'var(--color-text-primary)' }}
                    >
                        2. The "One Up" Story Check
                    </h2>
                    <p className="text-sm mb-4" style={{ color: 'var(--color-text-secondary)' }}>
                        Peter Lynch says: "Once I've established the size... I place it into one of six general categories." A Fast Grower should sell for a higher multiple than a Slow Grower.
                    </p>

                    <div className="space-y-4 mb-4">
                        <div>
                            <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                                Which category best fits this company?
                            </label>
                            <select
                                value={category}
                                onChange={(e) => updateCategory(e.target.value)}
                                className="w-full px-4 py-2 rounded-lg border"
                                style={{
                                    backgroundColor: 'var(--color-bg-tertiary)',
                                    borderColor: 'var(--color-border-subtle)',
                                    color: 'var(--color-text-primary)',
                                }}
                            >
                                <option value="slow">Slow Grower (Utilities, Mature)</option>
                                <option value="stalwart">Stalwart (Coke, P&G - Steady)</option>
                                <option value="fast">Fast Grower (20-25% growth)</option>
                                <option value="cyclical">Cyclical (Autos, Airlines)</option>
                            </select>
                        </div>

                        <div 
                            className="p-4 rounded-lg border-l-4"
                            style={{
                                backgroundColor: '#f5eef8',
                                borderLeftColor: '#8e44ad',
                            }}
                        >
                            <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                                <strong>
                                    {category === 'slow' ? 'Slow Grower' :
                                     category === 'stalwart' ? 'Stalwart' :
                                     category === 'fast' ? 'Fast Grower' :
                                     'Cyclical'}:
                                </strong>{' '}
                                {CATEGORY_DATA[category].text}
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                                Assign P/E Ratio: {peRatio}x
                            </label>
                            <input
                                type="range"
                                min="5"
                                max="40"
                                step="1"
                                value={peRatio}
                                onChange={(e) => setPeRatio(parseInt(e.target.value))}
                                className="w-full"
                                style={{ accentColor: '#8e44ad' }}
                            />
                        </div>
                    </div>

                    <div 
                        className="p-6 rounded-lg text-center mb-4"
                        style={{ backgroundColor: '#8e44ad', color: '#FFFFFF' }}
                    >
                        <h3 className="text-sm font-semibold mb-2 uppercase tracking-wide">Fair Market Price</h3>
                        <div className="text-4xl font-bold font-mono">
                            {formatMoney(fairPrice)}
                        </div>
                    </div>

                    <button
                        onClick={() => revealStep('step3')}
                        className="w-full py-3 px-6 rounded-lg font-medium transition-all"
                        style={{
                            backgroundColor: '#8e44ad',
                            color: '#FFFFFF',
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.opacity = '0.9';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.opacity = '1';
                        }}
                    >
                        Next: Verify with Cash (Tracy) ↓
                    </button>
                </div>

                {/* Step 3: The Quality of Earnings */}
                <div 
                    id="step3"
                    className="transition-all duration-300 rounded-lg p-6"
                    style={{
                        borderLeft: isStepActive('step3') ? '5px solid var(--color-accent-primary)' : '5px solid var(--color-bg-tertiary)',
                        backgroundColor: isStepActive('step3') ? 'var(--color-bg-secondary)' : 'transparent',
                        opacity: isStepActive('step3') ? 1 : 0.6,
                    }}
                >
                    <h2 
                        className="text-lg font-semibold mb-3"
                        style={{ color: 'var(--color-text-primary)' }}
                    >
                        3. The Quality of Earnings
                    </h2>
                    <p className="text-sm mb-4" style={{ color: 'var(--color-text-secondary)' }}>
                        John Tracy warns: "Profit and cash flow are not identical twins." If the company reports ${formatNum(netIncome)} in profit but has $0 cash flow, that profit might be fake (or "massaged").
                    </p>

                    <div className="space-y-4 mb-4">
                        <div>
                            <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                                Actual Cash Flow from Operations: {formatNum(cashFlow)}
                            </label>
                            <input
                                type="range"
                                min="0"
                                max="6000000"
                                step="100000"
                                value={cashFlow}
                                onChange={(e) => setCashFlow(parseInt(e.target.value))}
                                className="w-full"
                                style={{ accentColor: 'var(--color-accent-primary)' }}
                            />
                        </div>
                    </div>

                    <div 
                        className="flex justify-between items-center p-4 rounded-lg mb-4"
                        style={{ 
                            backgroundColor: 'var(--color-bg-tertiary)',
                            borderLeft: `5px solid ${qualityColor}`,
                        }}
                    >
                        <span className="text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>
                            Earnings Status:
                        </span>
                        <span 
                            className="text-lg font-bold"
                            style={{ color: qualityColor }}
                        >
                            {qualityStatus}
                        </span>
                    </div>

                    <div 
                        className="p-4 rounded-lg border-l-4 mb-4"
                        style={{
                            backgroundColor: '#e8f6f3',
                            borderLeftColor: '#1abc9c',
                        }}
                    >
                        <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                            <strong>Tracy Analysis:</strong> {qualityMessage}
                        </div>
                    </div>

                    <button
                        onClick={() => revealStep('step4')}
                        className="w-full py-3 px-6 rounded-lg font-medium transition-all"
                        style={{
                            backgroundColor: 'var(--color-accent-primary)',
                            color: '#FFFFFF',
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.opacity = '0.9';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.opacity = '1';
                        }}
                    >
                        Next: The Lynch Formula ↓
                    </button>
                </div>

                {/* Step 4: The Lynch Valuation Test */}
                <div 
                    id="step4"
                    className="transition-all duration-300 rounded-lg p-6"
                    style={{
                        borderLeft: isStepActive('step4') ? '5px solid #8e44ad' : '5px solid var(--color-bg-tertiary)',
                        backgroundColor: isStepActive('step4') ? '#fcf4ff' : 'transparent',
                        opacity: isStepActive('step4') ? 1 : 0.6,
                    }}
                >
                    <h2 
                        className="text-lg font-semibold mb-3"
                        style={{ color: 'var(--color-text-primary)' }}
                    >
                        4. The Lynch Valuation Test
                    </h2>
                    <p className="text-sm mb-4" style={{ color: 'var(--color-text-secondary)' }}>
                        In <em>One Up on Wall Street</em> (p. 189), Lynch gives his personal formula for valuing growth stocks. He says: "Find the long-term growth rate, add the dividend yield, and divide by the P/E ratio."
                    </p>

                    <div className="space-y-4 mb-4">
                        <div>
                            <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                                Growth Rate: {growthRate}%
                            </label>
                            <input
                                type="range"
                                min="0"
                                max="40"
                                step="1"
                                value={growthRate}
                                onChange={(e) => setGrowthRate(parseInt(e.target.value))}
                                className="w-full"
                                style={{ accentColor: '#8e44ad' }}
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-2" style={{ color: 'var(--color-text-secondary)' }}>
                                Dividend Yield: {dividendYield}%
                            </label>
                            <input
                                type="range"
                                min="0"
                                max="10"
                                step="0.5"
                                value={dividendYield}
                                onChange={(e) => setDividendYield(parseFloat(e.target.value))}
                                className="w-full"
                                style={{ accentColor: '#8e44ad' }}
                            />
                        </div>
                    </div>

                    <div 
                        className="p-6 rounded-lg text-center mb-4"
                        style={{ backgroundColor: '#8e44ad', color: '#FFFFFF' }}
                    >
                        <h3 className="text-sm font-semibold mb-2 uppercase tracking-wide">Lynch Score</h3>
                        <div className="text-4xl font-bold font-mono mb-2">
                            {lynchScore.toFixed(2)}
                        </div>
                        <div 
                            className="text-lg font-semibold"
                            style={{ color: lynchVerdictColor }}
                        >
                            {lynchVerdict}
                        </div>
                    </div>

                    <div 
                        className="p-4 rounded-lg border-l-4"
                        style={{
                            backgroundColor: '#f5eef8',
                            borderLeftColor: '#8e44ad',
                        }}
                    >
                        <div className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                            <strong>The Rule:</strong><br />
                            Less than 1 = Poor<br />
                            1.5 = Okay<br />
                            <strong>2 or better = What you're looking for</strong>
                        </div>
                    </div>
                </div>
            </div>
        </WidgetCard>
    );
};

export const marketIndices = [
    { name: 'S&P 500', value: '5,487.03', change: '+0.77%', positive: true },
    { name: 'NASDAQ', value: '17,857.02', change: '+0.95%', positive: true },
    { name: 'DOW JONES', value: '38,778.10', change: '-0.15%', positive: false },
    { name: 'RUSSELL 2000', value: '2,022.03', change: '-0.15%', positive: false },
];

// Helper to generate more realistic, random-walk time-series data
const generatePerformanceData = (days: number, startValue = 100, volatility = 0.02) => {
    const data = [];
    let currentValue = startValue;
    const today = new Date();

    for (let i = days - 1; i >= 0; i--) {
        const date = new Date(today);
        date.setDate(today.getDate() - i);
        
        const changePercent = 2 * volatility * Math.random() - volatility;
        currentValue *= (1 + changePercent);
        
        data.push({ date: date.toISOString().split('T')[0], value: parseFloat(currentValue.toFixed(2)) });
    }
    return data;
};

export const performanceData = {
    '1M': generatePerformanceData(30, 110, 0.015),
    '6M': generatePerformanceData(180, 100, 0.018),
    '1Y': generatePerformanceData(365, 90, 0.012),
};

export const portfolioData = {
    allocation: [
        { name: 'Stocks', value: 65, color: '#3b82f6' }, // blue-500
        { name: 'Bonds', value: 20, color: '#14b8a6' }, // teal-500
        { name: 'Crypto', value: 10, color: '#a855f7' }, // purple-500
        { name: 'Cash', value: 5, color: '#64748b' },   // slate-500
    ],
    performance: performanceData['6M'], // Default to 6M
};

export const topMovers = [
    { ticker: 'TSLA', name: 'Tesla Inc.', price: 187.44, change: 5.30, positive: true },
    { ticker: 'AAPL', name: 'Apple Inc.', price: 214.29, change: 2.19, positive: true },
    { ticker: 'GOOGL', name: 'Alphabet Inc.', price: 180.26, change: -1.21, positive: false },
];

export const globalMarketIndices = [
    { name: 'S&P 500', region: 'USA', value: '5,487.03', change: '+0.77%', positive: true },
    { name: 'STOXX 600', region: 'Europe', value: '514.94', change: '-0.73%', positive: false },
    { name: 'S&P IPSA', region: 'Chile', value: '6,634.35', change: '+0.21%', positive: true },
];

// FIX: Add missing 'communityIdeas' export to resolve import error.
export const communityIdeas = [
    { 
        title: "Thesis: Is $SNOW overvalued post-partnership news?", 
        author: "by @value_investor", 
        summary: "Diving deep into Snowflake's valuation. The recent AI partnerships are priced in, but what about execution risk? Let's discuss the moat..."
    },
    { 
        title: "Long-term hold: $MSFT's AI integration", 
        author: "by @tech_growth", 
        summary: "Microsoft's integration of Co-pilot across its entire suite is a game-changer for productivity. The TAM is huge, and they have the distribution."
    },
    { 
        title: "Undervalued small-cap: The case for $ETSY", 
        author: "by @deep_dive_dave", 
        summary: "Etsy is getting punished by the market, but its niche is defensible. GMS is stabilizing, and the take rate continues to grow. Contrarian bet?"
    },
];

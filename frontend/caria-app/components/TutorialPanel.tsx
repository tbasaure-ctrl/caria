import React, { useState } from 'react';
import { X, BookOpen, TrendingUp, Search, Activity, Brain, DollarSign, BarChart3, HelpCircle } from 'lucide-react';

interface TutorialSection {
    id: string;
    title: string;
    icon: React.ReactNode;
    content: string;
    features: string[];
}

const TUTORIAL_SECTIONS: TutorialSection[] = [
    {
        id: 'overview',
        title: 'Platform Overview',
        icon: <BookOpen className="h-6 w-6" />,
        content: 'Caria is an AI-powered investment research platform that combines real-time market data, advanced analytics, and machine learning to help you make informed investment decisions.',
        features: [
            'Real-time market data from FMP (Financial Modeling Prep)',
            'AI-driven stock analysis and scoring',
            'Portfolio stress testing and simulations',
            'Topological market analysis for anomaly detection',
            'Liquidity-based strategy recommendations'
        ]
    },
    {
        id: 'crisis_simulator',
        title: 'Crisis Simulator',
        icon: <TrendingUp className="h-6 w-6" />,
        content: 'Test how your portfolio would perform during historical crises like 2008 GFC, COVID-19 crash, or 9/11 attacks.',
        features: [
            'Select from 12+ historical crisis events',
            'Adaptive timelines (acute events show days, recessions show months)',
            'Compare portfolio vs S&P 500 benchmark',
            'See max drawdown and recovery metrics',
            'Understand which holdings drive risk'
        ]
    },
    {
        id: 'alpha_picker',
        title: 'Alpha Stock Picker',
        icon: <Search className="h-6 w-6" />,
        content: 'Discover emerging winners using our C-Score v2 engine that finds companies becoming great, not already established.',
        features: [
            'C-Quality (30%): Business durability & execution',
            'C-Delta (50%): Improvement momentum - THE ALPHA ENGINE',
            'Mispricing (20%): Contrarian signals',
            'Formula: (Quality^0.6) × (Delta^1.2) × Mispricing',
            'Avoids MAG7 trap - finds next SHOP, not current GOOGL'
        ]
    },
    {
        id: 'hidden_gems',
        title: 'Hidden Gems Screener',
        icon: <DollarSign className="h-6 w-6" />,
        content: 'Find undervalued mid-cap stocks with high C-Delta scores that the market hasn\'t discovered yet.',
        features: [
            'Scans 30+ large & mid-cap stocks',
            'Filters for improving fundamentals',
            'High revenue/employee growth',
            'Margin expansion streaks',
            'Insider buying activity'
        ]
    },
    {
        id: 'hydraulic_stack',
        title: 'AI-Hydraulic Stack',
        icon: <Activity className="h-6 w-6" />,
        content: 'Real-time market regime indicator based on Fed liquidity (Assets - TGA - RRP) and yield curve slope.',
        features: [
            'Score >60 = EXPANSION (growth mode)',
            'Score <40 = CONTRACTION (defensive mode)',
            'Updates every 4 hours with FRED data',
            'Guides strategy: Aggressive vs Defensive',
            'Combines Net Liquidity + Yield Curve signals'
        ]
    },
    {
        id: 'caria_cortex',
        title: 'Caria Cortex (Topological MRI)',
        icon: <Brain className="h-6 w-6" />,
        content: 'Uses Topological Data Analysis (TDA) to detect "alien" stocks - companies with abnormal behavioral patterns.',
        features: [
            'Identifies structural market anomalies',
            'High isolation scores = disconnected from sector',
            'Detect stocks moving independently',
            'Visualizes network topology',
            'Powered by persistent homology (gudhi library)'
        ]
    },
    {
        id: 'glossary',
        title: 'Financial Glossary',
        icon: <BarChart3 className="h-6 w-6" />,
        content: 'Key terms and concepts used throughout the platform.',
        features: [
            'ROIC: Return on Invested Capital',
            'FCF: Free Cash Flow (cash after expenses & capex)',
            'EV/S: Enterprise Value to Sales ratio',
            'TDA: Topological Data Analysis',
            'Betti Numbers: Connectivity metrics in topology',
            'Net Liquidity: Fed Assets - TGA - Reverse Repo'
        ]
    },
    {
        id: 'faq',
        title: 'FAQ',
        icon: <HelpCircle className="h-6 w-6" />,
        content: 'Frequently asked questions about using Caria.',
        features: [
            'Q: Why doesn\'t Google rank #1 in Alpha Picker?',
            'A: C-Score v2 rewards improvement momentum (Delta), not static quality. MAG7 stocks have plateaued growth.',
            '',
            'Q: What does "OFFLINE" mean in Caria Cortex?',
            'A: The topology engine requires gudhi library. Check backend logs.',
            '',
            'Q: How often does data update?',
            'A: Market data: real-time. Liquidity: every 4hrs. Topology: every 10s.'
        ]
    }
];

export default function TutorialPanel({ onClose }: { onClose: () => void }) {
    const [activeSection, setActiveSection] = useState('overview');

    const currentSection = TUTORIAL_SECTIONS.find(s => s.id === activeSection) || TUTORIAL_SECTIONS[0];

    return (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-slate-900 border border-gray-700 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-gray-700">
                    <div className="flex items-center gap-3">
                        <BookOpen className="h-8 w-8 text-cyan-400" />
                        <h2 className="text-2xl font-bold text-white">Caria Tutorial</h2>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
                    >
                        <X className="h-6 w-6 text-gray-400" />
                    </button>
                </div>

                <div className="flex flex-1 overflow-hidden">
                    {/* Sidebar */}
                    <div className="w-64 border-r border-gray-700 p-4 overflow-y-auto">
                        <nav className="space-y-2">
                            {TUTORIAL_SECTIONS.map(section => (
                                <button
                                    key={section.id}
                                    onClick={() => setActiveSection(section.id)}
                                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors text-left ${activeSection === section.id
                                            ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                                            : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                                        }`}
                                >
                                    {section.icon}
                                    <span className="text-sm font-medium">{section.title}</span>
                                </button>
                            ))}
                        </nav>
                    </div>

                    {/* Content */}
                    <div className="flex-1 p-6 overflow-y-auto">
                        <div className="flex items-center gap-3 mb-4">
                            <div className="text-cyan-400">{currentSection.icon}</div>
                            <h3 className="text-xl font-bold text-white">{currentSection.title}</h3>
                        </div>

                        <p className="text-gray-300 mb-6 leading-relaxed">
                            {currentSection.content}
                        </p>

                        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
                            <h4 className="text-sm font-semibold text-gray-400 mb-3 uppercase tracking-wide">
                                Key Features
                            </h4>
                            <ul className="space-y-2">
                                {currentSection.features.map((feature, idx) => (
                                    feature ? (
                                        <li key={idx} className="flex items-start gap-3 text-gray-300">
                                            <span className="text-cyan-400 mt-1">•</span>
                                            <span className="text-sm">{feature}</span>
                                        </li>
                                    ) : (
                                        <div key={idx} className="h-2" />
                                    )
                                ))}
                            </ul>
                        </div>

                        {/* Getting Started hint */}
                        {activeSection === 'overview' && (
                            <div className="mt-6 bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-4">
                                <p className="text-sm text-cyan-300">
                                    💡 <strong>Pro Tip:</strong> Start with Alpha Stock Picker to discover emerging opportunities,
                                    then use Crisis Simulator to test portfolio resilience.
                                </p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Footer */}
                <div className="border-t border-gray-700 p-4 bg-gray-800/50">
                    <p className="text-xs text-gray-500 text-center">
                        Need help? Contact support or visit our documentation at{' '}
                        <a href="https://caria.ai/docs" className="text-cyan-400 hover:underline">
                            caria.ai/docs
                        </a>
                    </p>
                </div>
            </div>
        </div>
    );
}

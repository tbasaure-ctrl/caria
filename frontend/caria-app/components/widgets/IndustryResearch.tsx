import React, { useState } from 'react';
import { WidgetCard } from './WidgetCard';

// Interfaces for structured report
interface StockPick {
    ticker: string;
    name: string;
    thesis: string;
    type: 'Value' | 'Growth' | 'Turnaround' | 'Defensive' | 'Speculative';
}

interface IndustryReport {
    id: string;
    title: string;
    subtitle: string;
    icon: string;
    readTime: string;
    tags: string[];
    isFeatured?: boolean;
    content: {
        overview: string;
        fullText: string;
        trends: { title: string; description: string }[];
        picks: StockPick[];
        conclusion?: string;
    };
}

// DATA: Content Translated to Professional English
const REPORT_DATA: IndustryReport[] = [
    {
        id: 'staples-nov-2025',
        title: 'Consumer Staples',
        subtitle: 'INDUSTRY OF THE MONTH: Tactical Value & Safety',
        icon: '🛒',
        readTime: '8 min read',
        tags: ['Industry of the Month', 'Defensive', 'High Conviction'],
        isFeatured: true,
        content: {
            overview: `Designated as the focal industry for November 2025. Historically, this sector acts as a "bond proxy" with dividend growth upside. Amid economic uncertainty, investors are rotating into the safety of inelastic demand.`,
            fullText: `1. Industry of the Month: Consumer Staples

1.1 Investment Thesis & Selection Rationale
The designation of Consumer Staples as the focal industry for November 2025 responds to a confluence of technical, fundamental, and seasonal factors. Often misunderstood as a boring haven, the sector is undergoing an internal transformation and valuation dispersion that offers significant alpha opportunities.

1.1.1 The Defensive Rotation
Through 2024, capital flowed disproportionately into technology. However, as valuations stretched, a classic rotation towards defense has been observed.
The mechanism is twofold: First, compressing Treasury yields make staples' dividends comparatively more attractive. Second, amidst economic uncertainty, investors seek the safety of inelastic demand—people buy toothpaste and food regardless of GDP.

1.1.2 Historical Seasonality: The "November Effect"
Quantitative analysis reveals November is statistically an exceptionally strong month for Consumer Staples (XLP).
November shows a robust continuation of Q4 momentum with a 75% historical win rate over the last 25 years, often attributed to fund manager positioning before fiscal year-end.

1.2 Deep Fundamental Analysis
The sector is not monolithic. A critical divergence exists between retailers and packaged goods manufacturers.

1.2.1 Valuation Bifurcation: Retailers vs. Manufacturers
Overvalued (Retailers): Costco (COST) and Walmart (WMT) trade at tech-like multiples (>40x P/E), discounting a perfect execution scenario that is hard to justify.
Undervalued (Packaged Food): Conversely, excluding retailers, the sector trades at an attractive discount (~11% below fair value). Solid firms like Kraft Heinz (KHC) and General Mills (GIS) have been excessively penalized.

1.2.2 The GLP-1 Impact: Reality vs. Hysteria
The fear that weight-loss drugs would decimate snack volumes has nuanced. Majors like Nestlé and General Mills are pivoting, launching high-protein products designed for GLP-1 users, stabilizing volumes.

1.2.3 Margin Compression & Private Label War
Inflation has led to trade-downs. Companies with strong pricing power have maintained gross margins through operational efficiencies, beating EPS estimates despite modest top-line growth.`,
            trends: [
                {
                    title: "The November Effect",
                    description: "Statistically, November is exceptionally strong for the sector (75% historical win rate)."
                },
                {
                    title: "GLP-1 Adaptation",
                    description: "Launch of high-protein products to accompany Ozempic/Wegovy users."
                }
            ],
            picks: [
                { ticker: 'KHC', name: 'Kraft Heinz', type: 'Value', thesis: 'Extreme undervaluation. Market ignoring successful debt restructuring and margin improvement.' },
                { ticker: 'GIS', name: 'General Mills', type: 'Defensive', thesis: 'Classic defensive player. Superior adaptation to health trends (Blue Buffalo).' },
                { ticker: 'SFM', name: 'Sprouts Farmers Market', type: 'Growth', thesis: 'Beneficiary of healthy eating/GLP-1 boom. Margin expansion with fresh produce.' },
                { ticker: 'OLLI', name: "Ollie's Bargain Outlet", type: 'Growth', thesis: '"Treasure hunt" model ideal for price-sensitive consumers. Inventory acquisition upside.' },
                { ticker: 'EL', name: 'Estée Lauder', type: 'Turnaround', thesis: 'Depressed valuation due to Asia weakness. Potential violent rebound if inventory stabilizes.' }
            ]
        }
    },
    {
        id: 'macro-nov-2025',
        title: 'Global Macro Strategy',
        subtitle: 'Economic Outlook & Asset Allocation',
        icon: '🌍',
        readTime: '5 min read',
        tags: ['Macro', 'Strategy'],
        content: {
            overview: `Markets are entering a tactical rotation phase. With the Fed adjusting rates to 3.75%-4.00%, capital is moving from "growth at any price" toward quality, balance sheet strength, and cash flow predictability.`,
            fullText: `Global Investment Strategy Report - November 2025

The penultimate month of 2025 unfolds in an economic context defying simple "soft landing" or "recession" labels. Financial markets, after a year marked by tech euphoria, have entered a distinctive tactical rotation phase. As the Federal Reserve adjusts rates to the 3.75%-4.00% range, investors are re-evaluating risk premiums.

The prevailing narrative has shifted from "growth at any price" to a renewed appreciation for balance sheet quality and cash flow predictability. This sentiment shift is a rational response to an environment where, while inflation has cooled, borrowing costs remain high, penalizing leverage.

November 2025 emerges as a critical inflection point. Historically a barometer for year-end positioning, current data suggests a clear bifurcation: while cyclical sectors face headwinds from projected 2026 slowing, defensive sectors and healthcare innovation are capturing institutional capital.`,
            trends: [
                { title: "Flight to Quality", description: "Preference for predictable cash flows and solid balance sheets over speculative growth." },
                { title: "Sector Bifurcation", description: "Defensive and Healthcare sectors capturing institutional flows vs. Cyclicals." }
            ],
            picks: [],
            conclusion: "Final Recommendation: Build a 'barbell' portfolio: a robust defensive core in staples and niche insurance, balanced with high-growth satellite bets in medical robotics and biotech with near-term catalysts."
        }
    },
    {
        id: 'pharma-nov-2025',
        title: 'Healthcare & Pharma',
        subtitle: 'Innovation & M&A Boom',
        icon: '🧬',
        readTime: '4 min read',
        tags: ['Biotech', 'M&A'],
        content: {
            overview: `Ecosystem under pressure from patent cliffs and regulation, catalyzing rampant innovation and M&A. Big Pharma is deploying balance sheets to buy growth (Oncology, Neuro, Obesity).`,
            fullText: `2. Healthcare & Pharma: Innovation Under Pressure

2.1 Industry Status: An Ecosystem in Tension
The sector presents a fascinating dichotomy. It faces significant regulatory headwinds and patent cliffs, yet this pressure is acting as a catalyst for rampant innovation and aggressive consolidation.

2.2 Dominant Trends
2.2.1 The M&A Renaissance
Facing loss of exclusivity on blockbusters, Big Pharma is deploying balance sheets to buy growth. 2025 has seen a wave of strategic deals (Merck, Sanofi, Novartis, Lilly), validating that valuable innovation is happening in the mid-cap biotech ecosystem.

2.2.2 High-Value Therapeutic Areas
Investment is concentrated where science is breaking barriers:
- Oncology: ADCs and T-cell engagers.
- Neuroscience: Renaissance in Alzheimer's and Schizophrenia treatments.
- Obesity & Metabolism: Next-gen metabolic treatments (better tolerability/muscle preservation).

2.3 Investment Opportunities
Outlook for late 2025 is continued volatility but with asymmetric opportunities in biotech. Look for companies with binary catalysts (clinical data readouts).`,
            trends: [
                { title: "M&A Renaissance", description: "Big Pharma deploying capital to acquire external innovation." },
                { title: "Hot Areas", description: "Oncology, Neuroscience, and Metabolism (Next-gen Obesity)." }
            ],
            picks: [
                { ticker: 'KALA', name: 'Kala Bio', type: 'Speculative', thesis: 'Binary catalyst end of 2025 (Phase 2b CHASE). Rare ocular disease with no cure.' },
                { ticker: 'KAPA', name: 'Kairos Pharma', type: 'Speculative', thesis: 'Phase 2 interim data in prostate cancer. Lucrative oncology niche.' }
            ]
        }
    },
    {
        id: 'medtech-nov-2025',
        title: 'Medical Devices',
        subtitle: 'The Silent Revolution: Robotics & AI',
        icon: '🦾',
        readTime: '4 min read',
        tags: ['Growth', 'Tech'],
        content: {
            overview: `Predictable structural growth (6% CAGR). AI operational in diagnostics and surge in surgical robotics and single-use devices driving hospital efficiency.`,
            fullText: `3. Medical Devices: The Silent Health-Tech Revolution

3.1 Characterization: Structural Growth & Resilience
Unlike binary biotech, MedTech offers predictable growth driven by demographics (aging) and hospital efficiency needs. Global market projected to reach $678.8B in 2025.

3.2 Tech Trends
3.2.1 AI & Surgical Robotics
AI has moved from promise to reality (e.g., pathology). In the OR, robotics allow minimally invasive procedures reducing hospital stays—critical for efficiency.

3.2.2 The Rise of Single-Use Devices
Massive shift towards replacing reusable instruments with single-use devices to eliminate cross-contamination and sterilization costs.

3.3 High-Growth Opportunities
We look for companies redefining the "Standard of Care".`,
            trends: [
                { title: "Surgical Robotics", description: "Minimally invasive procedures reducing hospital stay duration." },
                { title: "Single-Use Devices", description: "Eliminating contamination risks and sterilization costs." }
            ],
            picks: [
                { ticker: 'TMDX', name: 'TransMedics Group', type: 'Growth', thesis: 'OCS system keeps organs alive. Creating its own market (transplant logistics).' },
                { ticker: 'PRCT', name: 'PROCEPT BioRobotics', type: 'Growth', thesis: 'Robotics in Urology (Aquablation). 43% YoY revenue growth.' },
                { ticker: 'DCTH', name: 'Delcath Systems', type: 'Speculative', thesis: 'Interventional oncology (liver). Unique technological approach.' }
            ]
        }
    },
    {
        id: 'insurance-nov-2025',
        title: 'Insurance & Insurtech',
        subtitle: 'Efficiency, AI & Profitable Niches',
        icon: '🛡️',
        readTime: '3 min read',
        tags: ['Financials', 'AI'],
        content: {
            overview: `Forced modernization due to rising costs. Key: Niche specialists (E&S) and Insurtech 2.0. Generative AI reducing claims processing time by 80%.`,
            fullText: `4. Insurance & Managed Care: Efficiency, AI & Profitable Niches

4.1 Sector Overview: Forced Modernization
The insurance sector is undergoing a quiet revolution driven by necessity. Rising claims costs (social inflation, climate) force modernization. 2025 is mixed: generalists struggle, while niche specialists and "Insurtech 2.0" thrive.

4.2 Transformative Trends
4.2.1 Generative AI in Claims Processing
Operational AI is the biggest trend. Full automation of claims handling (80% time reduction, 30% cost reduction) and real-time fraud detection.

4.2.2 The Rise of E&S (Excess & Surplus)
As climate risks make certain regions "uninsurable" for standard carriers, the E&S market explodes. They have pricing freedom to take on complex risks profitably.

4.3 Investment Opportunities
Avoid property insurers exposed to catastrophes without pricing power. Seek specialists.`,
            trends: [
                { title: "Operational AI", description: "Drastic reduction in claims time and fraud detection." },
                { title: "E&S Market", description: "Growth in surplus lines due to complex risk pricing power." }
            ],
            picks: [
                { ticker: 'SKWD', name: 'Skyward Specialty', type: 'Growth', thesis: 'King of Niche E&S. 26% annual premium growth.' },
                { ticker: 'PRI', name: 'Primerica', type: 'Defensive', thesis: 'Efficient distribution model. Industry-leading 27.2% ROE. Cash flow machine.' },
                { ticker: 'CB', name: 'Chubb', type: 'Value', thesis: 'The Gold Standard. Legendary underwriting discipline and global balance sheet.' }
            ]
        }
    }
];

// Modal Component (Keeping structure, updating content rendering for English)
const ReportModal: React.FC<{ report: IndustryReport; onClose: () => void }> = ({ report, onClose }) => {
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-sm animate-fade-in">
            <div 
                className="w-full max-w-4xl max-h-[90vh] overflow-y-auto bg-[#050A14] border border-accent-gold/30 rounded-xl shadow-2xl custom-scrollbar"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Report Header */}
                <div className="sticky top-0 z-10 bg-[#050A14]/95 backdrop-blur border-b border-white/10 px-8 py-6 flex justify-between items-start">
                    <div>
                        <div className="flex items-center gap-3 mb-2">
                            <span className="text-3xl">{report.icon}</span>
                            <h2 className="text-2xl md:text-3xl font-display text-white tracking-wide">
                                {report.title}
                            </h2>
                        </div>
                        <p className="text-accent-gold font-medium text-sm uppercase tracking-widest">
                            {report.subtitle}
                        </p>
                    </div>
                    <button 
                        onClick={onClose}
                        className="p-2 rounded-full hover:bg-white/10 text-text-muted hover:text-white transition-colors"
                    >
                        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {/* Report Content */}
                <div className="p-8 space-y-10">
                    {/* Full Text */}
                    <div className="prose prose-invert max-w-none">
                        <p className="text-text-secondary text-lg leading-relaxed whitespace-pre-line font-serif">
                            {report.content.fullText}
                        </p>
                    </div>

                    {/* Key Trends */}
                    {report.content.trends.length > 0 && (
                        <div className="grid md:grid-cols-2 gap-6">
                            {report.content.trends.map((trend, idx) => (
                                <div key={idx} className="bg-bg-tertiary/50 p-6 rounded-lg border border-white/5 hover:border-accent-cyan/30 transition-colors">
                                    <h4 className="text-accent-cyan font-bold text-xs uppercase tracking-wider mb-3 flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-accent-cyan"></span>
                                        Trend {idx + 1}
                                    </h4>
                                    <h3 className="text-white font-display text-xl mb-2">{trend.title}</h3>
                                    <p className="text-sm text-text-muted leading-relaxed">{trend.description}</p>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Stock Picks */}
                    {report.content.picks.length > 0 && (
                        <div className="bg-white/5 rounded-xl p-8 border border-white/10">
                            <div className="flex items-center gap-4 mb-8">
                                <div className="h-px flex-1 bg-white/10"></div>
                                <span className="text-accent-gold font-display text-2xl">Top Picks & Thesis</span>
                                <div className="h-px flex-1 bg-white/10"></div>
                            </div>

                            <div className="space-y-6">
                                {report.content.picks.map((pick) => (
                                    <div 
                                        key={pick.ticker} 
                                        className="group relative overflow-hidden rounded-lg bg-[#0B1221] border border-white/10 hover:border-accent-gold/50 transition-all duration-300 p-6"
                                    >
                                        <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 mb-4">
                                            <div className="flex items-center gap-4">
                                                <span className="text-3xl font-display font-bold text-white group-hover:text-accent-gold transition-colors tracking-tight">
                                                    {pick.ticker}
                                                </span>
                                                <div className="flex flex-col">
                                                    <span className="text-sm font-medium text-text-primary">{pick.name}</span>
                                                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider w-fit mt-1 ${
                                                        pick.type === 'Value' ? 'bg-blue-500/20 text-blue-400' :
                                                        pick.type === 'Growth' ? 'bg-green-500/20 text-green-400' :
                                                        pick.type === 'Turnaround' ? 'bg-orange-500/20 text-orange-400' :
                                                        pick.type === 'Speculative' ? 'bg-purple-500/20 text-purple-400' :
                                                        'bg-gray-500/20 text-gray-400'
                                                    }`}>
                                                        {pick.type}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                        <p className="text-sm text-text-secondary leading-relaxed border-l-2 border-white/10 pl-4">
                                            {pick.thesis}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Conclusion */}
                    {report.content.conclusion && (
                        <div className="bg-accent-gold/5 border border-accent-gold/20 rounded-lg p-8 text-center">
                            <p className="text-accent-gold font-medium italic font-display text-xl leading-relaxed">
                                "{report.content.conclusion}"
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export const IndustryResearch: React.FC = () => {
    const [selectedReport, setSelectedReport] = useState<IndustryReport | null>(null);

    const featuredReport = REPORT_DATA.find(r => r.isFeatured);
    const otherReports = REPORT_DATA.filter(r => !r.isFeatured);

    return (
        <WidgetCard 
            title="Industry Research" 
            tooltip="Deep dive analysis into sectors with high alpha potential. Updated monthly."
        >
            <div className="space-y-6">
                <div className="flex justify-between items-end">
                    <h4 className="text-xs text-text-muted uppercase tracking-widest">November 2025 Edition</h4>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-accent-primary/10 text-accent-primary font-medium">
                        Strategy Report
                    </span>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
                    {/* FEATURED CARD (Left/Top - Large) */}
                    {featuredReport && (
                        <div 
                            onClick={() => setSelectedReport(featuredReport)}
                            className="lg:col-span-2 group cursor-pointer rounded-xl p-6 bg-gradient-to-br from-bg-tertiary to-[#0F1623] border border-white/10 hover:border-accent-gold/40 transition-all duration-300 relative overflow-hidden min-h-[200px] flex flex-col justify-between"
                        >
                            <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                                <span className="text-8xl">{featuredReport.icon}</span>
                            </div>
                            <div>
                                <div className="flex items-center gap-2 mb-3">
                                    <span className="text-xs font-bold bg-accent-gold/20 text-accent-gold px-2 py-1 rounded uppercase tracking-wider">
                                        Industry of the Month
                                    </span>
                                    <span className="text-[10px] text-text-muted flex items-center gap-1">
                                        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                                        {featuredReport.readTime}
                                    </span>
                                </div>
                                <h3 className="text-2xl font-display font-bold text-white mb-2 group-hover:text-accent-gold transition-colors">
                                    {featuredReport.title}
                                </h3>
                                <p className="text-sm text-text-secondary leading-relaxed max-w-md">
                                    {featuredReport.content.overview}
                                </p>
                            </div>
                            <div className="mt-6 flex items-center text-xs font-bold text-accent-gold uppercase tracking-wider">
                                Read Full Analysis <span className="ml-2 group-hover:translate-x-1 transition-transform">→</span>
                            </div>
                        </div>
                    )}

                    {/* OTHER REPORTS GRID */}
                    {otherReports.map((report) => (
                        <div 
                            key={report.id}
                            onClick={() => setSelectedReport(report)}
                            className="group cursor-pointer rounded-lg p-5 bg-bg-tertiary border border-white/5 hover:border-accent-cyan/30 hover:bg-white/5 transition-all duration-300 flex flex-col h-full"
                        >
                            <div className="flex justify-between items-start mb-3">
                                <div className="w-10 h-10 rounded-full bg-bg-primary flex items-center justify-center text-xl border border-white/10 group-hover:border-accent-cyan/50 transition-colors">
                                    {report.icon}
                                </div>
                                <span className="text-[10px] text-text-subtle">{report.readTime}</span>
                            </div>
                            
                            <h3 className="text-sm font-bold text-white group-hover:text-accent-cyan transition-colors font-display tracking-wide mb-1">
                                {report.title}
                            </h3>
                            <p className="text-xs text-text-muted line-clamp-2 mb-3 flex-grow">
                                {report.subtitle}
                            </p>
                            
                            <div className="flex gap-1 mt-auto">
                                {report.tags.slice(0, 1).map(tag => (
                                    <span key={tag} className="text-[9px] px-1.5 py-0.5 rounded bg-white/5 text-text-muted border border-white/5">
                                        {tag}
                                    </span>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {selectedReport && (
                <ReportModal 
                    report={selectedReport} 
                    onClose={() => setSelectedReport(null)} 
                />
            )}
        </WidgetCard>
    );
};

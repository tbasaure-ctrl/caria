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
        id: 'staples-dec-2025',
        title: 'Consumer Staples',
        subtitle: 'INDUSTRY OF THE MONTH: The Strategic Rotation to Resilience',
        icon: '🛒',
        readTime: '8 min read',
        tags: ['Industry of the Month', 'Defensive', 'High Conviction'],
        isFeatured: true,
        content: {
            overview: `Designated as the focal industry for December 2025. Amidst a "Great Rotation" out of overextended tech momentum, capital is seeking the safety of inelastic demand. Historically the best-performing sector during volatility.`,
            fullText: `1. Industry of the Month: Consumer Staples - The Strategic Rotation to Resilience

1.1 Executive Summary: Navigating the Anti-Momentum Pivot
As the global financial markets pivot into the final trading month of 2025, a profound regime change is underway. The aggressive "growth-at-any-cost" momentum is yielding to a disciplined, valuation-sensitive environment. Institutional capital is seeking refuge in sectors offering earnings visibility and balance sheet fortitude.

The December 2025 Industry Report identifies a "Great Rotation" out of overextended technology momentum plays and into the bedrock of the economy: Consumer Staples, Healthcare & Pharma, Medical Devices, and Insurance & Managed Care.

1.2 The Overarching Theme: "Rational Exuberance"
While indices remain near highs, internal breadth shows bifurcation. Investors grapple with a divided Fed, geopolitical friction ("Liberation Day" tariffs), and a resilient but value-conscious consumer. In this climate, the allure of speculative growth diminishes in favor of industries providing essential services—sectors where demand is inelastic.

1.3 Why Consumer Staples?
Selected as the "Industry of the Month" for its unmatched historical resilience.
- **Recession Performance:** In the last seven recessions, Consumer Staples was the #1 performing sector (+1% avg return vs double-digit declines elsewhere).
- **Long-Term Compounder:** Over 20 years, staples often outperform high-growth tech due to reinvested dividends and lower volatility.

2. Trends Defining the Sector in December 2025

2.1 The Rise of the "Value-Seeking" Consumer
High interest rates and inflation have eroded excess savings. Shoppers are focusing on value and trimming seasonal extras. This favors giants with scale (Costco, Walmart) and private label manufacturers. The "trade-down" effect accelerates market share gains for efficient operators.

2.2 Cost Input Stabilization vs. Tariff Threats
Input costs (commodities, energy) have stabilized, allowing margin expansion as 2024 price hikes stick. However, new tariffs could inflate packaging costs. Companies with domestic sourcing networks trade at a premium due to "supply chain security."

2.3 The "Mobile-First" Grocery Shift
Digital transformation has permeated the sector. Mobile devices will account for >50% of online holiday spending. Retailers optimizing apps for reordering and loyalty are seeing higher retention.

3. Outlook for 2026: Stability in a Volatile World
- **Earnings Visibility:** Tied to population growth, not ad spending cycles.
- **Dividend Yield:** Attractive 3-4% yields as Treasury yields cap out.
- **M&A Potential:** Consolidation in "Better-for-You" food space.`,
            trends: [
                {
                    title: "Value-Seeking Consumer",
                    description: "Accelerated trade-down to private labels and value retailers due to eroded savings."
                },
                {
                    title: "Domestic Supply Chain",
                    description: "Premium on companies insulated from 'Liberation Day' tariffs and import costs."
                }
            ],
            picks: [
                { ticker: 'CALM', name: 'Cal-Maine Foods', type: 'Value', thesis: 'Pure play on protein inelasticity. Small-cap value with pristine balance sheet. Holiday baking demand catalyst.' },
                { ticker: 'COST', name: 'Costco Wholesale', type: 'Defensive', thesis: 'Primary conduit for staples consumption. Wins in a value-seeking environment. Better risk-adjusted return than Nvidia.' },
                { ticker: 'WMT', name: 'Walmart', type: 'Defensive', thesis: 'Scale winner. Assigning higher quality premium to earnings safety than AI growth.' },
                { ticker: 'THS', name: 'TreeHouse Foods', type: 'Growth', thesis: 'Private label manufacturer leader. Capitalizing on the trade-down phenomenon without brand marketing overhead.' }
            ]
        }
    },
    {
        id: 'macro-dec-2025',
        title: 'Global Macro Strategy',
        subtitle: 'Rational Exuberance & Anti-Momentum',
        icon: '🌍',
        readTime: '6 min read',
        tags: ['Macro', 'Strategy'],
        content: {
            overview: `Theme: "Rational Exuberance". Markets are bifurcated. Fed is divided. Strategy: "Anti-Momentum" posture for 2026, favoring value and domestic resilience against tariffs.`,
            fullText: `Global Investment Strategy Report - December 2025

The overarching theme is "Rational Exuberance". While indices are high, internal breadth tells a story of bifurcation.

The December Catalyst: Federal Reserve & Seasonality
The immediate catalyst is the FOMC meeting (Dec 9-10). A "hawkish pause" could trigger rapid repricing.
- **Seasonality:** The "Santa Claus Rally" (last 5 days Dec + first 2 Jan) has a 79% win rate.
- **Tax-Loss Harvesting:** Early December selling creates artificial price dislocations in small-cap value (Insurance/Biotech), setting the stage for the "January Effect".

The "Anti-Momentum" Market Regime
Strategists advise an "Anti-Momentum" posture. The "Magnificent Seven" trade at elevated multiples (22.5x), while the "S&P 493" and small-caps offer attractive discounts. We favor industries monetizing immediate human needs over future tech promises.

Tariffs and the Real Economy
"Liberation Day" tariffs suggest a shift to higher import costs. Domestic-focused insurers, healthcare, and staples are relative beneficiaries.`,
            trends: [
                { title: "Anti-Momentum", description: "Rotation from crowded AI trade to low P/E, high yield defensive sectors." },
                { title: "January Effect Setup", description: "Harvesting tax losses in early Dec to buy small-cap value for Jan rebound." }
            ],
            picks: [],
            conclusion: "Strategic Action: Harvest losses in early Dec. Aggressively accumulate small-cap value (Biotech/Insurance) in final weeks. Anchor with Staples to reduce beta."
        }
    },
    {
        id: 'pharma-dec-2025',
        title: 'Healthcare & Pharma',
        subtitle: 'M&A Supercycle & Innovation',
        icon: '🧬',
        readTime: '5 min read',
        tags: ['Biotech', 'M&A'],
        content: {
            overview: `Sector is "Under Pressure" but poised for reversal. Catalyst: J.P. Morgan Healthcare Conference (Jan 2026). Big Pharma must deploy $1.5T dry powder to address Patent Cliff.`,
            fullText: `2. Healthcare & Pharma: Innovation Under Pressure

2.1 Industry Status: Contrarian Setup
Currently "Under Pressure" due to regulatory concerns, but revenue growth is re-accelerating. The primary driver is the lead-up to the J.P. Morgan Healthcare Conference (Jan 12-15, 2026).

2.2 Key Trends
- **The $200 Billion Patent Cliff:** By 2030, $200B/year revenue is at risk. Big Pharma has $1.5 Trillion in dry powder and MUST acquire growth. M&A is a mathematical necessity.
- **Obesity & Metabolic:** Focus shifting to "Next-Gen" MASH and muscle-preserving therapies. Halo effect on metabolic pipelines.
- **Neuroscience Renaissance:** New modalities for Alzheimer's and CNS disorders (e.g., Anavex) are opening up the "undruggable" brain.`,
            trends: [
                { title: "Patent Cliff Panic", description: "$200B revenue at risk forcing aggressive M&A before 2030." },
                { title: "JPM Week Run-up", description: "Speculation and deal announcements peak before the Jan 12 conference." }
            ],
            picks: [
                { ticker: 'IMVT', name: 'Immunovant', type: 'Speculative', thesis: 'Catalyst-rich 2026. "Pipeline-in-a-product" for autoimmune. Prime takeover target.' },
                { ticker: 'XBI', name: 'SPDR Biotech ETF', type: 'Growth', thesis: 'Basket play for "January Effect" and M&A lottery tickets. Trading at historical lows.' },
                { ticker: 'PRAX', name: 'Praxis Precision', type: 'Speculative', thesis: 'Genetic epilepsies. Direct catalysts in next 12 months. Precision medicine approach.' }
            ]
        }
    },
    {
        id: 'medtech-dec-2025',
        title: 'Medical Devices',
        subtitle: 'AI Integration & Procedure Recovery',
        icon: '🦾',
        readTime: '4 min read',
        tags: ['Growth', 'Tech'],
        content: {
            overview: `Convergence of aging demographics and AI. "Personalized Medicine" market growing to $206B. Elective procedure backlog driving volume acceleration.`,
            fullText: `3. Medical Devices: The Intersection of Tech and Health

3.1 Characterization: Accelerating Growth
Revenue growth is accelerating despite stock underperformance. Driven by aging demographics and AI integration into hardware.

3.2 Key Trends
- **Intelligence at the Edge:** Devices becoming connected sensors feeding AI models. Personalized medicine market to reach $206B.
- **Elective Procedure Supercycle:** Post-pandemic backlog (hips, cataracts, cardiac) driving volume.
- **"Pick and Shovel":** Life Sciences Tools providing equipment for biologic drug manufacturing.`,
            trends: [
                { title: "Personalized Medicine", description: "AI-driven diagnostics and therapy matching. Devices as data platforms." },
                { title: "Procedure Supercycle", description: "Sustained volume growth from post-pandemic elective surgery backlog." }
            ],
            picks: [
                { ticker: 'DHR', name: 'Danaher Corp', type: 'Growth', thesis: '"Arms dealer" to biopharma. Wide-moat compounder. Destocking ending in late 2025.' },
                { ticker: 'AXGN', name: 'Axogen', type: 'Speculative', thesis: 'Nerve repair niche dominance. Acquisition target for Stryker/J&J.' },
                { ticker: 'TMO', name: 'Thermo Fisher', type: 'Defensive', thesis: 'Scale behemoth. Recurring revenue safety net with aggressive capital deployment.' }
            ]
        }
    },
    {
        id: 'insurance-dec-2025',
        title: 'Insurance & Managed Care',
        subtitle: 'The "Hard Market" & Rates Play',
        icon: '🛡️',
        readTime: '4 min read',
        tags: ['Financials', 'Value'],
        content: {
            overview: `The "Sleeper" hit of late 2025. "Hard Market" pricing power + "Higher-for-Longer" investment income. Unique beneficiaries of the Anti-Momentum rotation.`,
            fullText: `4. Insurance & Managed Care: The "Hard Market" Cycle

4.1 Sector Overview: The Sleeper Hit
Experiencing accelerating revenue due to a "Hard Market" (rising premiums). Insurers are passing inflation costs to consumers.

4.2 Key Trends
- **Pricing Power:** MCOs repricing for higher utilization. P&C insurers raising rates for climate risks.
- **Investment Income:** "Higher-for-Longer" rates mean insurers earn massive yield on their float.
- **Asset Quality:** Financials seeing improved margins; feared credit defaults contained.`,
            trends: [
                { title: "Hard Market", description: "Aggressive premium hikes passing inflation costs to consumers." },
                { title: "Float Yield", description: "High interest rates boosting investment income component of earnings." }
            ],
            picks: [
                { ticker: 'UNH', name: 'UnitedHealth', type: 'Value', thesis: '"Dog of the Dow" reversal play. Massive valuation gap. Vertical integration advantage.' },
                { ticker: 'UVE', name: 'Universal Insurance', type: 'Speculative', thesis: 'Florida niche. Deep value (7x P/E). Cash machine if no Q4 hurricane.' },
                { ticker: 'IYF', name: 'US Financials ETF', type: 'Defensive', thesis: 'Broad exposure to sector alpha. Betting on resilient US consumer borrower.' }
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
                                                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider w-fit mt-1 ${pick.type === 'Value' ? 'bg-blue-500/20 text-blue-400' :
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

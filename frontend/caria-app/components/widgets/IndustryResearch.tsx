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
        id: 'macro-outlook-2026',
        title: '2026 Macro Outlook',
        subtitle: 'INDUSTRY OF THE MONTH: Navigating a Transforming Global Economy',
        icon: '🌐',
        readTime: '12 min read',
        tags: ['Industry of the Month', 'Macro', 'High Conviction'],
        isFeatured: true,
        content: {
            overview: `A comprehensive synthesis of leading economic forecasts for 2026. Drawing insights from major institutions including S&P Global, Deloitte, J.P. Morgan, Invesco, and Allianz Research, this report identifies consensus views, original insights, and key market outlooks across all asset classes.`,
            fullText: `2026 Macro Outlook: Navigating a Transforming Global Economy

Executive Summary: The Consensus View

After synthesizing over a dozen leading economic forecasts from institutions including S&P Global Ratings, Deloitte, J.P. Morgan Asset Management, Invesco, and Allianz Research, a clear picture emerges for 2026: a year of continued resilience marked by moderate growth, persistent but moderating inflation, and a gradual shift toward less restrictive monetary policy. The global economy is expected to demonstrate stability, though regional divergence will persist, with Asia-Pacific and emerging markets showing particular strength while developed economies navigate policy transitions.

Key Consensus Points (Supported by Multiple Sources)

1. **Moderate Global Growth with Regional Divergence**
   Multiple sources converge on a baseline of steady, if unspectacular, global expansion. S&P Global forecasts U.S. GDP growth of approximately 1.4-1.6% in 2026, while Asia-Pacific economies are expected to grow at 4.2-4.4%, driven by resilient domestic demand and tech export strength.¹ Europe is projected to see stable growth around 0.9-1.0%, with Germany's fiscal reawakening providing regional support.² The consensus view suggests that while growth will not match 2025's pace, a recession remains unlikely in most major economies.

2. **Inflation: Stuck Above Target, But Directionally Improving**
   A near-universal finding across reports is that inflation will remain above central bank targets (2%) but show directional improvement. Core PCE inflation is forecast to decline from current levels around 3% to approximately 2.6% by year-end 2026.³ The tug-of-war between slowing services prices and tariff-induced goods price increases is expected to gradually resolve as trade policy stabilizes. This "sticky but improving" inflation narrative supports a shallow Fed easing cycle rather than aggressive rate cuts.

3. **Monetary Policy: Shallow Easing Path**
   There is broad agreement that the Federal Reserve will implement 2-3 rate cuts totaling 50-75 basis points through 2026, bringing the terminal fed funds rate to approximately 3.00-3.25%.⁴ The Fed's cautious approach reflects balanced risks: cutting too aggressively could rekindle inflation, while staying too restrictive risks unnecessary economic slowdown. International central banks face limited room for further cuts, with many rates approaching neutral levels.⁵

4. **Trade Policy: Peak Tariff Impact Behind Us**
   Multiple sources identify 2025 as the peak year for tariff escalation, with 2026 expected to show stabilization or modest improvement. While tariff rates will not return to 2024 levels, the effective U.S. tariff rate is forecast to stabilize around 11-14%, down from peak escalation.⁶ Trade rerouting through third countries (India, Vietnam) has mitigated initial impacts, though export-dependent economies (Canada, France, Spain, Netherlands) remain vulnerable.⁷

5. **AI-Driven Growth: Real but Valuation Concerns**
   Artificial intelligence continues to be identified as a primary growth driver across multiple reports, particularly in business investment and productivity gains. However, there is growing consensus that AI-related valuations appear stretched, with concerns about potential bubble formation similar to the dot-com era.⁸ The key distinction: current AI investments are underpinned by actual infrastructure spending (data centers, semiconductor manufacturing) rather than pure speculation, suggesting a more durable foundation than previous tech booms.

Original Insights Supported by Data

**Insolvency Trends: The Hidden Risk**
Allianz Research provides original analysis showing global business insolvencies rising +5% in 2026, marking five consecutive years of increases to reach record highs (+24% above pre-pandemic average).⁹ The U.S. and China are expected to drive the bulk of this increase (+8% and +10% respectively), while Western Europe may see modest declines (-2%). This divergence reflects varying economic resilience and financial conditions. The report identifies a concerning scenario: if AI-induced boom were to burst similar to the 2001-2002 dot-com bubble, bankruptcies could surge by +4,500 companies in the U.S. alone.¹⁰

**Banking Sector Transformation**
Deloitte's analysis reveals that 2026 will be a defining year for banks, balancing macro headwinds, AI industrialization, and the disruptive entrance of stablecoins.¹¹ Banks face pressure to scale AI beyond pilots while maintaining robust data infrastructure. The stablecoin legislation (GENIUS Act) could reshape deposit flows and payment rails, forcing banks to decide quickly whether to issue, custody, process, or partner. Financial crime risks are escalating, fueled by AI-enabled fraud, requiring integrated tech-driven defenses.

**FX Markets: Return to Fundamentals**
Multiple FX outlooks converge on a "play the ball, not the man" theme for 2026—markets refocusing on fundamentals (rate differentials, growth trajectories, debt sustainability) rather than political rhetoric.¹² The U.S. dollar is expected to weaken further, particularly against Asian currencies benefiting from tech export strength and improved trade relations. This dollar weakness supports international equity returns and commodity prices.

Market Outlooks by Asset Class

**Equities: Quality and International Focus**
The consensus favors international equities over U.S. markets, driven by narrowing earnings growth gaps, favorable valuations, and dollar weakness. Asian emerging markets are particularly favored, benefiting from AI-related tech exports and resilient domestic demand.¹³ European value stocks and Japanese equities also receive support. Within U.S. equities, quality remains paramount—focus on secular themes (AI ecosystem broadening, financial deregulation) rather than cyclical plays. Valuations are elevated but supported by solid fundamentals in quality names.

**Fixed Income: Embrace Income, Active Selection**
Long-term rates are expected to remain range-bound with modest curve steepening. The consensus view emphasizes embracing income in fixed income rather than trying to perfectly time yield levels. Active security selection is recommended across credit, securitized assets, global bonds, and municipals. High-yield spreads are tight but may remain so given resilient fundamentals. Investment-grade credit offers similar profile to government bonds—modest returns with lower volatility.

**Commodities: Beneficiaries of Global Acceleration**
Commodities are expected to benefit as the global economy improves, with industrial commodities particularly favored.¹⁴ Real estate (REITs) may benefit as rates fall and economies accelerate. Bank loans offer attractive risk-reward trade-offs given floating rate structures. Gold faces mixed signals: soft dollar helps, but improving geopolitics and expensive valuations limit upside.

**Private Markets: Evolving Landscape**
Private markets are becoming more transparent, accessible, and integrated into traditional portfolio models.¹⁵ The asset class offers thematic exposures (AI infrastructure, real estate) and diversification benefits for concentrated portfolios. However, investors must navigate increased complexity and ensure proper due diligence as the asset class democratizes.

**Foreign Exchange: Dollar Weakness Theme**
The U.S. dollar is expected to weaken further in 2026, driven by Fed easing, improving international growth prospects, and narrowing rate differentials.¹⁶ Asian currencies (particularly those benefiting from tech exports) and European currencies are favored. The yen receives support as a partial hedge, though Japan's monetary policy normalization remains gradual.

Regional Outlooks

**United States: Steady as She Goes**
The U.S. economy is forecast to grow 1.4-2.3% in 2026, supported by fiscal policy (OBBBA tax changes), less restrictive monetary policy, and stabilized trade policy.¹⁷ Consumer spending growth will be modest (1.4%), with a K-shaped economy where affluent consumers continue spending while middle-income households feel pressure. Business investment will be sustained by AI-related capex, though traditional categories may struggle. Unemployment is expected to rise modestly to 4.5% as labor market conditions normalize.

**Asia-Pacific: Signs of Relief**
S&P Global raised 2026 GDP forecasts for Asia-Pacific to 4.2% (from 4.0%), driven by reduced tariff uncertainty, tech export strength, and resilient domestic demand.¹⁸ China's growth forecast was lifted to 4.4% (from 4.0%) due to lower U.S. tariffs. However, central banks have limited scope for further rate cuts as rates approach neutral levels and exchange rates have weakened. The region benefits from AI-driven trade, with tech exports outperforming.

**Europe: Germany's Fiscal Reawakening**
Europe will see stable economic growth in 2026, though geographical composition is shifting. Germany's expansive fiscal policy is expected to boost growth and provide positive spillover effects, particularly in Central and Eastern Europe.¹⁹ Spain's growth will slow, reflecting different fiscal positions. The region continues to contend with tariff aftereffects, though less severely than initially feared.

**Emerging Markets: AI Will Drive Trade Divergence**
Emerging markets are expected to see modestly slower growth in 2026 compared to 2025, but remain resilient.²⁰ AI and tech-related exports will continue to outperform, benefiting mostly Asian EMs. Countries with diversified export markets and strong domestic bases show better resilience than export-dependent economies.

Key Risks and Opportunities

**Risks:**
- AI bubble burst scenario (similar to dot-com 2001-2002) could trigger widespread insolvencies
- Tariff policy remains unpredictable despite stabilization expectations
- Inflation could prove stickier than forecast, delaying Fed easing
- Geopolitical tensions could resurface, disrupting trade flows
- Banking sector faces multiple headwinds (macro uncertainty, AI scaling challenges, stablecoin disruption)

**Opportunities:**
- International equity markets offer better risk-adjusted returns than U.S.
- Quality U.S. equities with secular themes (AI ecosystem, financial deregulation)
- Fixed income income generation in a range-bound rate environment
- Commodities and real assets benefiting from global acceleration
- Asian emerging markets leveraging AI-driven trade

Conclusion: A Year of Careful Navigation

2026 presents a landscape of moderate growth, persistent but improving inflation, and gradual policy normalization. The consensus view suggests resilience rather than exuberance, with quality and diversification remaining paramount. International markets offer compelling opportunities as earnings growth gaps narrow and the dollar weakens. Within the U.S., secular themes (AI, financial deregulation) outweigh cyclical plays.

The year ahead will reward investors who focus on fundamentals, embrace diversification, and maintain discipline around valuations. While risks exist—particularly around AI valuations, insolvency trends, and policy unpredictability—the baseline scenario supports continued risk asset performance with appropriate positioning.

The transformation underway is real: AI is driving productivity gains, trade relationships are recalibrating, and monetary policy is normalizing. Success in 2026 will come not from predicting perfect outcomes, but from navigating the interpretation game with discipline, diversification, and focus on quality.

---

Footnotes:
¹ S&P Global Ratings, "Economic Outlook Asia-Pacific Q1 2026: Signs Of Relief" (November 2025)
² S&P Global Ratings, "Economic Outlook Europe Q1 2026: Germany's Fiscal Reawakening" (November 2025)
³ Wells Fargo Economics, "2026 Annual Economic Outlook: Policy Reset" (November 2025)
⁴ J.P. Morgan Asset Management, "2026 Year-Ahead Investment Outlook: AI Lift and Economic Drift" (2025)
⁵ S&P Global Ratings, "Economic Outlook Asia-Pacific Q1 2026: Signs Of Relief" (November 2025)
⁶ Allianz Research, "Global Insolvency Outlook 2026-27: Don't look down!" (October 2025)
⁷ Allianz Research, "Global Insolvency Outlook 2026-27: Don't look down!" (October 2025)
⁸ Allianz Research, "Global Insolvency Outlook 2026-27: Don't look down!" (October 2025)
⁹ Allianz Research, "Global Insolvency Outlook 2026-27: Don't look down!" (October 2025)
¹⁰ Allianz Research, "Global Insolvency Outlook 2026-27: Don't look down!" (October 2025)
¹¹ Deloitte Center for Financial Services, "2026 banking and capital markets outlook" (October 2025)
¹² ING THINK, "FX Outlook 2026: Play the ball, not the man" (November 2025)
¹³ S&P Global Ratings, "Economic Outlook Emerging Markets Q1 2026: AI Will Drive Trade Divergence" (November 2025)
¹⁴ Invesco Strategy & Insights, "Global Asset Allocation 2026 Outlook: The Big Picture" (November 2025)
¹⁵ BlackRock, "Private Markets Outlook 2026: A New Continuum" (2025)
¹⁶ Invesco Strategy & Insights, "Global Asset Allocation 2026 Outlook: The Big Picture" (November 2025)
¹⁷ Wells Fargo Economics, "2026 Annual Economic Outlook: Policy Reset" (November 2025)
¹⁸ S&P Global Ratings, "Economic Outlook Asia-Pacific Q1 2026: Signs Of Relief" (November 2025)
¹⁹ S&P Global Ratings, "Economic Outlook Europe Q1 2026: Germany's Fiscal Reawakening" (November 2025)
²⁰ S&P Global Ratings, "Economic Outlook Emerging Markets Q1 2026: AI Will Drive Trade Divergence" (November 2025)`,
            trends: [
                {
                    title: "Moderate Growth with Regional Divergence",
                    description: "U.S. steady at 1.4-2.3%, Asia-Pacific strong at 4.2-4.4%, Europe stable. Consensus: no recession, but growth below 2025 pace."
                },
                {
                    title: "Shallow Fed Easing Path",
                    description: "2-3 rate cuts (50-75bps) expected through 2026, bringing terminal rate to 3.00-3.25%. Cautious approach reflects balanced inflation/growth risks."
                },
                {
                    title: "International Equity Outperformance",
                    description: "Earnings growth gaps narrowing, dollar weakness, and favorable valuations support international markets, particularly Asian EMs and European value."
                },
                {
                    title: "AI-Driven Growth with Valuation Concerns",
                    description: "Real infrastructure spending underpins AI boom, but stretched valuations raise bubble concerns. Quality and secular themes favored over speculation."
                },
                {
                    title: "Rising Insolvencies Despite Growth",
                    description: "Global business insolvencies rising +5% in 2026, marking five consecutive years of increases. U.S. (+8%) and China (+10%) driving bulk of increase."
                }
            ],
            picks: []
        }
    },
    {
        id: 'staples-dec-2025',
        title: 'Consumer Staples',
        subtitle: 'The Strategic Rotation to Resilience',
        icon: '🛒',
        readTime: '8 min read',
        tags: ['Defensive', 'Value'],
        content: {
            overview: `Amidst a "Great Rotation" out of overextended tech momentum, capital is seeking the safety of inelastic demand. Historically the best-performing sector during volatility.`,
            fullText: `Consumer Staples: The Strategic Rotation to Resilience

Executive Summary: Navigating the Anti-Momentum Pivot
As the global financial markets pivot into the final trading month of 2025, a profound regime change is underway. The aggressive "growth-at-any-cost" momentum is yielding to a disciplined, valuation-sensitive environment. Institutional capital is seeking refuge in sectors offering earnings visibility and balance sheet fortitude.

The December 2025 Industry Report identifies a "Great Rotation" out of overextended technology momentum plays and into the bedrock of the economy: Consumer Staples, Healthcare & Pharma, Medical Devices, and Insurance & Managed Care.

The Overarching Theme: "Rational Exuberance"
While indices remain near highs, internal breadth shows bifurcation. Investors grapple with a divided Fed, geopolitical friction ("Liberation Day" tariffs), and a resilient but value-conscious consumer. In this climate, the allure of speculative growth diminishes in favor of industries providing essential services—sectors where demand is inelastic.

Why Consumer Staples?
Selected for its unmatched historical resilience.
- **Recession Performance:** In the last seven recessions, Consumer Staples was the #1 performing sector (+1% avg return vs double-digit declines elsewhere).
- **Long-Term Compounder:** Over 20 years, staples often outperform high-growth tech due to reinvested dividends and lower volatility.

Trends Defining the Sector in December 2025

The Rise of the "Value-Seeking" Consumer
High interest rates and inflation have eroded excess savings. Shoppers are focusing on value and trimming seasonal extras. This favors giants with scale (Costco, Walmart) and private label manufacturers. The "trade-down" effect accelerates market share gains for efficient operators.

Cost Input Stabilization vs. Tariff Threats
Input costs (commodities, energy) have stabilized, allowing margin expansion as 2024 price hikes stick. However, new tariffs could inflate packaging costs. Companies with domestic sourcing networks trade at a premium due to "supply chain security."

The "Mobile-First" Grocery Shift
Digital transformation has permeated the sector. Mobile devices will account for >50% of online holiday spending. Retailers optimizing apps for reordering and loyalty are seeing higher retention.

Outlook for 2026: Stability in a Volatile World
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
                    <h4 className="text-xs text-text-muted uppercase tracking-widest">December 2025 Edition</h4>
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

"""
Industry Research / Sector Deep Dives endpoints.
Provides sector-level analysis and insights.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging

from api.dependencies import get_current_user, get_optional_current_user

router = APIRouter(prefix="/api/industry-research", tags=["industry-research"])
LOGGER = logging.getLogger("caria.api.industry_research")

# Base industry list (configurable)
INDUSTRIES = [
    "Consumer Staples",
    "Healthcare & Pharma",
    "Medical Devices",
    "Insurance & Managed Care",
    "Semiconductors & AI Hardware",
    "Cloud & SaaS",
    "Cybersecurity",
    "Digital Payments & Fintech",
    "E-commerce & Platforms",
    "Renewables & Grid",
    "Metals & Mining",
    "Luxury & Premium Brands",
    "Travel & Leisure",
    "Space & Satellite",
    "Genomics & Precision Medicine",
    "Climate Tech / Carbon Solutions",
    "Defense & Dual-Use Tech",
]

class IndustryCard(BaseModel):
    name: str
    status: str  # Emerging, Mature, Overheated, Under Pressure
    growth_signal: str
    valuation_context: str
    flows_activity: str
    representative_tickers: List[str]

class IndustrySummary(BaseModel):
    name: str
    thesis_summary: List[str]
    aggregate_kpis: Dict[str, Any]
    stage: str  # Early, Mid, Late, Turning
    leaders_challengers: List[Dict[str, str]]
    key_risks: List[str]
    caria_signals: Dict[str, Any]
    recent_headlines: List[str]
    learning_resources: Dict[str, List[str]]

@router.get("/industries")
def get_industries(
    current_user = Depends(get_optional_current_user)
) -> List[IndustryCard]:
    """
    Get list of all industries with summary cards.
    Each card shows status, growth signals, valuation context, and representative tickers.
    """
    # For now, return mock data structure - can be enhanced with real data later
    industry_cards = []
    
    for industry in INDUSTRIES:
        # Determine status based on industry (simplified logic)
        if "AI" in industry or "Cloud" in industry or "Cybersecurity" in industry:
            status = "Emerging"
        elif "Staples" in industry or "Insurance" in industry:
            status = "Mature"
        elif "Semiconductors" in industry or "E-commerce" in industry:
            status = "Overheated"
        else:
            status = "Under Pressure"
        
        # Mock representative tickers (would come from actual data)
        ticker_map = {
            "Semiconductors & AI Hardware": ["NVDA", "AMD", "INTC"],
            "Cloud & SaaS": ["MSFT", "CRM", "NOW"],
            "Healthcare & Pharma": ["JNJ", "PFE", "UNH"],
            "Consumer Staples": ["PG", "KO", "PEP"],
        }
        tickers = ticker_map.get(industry, ["AAPL", "MSFT", "GOOGL"])
        
        industry_cards.append(IndustryCard(
            name=industry,
            status=status,
            growth_signal=f"Revenue growth accelerating in {industry.lower()}",
            valuation_context="Trading at 15% premium to 5-year average",
            flows_activity="ETF inflows +$2.3B this quarter",
            representative_tickers=tickers
        ))
    
    return industry_cards

@router.get("/industries/{industry_name}")
def get_industry_detail(
    industry_name: str,
    current_user = Depends(get_optional_current_user)
) -> IndustrySummary:
    """
    Get detailed analysis for a specific industry.
    Includes thesis, KPIs, leaders/challengers, risks, Caria signals, headlines, and learning resources.
    """
    if industry_name not in INDUSTRIES:
        raise HTTPException(status_code=404, detail=f"Industry '{industry_name}' not found")
    
    # Mock detailed data - would be generated from real data/LLM in production
    summary = IndustrySummary(
        name=industry_name,
        thesis_summary=[
            f"{industry_name} is experiencing structural shifts driven by technology adoption",
            "Valuation multiples are elevated but supported by strong fundamentals",
            "Key risks include regulatory changes and competitive dynamics"
        ],
        aggregate_kpis={
            "revenue_growth": 12.5,
            "margins": 18.3,
            "ev_ebitda_median": 14.2,
            "market_cap_total": 2500000000000
        },
        stage="Mid",
        leaders_challengers=[
            {"ticker": "LEAD1", "name": "Market Leader 1", "market_share": "25%"},
            {"ticker": "LEAD2", "name": "Market Leader 2", "market_share": "18%"},
            {"ticker": "CHAL1", "name": "Emerging Challenger", "market_share": "5%"},
        ],
        key_risks=[
            "Regulatory scrutiny increasing",
            "Technology disruption risk",
            "Macro sensitivity to interest rates"
        ],
        caria_signals={
            "alpha_picker_appearances": 15,
            "screener_appearances": 32,
            "crisis_sensitivity": "Medium"
        },
        recent_headlines=[
            f"Major M&A activity in {industry_name}",
            "New regulations impact sector outlook",
            "Earnings season shows mixed results"
        ],
        learning_resources={
            "lectures": [
                f"Understanding {industry_name} Dynamics",
                "Sector Analysis Framework"
            ],
            "videos": [
                f"{industry_name} Deep Dive"
            ]
        }
    )
    
    return summary

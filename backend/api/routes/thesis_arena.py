"""
Thesis Arena endpoints.
Challenge investment thesis with parallel community responses.
"""

from __future__ import annotations

import asyncio
import logging
import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import psycopg2
import requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from urllib.parse import urlparse, parse_qs

from api.dependencies import get_current_user
from api.services.conviction_service import get_conviction_service
from caria.models.auth import UserInDB

LOGGER = logging.getLogger("caria.api.thesis_arena")

router = APIRouter(prefix="/api/thesis", tags=["Thesis Arena"])

# Common English words to ignore when extracting tickers
COMMON_WORDS = {
    "I", "A", "AM", "AN", "AT", "BE", "BY", "DO", "GO", "HE", "HI", "IF", "IN", "IS", "IT", "ME", "MY", 
    "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE", "WANT", "BUY", "SELL", "HOLD", "GET", "PUT", 
    "SEE", "SAY", "ASK", "CAN", "DID", "LET", "RUN", "TRY", "USE", "WAY", "WHO", "WHY", "YES", "YOU", 
    "THE", "AND", "BUT", "FOR", "NOT", "NOW", "OUT", "TOO", "THAT", "THIS", "ARE", "WAS", "WERE", "HAVE",
    "HAS", "HAD", "WILL", "WOULD", "SHOULD", "COULD", "BEEN", "BEING", "DOES", "DID", "DOING", "MAKING",
    "MADE", "MAKE", "LOOK", "LOOKING", "LIKE", "LIKES", "NEED", "NEEDS", "KNOW", "KNOWS", "THINK", "THINKS"
}

def _extract_ticker(text: str) -> Optional[str]:
    """Extract potential ticker from text."""
    # Look for 2-5 uppercase letters
    candidates = re.findall(r'\b[A-Z]{2,5}\b', text)
    # Filter out common words
    valid_candidates = [c for c in candidates if c not in COMMON_WORDS]
    
    if valid_candidates:
        # Return the first one that looks most like a ticker (heuristic)
        # Ideally we would validate against a ticker database
        return valid_candidates[0]
    return None

# Dynamic community loading
def _get_active_communities() -> List[str]:
    """Load communities from config or use expanded defaults."""
    config_path = Path(__file__).parent.parent.parent / "config" / "communities.json"

    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                return [c["id"] for c in config["communities"] if c.get("active", True)]
        except Exception as e:
            LOGGER.warning(f"Failed to load communities config: {e}")

    # Fallback to expanded default list
    return [
        "value_investor", "crypto_bro", "growth_investor", "contrarian",
        "technical_analyst", "dividend_investor", "esg_advocate", "risk_manager"
    ]

COMMUNITIES = _get_active_communities()


def _load_community_prompt(community: str) -> str:
    """Load community prompt from file."""
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "communities" / f"{community}.txt"
    base_prompt = ""
    try:
        if prompt_path.exists():
            base_prompt = prompt_path.read_text(encoding="utf-8")
        else:
            LOGGER.warning(f"Prompt file not found: {prompt_path}")
            base_prompt = f"You are a {community.replace('_', ' ')} investor."
    except Exception as e:
        LOGGER.exception(f"Error loading prompt for {community}: {e}")
        base_prompt = f"You are a {community.replace('_', ' ')} investor."
        
    # Add Socratic instructions
    socratic_instruction = (
        "\n\nIMPORTANT INSTRUCTIONS:\n"
        "1. Be CONCISE. Keep responses short (2-4 sentences).\n"
        "2. Use the SOCRATIC METHOD. Ask thought-provoking questions to guide the user's reflection.\n"
        "3. Do not lecture. Engage in a dialogue.\n"
        "4. Challenge the user's assumptions based on your persona.\n"
        "5. Focus on the investment thesis logic, not just the ticker."
    )
    
    return base_prompt + socratic_instruction


def _call_llm(prompt: str, community: str, provider: str = "auto") -> Optional[str]:
    """Call LLM with automatic provider fallback."""
    providers = ["groq", "openai", "anthropic"] if provider == "auto" else [provider]

    for prov in providers:
        try:
            if prov == "groq":
                return _call_groq(prompt, community)
            elif prov == "openai":
                return _call_openai(prompt, community)
            elif prov == "anthropic":
                return _call_anthropic(prompt, community)
        except Exception as e:
            LOGGER.warning(f"{prov} provider failed for {community}: {e}")
            continue

    LOGGER.error(f"All LLM providers failed for {community}")
    return None


def _call_groq(prompt: str, community: str) -> Optional[str]:
    """Call Groq/Llama API."""
    api_key = os.getenv("LLAMA_API_KEY", "").strip()
    api_url = os.getenv("LLAMA_API_URL", "https://api.groq.com/openai/v1/chat/completions").strip()
    model = os.getenv("LLAMA_MODEL", "llama-3.1-8b-instant").strip()

    if not api_key:
        raise ValueError("LLAMA_API_KEY not configured")

    community_prompt = _load_community_prompt(community)
    system_prompt = f"{community_prompt}\n\nYou are a {community.replace('_', ' ')} investor."

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
    }

    resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    LOGGER.info(f"Groq returned response for {community} ({len(text)} chars)")
    return text


def _call_openai(prompt: str, community: str) -> Optional[str]:
    """OpenAI GPT-4 fallback."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY not configured")

    community_prompt = _load_community_prompt(community)

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4-turbo-preview",
            "messages": [
                {"role": "system", "content": community_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        },
        timeout=30
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    LOGGER.info(f"OpenAI returned response for {community} ({len(text)} chars)")
    return text


def _call_anthropic(prompt: str, community: str) -> Optional[str]:
    """Claude fallback."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not configured")

    community_prompt = _load_community_prompt(community)

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        },
        json={
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": f"{community_prompt}\n\n{prompt}"}]
        },
        timeout=30
    )
    resp.raise_for_status()
    text = resp.json()["content"][0]["text"]
    LOGGER.info(f"Anthropic returned response for {community} ({len(text)} chars)")
    return text


# Keep old function name for backwards compatibility
def _call_llama_parallel(prompt: str, community: str) -> Optional[str]:
    """Legacy function - calls new _call_llm with auto provider."""
    return _call_llm(prompt, community, provider="auto")


def _get_db_connection():
    """Get database connection."""
    database_url = os.getenv("DATABASE_URL")
    conn = None
    
    if database_url:
        try:
            parsed = urlparse(database_url)
            query_params = parse_qs(parsed.query)
            unix_socket_host = query_params.get('host', [None])[0]
            
            if unix_socket_host:
                conn = psycopg2.connect(
                    host=unix_socket_host,
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path.lstrip('/'),
                )
            elif parsed.hostname:
                conn = psycopg2.connect(
                    host=parsed.hostname,
                    port=parsed.port or 5432,
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path.lstrip('/'),
                )
        except Exception as e:
            LOGGER.warning(f"Error using DATABASE_URL: {e}")
    
    if conn is None:
        password = os.getenv("POSTGRES_PASSWORD")
        if not password:
            raise RuntimeError("POSTGRES_PASSWORD environment variable is required")
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "caria_user"),
            password=password,
            database=os.getenv("POSTGRES_DB", "caria"),
        )
    
    return conn


async def _save_arena_thread(
    user_id: str,
    thesis: str,
    ticker: Optional[str],
    initial_conviction: float,
    current_conviction: float,
    community_responses: Dict[str, str],
    conviction_before: float,
    conviction_after: float,
) -> Optional[str]:
    """Save arena thread and first round to database."""
    conn = None
    try:
        conn = _get_db_connection()
        
        with conn.cursor() as cur:
            # Insert thread
            thread_id = uuid4()
            cur.execute(
                """
                INSERT INTO thesis_arena_threads 
                (id, user_id, thesis, ticker, initial_conviction, current_conviction, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'active')
                RETURNING id
                """,
                (thread_id, user_id, thesis, ticker, initial_conviction, current_conviction)
            )
            
            conn.commit()
            return str(thread_id)
    except Exception as e:
        LOGGER.exception(f"Error saving arena thread: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            conn.close()


class ThesisArenaChallengeRequest(BaseModel):
    """Request model for thesis arena challenge."""
    thesis: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Investment thesis to challenge"
    )
    ticker: Optional[str] = Field(
        None,
        description="Optional ticker symbol for context"
    )
    initial_conviction: float = Field(
        50.0,
        ge=0,
        le=100,
        description="Initial conviction level (0-100)"
    )


class CommunityResponse(BaseModel):
    """Response from a single community."""
    community: str
    response: str
    impact_score: float = Field(..., description="Impact on conviction (-1 to +1)")


class ThesisArenaChallengeResponse(BaseModel):
    """Response model for thesis arena challenge."""
    thesis: str
    ticker: Optional[str]
    initial_conviction: float
    community_responses: List[CommunityResponse]
    conviction_impact: Dict[str, Any]
    arena_id: Optional[str] = Field(None, description="Arena thread ID for multi-round conversations")
    round_number: int = Field(1, description="Current round number")


class ArenaRespondRequest(BaseModel):
    """Request model for responding in an arena thread."""
    thread_id: str = Field(..., description="Arena thread ID")
    user_message: str = Field(..., min_length=1, max_length=1000, description="User's follow-up message")


class ArenaRespondResponse(BaseModel):
    """Response model for arena respond."""
    thread_id: str
    round_number: int
    user_message: str
    community_responses: List[CommunityResponse]
    conviction_impact: Dict[str, Any]
    current_conviction: float


@router.get("/ticker-support/{ticker}")
async def check_ticker_support(
    ticker: str,
    current_user: UserInDB = Depends(get_current_user),
):
    """Check if ticker is supported by data sources."""
    ticker = ticker.upper()
    support_status = {
        "ticker": ticker,
        "supported": False,
        "data_sources": {}
    }

    # Check FMP
    try:
        from api.services.scoring_service import ScoringService
        scoring = ScoringService()
        quote = scoring.fmp.get_realtime_price(ticker)
        support_status["data_sources"]["fmp"] = bool(quote)
        support_status["supported"] = support_status["supported"] or bool(quote)
    except Exception as e:
        LOGGER.debug(f"FMP check failed for {ticker}: {e}")
        support_status["data_sources"]["fmp"] = False

    # Check OpenBB
    try:
        from api.services.openbb_client import OpenBBClient
        obb = OpenBBClient()
        data = obb.get_ticker_data(ticker)
        support_status["data_sources"]["openbb"] = bool(data)
        support_status["supported"] = support_status["supported"] or bool(data)
    except Exception as e:
        LOGGER.debug(f"OpenBB check failed for {ticker}: {e}")
        support_status["data_sources"]["openbb"] = False

    if not support_status["supported"]:
        support_status["message"] = f"{ticker} not available. Try major exchange tickers (NASDAQ, NYSE, AMEX)."

    return support_status


@router.post("/arena/challenge", response_model=ThesisArenaChallengeResponse)
async def challenge_thesis_arena(
    request: ThesisArenaChallengeRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> ThesisArenaChallengeResponse:
    """
    Challenge investment thesis with parallel community responses.
    """
    # Build prompt with ticker if provided
    ticker = request.ticker
    
    # If no ticker provided, try to extract it
    if not ticker:
        extracted = _extract_ticker(request.thesis)
        if extracted:
            ticker = extracted
            LOGGER.info(f"Extracted ticker {ticker} from thesis")
            
    prompt = request.thesis
    if ticker:
        prompt = f"Ticker: {ticker.upper()}\n\nThesis: {request.thesis}"
    
    # Call all communities in parallel
    LOGGER.info(f"Challenging thesis with {len(COMMUNITIES)} communities in parallel")
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    tasks = [
        loop.run_in_executor(None, _call_llama_parallel, prompt, community)
        for community in COMMUNITIES
    ]
    
    responses_list = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process responses
    community_responses_dict = {}
    community_responses_list = []
    
    for i, community in enumerate(COMMUNITIES):
        response_text = responses_list[i]
        
        if isinstance(response_text, Exception):
            LOGGER.error(f"Error getting response from {community}: {response_text}")
            response_text = f"Error: Could not get response from {community} community."
        elif response_text is None:
            response_text = f"No response available from {community} community."
        
        community_responses_dict[community] = response_text
        
        community_responses_list.append(
            CommunityResponse(
                community=community,
                response=response_text,
                impact_score=0.0,
            )
        )
    
    # Calculate conviction impact
    conviction_service = get_conviction_service()
    conviction_impact = conviction_service.calculate_conviction_impact(
        responses=community_responses_dict,
        initial_conviction=request.initial_conviction,
    )
    
    # Update impact scores
    for response in community_responses_list:
        impact_data = conviction_impact["community_impacts"].get(response.community, {})
        response.impact_score = impact_data.get("weighted_impact", 0.0)
    
    # Save to database
    thread_id = await _save_arena_thread(
        user_id=current_user.id,
        thesis=request.thesis,
        ticker=ticker,  # Save inferred or provided ticker
        initial_conviction=request.initial_conviction,
        current_conviction=conviction_impact["new_conviction"],
        community_responses=community_responses_dict,
        conviction_before=request.initial_conviction,
        conviction_after=conviction_impact["new_conviction"],
    )
    
    return ThesisArenaChallengeResponse(
        thesis=request.thesis,
        ticker=ticker,
        initial_conviction=request.initial_conviction,
        community_responses=community_responses_list,
        conviction_impact=conviction_impact,
        arena_id=str(thread_id) if thread_id else None,
        round_number=1,
    )


@router.post("/arena/respond", response_model=ArenaRespondResponse)
async def respond_in_arena(
    request: ArenaRespondRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> ArenaRespondResponse:
    """
    Respond in an existing arena thread with a follow-up message.
    """
    conn = None
    try:
        conn = _get_db_connection()
        
        # Get thread and current conviction
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, thesis, ticker, current_conviction, user_id
                FROM thesis_arena_threads
                WHERE id = %s AND user_id = %s AND status = 'active'
                """,
                (request.thread_id, current_user.id)
            )
            thread_row = cur.fetchone()
            
            if not thread_row:
                raise HTTPException(
                    status_code=404,
                    detail="Arena thread not found or not accessible"
                )
            
            thread_id, thesis, ticker, current_conviction, user_id = thread_row
            
            # Get last round number
            cur.execute(
                """
                SELECT MAX(round_number) FROM arena_rounds WHERE thread_id = %s
                """,
                (thread_id,)
            )
            last_round_row = cur.fetchone()
            next_round = (last_round_row[0] or 0) + 1
            
            # Build prompt with context - Including Socratic instructions implicitly via system prompt load
            context_prompt = f"""Previous Thesis: {thesis}
"""
            if ticker:
                context_prompt += f"Ticker: {ticker}\n"
            context_prompt += f"""
User's Follow-up Question/Response:
{request.user_message}

Please respond to the user's follow-up from your community perspective. 
Remember to be Socratic, concise (2-4 sentences), and challenge assumptions. 
Do not give financial advice, but guide the user's thinking."""
        
        # Call all communities
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        tasks = [
            loop.run_in_executor(None, _call_llama_parallel, context_prompt, community)
            for community in COMMUNITIES
        ]
        
        responses_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process responses
        community_responses_dict = {}
        community_responses_list = []
        
        for i, community in enumerate(COMMUNITIES):
            response_text = responses_list[i]
            
            if isinstance(response_text, Exception):
                LOGGER.error(f"Error getting response from {community}: {response_text}")
                response_text = f"Error: Could not get response from {community} community."
            elif response_text is None:
                response_text = f"No response available from {community} community."
            
            community_responses_dict[community] = response_text
            
            community_responses_list.append(
                CommunityResponse(
                    community=community,
                    response=response_text,
                    impact_score=0.0,
                )
            )
        
        # Calculate conviction impact
        conviction_service = get_conviction_service()
        conviction_impact = conviction_service.calculate_conviction_impact(
            responses=community_responses_dict,
            initial_conviction=current_conviction,
        )
        
        new_conviction = conviction_impact["new_conviction"]
        
        for response in community_responses_list:
            impact_data = conviction_impact["community_impacts"].get(response.community, {})
            response.impact_score = impact_data.get("weighted_impact", 0.0)
        
        # Save round
        with conn.cursor() as cur:
            round_id = uuid4()
            cur.execute(
                """
                INSERT INTO arena_rounds
                (id, thread_id, round_number, user_message, community_responses, conviction_before, conviction_after, conviction_change)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                """,
                (
                    round_id,
                    thread_id,
                    next_round,
                    request.user_message,
                    json.dumps(community_responses_dict),
                    current_conviction,
                    new_conviction,
                    new_conviction - current_conviction,
                )
            )
            
            cur.execute(
                """
                UPDATE thesis_arena_threads
                SET current_conviction = %s, updated_at = NOW()
                WHERE id = %s
                """,
                (new_conviction, thread_id)
            )
            
            conn.commit()
        
        return ArenaRespondResponse(
            thread_id=str(thread_id),
            round_number=next_round,
            user_message=request.user_message,
            community_responses=community_responses_list,
            conviction_impact=conviction_impact,
            current_conviction=new_conviction,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception(f"Error responding in arena: {e}")
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error responding in arena: {str(e)}"
        ) from e
    finally:
        if conn:
            conn.close()

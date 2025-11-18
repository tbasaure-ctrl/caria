"""
Thesis Arena endpoints.
Challenge investment thesis with parallel community responses.
"""

from __future__ import annotations

import asyncio
import logging
import os
import json
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

# Community names
COMMUNITIES = ["value_investor", "crypto_bro", "growth_investor", "contrarian"]


def _load_community_prompt(community: str) -> str:
    """Load community prompt from file."""
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "communities" / f"{community}.txt"
    try:
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        else:
            LOGGER.warning(f"Prompt file not found: {prompt_path}")
            return f"You are a {community.replace('_', ' ')} investor. Analyze the investment thesis."
    except Exception as e:
        LOGGER.exception(f"Error loading prompt for {community}: {e}")
        return f"You are a {community.replace('_', ' ')} investor. Analyze the investment thesis."


def _call_gemini_parallel(prompt: str, community: str) -> Optional[str]:
    """Call Gemini API with community-specific prompt."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        LOGGER.warning("GEMINI_API_KEY not configured")
        return None
    
    # Load community prompt
    community_prompt = _load_community_prompt(community)
    
    # Combine prompts
    full_prompt = f"""{community_prompt}

Investment Thesis to Analyze:
{prompt}

Provide your analysis and response as a {community.replace('_', ' ')} investor. Be specific, challenge assumptions, and provide actionable insights."""
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{"text": full_prompt}]
        }]
    }
    
    try:
        resp = requests.post(api_url, json=payload, timeout=30)
        
        if resp.status_code == 503:
            LOGGER.warning(f"Gemini 503 for {community}, skipping")
            return None
        
        resp.raise_for_status()
        data = resp.json()
        
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        
        if not text:
            LOGGER.warning(f"Gemini returned empty text for {community}")
            return None
        
        return text
    except Exception as e:
        LOGGER.exception(f"Error calling Gemini for {community}: {e}")
        return None


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
            
            # Insert first round
            round_id = uuid4()
            cur.execute(
                """
                INSERT INTO arena_rounds
                (id, thread_id, round_number, community_responses, conviction_before, conviction_after, conviction_change)
                VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s)
                """,
                (
                    round_id,
                    thread_id,
                    1,
                    json.dumps(community_responses),
                    conviction_before,
                    conviction_after,
                    conviction_after - conviction_before,
                )
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


@router.post("/arena/challenge", response_model=ThesisArenaChallengeResponse)
async def challenge_thesis_arena(
    request: ThesisArenaChallengeRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> ThesisArenaChallengeResponse:
    """
    Challenge investment thesis with parallel community responses.
    
    Calls Gemini API in parallel for each of the 4 communities:
    - value_investor
    - crypto_bro
    - growth_investor
    - contrarian
    
    Each community responds from their perspective, and conviction impact
    is calculated from their responses.
    
    Args:
        request: Thesis challenge request
        current_user: Current authenticated user
        
    Returns:
        Response with all community responses and conviction impact
    """
    # Build prompt with ticker if provided
    prompt = request.thesis
    if request.ticker:
        prompt = f"Ticker: {request.ticker.upper()}\n\nThesis: {request.thesis}"
    
    # Call all communities in parallel
    LOGGER.info(f"Challenging thesis with {len(COMMUNITIES)} communities in parallel")
    
    # Use asyncio to run requests in parallel
    # Note: FastAPI runs in async context, so we can use asyncio directly
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop exists, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    tasks = [
        loop.run_in_executor(None, _call_gemini_parallel, prompt, community)
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
        
        # Calculate impact score (will be refined by conviction service)
        community_responses_list.append(
            CommunityResponse(
                community=community,
                response=response_text,
                impact_score=0.0,  # Will be updated below
            )
        )
    
    # Calculate conviction impact
    conviction_service = get_conviction_service()
    conviction_impact = conviction_service.calculate_conviction_impact(
        responses=community_responses_dict,
        initial_conviction=request.initial_conviction,
    )
    
    # Update impact scores in responses
    for response in community_responses_list:
        impact_data = conviction_impact["community_impacts"].get(response.community, {})
        response.impact_score = impact_data.get("weighted_impact", 0.0)
    
    # Save to database for multi-round conversations
    thread_id = await _save_arena_thread(
        user_id=current_user.id,
        thesis=request.thesis,
        ticker=request.ticker,
        initial_conviction=request.initial_conviction,
        current_conviction=conviction_impact["new_conviction"],
        community_responses=community_responses_dict,
        conviction_before=request.initial_conviction,
        conviction_after=conviction_impact["new_conviction"],
    )
    
    return ThesisArenaChallengeResponse(
        thesis=request.thesis,
        ticker=request.ticker,
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
    
    Communities will respond to the user's follow-up, and conviction
    will be updated based on their responses.
    
    Args:
        request: Arena respond request with thread_id and user_message
        current_user: Current authenticated user
        
    Returns:
        Response with new round of community responses and updated conviction
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
            
            # Build prompt with context
            context_prompt = f"""Previous Thesis: {thesis}
"""
            if ticker:
                context_prompt += f"Ticker: {ticker}\n"
            context_prompt += f"""
User's Follow-up Question/Response:
{request.user_message}

Please respond to the user's follow-up from your community perspective. Consider the previous thesis and provide a thoughtful response."""
        
        # Call all communities in parallel with follow-up
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        tasks = [
            loop.run_in_executor(None, _call_gemini_parallel, context_prompt, community)
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
                    impact_score=0.0,  # Will be updated below
                )
            )
        
        # Calculate conviction impact
        conviction_service = get_conviction_service()
        conviction_impact = conviction_service.calculate_conviction_impact(
            responses=community_responses_dict,
            initial_conviction=current_conviction,
        )
        
        new_conviction = conviction_impact["new_conviction"]
        
        # Update impact scores
        for response in community_responses_list:
            impact_data = conviction_impact["community_impacts"].get(response.community, {})
            response.impact_score = impact_data.get("weighted_impact", 0.0)
        
        # Save round to database
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
            
            # Update thread current conviction
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

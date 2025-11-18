"""
Community Rankings endpoints.
Get top communities, hot theses, and survivors.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from psycopg2.extras import RealDictCursor

from api.dependencies import get_current_user
# Import _get_db_connection from community module
# We'll define it here to avoid circular imports
def _get_db_connection():
    """Get database connection using DATABASE_URL or fallback."""
    import psycopg2
    import os
    from urllib.parse import urlparse, parse_qs

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
            import logging
            logging.getLogger(__name__).warning(f"Error using DATABASE_URL: {e}")
    
    if conn is None:
        password = os.getenv("POSTGRES_PASSWORD")
        if not password:
            raise HTTPException(status_code=500, detail="Database password not configured")
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "caria_user"),
            password=password,
            database=os.getenv("POSTGRES_DB", "caria"),
        )
    
    return conn


from caria.models.auth import UserInDB

LOGGER = logging.getLogger("caria.api.community_rankings")

router = APIRouter(prefix="/api/community", tags=["Community"])


class RankingsResponse(BaseModel):
    """Response model for community rankings."""
    top_communities: list[dict] = Field(..., description="Top communities by activity")
    hot_theses: list[dict] = Field(..., description="Hot theses (trending)")
    survivors: list[dict] = Field(..., description="Survivor theses (high conviction maintained)")


@router.get("/rankings", response_model=RankingsResponse)
async def get_community_rankings(
    current_user: Optional[UserInDB] = Depends(get_current_user),
) -> RankingsResponse:
    """
    Get community rankings: top communities, hot theses, and survivors.
    
    - Top communities: Communities with most activity (posts + votes)
    - Hot theses: Theses with recent high engagement
    - Survivors: Theses that maintained high conviction over time
    """
    conn = None
    try:
        conn = _get_db_connection()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Top Communities (by arena activity)
            cursor.execute("""
                SELECT 
                    cp.arena_community as community,
                    COUNT(DISTINCT cp.id) as post_count,
                    SUM(cp.upvotes) as total_upvotes,
                    COUNT(DISTINCT cp.user_id) as unique_users
                FROM community_posts cp
                WHERE cp.is_arena_post = TRUE 
                    AND cp.arena_community IS NOT NULL
                    AND cp.is_active = TRUE
                GROUP BY cp.arena_community
                ORDER BY (post_count * 2 + total_upvotes) DESC
                LIMIT 10
            """)
            top_communities = [
                {
                    "community": row["community"],
                    "post_count": row["post_count"],
                    "total_upvotes": row["total_upvotes"],
                    "unique_users": row["unique_users"],
                }
                for row in cursor.fetchall()
            ]

            # Hot Theses (recent high engagement)
            seven_days_ago = datetime.utcnow() - timedelta(days=7)
            cursor.execute("""
                SELECT 
                    cp.id,
                    cp.title,
                    cp.thesis_preview,
                    cp.ticker,
                    cp.upvotes,
                    cp.created_at,
                    u.username
                FROM community_posts cp
                LEFT JOIN users u ON cp.user_id = u.id
                WHERE cp.is_active = TRUE
                    AND cp.created_at >= %s
                ORDER BY cp.upvotes DESC, cp.created_at DESC
                LIMIT 10
            """, (seven_days_ago,))
            hot_theses = [
                {
                    "id": str(row["id"]),
                    "title": row["title"],
                    "thesis_preview": row["thesis_preview"],
                    "ticker": row.get("ticker"),
                    "upvotes": row["upvotes"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "username": row.get("username"),
                }
                for row in cursor.fetchall()
            ]

            # Survivors (high conviction maintained - from arena threads)
            cursor.execute("""
                SELECT 
                    tat.id as thread_id,
                    tat.thesis,
                    tat.ticker,
                    tat.initial_conviction,
                    tat.current_conviction,
                    COUNT(DISTINCT ar.id) as round_count,
                    tat.created_at,
                    u.username
                FROM thesis_arena_threads tat
                LEFT JOIN users u ON tat.user_id = u.id
                LEFT JOIN arena_rounds ar ON ar.thread_id = tat.id
                WHERE tat.status = 'active'
                    AND tat.current_conviction >= 70
                    AND tat.current_conviction >= tat.initial_conviction
                GROUP BY tat.id, tat.thesis, tat.ticker, tat.initial_conviction, 
                         tat.current_conviction, tat.created_at, u.username
                HAVING COUNT(DISTINCT ar.id) >= 2
                ORDER BY tat.current_conviction DESC, round_count DESC
                LIMIT 10
            """)
            survivors = [
                {
                    "thread_id": str(row["thread_id"]),
                    "thesis": row["thesis"][:200] + "..." if len(row["thesis"]) > 200 else row["thesis"],
                    "ticker": row.get("ticker"),
                    "initial_conviction": row["initial_conviction"],
                    "current_conviction": row["current_conviction"],
                    "round_count": row["round_count"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "username": row.get("username"),
                }
                for row in cursor.fetchall()
            ]

            return RankingsResponse(
                top_communities=top_communities,
                hot_theses=hot_theses,
                survivors=survivors,
            )
    except Exception as e:
        LOGGER.exception(f"Error getting rankings: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting rankings: {str(e)}") from e
    finally:
        if conn:
            conn.close()


class PostValidateRequest(BaseModel):
    """Request model for post validation."""
    title: str = Field(..., min_length=5, max_length=255)
    thesis_preview: str = Field(..., min_length=10, max_length=500)
    full_thesis: Optional[str] = Field(None, max_length=5000)


class PostValidateResponse(BaseModel):
    """Response model for post validation."""
    is_valid: bool
    quality_score: float = Field(..., ge=0.0, le=1.0)
    feedback: list[str] = Field(..., description="Feedback on post quality")
    recommendation: str = Field(..., description="Recommendation: approve, revise, or reject")


@router.post("/posts/validate", response_model=PostValidateResponse)
async def validate_post(
    request: PostValidateRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> PostValidateResponse:
    """
    Validate a post using LLM to filter low-quality posts.
    
    Checks for:
    - Minimum quality threshold
    - Spam detection
    - Relevance to investment thesis
    - Completeness of analysis
    """
    import os
    import requests
    import json
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Fallback: approve if no API key
        LOGGER.warning("GEMINI_API_KEY not configured, approving post by default")
        return PostValidateResponse(
            is_valid=True,
            quality_score=0.7,
            feedback=["Validation service unavailable, post approved by default"],
            recommendation="approve",
        )
    
    # Build validation prompt
    validation_prompt = f"""You are a quality moderator for an investment thesis sharing platform.

Evaluate the following post:

Title: {request.title}
Preview: {request.thesis_preview}
Full Thesis: {request.full_thesis or request.thesis_preview}

Rate the post on:
1. Quality and depth of analysis (0-1)
2. Relevance to investment thesis (0-1)
3. Completeness of reasoning (0-1)
4. Spam/low-effort detection (0-1, lower is better)

Return a JSON response with:
{{
    "quality_score": 0.0-1.0,
    "is_valid": true/false,
    "feedback": ["feedback point 1", "feedback point 2", ...],
    "recommendation": "approve" or "revise" or "reject"
}}

Minimum quality threshold: 0.5
Reject if: spam, low-effort, irrelevant, or quality_score < 0.5"""
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{"text": validation_prompt}]
        }]
    }
    
    try:
        resp = requests.post(api_url, json=payload, timeout=30)
        
        if resp.status_code != 200:
            LOGGER.warning(f"Gemini validation failed: {resp.status_code}")
            # Fallback: approve if validation fails
            return PostValidateResponse(
                is_valid=True,
                quality_score=0.7,
                feedback=["Validation service error, post approved by default"],
                recommendation="approve",
            )
        
        data = resp.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        
        if not text:
            LOGGER.warning("Gemini returned empty validation response")
            return PostValidateResponse(
                is_valid=True,
                quality_score=0.7,
                feedback=["Validation service error, post approved by default"],
                recommendation="approve",
            )
        
        # Parse JSON response
        try:
            # Extract JSON from text (might have markdown formatting)
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_text = text[json_start:json_end]
                validation_result = json.loads(json_text)
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            LOGGER.warning(f"Failed to parse validation JSON: {e}")
            # Fallback: approve if parsing fails
            return PostValidateResponse(
                is_valid=True,
                quality_score=0.7,
                feedback=["Validation parsing error, post approved by default"],
                recommendation="approve",
            )
        
        return PostValidateResponse(
            is_valid=validation_result.get("is_valid", True),
            quality_score=float(validation_result.get("quality_score", 0.7)),
            feedback=validation_result.get("feedback", []),
            recommendation=validation_result.get("recommendation", "approve"),
        )
        
    except Exception as e:
        LOGGER.exception(f"Error validating post: {e}")
        # Fallback: approve if validation fails
        return PostValidateResponse(
            is_valid=True,
            quality_score=0.7,
            feedback=[f"Validation error: {str(e)}"],
            recommendation="approve",
        )


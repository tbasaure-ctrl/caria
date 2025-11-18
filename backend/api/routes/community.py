"""
Community endpoints - Shared investment thesis/ideas with Reddit-style voting.
Per user requirements: posts show title/preview, users can vote UP.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.dependencies import get_current_user
from caria.models.auth import UserInDB

router = APIRouter(prefix="/api/community", tags=["Community"])


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


# Request/Response Models
class CommunityPostCreate(BaseModel):
    title: str = Field(..., min_length=5, max_length=255)
    thesis_preview: str = Field(..., min_length=10, max_length=500)
    full_thesis: Optional[str] = Field(None, max_length=5000)
    ticker: Optional[str] = Field(None, max_length=10)
    analysis_merit_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class CommunityPostResponse(BaseModel):
    id: str
    user_id: str
    username: Optional[str] = None
    title: str
    thesis_preview: str
    full_thesis: Optional[str] = None
    ticker: Optional[str] = None
    analysis_merit_score: float
    upvotes: int
    user_has_voted: bool = False
    created_at: datetime
    updated_at: datetime


class VoteRequest(BaseModel):
    vote_type: str = Field(default="up", pattern="^up$")  # Only UP votes per requirements


@router.get("/posts", response_model=list[CommunityPostResponse])
async def get_community_posts(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("upvotes", pattern="^(upvotes|created_at|analysis_merit_score)$"),
    ticker: Optional[str] = Query(None, max_length=10),
    current_user: Optional[UserInDB] = Depends(get_current_user),
) -> list[CommunityPostResponse]:
    """
    Get community posts (top ideas).
    Shows title and preview only per user requirements.
    """
    from psycopg2.extras import RealDictCursor

    conn = _get_db_connection()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Build query
            query = """
                SELECT 
                    cp.id,
                    cp.user_id,
                    u.username,
                    cp.title,
                    cp.thesis_preview,
                    cp.full_thesis,
                    cp.ticker,
                    cp.analysis_merit_score,
                    cp.upvotes,
                    cp.created_at,
                    cp.updated_at,
                    CASE WHEN cv.id IS NOT NULL THEN TRUE ELSE FALSE END as user_has_voted
                FROM community_posts cp
                LEFT JOIN users u ON cp.user_id = u.id
                LEFT JOIN community_votes cv ON cv.post_id = cp.id AND cv.user_id = %s
                WHERE cp.is_active = TRUE
            """
            params = [str(current_user.id) if current_user else None]

            if ticker:
                query += " AND cp.ticker = %s"
                params.append(ticker.upper())

            # Order by
            if sort_by == "upvotes":
                query += " ORDER BY cp.upvotes DESC, cp.created_at DESC"
            elif sort_by == "created_at":
                query += " ORDER BY cp.created_at DESC"
            elif sort_by == "analysis_merit_score":
                query += " ORDER BY cp.analysis_merit_score DESC, cp.upvotes DESC"

            query += " LIMIT %s OFFSET %s"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                CommunityPostResponse(
                    id=str(row["id"]),
                    user_id=str(row["user_id"]),
                    username=row.get("username"),
                    title=row["title"],
                    thesis_preview=row["thesis_preview"],
                    full_thesis=row.get("full_thesis"),
                    ticker=row.get("ticker"),
                    analysis_merit_score=row["analysis_merit_score"] or 0.0,
                    upvotes=row["upvotes"],
                    user_has_voted=row.get("user_has_voted", False),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]
    finally:
        conn.close()


@router.post("/posts", response_model=CommunityPostResponse, status_code=status.HTTP_201_CREATED)
async def create_community_post(
    post_data: CommunityPostCreate,
    current_user: UserInDB = Depends(get_current_user),
) -> CommunityPostResponse:
    """
    Create a new community post (share investment thesis).
    Per user requirements: chat can offer to share thesis based on analysis merit.
    """
    from psycopg2.extras import RealDictCursor

    conn = _get_db_connection()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                INSERT INTO community_posts (
                    user_id, title, thesis_preview, full_thesis, ticker, analysis_merit_score,
                    arena_thread_id, arena_round_id, arena_community, is_arena_post
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, user_id, title, thesis_preview, full_thesis, ticker,
                          analysis_merit_score, upvotes, created_at, updated_at
                """,
                (
                    str(current_user.id),
                    post_data.title,
                    post_data.thesis_preview,
                    post_data.full_thesis,
                    post_data.ticker.upper() if post_data.ticker else None,
                    post_data.analysis_merit_score or 0.0,
                    post_data.arena_thread_id,
                    post_data.arena_round_id,
                    post_data.arena_community,
                    bool(post_data.arena_thread_id),  # is_arena_post = True if linked to arena
                ),
            )
            row = cursor.fetchone()
            conn.commit()

            return CommunityPostResponse(
                id=str(row["id"]),
                user_id=str(row["user_id"]),
                username=current_user.username,
                title=row["title"],
                thesis_preview=row["thesis_preview"],
                full_thesis=row.get("full_thesis"),
                ticker=row.get("ticker"),
                analysis_merit_score=row["analysis_merit_score"],
                upvotes=row["upvotes"],
                user_has_voted=False,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
    finally:
        conn.close()


@router.post("/posts/{post_id}/vote", status_code=status.HTTP_200_OK)
async def vote_on_post(
    post_id: UUID,
    vote_data: VoteRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> dict:
    """
    Vote on a community post (UP vote only per requirements).
    Users can vote if they click into the module.
    """
    import psycopg2
    import os
    from urllib.parse import urlparse, parse_qs

    # Use DATABASE_URL first (Cloud SQL format)
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

    try:
        with conn.cursor() as cursor:
            # Check if post exists
            cursor.execute("SELECT id FROM community_posts WHERE id = %s AND is_active = TRUE", (str(post_id),))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Post not found")

            # Check if user already voted
            cursor.execute(
                "SELECT id FROM community_votes WHERE post_id = %s AND user_id = %s",
                (str(post_id), str(current_user.id)),
            )
            existing_vote = cursor.fetchone()

            if existing_vote:
                # Remove vote (toggle off)
                cursor.execute(
                    "DELETE FROM community_votes WHERE post_id = %s AND user_id = %s",
                    (str(post_id), str(current_user.id)),
                )
                action = "removed"
            else:
                # Add vote
                cursor.execute(
                    "INSERT INTO community_votes (post_id, user_id, vote_type) VALUES (%s, %s, %s)",
                    (str(post_id), str(current_user.id), vote_data.vote_type),
                )
                action = "added"

            conn.commit()

            # Get updated upvote count
            cursor.execute("SELECT upvotes FROM community_posts WHERE id = %s", (str(post_id),))
            upvotes = cursor.fetchone()[0]

            return {"action": action, "upvotes": upvotes, "user_has_voted": action == "added"}
    finally:
        conn.close()


@router.get("/posts/{post_id}", response_model=CommunityPostResponse)
async def get_post_details(
    post_id: UUID,
    current_user: Optional[UserInDB] = Depends(get_current_user),
) -> CommunityPostResponse:
    """
    Get full details of a community post (including full thesis).
    """
    import psycopg2
    import os
    from psycopg2.extras import RealDictCursor

    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "caria_user"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DB", "caria"),
    )

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT 
                    cp.id,
                    cp.user_id,
                    u.username,
                    cp.title,
                    cp.thesis_preview,
                    cp.full_thesis,
                    cp.ticker,
                    cp.analysis_merit_score,
                    cp.upvotes,
                    cp.created_at,
                    cp.updated_at,
                    CASE WHEN cv.id IS NOT NULL THEN TRUE ELSE FALSE END as user_has_voted
                FROM community_posts cp
                LEFT JOIN users u ON cp.user_id = u.id
                LEFT JOIN community_votes cv ON cv.post_id = cp.id AND cv.user_id = %s
                WHERE cp.id = %s AND cp.is_active = TRUE
                """,
                (str(current_user.id) if current_user else None, str(post_id)),
            )
            row = cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Post not found")

            return CommunityPostResponse(
                id=str(row["id"]),
                user_id=str(row["user_id"]),
                username=row.get("username"),
                title=row["title"],
                thesis_preview=row["thesis_preview"],
                full_thesis=row.get("full_thesis"),
                ticker=row.get("ticker"),
                analysis_merit_score=row["analysis_merit_score"] or 0.0,
                upvotes=row["upvotes"],
                user_has_voted=row.get("user_has_voted", False),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
    finally:
        conn.close()


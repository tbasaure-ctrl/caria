"""
Chat endpoints - REST API for chat history recovery per audit document.
Problem #3: Recovery of lost messages on reconnection.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_current_user
from api.websocket_chat import get_chat_history
from caria.models.auth import UserInDB

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.get("/history")
async def get_chat_history_endpoint(
    since: Optional[str] = Query(None, description="ISO timestamp to get messages since"),
    current_user: UserInDB = Depends(get_current_user),
) -> dict:
    """
    Get chat history for current user since a timestamp.
    
    Per audit document Problem #3: This endpoint is called by the frontend
    in the 'connect' event to recover messages lost during disconnection.
    
    Args:
        since: ISO timestamp string (e.g., "2024-01-01T00:00:00Z")
               If provided, returns only messages after this timestamp.
               If not provided, returns last 50 messages.
    
    Returns:
        List of chat messages with id, message, timestamp, role.
    """
    try:
        user_id = str(current_user.id)
        messages = get_chat_history(user_id, since)
        
        return {
            "messages": messages,
            "count": len(messages),
            "user_id": user_id,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving chat history: {str(e)}"
        ) from e


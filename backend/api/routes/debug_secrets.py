"""
Debug endpoint to verify secrets are being read correctly (without exposing values).
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
import logging

router = APIRouter(prefix="/api/debug", tags=["debug"])
logger = logging.getLogger(__name__)

@router.get("/secrets-status")
async def check_secrets_status():
    """
    Check if API secrets are configured (without exposing actual values).
    Returns boolean status for each secret.
    """
    secrets_status = {
        "reddit_client_id": bool(os.getenv("REDDIT_CLIENT_ID")),
        "reddit_client_secret": bool(os.getenv("REDDIT_CLIENT_SECRET")),
        "reddit_user_agent": bool(os.getenv("REDDIT_USER_AGENT")),
        "fmp_api_key": bool(os.getenv("FMP_API_KEY")),
        "llama_api_key": bool(os.getenv("LLAMA_API_KEY")),
        "postgres_password": bool(os.getenv("POSTGRES_PASSWORD")),
        "jwt_secret_key": bool(os.getenv("JWT_SECRET_KEY")),
    }
    
    # Show first few characters for debugging (safe)
    reddit_id = os.getenv("REDDIT_CLIENT_ID", "")
    reddit_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    
    return JSONResponse(content={
        "secrets_configured": secrets_status,
        "reddit_client_id_preview": f"{reddit_id[:4]}..." if reddit_id else "NOT SET",
        "reddit_client_secret_preview": f"{reddit_secret[:4]}..." if reddit_secret else "NOT SET",
        "reddit_user_agent": os.getenv("REDDIT_USER_AGENT", "NOT SET"),
        "all_secrets_present": all(secrets_status.values()),
    })


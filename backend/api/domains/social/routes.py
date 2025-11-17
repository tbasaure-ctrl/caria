"""
Social Domain Routes.
Per audit document (4.1): Strict domain boundaries.
"""

from fastapi import APIRouter

# Import existing social routes (already have their prefixes)
from api.routes.community import router as community_router
from api.routes.chat import router as chat_router

# Combine into social domain router
router = APIRouter()
router.include_router(community_router)
router.include_router(chat_router)


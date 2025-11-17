"""
Identity Domain Routes - Authentication endpoints.
Per audit document (4.1): Strict domain boundaries.
"""

from fastapi import APIRouter

# Import existing auth routes (already has /api/auth prefix)
from api.routes.auth import router as auth_router

# Re-export as identity router (no additional prefix needed)
router = APIRouter()
router.include_router(auth_router)


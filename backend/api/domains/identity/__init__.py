"""
Identity Domain - Authentication and User Management.

Strict boundaries: This domain handles all authentication, user management,
and session management. Other domains should not directly access user data.
"""

from .routes import router as identity_router
from .services import AuthService, SessionService

__all__ = ["identity_router", "AuthService", "SessionService"]


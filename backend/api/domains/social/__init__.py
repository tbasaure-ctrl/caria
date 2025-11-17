"""
Social Domain - Community Posts and Chat.

Strict boundaries: This domain handles all social interactions.
Other domains should not directly access social data.
"""

from .routes import router as social_router

__all__ = ["social_router"]


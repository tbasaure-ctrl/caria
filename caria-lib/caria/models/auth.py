"""Authentication models for Caria multi-user system."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator


# ============================================================================
# REQUEST MODELS
# ============================================================================

class UserRegister(BaseModel):
    """User registration request."""
    email: EmailStr
    username: str = Field(min_length=3, max_length=50, pattern=r'^[a-zA-Z0-9_-]+$')
    password: str = Field(
        min_length=8,
        max_length=72,
        description="Password must be 8-72 characters. Note: bcrypt has a 72-byte limit, so some Unicode characters may reduce the effective maximum length."
    )
    full_name: Optional[str] = None
    
    @field_validator('password')
    @classmethod
    def validate_password_bytes(cls, v: str) -> str:
        """Validate that password doesn't exceed 72 bytes when encoded."""
        import logging
        logger = logging.getLogger("caria.models.auth")
        
        # Limpiar caracteres invisibles comunes
        cleaned = v.replace('\u200B', '').replace('\u200C', '').replace('\u200D', '').replace('\uFEFF', '').strip()
        
        # Si se limpiaron caracteres, usar la versión limpia
        if cleaned != v:
            logger.info(f"Password cleaned: {len(v)} -> {len(cleaned)} characters")
            v = cleaned
        
        # Primero verificar longitud de caracteres (validación básica de Pydantic ya lo hace)
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        # Luego verificar bytes (para caracteres especiales/Unicode)
        password_bytes = v.encode('utf-8')
        
        # Debug logging si excede el límite
        if len(password_bytes) > 72:
            # Log información de debugging (sin mostrar la contraseña completa)
            logger.warning(
                f"Password validation failed: {len(v)} chars, {len(password_bytes)} bytes. "
                f"First 20 chars: {repr(v[:20])}, Char codes: {[ord(c) for c in v[:30]]}"
            )
            raise ValueError(
                f"Password is too long. Maximum length is 72 bytes when encoded. "
                f"Your password has {len(v)} characters but is {len(password_bytes)} bytes when encoded. "
                f"Please use a shorter password (maximum ~50 characters for safety)."
            )
        return v


class UserLogin(BaseModel):
    """User login request."""
    username: str  # Can be username or email
    password: str


class TokenRefresh(BaseModel):
    """Token refresh request."""
    refresh_token: str


class PasswordReset(BaseModel):
    """Password reset request."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation with new password."""
    token: str
    new_password: str = Field(min_length=8, max_length=100)


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class Token(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class UserPublic(BaseModel):
    """Public user information (no sensitive data)."""
    id: UUID
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserWithToken(BaseModel):
    """User info with authentication token."""
    user: UserPublic
    token: Token


# ============================================================================
# DATABASE MODELS
# ============================================================================

class UserInDB(BaseModel):
    """User as stored in database."""
    id: UUID
    email: str
    username: str
    full_name: Optional[str] = None
    hashed_password: str
    is_active: bool = True
    is_verified: bool = False
    is_superuser: bool = False
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


# ============================================================================
# TOKEN PAYLOAD
# ============================================================================

class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # User ID
    username: str
    email: str
    exp: datetime
    iat: datetime
    type: str  # "access" or "refresh"

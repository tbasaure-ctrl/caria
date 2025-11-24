"""FastAPI dependencies for authentication and authorization."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional
from uuid import UUID

# Configure sys.path before importing caria modules
# This file is in backend/api/, we need to find caria-lib/
CURRENT_FILE = Path(__file__).resolve()
API_DIR = CURRENT_FILE.parent  # backend/api/
BACKEND_DIR = API_DIR.parent   # backend/

# Add backend/ to path for 'api' module
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Go up from backend/api/ to notebooks/, then into caria-lib/
CARIA_LIB = CURRENT_FILE.parent.parent.parent / "caria-lib"
if CARIA_LIB.exists() and str(CARIA_LIB) not in sys.path:
    sys.path.insert(0, str(CARIA_LIB))

# Also check /app/caria-lib (Docker path)
DOCKER_CARIA_LIB = Path("/app/caria-lib")
if DOCKER_CARIA_LIB.exists() and str(DOCKER_CARIA_LIB) not in sys.path:
    sys.path.insert(0, str(DOCKER_CARIA_LIB))

import psycopg2
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from caria.models.auth import UserInDB, UserPublic
from caria.services.auth_service import AuthService

LOGGER = logging.getLogger("caria.api.auth")

# Security scheme for Bearer token
security = HTTPBearer()

_ENV_LOADED = False


def _ensure_env_loaded():
    """Load backend .env file once so local development has credentials."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    import os
    from pathlib import Path

    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key not in os.environ:
                        os.environ[key] = value
                        if key == "POSTGRES_PASSWORD":
                            LOGGER.debug("Loaded POSTGRES_PASSWORD from .env (len=%s)", len(value))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Error loading backend .env: %s", exc)

    _ENV_LOADED = True


def open_db_connection() -> psycopg2.extensions.connection:
    """Return a synchronous psycopg2 connection honoring DATABASE_URL + SSL."""
    import os

    _ensure_env_loaded()

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        try:
            # psycopg2 supports libpq-style URIs (including sslmode & cloudsql host)
            return psycopg2.connect(database_url)
        except psycopg2.Error as exc:
            LOGGER.warning("Failed connecting with DATABASE_URL (%s). Falling back to discrete vars.", exc)

    postgres_password = os.getenv("POSTGRES_PASSWORD")
    if not postgres_password:
        available = [k for k in os.environ.keys() if "POSTGRES" in k or "DATABASE" in k]
        raise RuntimeError(
            f"POSTGRES_PASSWORD or DATABASE_URL must be configured. Available env vars: {available}"
        )

    host = os.getenv("POSTGRES_HOST", "localhost")
    sslmode = os.getenv("POSTGRES_SSLMODE")
    if not sslmode:
        sslmode = "require" if "neon.tech" in host else "prefer"

    return psycopg2.connect(
        host=host,
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "caria_user"),
        password=postgres_password,
        dbname=os.getenv("POSTGRES_DB", "caria"),
        sslmode=sslmode,
    )


def get_db_connection():
    """Yield a managed database connection for FastAPI dependencies."""
    conn = open_db_connection()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_auth_service(db_conn=Depends(get_db_connection)) -> AuthService:
    """Get auth service instance."""
    return AuthService(db_conn)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> UserInDB:
    """
    Get current authenticated user from JWT token.

    Raises HTTPException if token is invalid or user not found.
    """
    token = credentials.credentials

    # Decode token
    try:
        token_payload = AuthService.decode_token(token)
    except ValueError as e:
        LOGGER.warning("Invalid token: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    # Verify token type
    if token_payload.type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user from database
    try:
        user_id = UUID(token_payload.sub)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user ID in token",
        ) from e

    user = auth_service.get_user_by_id(user_id)

    if not user:
        LOGGER.warning("User not found for token: %s", user_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    return user


async def get_current_active_user(
    current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    """Get current active user (alias for get_current_user)."""
    return current_user


async def get_current_verified_user(
    current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    """Get current verified user."""
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required",
        )
    return current_user


async def get_current_superuser(
    current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    """Get current superuser."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser privileges required",
        )
    return current_user


async def get_optional_current_user(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service)
) -> Optional[UserInDB]:
    """
    Get current user if authenticated, None otherwise.

    Useful for endpoints that work both authenticated and unauthenticated.
    """
    # Try to get token from Authorization header
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    token = auth_header.replace("Bearer ", "")

    try:
        token_payload = AuthService.decode_token(token)
        user_id = UUID(token_payload.sub)
        user = auth_service.get_user_by_id(user_id)
        return user if user and user.is_active else None
    except Exception:  # noqa: BLE001
        return None


# ============================================================================
# RATE LIMITING DEPENDENCY
# ============================================================================

class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """Initialize rate limiter."""
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = {}

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for identifier."""
        import time

        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ]
        else:
            self.requests[identifier] = []

        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False

        # Add current request
        self.requests[identifier].append(now)
        return True


# Global rate limiters
public_rate_limiter = RateLimiter(max_requests=30, window_seconds=60)  # 30 req/min for public
auth_rate_limiter = RateLimiter(max_requests=100, window_seconds=60)  # 100 req/min for authenticated


async def check_rate_limit(
    request: Request,
    current_user: Optional[UserInDB] = Depends(get_optional_current_user)
):
    """Check rate limit for current request."""
    # Use user ID if authenticated, otherwise IP address
    if current_user:
        identifier = str(current_user.id)
        rate_limiter = auth_rate_limiter
    else:
        identifier = request.client.host if request.client else "unknown"
        rate_limiter = public_rate_limiter

    if not rate_limiter.is_allowed(identifier):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
        )


# ============================================================================
# USAGE TRACKING DEPENDENCY
# ============================================================================

async def track_usage(
    request: Request,
    current_user: Optional[UserInDB] = Depends(get_optional_current_user),
    db_conn = Depends(get_db_connection)
):
    """Track API usage for analytics and billing."""
    # This would be called after the request completes
    # For now, just a placeholder
    pass

"""Authentication service with JWT, password hashing, and user management."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

import bcrypt
import jwt
from psycopg2.extras import RealDictCursor

from caria.models.auth import TokenPayload, UserInDB

LOGGER = logging.getLogger("caria.services.auth")

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    LOGGER.warning(
        "JWT_SECRET_KEY not set. Using default (INSECURE - change in production!). "
        "Set JWT_SECRET_KEY environment variable."
    )
    JWT_SECRET_KEY = "change-me-in-production-use-secrets"
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour
REFRESH_TOKEN_EXPIRE_DAYS = 30    # 30 days


class AuthService:
    """Service for authentication and user management."""

    def __init__(self, db_connection):
        """Initialize with database connection."""
        self.db = db_connection

    # ========================================================================
    # PASSWORD HASHING
    # ========================================================================

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt.
        
        Note: bcrypt has a 72-byte limit. Passwords longer than 72 bytes
        will be truncated, so we validate the length here.
        """
        password_bytes = password.encode('utf-8')
        
        # Validate password length (bcrypt limit is 72 bytes)
        if len(password_bytes) > 72:
            raise ValueError(
                f"Password is too long. Maximum length is 72 bytes when encoded. "
                f"Your password is {len(password_bytes)} bytes. "
                f"Please use a shorter password."
            )
        
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)

    # ========================================================================
    # JWT TOKEN MANAGEMENT
    # ========================================================================

    @staticmethod
    def create_access_token(user: UserInDB) -> str:
        """Create JWT access token."""
        now = datetime.utcnow()
        expires = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        payload = {
            "sub": str(user.id),
            "username": user.username,
            "email": user.email,
            "exp": expires,
            "iat": now,
            "type": "access"
        }

        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return token

    @staticmethod
    def create_refresh_token(user: UserInDB) -> str:
        """Create JWT refresh token."""
        now = datetime.utcnow()
        expires = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

        payload = {
            "sub": str(user.id),
            "username": user.username,
            "email": user.email,
            "exp": expires,
            "iat": now,
            "type": "refresh"
        }

        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return token

    @staticmethod
    def decode_token(token: str) -> TokenPayload:
        """Decode and validate JWT token."""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return TokenPayload(**payload)
        except jwt.ExpiredSignatureError as exc:
            raise ValueError("Token has expired") from exc
        except jwt.InvalidTokenError as exc:
            raise ValueError("Invalid token") from exc

    # ========================================================================
    # USER MANAGEMENT
    # ========================================================================

    def get_user_by_id(self, user_id: UUID) -> Optional[UserInDB]:
        """Get user by ID."""
        with self.db.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT id, email, username, full_name, hashed_password,
                       is_active, is_verified, is_superuser,
                       created_at, updated_at, last_login
                FROM users
                WHERE id = %s AND is_active = TRUE
                """,
                (str(user_id),)
            )
            row = cursor.fetchone()
            if row:
                return UserInDB(**dict(row))
            return None

    def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """Get user by username."""
        with self.db.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT id, email, username, full_name, hashed_password,
                       is_active, is_verified, is_superuser,
                       created_at, updated_at, last_login
                FROM users
                WHERE username = %s AND is_active = TRUE
                """,
                (username,)
            )
            row = cursor.fetchone()
            if row:
                return UserInDB(**dict(row))
            return None

    def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """Get user by email."""
        with self.db.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT id, email, username, full_name, hashed_password,
                       is_active, is_verified, is_superuser,
                       created_at, updated_at, last_login
                FROM users
                WHERE email = %s AND is_active = TRUE
                """,
                (email,)
            )
            row = cursor.fetchone()
            if row:
                return UserInDB(**dict(row))
            return None

    def get_user_by_username_or_email(self, identifier: str) -> Optional[UserInDB]:
        """Get user by username or email."""
        # Try username first
        user = self.get_user_by_username(identifier)
        if user:
            return user
        # Try email
        return self.get_user_by_email(identifier)

    def create_user(
        self,
        email: str,
        username: str,
        password: str,
        full_name: Optional[str] = None
    ) -> UserInDB:
        """Create a new user."""
        # Check if username or email already exists
        if self.get_user_by_username(username):
            raise ValueError(f"Username '{username}' already exists")
        if self.get_user_by_email(email):
            raise ValueError(f"Email '{email}' already registered")

        # Hash password
        hashed_password = self.hash_password(password)

        # Insert user
        with self.db.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                INSERT INTO users (email, username, full_name, hashed_password)
                VALUES (%s, %s, %s, %s)
                RETURNING id, email, username, full_name, hashed_password,
                          is_active, is_verified, is_superuser,
                          created_at, updated_at, last_login
                """,
                (email, username, full_name, hashed_password)
            )
            row = cursor.fetchone()
            self.db.commit()

            user = UserInDB(**dict(row))
            LOGGER.info("User created: %s (%s)", username, email)
            return user

    def update_last_login(self, user_id: UUID) -> None:
        """Update user's last login timestamp."""
        with self.db.cursor() as cursor:
            cursor.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s",
                (str(user_id),)
            )
            self.db.commit()

    # ========================================================================
    # AUTHENTICATION FLOWS
    # ========================================================================

    def authenticate(self, username_or_email: str, password: str) -> Optional[UserInDB]:
        """Authenticate user with username/email and password."""
        try:
            user = self.get_user_by_username_or_email(username_or_email)
        except Exception as e:
            LOGGER.exception("Error getting user during authentication: %s", e)
            return None

        if not user:
            LOGGER.warning("Authentication failed: user not found: %s", username_or_email)
            return None

        try:
            if not self.verify_password(password, user.hashed_password):
                LOGGER.warning("Authentication failed: invalid password for: %s", username_or_email)
                return None
        except Exception as e:
            LOGGER.exception("Error verifying password during authentication: %s", e)
            return None

        # Update last login (no fallar si esto falla)
        try:
            self.update_last_login(user.id)
        except Exception as e:
            LOGGER.warning("Failed to update last login: %s", e)
            # No fallar la autenticación por esto

        LOGGER.info("User authenticated: %s", user.username)
        return user

    def register_user(
        self,
        email: str,
        username: str,
        password: str,
        full_name: Optional[str] = None
    ) -> UserInDB:
        """Register a new user (wrapper for create_user)."""
        return self.create_user(email, username, password, full_name)

    # ========================================================================
    # REFRESH TOKEN MANAGEMENT
    # ========================================================================

    def store_refresh_token(self, user_id: UUID, token: str) -> None:
        """Store refresh token in database."""
        token_hash = self.hash_password(token)  # Reuse bcrypt for hashing
        expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

        with self.db.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
                VALUES (%s, %s, %s)
                """,
                (str(user_id), token_hash, expires_at)
            )
            self.db.commit()

    def verify_refresh_token(self, user_id: UUID, token: str) -> bool:
        """Verify if refresh token is valid and not revoked."""
        with self.db.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT token_hash, expires_at, revoked
                FROM refresh_tokens
                WHERE user_id = %s AND revoked IS NULL
                ORDER BY created_at DESC
                LIMIT 10
                """,
                (str(user_id),)
            )
            rows = cursor.fetchall()

            for row in rows:
                # Check if token matches
                if self.verify_password(token, row['token_hash']):
                    # Check if not expired
                    if row['expires_at'] > datetime.utcnow():
                        return True

            return False

    def revoke_refresh_token(self, user_id: UUID, token: str) -> None:
        """Revoke a refresh token."""
        token_hash = self.hash_password(token)

        with self.db.cursor() as cursor:
            cursor.execute(
                """
                UPDATE refresh_tokens
                SET revoked = CURRENT_TIMESTAMP
                WHERE user_id = %s AND token_hash = %s
                """,
                (str(user_id), token_hash)
            )
            self.db.commit()

    # ========================================================================
    # AUDIT LOGGING
    # ========================================================================

    def log_audit(
        self,
        user_id: Optional[UUID],
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[UUID] = None,
        details: Optional[dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """Log audit event."""
        import json

        with self.db.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO audit_logs (
                    user_id, action, resource_type, resource_id,
                    details, ip_address, user_agent
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(user_id) if user_id else None,
                    action,
                    resource_type,
                    str(resource_id) if resource_id else None,
                    json.dumps(details) if details else None,
                    ip_address,
                    user_agent
                )
            )
            self.db.commit()

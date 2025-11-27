"""Authentication endpoints for user registration, login, and token management."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.dependencies import get_auth_service, get_current_user
from caria.models.auth import (
    Token,
    TokenRefresh,
    UserInDB,
    UserLogin,
    UserPublic,
    UserRegister,
    UserUpdate,
    UserWithToken,
)
from caria.services.auth_service import AuthService

LOGGER = logging.getLogger("caria.api.auth")

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# ============================================================================
# REGISTRATION
# ============================================================================

@router.post("/register", response_model=UserWithToken, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserRegister,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Register a new user.

    - **email**: Valid email address (must be unique)
    - **username**: Username (3-50 chars, alphanumeric, must be unique)
    - **password**: Password (minimum 8 characters)
    - **full_name**: Optional full name

    Returns user info and JWT tokens.
    """
    try:
        # Create user
        user = auth_service.register_user(
            email=user_data.email,
            username=user_data.username,
            password=user_data.password,
            full_name=user_data.full_name
        )

        # Generate tokens
        access_token = AuthService.create_access_token(user)
        refresh_token = AuthService.create_refresh_token(user)

        # Store refresh token
        auth_service.store_refresh_token(user.id, refresh_token)

        # Log audit event
        auth_service.log_audit(
            user_id=user.id,
            action="user.register",
            resource_type="user",
            resource_id=user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )

        # Prepare response
        from caria.services.auth_service import ACCESS_TOKEN_EXPIRE_MINUTES

        token_response = Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )

        user_public = UserPublic(
            id=user.id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            is_active=user.is_active,
            is_verified=user.is_verified,
            created_at=user.created_at,
            last_login=user.last_login
        )

        return UserWithToken(user=user_public, token=token_response)

    except ValueError as e:
        LOGGER.warning("Registration validation error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        LOGGER.exception("Registration error: %s", e)
        error_detail = str(e) if str(e) else "Registration failed"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        )


# ============================================================================
# LOGIN
# ============================================================================

@router.post("/login", response_model=UserWithToken)
async def login(
    credentials: UserLogin,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Login with username/email and password.

    - **username**: Username or email address
    - **password**: Password

    Returns user info and JWT tokens.
    """
    # Authenticate user (authenticate ahora maneja sus propias excepciones)
    try:
        # Log para debugging (sin mostrar la contraseña completa)
        LOGGER.debug("Login attempt for username: %s, password length: %d chars, %d bytes", 
                    credentials.username, 
                    len(credentials.password),
                    len(credentials.password.encode('utf-8')))
        user = auth_service.authenticate(credentials.username, credentials.password)
    except Exception as e:
        # Esto no debería pasar porque authenticate maneja sus excepciones,
        # pero por si acaso lo capturamos aquí también
        LOGGER.exception("Unexpected error during authentication: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )

    if not user:
        # Log failed login attempt (no fallar si el logging falla)
        try:
            auth_service.log_audit(
                user_id=None,
                action="user.login.failed",
                details={"username": credentials.username},
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent")
            )
        except Exception as audit_error:
            LOGGER.warning("Failed to log audit event: %s", audit_error)
            # No fallar el login por un error de logging

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Generate tokens
    try:
        access_token = AuthService.create_access_token(user)
        refresh_token = AuthService.create_refresh_token(user)
    except Exception as e:
        LOGGER.exception("Error generating tokens: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating authentication tokens"
        )

    # Store refresh token (no fallar si esto falla)
    try:
        auth_service.store_refresh_token(user.id, refresh_token)
    except Exception as e:
        LOGGER.warning("Failed to store refresh token: %s", e)
        # Continuar de todas formas, el login puede funcionar sin refresh token

    # Log successful login (no fallar si esto falla)
    try:
        auth_service.log_audit(
            user_id=user.id,
            action="user.login.success",
            resource_type="user",
            resource_id=user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
    except Exception as audit_error:
        LOGGER.warning("Failed to log audit event: %s", audit_error)
        # No fallar el login por un error de logging

    # Prepare response
    from caria.services.auth_service import ACCESS_TOKEN_EXPIRE_MINUTES

    token_response = Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

    user_public = UserPublic(
        id=user.id,
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        is_active=user.is_active,
        is_verified=user.is_verified,
        created_at=user.created_at,
        last_login=user.last_login
    )

    return UserWithToken(user=user_public, token=token_response)


# ============================================================================
# TOKEN REFRESH
# ============================================================================

@router.post("/refresh", response_model=Token)
async def refresh_token(
    token_data: TokenRefresh,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Refresh access token using refresh token.

    - **refresh_token**: Valid refresh token

    Returns new access and refresh tokens.
    """
    # Decode refresh token
    try:
        token_payload = AuthService.decode_token(token_data.refresh_token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify token type
    if token_payload.type != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user
    from uuid import UUID
    user_id = UUID(token_payload.sub)
    user = auth_service.get_user_by_id(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    # Verify refresh token is valid and not revoked
    if not auth_service.verify_refresh_token(user_id, token_data.refresh_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked refresh token",
        )

    # Generate new tokens
    access_token = AuthService.create_access_token(user)
    new_refresh_token = AuthService.create_refresh_token(user)

    # Store new refresh token
    auth_service.store_refresh_token(user.id, new_refresh_token)

    # Optionally revoke old refresh token (for security)
    # auth_service.revoke_refresh_token(user.id, token_data.refresh_token)

    from caria.services.auth_service import ACCESS_TOKEN_EXPIRE_MINUTES

    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


# ============================================================================
# LOGOUT
# ============================================================================

@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    token_data: TokenRefresh,
    current_user: UserInDB = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Logout and revoke refresh token.

    - **refresh_token**: Refresh token to revoke

    Requires authentication.
    """
    # Revoke refresh token
    auth_service.revoke_refresh_token(current_user.id, token_data.refresh_token)

    # Log logout
    auth_service.log_audit(
        user_id=current_user.id,
        action="user.logout",
        resource_type="user",
        resource_id=current_user.id
    )

    return None


# ============================================================================
# CURRENT USER INFO
# ============================================================================

@router.get("/me", response_model=UserPublic)
async def get_current_user_info(
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Get current authenticated user information.

    Requires authentication.
    """
    return UserPublic(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )


@router.put("/me", response_model=UserPublic)
async def update_current_user_info(
    user_update: UserUpdate,
    current_user: UserInDB = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Update current authenticated user profile information.

    Requires authentication.
    """
    try:
        updated_user = auth_service.update_user(
            user_id=current_user.id,
            full_name=user_update.full_name
        )
        
        return UserPublic(
            id=updated_user.id,
            email=updated_user.email,
            username=updated_user.username,
            full_name=updated_user.full_name,
            is_active=updated_user.is_active,
            is_verified=updated_user.is_verified,
            created_at=updated_user.created_at,
            last_login=updated_user.last_login
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        LOGGER.exception("Error updating user profile: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile"
        )


# ============================================================================
# PASSWORD MANAGEMENT (TODO)
# ============================================================================

@router.post("/password/reset-request", status_code=status.HTTP_202_ACCEPTED)
async def request_password_reset(email: str):
    """
    Request password reset (sends email with reset token).

    TODO: Implement email sending.
    """
    # TODO: Generate reset token, send email
    return {"message": "If email exists, password reset instructions were sent"}


@router.post("/password/reset-confirm", status_code=status.HTTP_200_OK)
async def confirm_password_reset(token: str, new_password: str):
    """
    Confirm password reset with token.

    TODO: Implement token validation and password update.
    """
    # TODO: Validate token, update password
    return {"message": "Password reset successful"}

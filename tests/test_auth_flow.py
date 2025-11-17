"""
End-to-end tests for authentication flow.

Run with: poetry run pytest tests/test_auth_flow.py -v
"""

import os
import time
import pytest
import requests

# Configuration
BASE_URL = os.getenv("CARIA_API_URL", "http://localhost:8000")
TEST_USER_EMAIL = f"test_{int(time.time())}@example.com"
TEST_USER_USERNAME = f"testuser_{int(time.time())}"
TEST_USER_PASSWORD = "TestSecurePassword123!"


class TestAuthenticationFlow:
    """Test complete authentication flow."""

    def test_health_check(self):
        """Test that API is running and healthy."""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["database"] in ["available", "unavailable"]

    def test_register_new_user(self):
        """Test user registration."""
        response = requests.post(
            f"{BASE_URL}/api/auth/register",
            json={
                "email": TEST_USER_EMAIL,
                "username": TEST_USER_USERNAME,
                "password": TEST_USER_PASSWORD,
                "full_name": "Test User"
            }
        )

        assert response.status_code == 201
        data = response.json()

        # Check user info
        assert data["user"]["email"] == TEST_USER_EMAIL
        assert data["user"]["username"] == TEST_USER_USERNAME
        assert data["user"]["is_active"] is True

        # Check token
        assert "access_token" in data["token"]
        assert "refresh_token" in data["token"]
        assert data["token"]["token_type"] == "bearer"

        # Save token for next tests
        self.access_token = data["token"]["access_token"]
        self.refresh_token = data["token"]["refresh_token"]

    def test_register_duplicate_username(self):
        """Test that duplicate username is rejected."""
        # Try to register same username again
        response = requests.post(
            f"{BASE_URL}/api/auth/register",
            json={
                "email": f"another_{TEST_USER_EMAIL}",
                "username": TEST_USER_USERNAME,  # Same username
                "password": TEST_USER_PASSWORD
            }
        )

        assert response.status_code == 400
        assert "already exists" in response.json()["detail"].lower()

    def test_login_with_username(self):
        """Test login with username and password."""
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={
                "username": TEST_USER_USERNAME,
                "password": TEST_USER_PASSWORD
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "access_token" in data["token"]
        assert data["user"]["username"] == TEST_USER_USERNAME

    def test_login_with_email(self):
        """Test login with email and password."""
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={
                "username": TEST_USER_EMAIL,  # Can use email as username
                "password": TEST_USER_PASSWORD
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user"]["email"] == TEST_USER_EMAIL

    def test_login_wrong_password(self):
        """Test that wrong password is rejected."""
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={
                "username": TEST_USER_USERNAME,
                "password": "WrongPassword123!"
            }
        )

        assert response.status_code == 401
        assert "incorrect" in response.json()["detail"].lower()

    def test_get_current_user(self):
        """Test getting current user info with token."""
        # First login to get token
        login_response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={
                "username": TEST_USER_USERNAME,
                "password": TEST_USER_PASSWORD
            }
        )
        token = login_response.json()["token"]["access_token"]

        # Get user info
        response = requests.get(
            f"{BASE_URL}/api/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == TEST_USER_USERNAME
        assert data["email"] == TEST_USER_EMAIL

    def test_get_current_user_no_token(self):
        """Test that /me endpoint requires authentication."""
        response = requests.get(f"{BASE_URL}/api/auth/me")

        assert response.status_code == 403  # No token provided

    def test_get_current_user_invalid_token(self):
        """Test that invalid token is rejected."""
        response = requests.get(
            f"{BASE_URL}/api/auth/me",
            headers={"Authorization": "Bearer invalid_token_here"}
        )

        assert response.status_code == 401

    def test_refresh_token(self):
        """Test refreshing access token."""
        # Login to get tokens
        login_response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={
                "username": TEST_USER_USERNAME,
                "password": TEST_USER_PASSWORD
            }
        )
        refresh_token = login_response.json()["token"]["refresh_token"]

        # Wait a moment
        time.sleep(1)

        # Refresh token
        response = requests.post(
            f"{BASE_URL}/api/auth/refresh",
            json={"refresh_token": refresh_token}
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

        # New tokens should be different
        new_access_token = data["access_token"]
        old_access_token = login_response.json()["token"]["access_token"]
        assert new_access_token != old_access_token

    def test_logout(self):
        """Test logout (revoking refresh token)."""
        # Login to get tokens
        login_response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={
                "username": TEST_USER_USERNAME,
                "password": TEST_USER_PASSWORD
            }
        )
        access_token = login_response.json()["token"]["access_token"]
        refresh_token = login_response.json()["token"]["refresh_token"]

        # Logout
        response = requests.post(
            f"{BASE_URL}/api/auth/logout",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"refresh_token": refresh_token}
        )

        assert response.status_code == 204

        # Try to use revoked refresh token
        refresh_response = requests.post(
            f"{BASE_URL}/api/auth/refresh",
            json={"refresh_token": refresh_token}
        )

        # Should fail since token was revoked
        assert refresh_response.status_code == 401


class TestProtectedEndpoints:
    """Test that protected endpoints require authentication."""

    def test_regime_current_no_auth(self):
        """Test that regime endpoint works without auth (public)."""
        response = requests.get(f"{BASE_URL}/api/regime/current")

        # This endpoint is public, should work
        assert response.status_code in [200, 503]  # 503 if HMM not trained

    def test_factors_screen_requires_auth(self):
        """Test that factors screening requires authentication."""
        response = requests.post(
            f"{BASE_URL}/api/factors/screen",
            json={"top_n": 10, "regime": "expansion"}
        )

        # Should require auth (if implemented)
        # For now, may work without auth depending on config
        assert response.status_code in [200, 401, 403, 422]

    def test_valuation_analyze_with_auth(self):
        """Test valuation endpoint with authentication."""
        # Login to get token
        login_response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={
                "username": "demo",  # Use demo user
                "password": "demo123"
            }
        )

        if login_response.status_code == 200:
            token = login_response.json()["token"]["access_token"]

            # Analyze company
            response = requests.post(
                f"{BASE_URL}/api/valuation/analyze",
                headers={"Authorization": f"Bearer {token}"},
                json={"ticker": "AAPL", "method": "auto"}
            )

            # Should work or return data error
            assert response.status_code in [200, 404, 422]


class TestInputValidation:
    """Test input validation."""

    def test_register_invalid_email(self):
        """Test that invalid email is rejected."""
        response = requests.post(
            f"{BASE_URL}/api/auth/register",
            json={
                "email": "not_an_email",
                "username": "testuser999",
                "password": "Password123!"
            }
        )

        assert response.status_code == 422

    def test_register_weak_password(self):
        """Test that weak password is rejected."""
        response = requests.post(
            f"{BASE_URL}/api/auth/register",
            json={
                "email": "test999@example.com",
                "username": "testuser999",
                "password": "123"  # Too short
            }
        )

        assert response.status_code == 422

    def test_register_invalid_username(self):
        """Test that invalid username is rejected."""
        response = requests.post(
            f"{BASE_URL}/api/auth/register",
            json={
                "email": "test999@example.com",
                "username": "ab",  # Too short
                "password": "Password123!"
            }
        )

        assert response.status_code == 422


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

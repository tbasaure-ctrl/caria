"""
Comprehensive API Testing Script for Caria Backend
Tests all endpoints used by the frontend
"""

import requests
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
import os

# Configuration
# Update this to your Render URL after deployment: https://caria-api.onrender.com
API_BASE_URL = os.getenv("API_BASE_URL", "https://caria-api.onrender.com")
TEST_USER_EMAIL = "test@caria.com"
TEST_USER_PASSWORD = "testpassword123"

@dataclass
class TestResult:
    endpoint: str
    method: str
    status_code: int
    success: bool
    response_data: Optional[Dict[str, Any]]
    error: Optional[str]

    def __str__(self):
        status_emoji = "✅" if self.success else "❌"
        return f"{status_emoji} {self.method} {self.endpoint} -> {self.status_code}"


class CariaAPITester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.auth_token = None
        self.results = []

    def log_result(self, result: TestResult):
        """Log test result"""
        self.results.append(result)
        print(result)
        if result.error:
            print(f"   Error: {result.error}")

    def test_endpoint(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        auth_required: bool = True,
        expected_status: int = 200
    ) -> TestResult:
        """Test a single endpoint"""
        url = f"{self.base_url}{endpoint}"
        headers = {}

        if auth_required and self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        try:
            if method == "GET":
                response = self.session.get(url, headers=headers)
            elif method == "POST":
                headers["Content-Type"] = "application/json"
                response = self.session.post(url, json=data, headers=headers)
            elif method == "DELETE":
                response = self.session.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")

            success = response.status_code == expected_status or (200 <= response.status_code < 300)

            try:
                response_data = response.json()
            except:
                response_data = {"text": response.text[:200]}

            result = TestResult(
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
                success=success,
                response_data=response_data,
                error=None if success else response_data.get("detail", "Unknown error")
            )

        except Exception as e:
            result = TestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                success=False,
                response_data=None,
                error=str(e)
            )

        self.log_result(result)
        return result

    def test_authentication(self):
        """Test authentication endpoints"""
        print("\n" + "="*60)
        print("🔐 TESTING AUTHENTICATION")
        print("="*60)

        # Test health check first (no auth required)
        self.test_endpoint("GET", "/health", auth_required=False)

        # Test login (should fail if user doesn't exist)
        login_result = self.test_endpoint(
            "POST",
            "/api/auth/login",
            data={
                "email": TEST_USER_EMAIL,
                "password": TEST_USER_PASSWORD
            },
            auth_required=False,
            expected_status=200
        )

        if login_result.success and login_result.response_data:
            self.auth_token = login_result.response_data.get("access_token")
            print(f"   ✅ Authenticated successfully! Token: {self.auth_token[:20]}...")

    def test_market_data(self):
        """Test market data endpoints"""
        print("\n" + "="*60)
        print("📈 TESTING MARKET DATA")
        print("="*60)

        # Fear & Greed Index
        self.test_endpoint("GET", "/api/market/fear-greed")

        # Price data for specific ticker
        self.test_endpoint("GET", "/api/prices/realtime/AAPL")
        self.test_endpoint("GET", "/api/prices/realtime/SPY")

        # Reddit sentiment
        self.test_endpoint("GET", "/api/social/reddit?timeframe=day")

    def test_regime_endpoints(self):
        """Test regime and analysis endpoints"""
        print("\n" + "="*60)
        print("🧠 TESTING REGIME & ANALYSIS")
        print("="*60)

        # Current regime
        self.test_endpoint("GET", "/api/regime/current")

        # Valuation
        self.test_endpoint("GET", "/api/valuation/AAPL")

        # Monte Carlo simulation
        self.test_endpoint(
            "POST",
            "/api/montecarlo/simulate",
            data={
                "initial_value": 100000,
                "mu": 0.10,
                "sigma": 0.25,
                "years": 5,
                "simulations": 1000,
                "contributions_per_year": 0,
                "annual_fee": 0
            }
        )

    def test_portfolio_endpoints(self):
        """Test portfolio management endpoints"""
        print("\n" + "="*60)
        print("🏦 TESTING PORTFOLIO")
        print("="*60)

        # Holdings
        self.test_endpoint("GET", "/api/portfolio/holdings")

        # Portfolio analytics
        self.test_endpoint("GET", "/api/portfolio/analysis/metrics?benchmark=SPY")

        # Tactical allocation
        self.test_endpoint("GET", "/api/portfolio/tactical/allocation?regime=expansion")

        # Model portfolio list
        self.test_endpoint("GET", "/api/portfolio/model/list?status=active")

        # Regime test
        self.test_endpoint(
            "POST",
            "/api/portfolio/regime-test",
            data={"regime": "recession"}
        )

    def test_community_endpoints(self):
        """Test community features"""
        print("\n" + "="*60)
        print("👥 TESTING COMMUNITY")
        print("="*60)

        # Community posts
        self.test_endpoint("GET", "/api/community/posts?limit=10&sort_by=upvotes")

        # Community rankings
        self.test_endpoint("GET", "/api/community/rankings")

    def test_thesis_arena(self):
        """Test thesis arena endpoints"""
        print("\n" + "="*60)
        print("⚔️ TESTING THESIS ARENA")
        print("="*60)

        # Challenge thesis
        challenge_result = self.test_endpoint(
            "POST",
            "/api/thesis/arena/challenge",
            data={
                "thesis": "Apple will outperform the market due to strong iPhone sales",
                "ticker": "AAPL",
                "initial_conviction": 75
            }
        )

    def generate_report(self):
        """Generate test report"""
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)

        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed

        print(f"\nTotal Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")

        if failed > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.results:
                if not result.success:
                    print(f"  - {result.method} {result.endpoint}")
                    print(f"    Status: {result.status_code}")
                    print(f"    Error: {result.error}")

        print("\n" + "="*80)

    def run_all_tests(self):
        """Run all tests"""
        print("\n🚀 CARIA API COMPREHENSIVE TEST SUITE")
        print(f"Testing: {self.base_url}")

        # Run test suites
        self.test_authentication()

        if self.auth_token:
            self.test_market_data()
            self.test_regime_endpoints()
            self.test_portfolio_endpoints()
            self.test_community_endpoints()
            self.test_thesis_arena()
        else:
            print("\n⚠️ Skipping authenticated tests (login failed)")
            print("Note: You may need to register a user first or check credentials")

        self.generate_report()


if __name__ == "__main__":
    # Check if backend is running
    print("Checking backend availability...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print(f"✅ Backend is running at {API_BASE_URL}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to backend at {API_BASE_URL}")
        print("Make sure the backend is running:")
        print("  cd backend/api")
        print("  poetry run uvicorn app:app --reload")
        exit(1)

    # Run tests
    tester = CariaAPITester(API_BASE_URL)
    tester.run_all_tests()

#!/usr/bin/env python3
"""
The Odds API Integration Tests
Unit tests and integration tests for The Odds API
"""

import unittest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from odds_api.client import OddsAPIClient
from odds_api.tools import (
    get_sports_list,
    get_live_odds,
    find_arbitrage_opportunities,
    calculate_implied_probability,
    compare_bookmaker_margins
)

class TestOddsAPIClient(unittest.TestCase):
    """Test The Odds API client functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_api_key = "test_api_key_12345"

    def test_client_initialization_with_key(self):
        """Test client initialization with API key"""
        client = OddsAPIClient(api_key=self.mock_api_key)
        self.assertEqual(client.api_key, self.mock_api_key)

    def test_client_initialization_without_key(self):
        """Test client initialization without API key raises error"""
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(ValueError):
                OddsAPIClient()

    @patch.dict('os.environ', {'THE_ODDS_API_KEY': 'env_test_key'})
    def test_client_initialization_from_env(self):
        """Test client initialization from environment variable"""
        client = OddsAPIClient()
        self.assertEqual(client.api_key, 'env_test_key')

    @patch('odds_api.client.requests.Session.get')
    def test_successful_api_request(self, mock_get):
        """Test successful API request"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"key": "test_sport", "title": "Test Sport"}]
        mock_response.headers = {'x-requests-remaining': '499', 'x-requests-used': '1'}
        mock_get.return_value = mock_response

        client = OddsAPIClient(api_key=self.mock_api_key)
        result = client.get_sports()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["key"], "test_sport")
        self.assertEqual(client.requests_remaining, '499')

    @patch('odds_api.client.requests.Session.get')
    def test_api_error_handling(self, mock_get):
        """Test API error handling"""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_get.return_value = mock_response

        client = OddsAPIClient(api_key=self.mock_api_key)

        with self.assertRaises(Exception):
            client.get_sports()

    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        client = OddsAPIClient(api_key=self.mock_api_key)
        client.min_request_interval = 0.1  # Short interval for testing

        import time
        start_time = time.time()
        client._wait_for_rate_limit()
        client._wait_for_rate_limit()
        elapsed = time.time() - start_time

        # Should take at least the minimum interval
        self.assertGreaterEqual(elapsed, 0.1)

class TestOddsAPITools(unittest.TestCase):
    """Test The Odds API tools functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.sample_sports_data = [
            {"key": "americanfootball_nfl", "title": "NFL", "group": "American Football", "active": True},
            {"key": "basketball_nba", "title": "NBA", "group": "Basketball", "active": True},
            {"key": "soccer_epl", "title": "EPL", "group": "Soccer", "active": False}
        ]

        self.sample_odds_data = [
            {
                "id": "event123",
                "sport_title": "NBA",
                "commence_time": "2025-01-15T20:00:00Z",
                "home_team": "Lakers",
                "away_team": "Warriors",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "title": "DraftKings",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Lakers", "price": 2.10},
                                    {"name": "Warriors", "price": 1.85}
                                ]
                            }
                        ]
                    },
                    {
                        "key": "fanduel",
                        "title": "FanDuel",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Lakers", "price": 2.05},
                                    {"name": "Warriors", "price": 1.90}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]

    @patch('odds_api.tools.get_client')
    def test_get_sports_list_success(self, mock_get_client):
        """Test successful sports list retrieval"""
        mock_client = Mock()
        mock_client.get_sports.return_value = self.sample_sports_data
        mock_client.get_usage_info.return_value = {"requests_remaining": 499}
        mock_get_client.return_value = mock_client

        result = get_sports_list()

        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['total_sports'], 3)
        self.assertEqual(result['active_sports'], 2)
        self.assertIn('American Football', result['sports_by_group'])

    @patch('odds_api.tools.get_client')
    def test_get_sports_list_error(self, mock_get_client):
        """Test sports list retrieval error handling"""
        mock_client = Mock()
        mock_client.get_sports.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client

        result = get_sports_list()

        self.assertEqual(result['status'], 'error')
        self.assertIn('error', result)

    @patch('odds_api.tools.get_client')
    def test_get_live_odds_success(self, mock_get_client):
        """Test successful live odds retrieval"""
        mock_client = Mock()
        mock_client.get_odds.return_value = self.sample_odds_data
        mock_client.get_usage_info.return_value = {"requests_remaining": 498}
        mock_get_client.return_value = mock_client

        result = get_live_odds("basketball_nba", "us", "h2h")

        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['sport'], 'basketball_nba')
        self.assertEqual(result['total_events'], 1)
        self.assertIn('analysis', result)

    def test_calculate_implied_probability_decimal(self):
        """Test implied probability calculation for decimal odds"""
        result = calculate_implied_probability(2.50, "decimal")

        self.assertEqual(result['status'], 'success')
        self.assertAlmostEqual(result['implied_probability'], 0.4, places=2)
        self.assertAlmostEqual(result['implied_probability_percent'], 40.0, places=1)

    def test_calculate_implied_probability_american_positive(self):
        """Test implied probability calculation for positive American odds"""
        result = calculate_implied_probability(150, "american")

        self.assertEqual(result['status'], 'success')
        self.assertAlmostEqual(result['implied_probability'], 0.4, places=2)

    def test_calculate_implied_probability_american_negative(self):
        """Test implied probability calculation for negative American odds"""
        result = calculate_implied_probability(-110, "american")

        self.assertEqual(result['status'], 'success')
        self.assertAlmostEqual(result['implied_probability'], 0.524, places=2)

    def test_calculate_implied_probability_invalid_format(self):
        """Test implied probability calculation with invalid format"""
        result = calculate_implied_probability(2.50, "invalid")

        self.assertEqual(result['status'], 'error')
        self.assertIn('error', result)

class TestArbitrageDetection(unittest.TestCase):
    """Test arbitrage detection functionality"""

    def setUp(self):
        """Set up arbitrage test fixtures"""
        # Arbitrage opportunity: Lakers 2.10 at DK, Warriors 1.95 at FD
        # Implied probs: 0.476 + 0.513 = 0.989 < 1.0 (profit!)
        self.arbitrage_event = {
            "id": "arb_event",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Lakers", "price": 2.10},
                                {"name": "Warriors", "price": 1.80}
                            ]
                        }
                    ]
                },
                {
                    "key": "fanduel",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Lakers", "price": 2.00},
                                {"name": "Warriors", "price": 1.95}
                            ]
                        }
                    ]
                }
            ]
        }

        # No arbitrage: higher implied probabilities
        self.no_arbitrage_event = {
            "id": "no_arb_event",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Lakers", "price": 1.90},
                                {"name": "Warriors", "price": 1.90}
                            ]
                        }
                    ]
                }
            ]
        }

    @patch('odds_api.tools.get_client')
    def test_find_arbitrage_opportunities_success(self, mock_get_client):
        """Test finding arbitrage opportunities"""
        mock_client = Mock()
        mock_client.get_odds.return_value = [self.arbitrage_event]
        mock_client.get_usage_info.return_value = {"requests_remaining": 497}
        mock_get_client.return_value = mock_client

        result = find_arbitrage_opportunities("basketball_nba", "us,uk", "h2h", 0.01)

        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['arbitrage_opportunities'], 1)
        self.assertTrue(len(result['opportunities']) > 0)

        opp = result['opportunities'][0]
        self.assertTrue(opp['arbitrage']['has_arbitrage'])
        self.assertGreater(opp['arbitrage']['profit_margin'], 0.01)

    @patch('odds_api.tools.get_client')
    def test_find_arbitrage_no_opportunities(self, mock_get_client):
        """Test when no arbitrage opportunities exist"""
        mock_client = Mock()
        mock_client.get_odds.return_value = [self.no_arbitrage_event]
        mock_client.get_usage_info.return_value = {"requests_remaining": 496}
        mock_get_client.return_value = mock_client

        result = find_arbitrage_opportunities("basketball_nba", "us", "h2h", 0.01)

        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['arbitrage_opportunities'], 0)

class TestIntegrationScenarios(unittest.TestCase):
    """Test real-world integration scenarios"""

    @patch('odds_api.tools.get_client')
    def test_full_betting_workflow(self, mock_get_client):
        """Test complete betting analysis workflow"""
        # Mock client with realistic data
        mock_client = Mock()
        mock_client.get_sports.return_value = [
            {"key": "basketball_nba", "title": "NBA", "active": True}
        ]
        mock_client.get_odds.return_value = [
            {
                "id": "nba123",
                "home_team": "Lakers",
                "away_team": "Warriors",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Lakers", "price": 2.20},
                                    {"name": "Warriors", "price": 1.75}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
        mock_client.get_usage_info.return_value = {"requests_remaining": 495}
        mock_get_client.return_value = mock_client

        # 1. Get available sports
        sports = get_sports_list()
        self.assertEqual(sports['status'], 'success')

        # 2. Get odds for NBA
        odds = get_live_odds("basketball_nba")
        self.assertEqual(odds['status'], 'success')

        # 3. Calculate probabilities
        prob_result = calculate_implied_probability(2.20, "decimal")
        self.assertEqual(prob_result['status'], 'success')

        # 4. Look for arbitrage
        arbitrage = find_arbitrage_opportunities("basketball_nba")
        self.assertEqual(arbitrage['status'], 'success')

    def test_kelly_criterion_calculation(self):
        """Test Kelly Criterion calculation logic"""
        # Test case: 60% true probability, 2.0 decimal odds
        true_prob = 0.60
        decimal_odds = 2.0

        # Calculate Kelly fraction
        b = decimal_odds - 1  # 1.0
        p = true_prob  # 0.60
        q = 1 - p  # 0.40

        kelly_fraction = (b * p - q) / b  # (1*0.6 - 0.4) / 1 = 0.2

        self.assertAlmostEqual(kelly_fraction, 0.2, places=3)

        # Test case: negative Kelly (don't bet)
        true_prob = 0.40
        kelly_fraction = (b * true_prob - (1 - true_prob)) / b
        self.assertLess(kelly_fraction, 0)

    def test_margin_calculation(self):
        """Test bookmaker margin calculation"""
        # Example: Lakers 1.90, Warriors 1.90
        # Implied probs: 0.526 + 0.526 = 1.052
        # Margin: 1.052 - 1 = 0.052 (5.2%)

        odds_1 = 1.90
        odds_2 = 1.90

        implied_prob_1 = 1 / odds_1
        implied_prob_2 = 1 / odds_2
        total_implied_prob = implied_prob_1 + implied_prob_2
        margin = total_implied_prob - 1

        self.assertAlmostEqual(margin, 0.053, places=3)  # ~5.3%

class TestMCPIntegration(unittest.TestCase):
    """Test MCP server integration"""

    def test_mcp_tool_imports(self):
        """Test that MCP tools can be imported"""
        try:
            from odds_api.tools import (
                get_sports_list,
                get_live_odds,
                find_arbitrage_opportunities,
                calculate_implied_probability
            )
            # If we get here, imports are successful
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import MCP tools: {e}")

    def test_environment_variable_handling(self):
        """Test environment variable handling"""
        # Test with missing env var
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(ValueError):
                OddsAPIClient()

        # Test with env var present
        with patch.dict('os.environ', {'THE_ODDS_API_KEY': 'test_key'}):
            client = OddsAPIClient()
            self.assertEqual(client.api_key, 'test_key')

def run_tests():
    """Run all tests"""
    print("🧪 Running The Odds API Integration Tests")
    print("=" * 50)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestOddsAPIClient,
        TestOddsAPITools,
        TestArbitrageDetection,
        TestIntegrationScenarios,
        TestMCPIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nResult: {'✅ PASSED' if success else '❌ FAILED'}")

    return success

if __name__ == "__main__":
    run_tests()
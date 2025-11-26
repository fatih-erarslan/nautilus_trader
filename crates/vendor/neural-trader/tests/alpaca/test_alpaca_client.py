"""
Tests for Alpaca Client
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpaca.alpaca_client import AlpacaClient, OrderSide, OrderType, TimeInForce, Position, Order

class TestAlpacaClient(unittest.TestCase):
    """Test cases for Alpaca Client"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock environment variables
        self.mock_env = {
            'ALPACA_API_KEY': 'test_api_key',
            'ALPACA_SECRET_KEY': 'test_secret_key',
            'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets'
        }

        with patch.dict(os.environ, self.mock_env):
            self.client = AlpacaClient()

    def test_client_initialization(self):
        """Test client initialization with environment variables"""
        self.assertEqual(self.client.api_key, 'test_api_key')
        self.assertEqual(self.client.secret_key, 'test_secret_key')
        self.assertEqual(self.client.base_url, 'https://paper-api.alpaca.markets')

    def test_client_initialization_with_params(self):
        """Test client initialization with explicit parameters"""
        client = AlpacaClient(
            api_key='custom_key',
            secret_key='custom_secret',
            base_url='https://api.alpaca.markets'
        )
        self.assertEqual(client.api_key, 'custom_key')
        self.assertEqual(client.secret_key, 'custom_secret')
        self.assertEqual(client.base_url, 'https://api.alpaca.markets')

    def test_missing_credentials_raises_error(self):
        """Test that missing credentials raise ValueError"""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                AlpacaClient()

    @patch('requests.request')
    def test_make_request_success(self, mock_request):
        """Test successful API request"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'{"test": "data"}'
        mock_response.json.return_value = {"test": "data"}
        mock_request.return_value = mock_response

        result = self.client._make_request('GET', 'test_endpoint')
        self.assertEqual(result, {"test": "data"})

    @patch('requests.request')
    def test_make_request_rate_limit(self, mock_request):
        """Test rate limit handling"""
        # Mock rate limit response followed by success
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429

        success_response = Mock()
        success_response.status_code = 200
        success_response.content = b'{"success": "true"}'
        success_response.json.return_value = {"success": "true"}

        mock_request.side_effect = [rate_limit_response, success_response]

        with patch('time.sleep') as mock_sleep:
            result = self.client._make_request('GET', 'test_endpoint')

        mock_sleep.assert_called_once_with(60)
        self.assertEqual(result, {"success": "true"})

    @patch('requests.request')
    def test_get_account(self, mock_request):
        """Test get account information"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "account_number": "123456789",
            "status": "ACTIVE",
            "currency": "USD",
            "buying_power": "100000.00",
            "cash": "50000.00",
            "portfolio_value": "150000.00"
        }
        mock_request.return_value = mock_response

        account = self.client.get_account()
        self.assertEqual(account['status'], 'ACTIVE')
        self.assertEqual(account['buying_power'], '100000.00')

    @patch('requests.request')
    def test_get_positions(self, mock_request):
        """Test get positions"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "symbol": "AAPL",
                "qty": "100",
                "market_value": "15000.00",
                "avg_entry_price": "150.00",
                "unrealized_pl": "1000.00",
                "unrealized_plpc": "0.0667",
                "side": "long"
            }
        ]
        mock_request.return_value = mock_response

        positions = self.client.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertIsInstance(positions[0], Position)
        self.assertEqual(positions[0].symbol, "AAPL")
        self.assertEqual(positions[0].qty, "100")

    @patch('requests.request')
    def test_place_market_order(self, mock_request):
        """Test placing market order"""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "order_123",
            "symbol": "AAPL",
            "qty": "10",
            "side": "buy",
            "order_type": "market",
            "time_in_force": "day",
            "status": "new",
            "filled_qty": "0",
            "filled_avg_price": "0"
        }
        mock_request.return_value = mock_response

        order = self.client.place_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )

        self.assertIsInstance(order, Order)
        self.assertEqual(order.symbol, "AAPL")
        self.assertEqual(order.side, "buy")
        self.assertEqual(order.order_type, "market")

    @patch('requests.request')
    def test_place_limit_order(self, mock_request):
        """Test placing limit order"""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "order_456",
            "symbol": "GOOGL",
            "qty": "5",
            "side": "sell",
            "order_type": "limit",
            "time_in_force": "gtc",
            "status": "new",
            "filled_qty": "0",
            "filled_avg_price": "0",
            "limit_price": "2500.00"
        }
        mock_request.return_value = mock_response

        order = self.client.place_order(
            symbol="GOOGL",
            qty=5,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            limit_price=2500.00,
            time_in_force=TimeInForce.GTC
        )

        self.assertEqual(order.symbol, "GOOGL")
        self.assertEqual(order.order_type, "limit")

    def test_place_limit_order_without_price_raises_error(self):
        """Test that limit order without price raises error"""
        with self.assertRaises(ValueError):
            self.client.place_order(
                symbol="AAPL",
                qty=10,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT
                # Missing limit_price
            )

    @patch('requests.get')
    def test_get_bars(self, mock_get):
        """Test getting historical bars"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "bars": {
                "AAPL": [
                    {
                        "t": "2023-01-01T00:00:00Z",
                        "o": 150.0,
                        "h": 155.0,
                        "l": 149.0,
                        "c": 153.0,
                        "v": 1000000
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        bars = self.client.get_bars("AAPL", timeframe="1Day", limit=10)

        self.assertIsInstance(bars, pd.DataFrame)
        self.assertIn('open', bars.columns)
        self.assertIn('high', bars.columns)
        self.assertIn('low', bars.columns)
        self.assertIn('close', bars.columns)
        self.assertIn('volume', bars.columns)

    @patch('requests.request')
    def test_is_market_open(self, mock_request):
        """Test market open status"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "timestamp": "2023-01-01T15:30:00Z",
            "is_open": True,
            "next_open": "2023-01-02T14:30:00Z",
            "next_close": "2023-01-01T21:00:00Z"
        }
        mock_request.return_value = mock_response

        is_open = self.client.is_market_open()
        self.assertTrue(is_open)

    @patch('websocket.WebSocketApp')
    def test_start_streaming(self, mock_websocket):
        """Test WebSocket streaming initialization"""
        mock_ws = Mock()
        mock_websocket.return_value = mock_ws

        def dummy_callback(data):
            pass

        self.client.start_streaming(['AAPL', 'GOOGL'], dummy_callback)

        # Verify WebSocket was created
        mock_websocket.assert_called_once()

    def test_order_enum_values(self):
        """Test order enum values"""
        self.assertEqual(OrderType.MARKET.value, "market")
        self.assertEqual(OrderType.LIMIT.value, "limit")
        self.assertEqual(OrderSide.BUY.value, "buy")
        self.assertEqual(OrderSide.SELL.value, "sell")
        self.assertEqual(TimeInForce.DAY.value, "day")
        self.assertEqual(TimeInForce.GTC.value, "gtc")

class TestIntegrationWithMocks(unittest.TestCase):
    """Integration tests using mocks"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_env = {
            'ALPACA_API_KEY': 'test_api_key',
            'ALPACA_SECRET_KEY': 'test_secret_key',
            'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets'
        }

    @patch('requests.request')
    @patch('requests.get')
    def test_complete_trading_workflow(self, mock_get, mock_request):
        """Test complete trading workflow"""
        with patch.dict(os.environ, self.mock_env):
            client = AlpacaClient()

        # Mock account response
        account_response = Mock()
        account_response.status_code = 200
        account_response.json.return_value = {
            "buying_power": "100000.00",
            "portfolio_value": "100000.00"
        }

        # Mock order response
        order_response = Mock()
        order_response.status_code = 201
        order_response.json.return_value = {
            "id": "order_123",
            "symbol": "AAPL",
            "qty": "10",
            "side": "buy",
            "order_type": "market",
            "time_in_force": "day",
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "150.00"
        }

        # Mock position response
        position_response = Mock()
        position_response.status_code = 200
        position_response.json.return_value = {
            "symbol": "AAPL",
            "qty": "10",
            "market_value": "1500.00",
            "avg_entry_price": "150.00",
            "unrealized_pl": "0.00",
            "unrealized_plpc": "0.00",
            "side": "long"
        }

        # Mock bars response
        bars_response = Mock()
        bars_response.status_code = 200
        bars_response.json.return_value = {
            "bars": {
                "AAPL": [
                    {
                        "t": "2023-01-01T00:00:00Z",
                        "o": 150.0, "h": 155.0, "l": 149.0, "c": 153.0, "v": 1000000
                    }
                ]
            }
        }

        mock_request.side_effect = [account_response, order_response, position_response]
        mock_get.return_value = bars_response

        # Test workflow
        # 1. Get account
        account = client.get_account()
        self.assertEqual(account['buying_power'], '100000.00')

        # 2. Place order
        order = client.place_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )
        self.assertEqual(order.symbol, "AAPL")
        self.assertEqual(order.status, "filled")

        # 3. Check position
        position = client.get_position("AAPL")
        self.assertEqual(position.symbol, "AAPL")
        self.assertEqual(position.qty, "10")

        # 4. Get market data
        bars = client.get_bars("AAPL")
        self.assertFalse(bars.empty)

if __name__ == '__main__':
    unittest.main()
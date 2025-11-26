"""
Unit tests for CCXT Interface
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ccxt_integration.interfaces.ccxt_interface import (
    CCXTInterface,
    ExchangeConfig,
    OrderType,
    TimeInForce
)


class TestCCXTInterface:
    """Test suite for CCXTInterface"""
    
    @pytest.fixture
    def exchange_config(self):
        """Create test exchange configuration"""
        return ExchangeConfig(
            name='binance',
            api_key='test_api_key',
            secret='test_secret',
            sandbox=True,
            enable_rate_limit=True
        )
        
    @pytest_asyncio.fixture
    async def ccxt_interface(self, exchange_config):
        """Create CCXTInterface instance"""
        with patch('ccxt_integration.interfaces.ccxt_interface.ccxt_async') as mock_ccxt:
            # Mock exchange class
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock(return_value={
                'BTC/USDT': {'symbol': 'BTC/USDT', 'base': 'BTC', 'quote': 'USDT'},
                'ETH/USDT': {'symbol': 'ETH/USDT', 'base': 'ETH', 'quote': 'USDT'}
            })
            mock_exchange.markets = {
                'BTC/USDT': {'symbol': 'BTC/USDT', 'base': 'BTC', 'quote': 'USDT'},
                'ETH/USDT': {'symbol': 'ETH/USDT', 'base': 'ETH', 'quote': 'USDT'}
            }
            mock_exchange.close = AsyncMock()
            
            mock_ccxt.binance = Mock(return_value=mock_exchange)
            
            interface = CCXTInterface(exchange_config)
            interface.exchange = mock_exchange
            interface._initialized = True
            
            yield interface
            
    @pytest.mark.asyncio
    async def test_initialization(self, exchange_config):
        """Test interface initialization"""
        with patch('ccxt_integration.interfaces.ccxt_interface.ccxt_async') as mock_ccxt:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock(return_value={})
            mock_exchange.close = AsyncMock()
            
            # Mock the exchange class with urls attribute
            mock_exchange_class = Mock()
            mock_exchange_class.urls = {'api': 'https://api.binance.com', 'test': 'https://testnet.binance.vision'}
            mock_exchange_class.return_value = mock_exchange
            
            mock_ccxt.binance = mock_exchange_class
            
            interface = CCXTInterface(exchange_config)
            await interface.initialize()
            
            assert interface._initialized is True
            assert interface.exchange is not None
            mock_exchange.load_markets.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_get_balance(self, ccxt_interface):
        """Test balance fetching"""
        # Mock balance response
        mock_balance = {
            'BTC': {'free': 1.5, 'used': 0.5, 'total': 2.0},
            'USDT': {'free': 10000, 'used': 5000, 'total': 15000},
            'ETH': {'free': 0, 'used': 0, 'total': 0}
        }
        ccxt_interface.exchange.fetch_balance = AsyncMock(return_value=mock_balance)
        
        # Test fetching specific asset
        btc_balance = await ccxt_interface.get_balance('BTC')
        assert btc_balance['asset'] == 'BTC'
        assert btc_balance['free'] == 1.5
        assert btc_balance['total'] == 2.0
        
        # Test fetching all balances
        all_balances = await ccxt_interface.get_balance()
        assert 'BTC' in all_balances
        assert 'USDT' in all_balances
        assert 'ETH' not in all_balances  # Zero balance excluded
        
    @pytest.mark.asyncio
    async def test_get_ticker(self, ccxt_interface):
        """Test ticker fetching"""
        mock_ticker = {
            'symbol': 'BTC/USDT',
            'bid': 45000,
            'ask': 45010,
            'last': 45005,
            'baseVolume': 1234.56,
            'quoteVolume': 55555555.55,
            'timestamp': 1234567890,
            'datetime': '2023-01-01T00:00:00Z',
            'high': 46000,
            'low': 44000,
            'open': 44500,
            'close': 45005,
            'change': 505,
            'percentage': 1.13
        }
        ccxt_interface.exchange.fetch_ticker = AsyncMock(return_value=mock_ticker)
        
        ticker = await ccxt_interface.get_ticker('BTC/USDT')
        
        assert ticker['symbol'] == 'BTC/USDT'
        assert ticker['bid'] == 45000
        assert ticker['ask'] == 45010
        assert ticker['last'] == 45005
        
    @pytest.mark.asyncio
    async def test_place_market_order(self, ccxt_interface):
        """Test placing market order"""
        mock_order = {
            'id': '12345',
            'symbol': 'BTC/USDT',
            'type': 'market',
            'side': 'buy',
            'amount': 0.1,
            'status': 'closed',
            'filled': 0.1,
            'remaining': 0,
            'timestamp': 1234567890,
            'datetime': '2023-01-01T00:00:00Z',
            'fee': {'cost': 0.0001, 'currency': 'BTC'}
        }
        ccxt_interface.exchange.create_market_order = AsyncMock(return_value=mock_order)
        
        order_params = {
            'symbol': 'BTC/USDT',
            'type': 'market',
            'side': 'buy',
            'amount': 0.1
        }
        
        result = await ccxt_interface.place_order(order_params)
        
        assert result['id'] == '12345'
        assert result['status'] == 'closed'
        assert result['filled'] == 0.1
        
    @pytest.mark.asyncio
    async def test_place_limit_order(self, ccxt_interface):
        """Test placing limit order"""
        mock_order = {
            'id': '67890',
            'symbol': 'BTC/USDT',
            'type': 'limit',
            'side': 'sell',
            'amount': 0.1,
            'price': 46000,
            'status': 'open',
            'filled': 0,
            'remaining': 0.1,
            'timestamp': 1234567890,
            'datetime': '2023-01-01T00:00:00Z'
        }
        ccxt_interface.exchange.create_limit_order = AsyncMock(return_value=mock_order)
        
        order_params = {
            'symbol': 'BTC/USDT',
            'type': 'limit',
            'side': 'sell',
            'amount': 0.1,
            'price': 46000
        }
        
        result = await ccxt_interface.place_order(order_params)
        
        assert result['id'] == '67890'
        assert result['status'] == 'open'
        assert result['price'] == 46000
        
    @pytest.mark.asyncio
    async def test_cancel_order(self, ccxt_interface):
        """Test order cancellation"""
        mock_result = {
            'id': '12345',
            'status': 'canceled',
            'info': {}
        }
        ccxt_interface.exchange.cancel_order = AsyncMock(return_value=mock_result)
        
        result = await ccxt_interface.cancel_order('12345', 'BTC/USDT')
        
        assert result['id'] == '12345'
        assert result['status'] == 'cancelled'
        
    @pytest.mark.asyncio
    async def test_get_order_status(self, ccxt_interface):
        """Test fetching order status"""
        mock_order = {
            'id': '12345',
            'symbol': 'BTC/USDT',
            'type': 'limit',
            'side': 'buy',
            'status': 'closed',
            'amount': 0.1,
            'filled': 0.1,
            'remaining': 0,
            'price': 45000,
            'average': 45005,
            'timestamp': 1234567890,
            'datetime': '2023-01-01T00:00:00Z'
        }
        ccxt_interface.exchange.fetch_order = AsyncMock(return_value=mock_order)
        
        status = await ccxt_interface.get_order_status('12345', 'BTC/USDT')
        
        assert status['id'] == '12345'
        assert status['status'] == 'closed'
        assert status['filled'] == 0.1
        
    @pytest.mark.asyncio
    async def test_get_orderbook(self, ccxt_interface):
        """Test orderbook fetching"""
        mock_orderbook = {
            'symbol': 'BTC/USDT',
            'bids': [[45000, 1.0], [44999, 2.0], [44998, 3.0]],
            'asks': [[45001, 1.0], [45002, 2.0], [45003, 3.0]],
            'timestamp': 1234567890,
            'datetime': '2023-01-01T00:00:00Z'
        }
        ccxt_interface.exchange.fetch_order_book = AsyncMock(return_value=mock_orderbook)
        
        orderbook = await ccxt_interface.get_orderbook('BTC/USDT', limit=100)
        
        assert orderbook['symbol'] == 'BTC/USDT'
        assert len(orderbook['bids']) == 3
        assert len(orderbook['asks']) == 3
        assert orderbook['bids'][0][0] == 45000
        
    @pytest.mark.asyncio
    async def test_get_trades(self, ccxt_interface):
        """Test fetching recent trades"""
        mock_trades = [
            {
                'id': '1',
                'timestamp': 1234567890,
                'datetime': '2023-01-01T00:00:00Z',
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'price': 45000,
                'amount': 0.1
            },
            {
                'id': '2',
                'timestamp': 1234567891,
                'datetime': '2023-01-01T00:00:01Z',
                'symbol': 'BTC/USDT',
                'side': 'sell',
                'price': 45001,
                'amount': 0.2
            }
        ]
        ccxt_interface.exchange.fetch_trades = AsyncMock(return_value=mock_trades)
        
        trades = await ccxt_interface.get_trades('BTC/USDT', limit=100)
        
        assert len(trades) == 2
        assert trades[0]['price'] == 45000
        assert trades[1]['side'] == 'sell'
        
    @pytest.mark.asyncio
    async def test_get_klines(self, ccxt_interface):
        """Test fetching OHLCV data"""
        mock_ohlcv = [
            [1234567890000, 45000, 45100, 44900, 45050, 100],
            [1234567950000, 45050, 45150, 45000, 45100, 150]
        ]
        ccxt_interface.exchange.fetch_ohlcv = AsyncMock(return_value=mock_ohlcv)
        
        klines = await ccxt_interface.get_klines('BTC/USDT', '1m', limit=2)
        
        assert len(klines) == 2
        assert klines[0][1] == 45000  # Open
        assert klines[0][2] == 45100  # High
        assert klines[0][3] == 44900  # Low
        assert klines[0][4] == 45050  # Close
        assert klines[0][5] == 100    # Volume
        
    @pytest.mark.asyncio
    async def test_get_account_info(self, ccxt_interface):
        """Test fetching account information"""
        mock_status = {'status': 'ok'}
        mock_fees = {
            'trading': {
                'maker': 0.001,
                'taker': 0.001
            }
        }
        
        ccxt_interface.exchange.fetch_status = AsyncMock(return_value=mock_status)
        ccxt_interface.exchange.fetch_trading_fees = AsyncMock(return_value=mock_fees)
        
        account_info = await ccxt_interface.get_account_info()
        
        assert account_info['exchange'] == 'binance'
        assert account_info['sandbox'] is True
        assert account_info['fees']['trading']['maker'] == 0.001
        
    @pytest.mark.asyncio
    async def test_error_handling(self, ccxt_interface):
        """Test error handling"""
        # Simulate API error
        ccxt_interface.exchange.fetch_ticker = AsyncMock(
            side_effect=Exception("API Error: Rate limit exceeded")
        )
        
        with pytest.raises(Exception) as exc_info:
            await ccxt_interface.get_ticker('BTC/USDT')
            
        assert "API Error" in str(exc_info.value)
        
    @pytest.mark.asyncio
    async def test_close_connection(self, ccxt_interface):
        """Test closing exchange connection"""
        await ccxt_interface.close()
        
        ccxt_interface.exchange.close.assert_called_once()
        assert ccxt_interface._initialized is False
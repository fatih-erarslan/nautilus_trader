"""
Comprehensive tests for Alpaca WebSocket client.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import websockets

from src.alpaca_trading.websocket_client import (
    AlpacaWebSocketClient,
    ConnectionManager,
    MessageProcessor,
    WebSocketError
)


@pytest.fixture
def mock_config():
    """Mock configuration."""
    return {
        'api_key': 'test_key',
        'api_secret': 'test_secret',
        'base_url': 'wss://stream.data.alpaca.markets/v2/test',
        'max_reconnect_attempts': 3,
        'reconnect_delay': 0.1
    }


@pytest.fixture
async def client(mock_config):
    """Create test client."""
    client = AlpacaWebSocketClient(
        api_key=mock_config['api_key'],
        api_secret=mock_config['api_secret'],
        feed='iex',
        raw_data=False
    )
    yield client
    if client._running:
        await client.stop()


class TestConnectionManager:
    """Test connection management."""
    
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_config):
        """Test successful connection."""
        manager = ConnectionManager(mock_config)
        
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({
            'T': 'success',
            'msg': 'authenticated'
        }))
        
        with patch('websockets.connect', AsyncMock(return_value=mock_ws)):
            ws = await manager.connect()
            assert ws is not None
            assert manager.is_connected
            mock_ws.send.assert_called()  # Auth message sent
    
    @pytest.mark.asyncio
    async def test_connect_auth_failure(self, mock_config):
        """Test authentication failure."""
        manager = ConnectionManager(mock_config)
        
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({
            'T': 'error',
            'msg': 'authentication failed'
        }))
        
        with patch('websockets.connect', AsyncMock(return_value=mock_ws)):
            with pytest.raises(WebSocketError):
                await manager.connect()
    
    @pytest.mark.asyncio
    async def test_reconnect_logic(self, mock_config):
        """Test reconnection logic."""
        manager = ConnectionManager(mock_config)
        
        # First connection fails
        connect_attempts = 0
        
        async def mock_connect(*args, **kwargs):
            nonlocal connect_attempts
            connect_attempts += 1
            
            if connect_attempts < 3:
                raise websockets.exceptions.ConnectionClosed(None, None)
            
            # Third attempt succeeds
            mock_ws = AsyncMock()
            mock_ws.recv = AsyncMock(return_value=json.dumps({
                'T': 'success',
                'msg': 'authenticated'
            }))
            return mock_ws
        
        with patch('websockets.connect', mock_connect):
            ws = await manager.connect()
            assert ws is not None
            assert connect_attempts == 3
    
    @pytest.mark.asyncio
    async def test_disconnect(self, mock_config):
        """Test disconnection."""
        manager = ConnectionManager(mock_config)
        
        mock_ws = AsyncMock()
        manager._websocket = mock_ws
        manager._connected = True
        
        await manager.disconnect()
        
        mock_ws.close.assert_called_once()
        assert not manager.is_connected


class TestMessageProcessor:
    """Test message processing."""
    
    def test_parse_trade_message(self):
        """Test parsing trade messages."""
        processor = MessageProcessor()
        
        raw_message = {
            'T': 't',
            'S': 'AAPL',
            'p': 150.25,
            's': 100,
            't': '2024-01-10T10:30:00Z',
            'c': ['@'],
            'i': 12345
        }
        
        trade = processor.parse_message(raw_message)
        
        assert trade['type'] == 'trade'
        assert trade['symbol'] == 'AAPL'
        assert trade['price'] == 150.25
        assert trade['size'] == 100
        assert trade['timestamp'] == '2024-01-10T10:30:00Z'
    
    def test_parse_quote_message(self):
        """Test parsing quote messages."""
        processor = MessageProcessor()
        
        raw_message = {
            'T': 'q',
            'S': 'AAPL',
            'bp': 150.20,
            'bs': 500,
            'ap': 150.25,
            'as': 300,
            't': '2024-01-10T10:30:00Z'
        }
        
        quote = processor.parse_message(raw_message)
        
        assert quote['type'] == 'quote'
        assert quote['symbol'] == 'AAPL'
        assert quote['bid_price'] == 150.20
        assert quote['bid_size'] == 500
        assert quote['ask_price'] == 150.25
        assert quote['ask_size'] == 300
    
    def test_parse_bar_message(self):
        """Test parsing bar messages."""
        processor = MessageProcessor()
        
        raw_message = {
            'T': 'b',
            'S': 'AAPL',
            'o': 150.00,
            'h': 151.00,
            'l': 149.50,
            'c': 150.75,
            'v': 10000,
            't': '2024-01-10T10:30:00Z'
        }
        
        bar = processor.parse_message(raw_message)
        
        assert bar['type'] == 'bar'
        assert bar['symbol'] == 'AAPL'
        assert bar['open'] == 150.00
        assert bar['high'] == 151.00
        assert bar['low'] == 149.50
        assert bar['close'] == 150.75
        assert bar['volume'] == 10000
    
    @pytest.mark.asyncio
    async def test_handle_trade_callback(self):
        """Test trade callback handling."""
        processor = MessageProcessor()
        
        callback_called = False
        received_data = None
        
        async def trade_callback(data):
            nonlocal callback_called, received_data
            callback_called = True
            received_data = data
        
        processor.on_trade(trade_callback)
        
        trade_data = {
            'type': 'trade',
            'symbol': 'AAPL',
            'price': 150.25
        }
        
        await processor.handle_message(trade_data)
        
        assert callback_called
        assert received_data == trade_data


class TestAlpacaWebSocketClient:
    """Test main WebSocket client."""
    
    @pytest.mark.asyncio
    async def test_subscribe_unsubscribe(self, client):
        """Test subscription management."""
        # Subscribe to trades
        await client.subscribe_trades(['AAPL', 'GOOGL'])
        assert 'AAPL' in client._subscriptions['trades']
        assert 'GOOGL' in client._subscriptions['trades']
        
        # Subscribe to quotes
        await client.subscribe_quotes(['AAPL'])
        assert 'AAPL' in client._subscriptions['quotes']
        
        # Unsubscribe
        await client.unsubscribe_trades(['AAPL'])
        assert 'AAPL' not in client._subscriptions['trades']
        assert 'GOOGL' in client._subscriptions['trades']
    
    @pytest.mark.asyncio
    async def test_message_handling(self, client):
        """Test message handling flow."""
        trade_received = asyncio.Event()
        trade_data = None
        
        async def on_trade(data):
            nonlocal trade_data
            trade_data = data
            trade_received.set()
        
        client.on_trade(on_trade)
        
        # Simulate receiving a trade message
        mock_message = json.dumps({
            'T': 't',
            'S': 'AAPL',
            'p': 150.25,
            's': 100,
            't': '2024-01-10T10:30:00Z'
        })
        
        # Mock WebSocket to test message flow
        with patch.object(client._connection_manager, '_websocket') as mock_ws:
            mock_ws.recv = AsyncMock(side_effect=[mock_message, asyncio.CancelledError()])
            
            # Start client in background
            run_task = asyncio.create_task(client.run())
            
            # Wait for trade to be received
            await asyncio.wait_for(trade_received.wait(), timeout=1.0)
            
            assert trade_data is not None
            assert trade_data['symbol'] == 'AAPL'
            assert trade_data['price'] == 150.25
            
            # Stop client
            await client.stop()
            
            try:
                await run_task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client):
        """Test error handling."""
        error_received = asyncio.Event()
        error_data = None
        
        def on_error(error):
            nonlocal error_data
            error_data = error
            error_received.set()
        
        client.on_error(on_error)
        
        # Simulate error message
        mock_message = json.dumps({
            'T': 'error',
            'code': 400,
            'msg': 'Invalid subscription'
        })
        
        with patch.object(client._connection_manager, '_websocket') as mock_ws:
            mock_ws.recv = AsyncMock(side_effect=[mock_message, asyncio.CancelledError()])
            
            run_task = asyncio.create_task(client.run())
            
            await asyncio.wait_for(error_received.wait(), timeout=1.0)
            
            assert error_data is not None
            assert 'Invalid subscription' in str(error_data)
            
            await client.stop()
            
            try:
                await run_task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, client):
        """Test connection lifecycle."""
        # Mock successful connection
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.CancelledError())
        
        with patch('websockets.connect', AsyncMock(return_value=mock_ws)):
            with patch.object(client._connection_manager, '_handle_auth', AsyncMock(return_value=True)):
                # Start client
                run_task = asyncio.create_task(client.run())
                
                # Give it time to connect
                await asyncio.sleep(0.1)
                
                assert client.is_connected
                
                # Stop client
                await client.stop()
                
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
                
                assert not client.is_connected
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, client):
        """Test concurrent subscription operations."""
        # Perform multiple operations concurrently
        await asyncio.gather(
            client.subscribe_trades(['AAPL', 'GOOGL']),
            client.subscribe_quotes(['MSFT', 'TSLA']),
            client.subscribe_bars(['SPY'])
        )
        
        assert len(client._subscriptions['trades']) == 2
        assert len(client._subscriptions['quotes']) == 2
        assert len(client._subscriptions['bars']) == 1
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, client):
        """Test rate limiting behavior."""
        # Configure rate limiting
        client._rate_limiter.max_messages_per_second = 2
        
        messages_sent = []
        
        async def mock_send(msg):
            messages_sent.append((datetime.now(), msg))
        
        with patch.object(client._connection_manager, 'send_message', mock_send):
            # Send multiple messages rapidly
            for i in range(5):
                await client._send_subscription_update()
            
            # Check that messages are rate limited
            assert len(messages_sent) == 5
            
            # Check timing between messages
            for i in range(1, len(messages_sent)):
                time_diff = (messages_sent[i][0] - messages_sent[i-1][0]).total_seconds()
                assert time_diff >= 0.4  # Should be ~0.5 seconds apart


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, client):
        """Test complete workflow."""
        received_messages = {
            'trades': [],
            'quotes': [],
            'bars': []
        }
        
        async def on_trade(data):
            received_messages['trades'].append(data)
        
        async def on_quote(data):
            received_messages['quotes'].append(data)
        
        async def on_bar(data):
            received_messages['bars'].append(data)
        
        # Set up callbacks
        client.on_trade(on_trade)
        client.on_quote(on_quote)
        client.on_bar(on_bar)
        
        # Mock WebSocket messages
        messages = [
            json.dumps({'T': 'success', 'msg': 'authenticated'}),
            json.dumps({'T': 't', 'S': 'AAPL', 'p': 150.25, 's': 100}),
            json.dumps({'T': 'q', 'S': 'AAPL', 'bp': 150.20, 'ap': 150.25}),
            json.dumps({'T': 'b', 'S': 'AAPL', 'o': 150, 'h': 151, 'l': 149, 'c': 150.5, 'v': 1000})
        ]
        
        message_index = 0
        
        async def mock_recv():
            nonlocal message_index
            if message_index < len(messages):
                msg = messages[message_index]
                message_index += 1
                return msg
            await asyncio.sleep(0.1)
            raise asyncio.CancelledError()
        
        mock_ws = AsyncMock()
        mock_ws.recv = mock_recv
        
        with patch('websockets.connect', AsyncMock(return_value=mock_ws)):
            # Subscribe to all data types
            await client.subscribe_trades(['AAPL'])
            await client.subscribe_quotes(['AAPL'])
            await client.subscribe_bars(['AAPL'])
            
            # Run client
            run_task = asyncio.create_task(client.run())
            
            # Wait for messages
            await asyncio.sleep(0.5)
            
            # Verify messages received
            assert len(received_messages['trades']) == 1
            assert len(received_messages['quotes']) == 1
            assert len(received_messages['bars']) == 1
            
            # Stop client
            await client.stop()
            
            try:
                await run_task
            except asyncio.CancelledError:
                pass
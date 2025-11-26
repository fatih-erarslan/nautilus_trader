"""
Test suite for the WebSocketClient implementation.

Tests the WebSocket client for:
- WebSocket connection and authentication
- Real-time market data streaming
- Order book updates
- Trade notifications
- Connection handling (reconnect, error recovery)
- Message parsing and validation
"""

import pytest
import json
import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Dict, List, Any, Callable

import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI, ProtocolError

from src.polymarket.api.websocket_client import (
    WebSocketClient,
    WebSocketError,
    ConnectionError as WSConnectionError,
    AuthenticationError as WSAuthenticationError,
    SubscriptionError,
    MessageParsingError,
    WebSocketConfig,
    MessageType,
    SubscriptionType,
    ConnectionState,
)
from src.polymarket.models.market import Market, Outcome
from src.polymarket.models.order import Order, OrderBook, OrderSide, OrderType, Trade
from src.polymarket.utils import PolymarketConfig


class TestWebSocketConfig:
    """Test cases for WebSocketConfig."""
    
    def test_config_initialization(self):
        """Test WebSocket configuration initialization."""
        config = WebSocketConfig(
            url="wss://ws.polymarket.com",
            ping_interval=30,
            ping_timeout=10,
            max_reconnect_attempts=5,
            reconnect_delay=2.0,
            heartbeat_interval=20
        )
        
        assert config.url == "wss://ws.polymarket.com"
        assert config.ping_interval == 30
        assert config.ping_timeout == 10
        assert config.max_reconnect_attempts == 5
        assert config.reconnect_delay == 2.0
        assert config.heartbeat_interval == 20
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = WebSocketConfig(url="wss://test.com")
        
        assert config.ping_interval == 60
        assert config.ping_timeout == 30
        assert config.max_reconnect_attempts == 3
        assert config.reconnect_delay == 1.0
        assert config.heartbeat_interval == 30


class TestWebSocketClient:
    """Test cases for the WebSocketClient implementation."""
    
    @pytest.fixture
    def polymarket_config(self):
        """Create test Polymarket configuration."""
        return PolymarketConfig(
            api_key="test_api_key",
            private_key="test_private_key",
            ws_url="wss://ws.polymarket.com"
        )
    
    @pytest.fixture
    def ws_config(self):
        """Create test WebSocket configuration."""
        return WebSocketConfig(
            url="wss://ws.polymarket.com",
            ping_interval=30,
            max_reconnect_attempts=3
        )
    
    @pytest.fixture
    def ws_client(self, polymarket_config, ws_config):
        """Create test WebSocket client."""
        return WebSocketClient(
            config=polymarket_config,
            ws_config=ws_config
        )
    
    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket connection."""
        ws = AsyncMock()
        ws.closed = False
        ws.close_code = None
        return ws
    
    def test_client_initialization(self, polymarket_config, ws_config):
        """Test WebSocket client initialization."""
        client = WebSocketClient(config=polymarket_config, ws_config=ws_config)
        
        assert client.config == polymarket_config
        assert client.ws_config == ws_config
        assert client.state == ConnectionState.DISCONNECTED
        assert client._websocket is None
        assert len(client._subscriptions) == 0
        assert len(client._message_handlers) == 0
    
    def test_client_default_initialization(self):
        """Test client with default configuration."""
        client = WebSocketClient()
        
        assert client.config is not None
        assert client.ws_config is not None
        assert client.state == ConnectionState.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_connect_success(self, ws_client, mock_websocket):
        """Test successful WebSocket connection."""
        async def mock_connect_func(*args, **kwargs):
            return mock_websocket
        
        with patch('websockets.connect', side_effect=mock_connect_func) as mock_connect:
            await ws_client.connect()
            
            assert ws_client.state == ConnectionState.CONNECTED
            assert ws_client._websocket == mock_websocket
            mock_connect.assert_called_once()
            
            # Verify connection parameters
            call_args = mock_connect.call_args
            assert 'wss://ws.polymarket.com' in call_args[0][0]
            assert 'ping_interval' in call_args[1]
            assert 'ping_timeout' in call_args[1]
    
    @pytest.mark.asyncio
    async def test_connect_with_authentication(self, ws_client, mock_websocket):
        """Test WebSocket connection with authentication."""
        auth_message = {
            "type": "auth",
            "api_key": "test_api_key",
            "timestamp": "2024-01-15T12:00:00Z",
            "signature": "test_signature"
        }
        
        with patch('websockets.connect', return_value=mock_websocket):
            with patch.object(ws_client, '_generate_auth_message', return_value=auth_message):
                with patch.object(ws_client, '_send_message') as mock_send:
                    await ws_client.connect(authenticate=True)
                    
                    assert ws_client.state == ConnectionState.CONNECTED
                    mock_send.assert_called_once_with(auth_message)
    
    @pytest.mark.asyncio
    async def test_connect_failure_invalid_uri(self, ws_client):
        """Test connection failure with invalid URI."""
        with patch('websockets.connect', side_effect=InvalidURI("Invalid URI")):
            with pytest.raises(WSConnectionError) as exc_info:
                await ws_client.connect()
            
            assert "Invalid URI" in str(exc_info.value)
            assert ws_client.state == ConnectionState.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_connect_failure_network_error(self, ws_client):
        """Test connection failure due to network error."""
        with patch('websockets.connect', side_effect=OSError("Network unreachable")):
            with pytest.raises(WSConnectionError):
                await ws_client.connect()
            
            assert ws_client.state == ConnectionState.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_disconnect_success(self, ws_client, mock_websocket):
        """Test successful disconnection."""
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        
        await ws_client.disconnect()
        
        assert ws_client.state == ConnectionState.DISCONNECTED
        assert ws_client._websocket is None
        mock_websocket.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, ws_client):
        """Test disconnect when not connected."""
        assert ws_client.state == ConnectionState.DISCONNECTED
        
        # Should not raise error
        await ws_client.disconnect()
        assert ws_client.state == ConnectionState.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_subscribe_market_data(self, ws_client, mock_websocket):
        """Test subscribing to market data."""
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        
        market_id = "0x1234567890"
        
        with patch.object(ws_client, '_send_message') as mock_send:
            await ws_client.subscribe_market_data(market_id)
            
            # Verify subscription message sent
            mock_send.assert_called_once()
            message = mock_send.call_args[0][0]
            assert message['type'] == 'subscribe'
            assert message['channel'] == 'market_data'
            assert message['market_id'] == market_id
            
            # Verify subscription tracked
            assert market_id in ws_client._subscriptions
            assert ws_client._subscriptions[market_id]['type'] == SubscriptionType.MARKET_DATA
    
    @pytest.mark.asyncio
    async def test_subscribe_order_book(self, ws_client, mock_websocket):
        """Test subscribing to order book updates."""
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        
        market_id = "0x1234567890"
        
        with patch.object(ws_client, '_send_message') as mock_send:
            await ws_client.subscribe_order_book(market_id, depth=10)
            
            message = mock_send.call_args[0][0]
            assert message['type'] == 'subscribe'
            assert message['channel'] == 'order_book'
            assert message['market_id'] == market_id
            assert message['depth'] == 10
    
    @pytest.mark.asyncio
    async def test_subscribe_trades(self, ws_client, mock_websocket):
        """Test subscribing to trade notifications."""
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        
        market_id = "0x1234567890"
        
        with patch.object(ws_client, '_send_message') as mock_send:
            await ws_client.subscribe_trades(market_id)
            
            message = mock_send.call_args[0][0]
            assert message['type'] == 'subscribe'
            assert message['channel'] == 'trades'
            assert message['market_id'] == market_id
    
    @pytest.mark.asyncio
    async def test_subscribe_user_orders(self, ws_client, mock_websocket):
        """Test subscribing to user order updates."""
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        
        with patch.object(ws_client, '_send_message') as mock_send:
            await ws_client.subscribe_user_orders()
            
            message = mock_send.call_args[0][0]
            assert message['type'] == 'subscribe'
            assert message['channel'] == 'user_orders'
    
    @pytest.mark.asyncio
    async def test_subscribe_when_disconnected(self, ws_client):
        """Test subscription when not connected."""
        assert ws_client.state == ConnectionState.DISCONNECTED
        
        with pytest.raises(WSConnectionError):
            await ws_client.subscribe_market_data("0x123")
    
    @pytest.mark.asyncio
    async def test_unsubscribe_success(self, ws_client, mock_websocket):
        """Test successful unsubscription."""
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        
        # Add subscription
        market_id = "0x1234567890"
        ws_client._subscriptions[market_id] = {
            'type': SubscriptionType.MARKET_DATA,
            'channel': 'market_data'
        }
        
        with patch.object(ws_client, '_send_message') as mock_send:
            await ws_client.unsubscribe(market_id)
            
            message = mock_send.call_args[0][0]
            assert message['type'] == 'unsubscribe'
            assert message['market_id'] == market_id
            
            # Verify subscription removed
            assert market_id not in ws_client._subscriptions
    
    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(self, ws_client, mock_websocket):
        """Test unsubscribing from non-existent subscription."""
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        
        # Should not raise error
        await ws_client.unsubscribe("0xnonexistent")
    
    @pytest.mark.asyncio
    async def test_send_message_success(self, ws_client, mock_websocket):
        """Test successful message sending."""
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        
        message = {"type": "test", "data": "test_data"}
        
        await ws_client._send_message(message)
        
        mock_websocket.send.assert_called_once()
        sent_data = mock_websocket.send.call_args[0][0]
        assert json.loads(sent_data) == message
    
    @pytest.mark.asyncio
    async def test_send_message_when_disconnected(self, ws_client):
        """Test sending message when disconnected."""
        message = {"type": "test"}
        
        with pytest.raises(WSConnectionError):
            await ws_client._send_message(message)
    
    @pytest.mark.asyncio
    async def test_send_message_connection_closed(self, ws_client, mock_websocket):
        """Test sending message when connection is closed."""
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        mock_websocket.send.side_effect = ConnectionClosed(None, None)
        
        message = {"type": "test"}
        
        with pytest.raises(WSConnectionError):
            await ws_client._send_message(message)
        
        # State should be updated
        assert ws_client.state == ConnectionState.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_receive_messages_market_data(self, ws_client, mock_websocket):
        """Test receiving and processing market data messages."""
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        
        # Mock message sequence
        market_data_message = json.dumps({
            "type": "market_data",
            "channel": "market_data",
            "market_id": "0x123",
            "data": {
                "price": "0.65",
                "volume": "1000.0",
                "timestamp": "2024-01-15T12:00:00Z"
            }
        })
        
        mock_websocket.recv = AsyncMock(side_effect=[
            market_data_message,
            ConnectionClosed(None, None)  # End the loop
        ])
        
        # Set up message handler
        received_messages = []
        def handler(message_type, data):
            received_messages.append((message_type, data))
        
        ws_client.add_message_handler(MessageType.MARKET_DATA, handler)
        
        # Start receiving (will stop on ConnectionClosed)
        try:
            await ws_client._receive_messages()
        except ConnectionClosed:
            pass
        
        # Verify message was processed
        assert len(received_messages) == 1
        message_type, data = received_messages[0]
        assert message_type == MessageType.MARKET_DATA
        assert data['market_id'] == "0x123"
        assert data['data']['price'] == "0.65"
    
    @pytest.mark.asyncio
    async def test_receive_messages_order_book(self, ws_client, mock_websocket):
        """Test receiving order book updates."""
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        
        order_book_message = json.dumps({
            "type": "order_book",
            "channel": "order_book",
            "market_id": "0x123",
            "data": {
                "bids": [
                    {"price": "0.65", "size": "1000.0"},
                    {"price": "0.64", "size": "2000.0"}
                ],
                "asks": [
                    {"price": "0.66", "size": "1500.0"},
                    {"price": "0.67", "size": "2500.0"}
                ],
                "timestamp": "2024-01-15T12:00:00Z"
            }
        })
        
        mock_websocket.recv = AsyncMock(side_effect=[
            order_book_message,
            ConnectionClosed(None, None)
        ])
        
        received_messages = []
        def handler(message_type, data):
            received_messages.append((message_type, data))
        
        ws_client.add_message_handler(MessageType.ORDER_BOOK, handler)
        
        try:
            await ws_client._receive_messages()
        except ConnectionClosed:
            pass
        
        assert len(received_messages) == 1
        message_type, data = received_messages[0]
        assert message_type == MessageType.ORDER_BOOK
        assert len(data['data']['bids']) == 2
        assert len(data['data']['asks']) == 2
    
    @pytest.mark.asyncio
    async def test_receive_messages_trade_notification(self, ws_client, mock_websocket):
        """Test receiving trade notifications."""
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        
        trade_message = json.dumps({
            "type": "trade",
            "channel": "trades", 
            "market_id": "0x123",
            "data": {
                "id": "trade_456",
                "price": "0.65",
                "size": "100.0",
                "side": "buy",
                "timestamp": "2024-01-15T12:00:00Z",
                "trader": "0x789"
            }
        })
        
        mock_websocket.recv = AsyncMock(side_effect=[
            trade_message,
            ConnectionClosed(None, None)
        ])
        
        received_messages = []
        def handler(message_type, data):
            received_messages.append((message_type, data))
        
        ws_client.add_message_handler(MessageType.TRADE, handler)
        
        try:
            await ws_client._receive_messages()
        except ConnectionClosed:
            pass
        
        assert len(received_messages) == 1
        message_type, data = received_messages[0]
        assert message_type == MessageType.TRADE
        assert data['data']['id'] == "trade_456"
        assert data['data']['side'] == "buy"
    
    @pytest.mark.asyncio
    async def test_receive_messages_auth_response(self, ws_client, mock_websocket):
        """Test receiving authentication response."""
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        
        auth_success_message = json.dumps({
            "type": "auth_response",
            "success": True,
            "message": "Authentication successful",
            "user_id": "user_123"
        })
        
        mock_websocket.recv = AsyncMock(side_effect=[
            auth_success_message,
            ConnectionClosed(None, None)
        ])
        
        received_messages = []
        def handler(message_type, data):
            received_messages.append((message_type, data))
        
        ws_client.add_message_handler(MessageType.AUTH_RESPONSE, handler)
        
        try:
            await ws_client._receive_messages()
        except ConnectionClosed:
            pass
        
        assert len(received_messages) == 1
        message_type, data = received_messages[0]
        assert message_type == MessageType.AUTH_RESPONSE
        assert data['success'] is True
    
    @pytest.mark.asyncio
    async def test_receive_invalid_json(self, ws_client, mock_websocket):
        """Test handling invalid JSON messages."""
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        
        invalid_json = "invalid json message"
        
        mock_websocket.recv = AsyncMock(side_effect=[
            invalid_json,
            ConnectionClosed(None, None)
        ])
        
        received_errors = []
        def error_handler(message_type, data):
            received_errors.append((message_type, data))
        
        ws_client.add_message_handler(MessageType.ERROR, error_handler)
        
        try:
            await ws_client._receive_messages()
        except ConnectionClosed:
            pass
        
        # Should receive error message for invalid JSON
        assert len(received_errors) == 1
        message_type, data = received_errors[0]
        assert message_type == MessageType.ERROR
        assert "JSON" in data.get('error', '')
    
    @pytest.mark.asyncio
    async def test_receive_unknown_message_type(self, ws_client, mock_websocket):
        """Test handling unknown message types."""
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        
        unknown_message = json.dumps({
            "type": "unknown_type",
            "data": "some data"
        })
        
        mock_websocket.recv = AsyncMock(side_effect=[
            unknown_message,
            ConnectionClosed(None, None)
        ])
        
        # Should not raise error, just log warning
        try:
            await ws_client._receive_messages()
        except ConnectionClosed:
            pass
        
        # Verify no handlers were called (since no handlers for unknown type)
        # This test mainly ensures the client doesn't crash on unknown messages
    
    @pytest.mark.asyncio
    async def test_heartbeat_mechanism(self, ws_client, mock_websocket):
        """Test heartbeat mechanism."""
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        
        with patch.object(ws_client, '_send_heartbeat') as mock_heartbeat:
            with patch('asyncio.sleep') as mock_sleep:
                # Mock sleep to control timing
                mock_sleep.side_effect = [None, asyncio.CancelledError()]
                
                try:
                    await ws_client._heartbeat_task()
                except asyncio.CancelledError:
                    pass
                
                # Verify heartbeat was sent
                mock_heartbeat.assert_called_once()
                mock_sleep.assert_called_with(30)  # Default heartbeat interval
    
    @pytest.mark.asyncio
    async def test_send_heartbeat(self, ws_client, mock_websocket):
        """Test sending heartbeat message."""
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        
        with patch.object(ws_client, '_send_message') as mock_send:
            await ws_client._send_heartbeat()
            
            mock_send.assert_called_once()
            message = mock_send.call_args[0][0]
            assert message['type'] == 'ping'
            assert 'timestamp' in message
    
    @pytest.mark.asyncio
    async def test_reconnection_success(self, ws_client):
        """Test successful reconnection after connection loss."""
        reconnect_count = 0
        
        async def mock_connect(*args, **kwargs):
            nonlocal reconnect_count
            reconnect_count += 1
            if reconnect_count <= 2:
                raise OSError("Connection failed")
            # Third attempt succeeds
            return AsyncMock()
        
        with patch('websockets.connect', side_effect=mock_connect):
            with patch('asyncio.sleep'):  # Speed up test
                success = await ws_client._attempt_reconnect()
                
                assert success is True
                assert ws_client.state == ConnectionState.CONNECTED
                assert reconnect_count == 3
    
    @pytest.mark.asyncio
    async def test_reconnection_max_attempts_exceeded(self, ws_client):
        """Test reconnection failure after max attempts."""
        with patch('websockets.connect', side_effect=OSError("Connection failed")):
            with patch('asyncio.sleep'):  # Speed up test
                success = await ws_client._attempt_reconnect()
                
                assert success is False
                assert ws_client.state == ConnectionState.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_auto_reconnect_on_connection_loss(self, ws_client, mock_websocket):
        """Test automatic reconnection when connection is lost."""
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        ws_client._auto_reconnect = True
        
        # Simulate connection loss during message receiving
        mock_websocket.recv.side_effect = ConnectionClosed(None, None)
        
        with patch.object(ws_client, '_attempt_reconnect', return_value=True) as mock_reconnect:
            with patch.object(ws_client, '_resubscribe_all') as mock_resubscribe:
                try:
                    await ws_client._receive_messages()
                except ConnectionClosed:
                    pass
                
                # Should attempt reconnection
                mock_reconnect.assert_called_once()
                mock_resubscribe.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_resubscribe_after_reconnect(self, ws_client, mock_websocket):
        """Test resubscription after successful reconnection."""
        # Set up existing subscriptions
        ws_client._subscriptions = {
            "0x123": {
                'type': SubscriptionType.MARKET_DATA,
                'channel': 'market_data'
            },
            "0x456": {
                'type': SubscriptionType.ORDER_BOOK,
                'channel': 'order_book',
                'depth': 10
            }
        }
        
        ws_client._websocket = mock_websocket
        ws_client.state = ConnectionState.CONNECTED
        
        with patch.object(ws_client, '_send_message') as mock_send:
            await ws_client._resubscribe_all()
            
            # Should send subscription message for each existing subscription
            assert mock_send.call_count == 2
    
    def test_add_message_handler(self, ws_client):
        """Test adding message handlers."""
        def handler1(msg_type, data):
            pass
        
        def handler2(msg_type, data):
            pass
        
        # Add handlers
        ws_client.add_message_handler(MessageType.MARKET_DATA, handler1)
        ws_client.add_message_handler(MessageType.TRADE, handler2)
        
        assert MessageType.MARKET_DATA in ws_client._message_handlers
        assert MessageType.TRADE in ws_client._message_handlers
        assert len(ws_client._message_handlers[MessageType.MARKET_DATA]) == 1
        assert len(ws_client._message_handlers[MessageType.TRADE]) == 1
    
    def test_remove_message_handler(self, ws_client):
        """Test removing message handlers."""
        def handler(msg_type, data):
            pass
        
        # Add and then remove handler
        ws_client.add_message_handler(MessageType.MARKET_DATA, handler)
        assert MessageType.MARKET_DATA in ws_client._message_handlers
        
        ws_client.remove_message_handler(MessageType.MARKET_DATA, handler)
        assert len(ws_client._message_handlers[MessageType.MARKET_DATA]) == 0
    
    def test_remove_nonexistent_handler(self, ws_client):
        """Test removing non-existent handler."""
        def handler(msg_type, data):
            pass
        
        # Should not raise error
        ws_client.remove_message_handler(MessageType.MARKET_DATA, handler)
    
    def test_get_subscriptions(self, ws_client):
        """Test getting current subscriptions."""
        # Add some subscriptions
        ws_client._subscriptions = {
            "0x123": {'type': SubscriptionType.MARKET_DATA},
            "0x456": {'type': SubscriptionType.ORDER_BOOK}
        }
        
        subscriptions = ws_client.get_subscriptions()
        assert len(subscriptions) == 2
        assert "0x123" in subscriptions
        assert "0x456" in subscriptions
    
    def test_is_connected(self, ws_client):
        """Test connection state checking."""
        # Initially disconnected
        assert ws_client.is_connected() is False
        
        # Set connected state
        ws_client.state = ConnectionState.CONNECTED
        assert ws_client.is_connected() is True
        
        # Set reconnecting state
        ws_client.state = ConnectionState.RECONNECTING
        assert ws_client.is_connected() is False
    
    def test_authentication_message_generation(self, ws_client):
        """Test authentication message generation."""
        with patch('time.time', return_value=1642248000):  # Fixed timestamp
            with patch.object(ws_client, '_sign_message', return_value="test_signature"):
                auth_msg = ws_client._generate_auth_message()
                
                assert auth_msg['type'] == 'auth'
                assert auth_msg['api_key'] == ws_client.config.api_key
                assert 'timestamp' in auth_msg
                assert auth_msg['signature'] == "test_signature"
    
    def test_message_signing(self, ws_client):
        """Test message signing for authentication."""
        message = "test_message_to_sign"
        
        with patch('hmac.new') as mock_hmac:
            mock_hmac.return_value.hexdigest.return_value = "test_signature"
            
            signature = ws_client._sign_message(message)
            
            assert signature == "test_signature"
            mock_hmac.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_manager_usage(self, polymarket_config, ws_config):
        """Test using client as async context manager."""
        with patch('websockets.connect', return_value=AsyncMock()) as mock_connect:
            async with WebSocketClient(config=polymarket_config, ws_config=ws_config) as client:
                assert client.state == ConnectionState.CONNECTED
                assert client._websocket is not None
            
            # Should be disconnected after context exit
            assert client.state == ConnectionState.DISCONNECTED


class TestWebSocketClientIntegration:
    """Integration tests for WebSocketClient."""
    
    @pytest.fixture
    def ws_client(self):
        """Create client for integration tests."""
        config = PolymarketConfig(
            api_key="test_key",
            private_key="test_private_key",
            ws_url="wss://ws.polymarket.com"
        )
        ws_config = WebSocketConfig(
            url="wss://ws.polymarket.com",
            ping_interval=10,
            max_reconnect_attempts=2
        )
        return WebSocketClient(config=config, ws_config=ws_config)
    
    @pytest.mark.asyncio
    async def test_full_trading_workflow(self, ws_client):
        """Test complete trading workflow with WebSocket updates."""
        mock_websocket = AsyncMock()
        
        # Simulate sequence of messages
        messages = [
            json.dumps({
                "type": "auth_response",
                "success": True
            }),
            json.dumps({
                "type": "market_data",
                "market_id": "0x123",
                "data": {"price": "0.65", "volume": "1000"}
            }),
            json.dumps({
                "type": "order_book",
                "market_id": "0x123", 
                "data": {
                    "bids": [{"price": "0.64", "size": "500"}],
                    "asks": [{"price": "0.66", "size": "750"}]
                }
            }),
            json.dumps({
                "type": "trade",
                "market_id": "0x123",
                "data": {
                    "id": "trade_1",
                    "price": "0.65",
                    "size": "100",
                    "side": "buy"
                }
            })
        ]
        
        mock_websocket.recv = AsyncMock(side_effect=messages + [ConnectionClosed(None, None)])
        
        with patch('websockets.connect', return_value=mock_websocket):
            # Track received messages
            received_messages = []
            
            def message_handler(msg_type, data):
                received_messages.append((msg_type, data))
            
            # Set up handlers
            ws_client.add_message_handler(MessageType.AUTH_RESPONSE, message_handler)
            ws_client.add_message_handler(MessageType.MARKET_DATA, message_handler)
            ws_client.add_message_handler(MessageType.ORDER_BOOK, message_handler)
            ws_client.add_message_handler(MessageType.TRADE, message_handler)
            
            # Connect and authenticate
            await ws_client.connect(authenticate=True)
            
            # Subscribe to market data
            await ws_client.subscribe_market_data("0x123")
            await ws_client.subscribe_order_book("0x123")
            await ws_client.subscribe_trades("0x123")
            
            # Start receiving messages
            try:
                await ws_client._receive_messages()
            except ConnectionClosed:
                pass
            
            # Verify all message types were received
            message_types = [msg[0] for msg in received_messages]
            assert MessageType.AUTH_RESPONSE in message_types
            assert MessageType.MARKET_DATA in message_types
            assert MessageType.ORDER_BOOK in message_types
            assert MessageType.TRADE in message_types
            
            await ws_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenario(self, ws_client):
        """Test error recovery and resilience."""
        connection_attempts = []
        
        async def mock_connect_with_failures(*args, **kwargs):
            connection_attempts.append(len(connection_attempts) + 1)
            if len(connection_attempts) <= 2:
                raise OSError("Connection failed")
            return AsyncMock()  # Third attempt succeeds
        
        with patch('websockets.connect', side_effect=mock_connect_with_failures):
            with patch('asyncio.sleep'):  # Speed up test
                # Initial connection should succeed after retries
                await ws_client.connect()
                assert ws_client.state == ConnectionState.CONNECTED
                assert len(connection_attempts) == 3
                
                # Simulate connection loss and auto-recovery
                ws_client._auto_reconnect = True
                mock_websocket = ws_client._websocket
                mock_websocket.recv.side_effect = ConnectionClosed(None, None)
                
                with patch.object(ws_client, '_attempt_reconnect', return_value=True):
                    try:
                        await ws_client._receive_messages()
                    except ConnectionClosed:
                        pass
                
                # Should remain connected after recovery
                assert ws_client.state == ConnectionState.CONNECTED


class TestMessageTypes:
    """Test message type enumeration and parsing."""
    
    def test_message_type_values(self):
        """Test message type enumeration values."""
        assert MessageType.MARKET_DATA == "market_data"
        assert MessageType.ORDER_BOOK == "order_book"
        assert MessageType.TRADE == "trade"
        assert MessageType.USER_ORDER == "user_order"
        assert MessageType.AUTH_RESPONSE == "auth_response"
        assert MessageType.ERROR == "error"
        assert MessageType.HEARTBEAT == "heartbeat"
    
    def test_subscription_type_values(self):
        """Test subscription type enumeration values."""
        assert SubscriptionType.MARKET_DATA == "market_data"
        assert SubscriptionType.ORDER_BOOK == "order_book"
        assert SubscriptionType.TRADES == "trades"
        assert SubscriptionType.USER_ORDERS == "user_orders"
    
    def test_connection_state_values(self):
        """Test connection state enumeration values."""
        assert ConnectionState.DISCONNECTED == "disconnected"
        assert ConnectionState.CONNECTING == "connecting"
        assert ConnectionState.CONNECTED == "connected"
        assert ConnectionState.RECONNECTING == "reconnecting"


class TestWebSocketExceptions:
    """Test WebSocket-specific exceptions."""
    
    def test_websocket_error(self):
        """Test WebSocketError base exception."""
        error = WebSocketError("Test error", {"key": "value"})
        assert str(error) == "Test error"
        assert error.details == {"key": "value"}
        assert isinstance(error.timestamp, datetime)
    
    def test_connection_error(self):
        """Test WSConnectionError exception."""
        error = WSConnectionError("Connection failed", original_error=OSError("Network error"))
        assert "Connection failed" in str(error)
        assert isinstance(error.original_error, OSError)
    
    def test_authentication_error(self):
        """Test WSAuthenticationError exception."""
        error = WSAuthenticationError("Auth failed", user_id="user123")
        assert "Auth failed" in str(error)
        assert error.user_id == "user123"
    
    def test_subscription_error(self):
        """Test SubscriptionError exception."""
        error = SubscriptionError("Sub failed", channel="market_data", market_id="0x123")
        assert "Sub failed" in str(error)
        assert error.channel == "market_data"
        assert error.market_id == "0x123"
    
    def test_message_parsing_error(self):
        """Test MessageParsingError exception."""
        raw_message = '{"invalid": json}'
        error = MessageParsingError("Parse failed", raw_message=raw_message)
        assert "Parse failed" in str(error)
        assert error.raw_message == raw_message
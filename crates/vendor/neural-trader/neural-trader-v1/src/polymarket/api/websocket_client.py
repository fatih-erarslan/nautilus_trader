"""
WebSocket Client for Polymarket Real-time Data

Client for real-time market data streaming:
- WebSocket connection and authentication
- Real-time market data streaming
- Order book updates
- Trade notifications
- Connection handling (reconnect, error recovery)
- Message parsing and validation
"""

import asyncio
import json
import logging
import hmac
import hashlib
import time
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set
from collections import defaultdict

import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI, ProtocolError

from ..utils import PolymarketConfig

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting" 
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


class MessageType(Enum):
    """WebSocket message types"""
    MARKET_DATA = "market_data"
    ORDER_BOOK = "order_book"
    TRADE = "trade"
    USER_ORDER = "user_order"
    AUTH_RESPONSE = "auth_response"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class SubscriptionType(Enum):
    """WebSocket subscription types"""
    MARKET_DATA = "market_data"
    ORDER_BOOK = "order_book"
    TRADES = "trades"
    USER_ORDERS = "user_orders"


@dataclass
class WebSocketConfig:
    """WebSocket client configuration"""
    url: str
    ping_interval: int = 60
    ping_timeout: int = 30
    max_reconnect_attempts: int = 3
    reconnect_delay: float = 1.0
    heartbeat_interval: int = 30


class WebSocketError(Exception):
    """Base WebSocket error"""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}
        self.timestamp = datetime.now()


class ConnectionError(WebSocketError):
    """WebSocket connection error"""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class AuthenticationError(WebSocketError):
    """WebSocket authentication error"""
    
    def __init__(self, message: str, user_id: Optional[str] = None):
        super().__init__(message)
        self.user_id = user_id


class SubscriptionError(WebSocketError):
    """WebSocket subscription error"""
    
    def __init__(self, message: str, channel: Optional[str] = None, market_id: Optional[str] = None):
        super().__init__(message)
        self.channel = channel
        self.market_id = market_id


class MessageParsingError(WebSocketError):
    """Message parsing error"""
    
    def __init__(self, message: str, raw_message: Optional[str] = None):
        super().__init__(message)
        self.raw_message = raw_message


class WebSocketClient:
    """
    WebSocket client for real-time Polymarket data
    
    Provides real-time streaming of market data, order books, trades,
    and user order updates with automatic reconnection and error recovery.
    """
    
    def __init__(
        self,
        config: Optional[PolymarketConfig] = None,
        ws_config: Optional[WebSocketConfig] = None
    ):
        """
        Initialize WebSocket client
        
        Args:
            config: Polymarket configuration
            ws_config: WebSocket-specific configuration
        """
        self.config = config or PolymarketConfig()
        self.ws_config = ws_config or WebSocketConfig(url=self.config.ws_url)
        
        # Connection state
        self.state = ConnectionState.DISCONNECTED
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._reconnect_attempts = 0
        self._auto_reconnect = True
        
        # Subscriptions and handlers
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        
        # Background tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        logger.info("Initialized WebSocketClient")
    
    async def connect(self, authenticate: bool = False) -> None:
        """
        Connect to WebSocket server
        
        Args:
            authenticate: Whether to authenticate after connecting
            
        Raises:
            ConnectionError: If connection fails
        """
        try:
            self.state = ConnectionState.CONNECTING
            logger.info(f"Connecting to WebSocket: {self.ws_config.url}")
            
            # Create WebSocket connection
            self._websocket = await websockets.connect(
                self.ws_config.url,
                ping_interval=self.ws_config.ping_interval,
                ping_timeout=self.ws_config.ping_timeout
            )
            
            self.state = ConnectionState.CONNECTED
            self._reconnect_attempts = 0
            
            logger.info("WebSocket connected successfully")
            
            # Authenticate if requested
            if authenticate:
                auth_message = self._generate_auth_message()
                await self._send_message(auth_message)
            
            # Start background tasks
            self._receive_task = asyncio.create_task(self._receive_messages())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_task_impl())
            
        except (InvalidURI, OSError) as e:
            self.state = ConnectionState.DISCONNECTED
            logger.error(f"Failed to connect to WebSocket: {str(e)}")
            raise ConnectionError(f"Connection failed: {str(e)}", e) from e
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket server"""
        if self.state == ConnectionState.DISCONNECTED:
            return
        
        logger.info("Disconnecting from WebSocket")
        
        # Cancel background tasks
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket connection
        if self._websocket and not self._websocket.closed:
            await self._websocket.close()
        
        self._websocket = None
        self.state = ConnectionState.DISCONNECTED
        self._subscriptions.clear()
        
        logger.info("WebSocket disconnected")
    
    async def subscribe_market_data(self, market_id: str) -> None:
        """
        Subscribe to market data updates
        
        Args:
            market_id: Market identifier
            
        Raises:
            ConnectionError: If not connected
        """
        if self.state != ConnectionState.CONNECTED:
            raise ConnectionError("Not connected to WebSocket")
        
        message = {
            "type": "subscribe",
            "channel": "market_data",
            "market_id": market_id
        }
        
        await self._send_message(message)
        
        self._subscriptions[market_id] = {
            "type": SubscriptionType.MARKET_DATA,
            "channel": "market_data"
        }
        
        logger.debug(f"Subscribed to market data for {market_id}")
    
    async def subscribe_order_book(self, market_id: str, depth: int = 5) -> None:
        """
        Subscribe to order book updates
        
        Args:
            market_id: Market identifier
            depth: Order book depth
            
        Raises:
            ConnectionError: If not connected
        """
        if self.state != ConnectionState.CONNECTED:
            raise ConnectionError("Not connected to WebSocket")
        
        message = {
            "type": "subscribe",
            "channel": "order_book",
            "market_id": market_id,
            "depth": depth
        }
        
        await self._send_message(message)
        
        self._subscriptions[market_id] = {
            "type": SubscriptionType.ORDER_BOOK,
            "channel": "order_book",
            "depth": depth
        }
        
        logger.debug(f"Subscribed to order book for {market_id} with depth {depth}")
    
    async def subscribe_trades(self, market_id: str) -> None:
        """
        Subscribe to trade notifications
        
        Args:
            market_id: Market identifier
            
        Raises:
            ConnectionError: If not connected
        """
        if self.state != ConnectionState.CONNECTED:
            raise ConnectionError("Not connected to WebSocket")
        
        message = {
            "type": "subscribe",
            "channel": "trades",
            "market_id": market_id
        }
        
        await self._send_message(message)
        
        self._subscriptions[market_id] = {
            "type": SubscriptionType.TRADES,
            "channel": "trades"
        }
        
        logger.debug(f"Subscribed to trades for {market_id}")
    
    async def subscribe_user_orders(self) -> None:
        """
        Subscribe to user order updates
        
        Raises:
            ConnectionError: If not connected
        """
        if self.state != ConnectionState.CONNECTED:
            raise ConnectionError("Not connected to WebSocket")
        
        message = {
            "type": "subscribe",
            "channel": "user_orders"
        }
        
        await self._send_message(message)
        
        self._subscriptions["user_orders"] = {
            "type": SubscriptionType.USER_ORDERS,
            "channel": "user_orders"
        }
        
        logger.debug("Subscribed to user orders")
    
    async def unsubscribe(self, identifier: str) -> None:
        """
        Unsubscribe from a channel
        
        Args:
            identifier: Market ID or subscription identifier
        """
        if self.state != ConnectionState.CONNECTED:
            return
        
        if identifier not in self._subscriptions:
            logger.warning(f"No subscription found for {identifier}")
            return
        
        message = {
            "type": "unsubscribe",
            "market_id": identifier
        }
        
        await self._send_message(message)
        
        del self._subscriptions[identifier]
        logger.debug(f"Unsubscribed from {identifier}")
    
    async def _send_message(self, message: Dict[str, Any]) -> None:
        """
        Send message to WebSocket server
        
        Args:
            message: Message to send
            
        Raises:
            ConnectionError: If not connected or send fails
        """
        if not self._websocket or self.state != ConnectionState.CONNECTED:
            raise ConnectionError("Not connected to WebSocket")
        
        try:
            message_str = json.dumps(message)
            await self._websocket.send(message_str)
            logger.debug(f"Sent WebSocket message: {message.get('type', 'unknown')}")
            
        except ConnectionClosed as e:
            self.state = ConnectionState.DISCONNECTED
            logger.error("WebSocket connection closed during send")
            raise ConnectionError("Connection closed during send") from e
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {str(e)}")
            raise ConnectionError(f"Send failed: {str(e)}") from e
    
    async def _receive_messages(self) -> None:
        """
        Receive and process WebSocket messages
        
        Runs continuously until connection is closed or error occurs.
        """
        if not self._websocket:
            return
        
        try:
            async for message in self._websocket:
                try:
                    await self._process_message(message)
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    await self._handle_message_error(str(message), e)
                    
        except ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            self.state = ConnectionState.DISCONNECTED
            
            # Attempt reconnection if auto-reconnect is enabled
            if self._auto_reconnect:
                await self._handle_connection_lost()
                
        except Exception as e:
            logger.error(f"Error in message receiving loop: {str(e)}")
            self.state = ConnectionState.DISCONNECTED
    
    async def _process_message(self, raw_message: str) -> None:
        """
        Process incoming WebSocket message
        
        Args:
            raw_message: Raw message string
        """
        try:
            message_data = json.loads(raw_message)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON message: {str(e)}")
            await self._handle_message_error(raw_message, MessageParsingError("Invalid JSON", raw_message))
            return
        
        message_type = message_data.get('type')
        if not message_type:
            logger.warning("Received message without type field")
            return
        
        try:
            msg_type_enum = MessageType(message_type)
        except ValueError:
            logger.warning(f"Unknown message type: {message_type}")
            return
        
        # Call registered handlers
        handlers = self._message_handlers.get(msg_type_enum, [])
        for handler in handlers:
            try:
                await asyncio.create_task(self._call_handler(handler, msg_type_enum, message_data))
            except Exception as e:
                logger.error(f"Error in message handler: {str(e)}")
    
    async def _call_handler(self, handler: Callable, message_type: MessageType, data: Dict[str, Any]) -> None:
        """Call message handler (sync or async)"""
        if asyncio.iscoroutinefunction(handler):
            await handler(message_type, data)
        else:
            handler(message_type, data)
    
    async def _handle_message_error(self, raw_message: str, error: Exception) -> None:
        """Handle message processing errors"""
        error_handlers = self._message_handlers.get(MessageType.ERROR, [])
        error_data = {
            "error": str(error),
            "raw_message": raw_message,
            "timestamp": datetime.now().isoformat()
        }
        
        for handler in error_handlers:
            try:
                await self._call_handler(handler, MessageType.ERROR, error_data)
            except Exception as e:
                logger.error(f"Error in error handler: {str(e)}")
    
    async def _handle_connection_lost(self) -> None:
        """Handle connection loss and attempt reconnection"""
        if not self._auto_reconnect:
            return
        
        logger.info("Attempting to reconnect after connection loss")
        success = await self._attempt_reconnect()
        
        if success:
            logger.info("Reconnection successful, resubscribing to channels")
            await self._resubscribe_all()
        else:
            logger.error("Failed to reconnect after maximum attempts")
    
    async def _attempt_reconnect(self) -> bool:
        """
        Attempt to reconnect to WebSocket server
        
        Returns:
            True if reconnection successful, False otherwise
        """
        self.state = ConnectionState.RECONNECTING
        
        for attempt in range(self.ws_config.max_reconnect_attempts):
            try:
                logger.info(f"Reconnection attempt {attempt + 1}/{self.ws_config.max_reconnect_attempts}")
                
                # Wait before reconnecting
                if attempt > 0:
                    delay = self.ws_config.reconnect_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                
                # Attempt to connect
                await self.connect()
                return True
                
            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {str(e)}")
                continue
        
        self.state = ConnectionState.DISCONNECTED
        return False
    
    async def _resubscribe_all(self) -> None:
        """Resubscribe to all previous subscriptions after reconnection"""
        subscriptions_copy = dict(self._subscriptions)
        self._subscriptions.clear()
        
        for identifier, sub_info in subscriptions_copy.items():
            try:
                sub_type = sub_info['type']
                
                if sub_type == SubscriptionType.MARKET_DATA:
                    await self.subscribe_market_data(identifier)
                elif sub_type == SubscriptionType.ORDER_BOOK:
                    depth = sub_info.get('depth', 5)
                    await self.subscribe_order_book(identifier, depth)
                elif sub_type == SubscriptionType.TRADES:
                    await self.subscribe_trades(identifier)
                elif sub_type == SubscriptionType.USER_ORDERS:
                    await self.subscribe_user_orders()
                    
            except Exception as e:
                logger.error(f"Failed to resubscribe to {identifier}: {str(e)}")
    
    async def _heartbeat_task_impl(self) -> None:
        """Send periodic heartbeat messages"""
        while self.state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(self.ws_config.heartbeat_interval)
                if self.state == ConnectionState.CONNECTED:
                    await self._send_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat task: {str(e)}")
                break
    
    async def _send_heartbeat(self) -> None:
        """Send heartbeat message"""
        message = {
            "type": "ping",
            "timestamp": int(time.time())
        }
        await self._send_message(message)
    
    def _generate_auth_message(self) -> Dict[str, Any]:
        """
        Generate authentication message
        
        Returns:
            Authentication message
        """
        timestamp = str(int(time.time()))
        message_to_sign = f"{timestamp}{self.config.api_key}"
        signature = self._sign_message(message_to_sign)
        
        return {
            "type": "auth",
            "api_key": self.config.api_key,
            "timestamp": timestamp,
            "signature": signature
        }
    
    def _sign_message(self, message: str) -> str:
        """
        Sign message using HMAC-SHA256
        
        Args:
            message: Message to sign
            
        Returns:
            Hex signature
        """
        secret_bytes = self.config.private_key.encode('utf-8')
        message_bytes = message.encode('utf-8')
        
        signature = hmac.new(
            secret_bytes,
            message_bytes,
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def add_message_handler(self, message_type: MessageType, handler: Callable) -> None:
        """
        Add handler for specific message type
        
        Args:
            message_type: Type of message to handle
            handler: Handler function (can be sync or async)
        """
        self._message_handlers[message_type].append(handler)
        logger.debug(f"Added handler for {message_type.value}")
    
    def remove_message_handler(self, message_type: MessageType, handler: Callable) -> None:
        """
        Remove handler for specific message type
        
        Args:
            message_type: Type of message
            handler: Handler function to remove
        """
        handlers = self._message_handlers.get(message_type, [])
        if handler in handlers:
            handlers.remove(handler)
            logger.debug(f"Removed handler for {message_type.value}")
    
    def get_subscriptions(self) -> Dict[str, Dict[str, Any]]:
        """Get current subscriptions"""
        return dict(self._subscriptions)
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.state == ConnectionState.CONNECTED
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
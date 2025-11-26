"""Alpaca WebSocket Client with MessagePack encoding and automatic reconnection."""

import asyncio
import json
import logging
import msgpack
import ssl
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Union
from urllib.parse import urljoin
import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = logging.getLogger(__name__)


class AlpacaWebSocketClient:
    """WebSocket client for Alpaca Markets real-time data streaming.
    
    Features:
    - MessagePack encoding/decoding for efficient data transfer
    - Automatic reconnection with exponential backoff
    - Connection pooling for data and trading streams
    - Health checks and heartbeat monitoring
    - Latency measurement
    """
    
    # Alpaca WebSocket endpoints
    DATA_STREAM_URL = "wss://stream.data.alpaca.markets/v2/iex"  # Default to IEX for free tier
    DATA_STREAM_URL_SIP = "wss://stream.data.alpaca.markets/v2/sip"  # SIP for paid tier
    TRADING_STREAM_URL = "wss://api.alpaca.markets/stream"
    
    # Message types
    AUTH_MESSAGE = "auth"
    SUBSCRIBE_MESSAGE = "subscribe"
    UNSUBSCRIBE_MESSAGE = "unsubscribe"
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        stream_type: str = "data",
        raw_data: bool = False,
        feed: str = "iex",
        max_reconnect_attempts: int = 10,
        initial_reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
        heartbeat_interval: float = 30.0
    ):
        """Initialize Alpaca WebSocket client.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            stream_type: "data" for market data or "trading" for account updates
            raw_data: Use raw data format (no MessagePack)
            feed: Data feed type ("sip", "iex", or "otc")
            max_reconnect_attempts: Maximum reconnection attempts
            initial_reconnect_delay: Initial delay between reconnect attempts
            max_reconnect_delay: Maximum delay between reconnect attempts
            heartbeat_interval: Interval for heartbeat checks
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.stream_type = stream_type
        self.raw_data = raw_data
        self.feed = feed
        
        # WebSocket connection
        self.ws: Optional[WebSocketClientProtocol] = None
        # Use IEX by default for data stream (free tier), SIP requires subscription
        if stream_type == "data":
            if feed == "sip":
                self.ws_url = self.DATA_STREAM_URL_SIP
            else:
                self.ws_url = self.DATA_STREAM_URL  # Default to IEX
        else:
            self.ws_url = self.TRADING_STREAM_URL
        
        # Reconnection settings
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_reconnect_delay = initial_reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.reconnect_attempt = 0
        self.reconnect_delay = initial_reconnect_delay
        
        # Connection state
        self.connected = False
        self.authenticated = False
        self.subscriptions: Dict[str, List[str]] = {
            "trades": [],
            "quotes": [],
            "bars": [],
            "dailyBars": [],
            "statuses": [],
            "lulds": []
        }
        
        # Callbacks
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.error_handler: Optional[Callable] = None
        
        # Performance monitoring
        self.heartbeat_interval = heartbeat_interval
        self.last_heartbeat = time.time()
        self.last_message_time = time.time()
        self.message_latencies: List[float] = []
        self.reconnect_count = 0
        
        # Tasks
        self.receive_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        
    async def connect(self) -> None:
        """Establish WebSocket connection with automatic reconnection."""
        while self.reconnect_attempt < self.max_reconnect_attempts:
            try:
                logger.info(f"Connecting to {self.ws_url} (attempt {self.reconnect_attempt + 1})")
                
                # SSL context for secure connection
                ssl_context = ssl.create_default_context()
                
                # Connect with appropriate subprotocol
                subprotocols = ["msgpack"] if not self.raw_data else []
                self.ws = await websockets.connect(
                    self.ws_url,
                    ssl=ssl_context,
                    subprotocols=subprotocols,
                    ping_interval=20,
                    ping_timeout=10
                )
                
                self.connected = True
                self.reconnect_attempt = 0
                self.reconnect_delay = self.initial_reconnect_delay
                
                logger.info(f"Connected to {self.ws_url}")
                
                # Authenticate
                await self._authenticate()
                
                # Start receiving messages
                self.receive_task = asyncio.create_task(self._receive_messages())
                self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
                
                # Resubscribe to previous subscriptions
                await self._resubscribe()
                
                break
                
            except Exception as e:
                logger.error(f"Connection failed: {e}")
                self.connected = False
                self.authenticated = False
                
                if self.reconnect_attempt < self.max_reconnect_attempts:
                    logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                    await asyncio.sleep(self.reconnect_delay)
                    
                    self.reconnect_attempt += 1
                    self.reconnect_delay = min(
                        self.reconnect_delay * 2,
                        self.max_reconnect_delay
                    )
                    self.reconnect_count += 1
                else:
                    logger.error("Max reconnection attempts reached")
                    raise
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self.connected = False
        self.authenticated = False
        
        # Cancel tasks
        if self.receive_task:
            self.receive_task.cancel()
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        # Close WebSocket
        if self.ws:
            await self.ws.close()
            self.ws = None
        
        logger.info("Disconnected from WebSocket")
    
    async def _authenticate(self) -> None:
        """Authenticate with Alpaca."""
        auth_message = {
            "action": self.AUTH_MESSAGE,
            "key": self.api_key,
            "secret": self.api_secret
        }
        
        if self.stream_type == "data":
            auth_message["feed"] = self.feed
        
        await self._send_message(auth_message)
        
        # Wait for authentication response
        timeout = 5.0
        start_time = time.time()
        
        while not self.authenticated and time.time() - start_time < timeout:
            await asyncio.sleep(0.1)
        
        if not self.authenticated:
            raise Exception("Authentication failed")
        
        logger.info("Successfully authenticated")
    
    async def subscribe(
        self,
        trades: Optional[List[str]] = None,
        quotes: Optional[List[str]] = None,
        bars: Optional[List[str]] = None,
        daily_bars: Optional[List[str]] = None,
        statuses: Optional[List[str]] = None,
        lulds: Optional[List[str]] = None
    ) -> None:
        """Subscribe to market data streams.
        
        Args:
            trades: List of symbols for trade updates
            quotes: List of symbols for quote updates
            bars: List of symbols for minute bar updates
            daily_bars: List of symbols for daily bar updates
            statuses: List of symbols for trading status updates
            lulds: List of symbols for LULD updates
        """
        if not self.connected or not self.authenticated:
            raise Exception("Not connected or authenticated")
        
        subscription = {}
        
        if trades:
            subscription["trades"] = trades
            self.subscriptions["trades"].extend(trades)
        
        if quotes:
            subscription["quotes"] = quotes
            self.subscriptions["quotes"].extend(quotes)
        
        if bars:
            subscription["bars"] = bars
            self.subscriptions["bars"].extend(bars)
        
        if daily_bars:
            subscription["dailyBars"] = daily_bars
            self.subscriptions["dailyBars"].extend(daily_bars)
        
        if statuses:
            subscription["statuses"] = statuses
            self.subscriptions["statuses"].extend(statuses)
        
        if lulds:
            subscription["lulds"] = lulds
            self.subscriptions["lulds"].extend(lulds)
        
        if subscription:
            message = {
                "action": self.SUBSCRIBE_MESSAGE,
                **subscription
            }
            await self._send_message(message)
            logger.info(f"Subscribed to: {subscription}")
    
    async def unsubscribe(
        self,
        trades: Optional[List[str]] = None,
        quotes: Optional[List[str]] = None,
        bars: Optional[List[str]] = None,
        daily_bars: Optional[List[str]] = None,
        statuses: Optional[List[str]] = None,
        lulds: Optional[List[str]] = None
    ) -> None:
        """Unsubscribe from market data streams."""
        if not self.connected or not self.authenticated:
            raise Exception("Not connected or authenticated")
        
        unsubscription = {}
        
        if trades:
            unsubscription["trades"] = trades
            self.subscriptions["trades"] = [s for s in self.subscriptions["trades"] if s not in trades]
        
        if quotes:
            unsubscription["quotes"] = quotes
            self.subscriptions["quotes"] = [s for s in self.subscriptions["quotes"] if s not in quotes]
        
        if bars:
            unsubscription["bars"] = bars
            self.subscriptions["bars"] = [s for s in self.subscriptions["bars"] if s not in bars]
        
        if daily_bars:
            unsubscription["dailyBars"] = daily_bars
            self.subscriptions["dailyBars"] = [s for s in self.subscriptions["dailyBars"] if s not in daily_bars]
        
        if statuses:
            unsubscription["statuses"] = statuses
            self.subscriptions["statuses"] = [s for s in self.subscriptions["statuses"] if s not in statuses]
        
        if lulds:
            unsubscription["lulds"] = lulds
            self.subscriptions["lulds"] = [s for s in self.subscriptions["lulds"] if s not in lulds]
        
        if unsubscription:
            message = {
                "action": self.UNSUBSCRIBE_MESSAGE,
                **unsubscription
            }
            await self._send_message(message)
            logger.info(f"Unsubscribed from: {unsubscription}")
    
    def register_handler(self, message_type: str, handler: Callable) -> None:
        """Register a message handler for specific message types.
        
        Args:
            message_type: Type of message (e.g., "trade", "quote", "bar")
            handler: Callback function to handle messages
        """
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
        logger.debug(f"Registered handler for {message_type}")
    
    def set_error_handler(self, handler: Callable) -> None:
        """Set error handler callback."""
        self.error_handler = handler
    
    async def _send_message(self, message: Dict[str, Any]) -> None:
        """Send message through WebSocket."""
        if not self.ws:
            raise Exception("WebSocket not connected")
        
        if self.raw_data:
            # Send as JSON
            await self.ws.send(json.dumps(message))
        else:
            # Send as MessagePack
            await self.ws.send(msgpack.packb(message))
        
        logger.debug(f"Sent message: {message}")
    
    async def _receive_messages(self) -> None:
        """Receive and process messages from WebSocket."""
        while self.connected:
            try:
                message = await self.ws.recv()
                receive_time = time.time()
                self.last_message_time = receive_time
                
                # Decode message
                if isinstance(message, bytes):
                    # MessagePack format
                    data = msgpack.unpackb(message, raw=False)
                else:
                    # JSON format
                    data = json.loads(message)
                
                # Process message
                await self._process_message(data, receive_time)
                
            except ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.connected = False
                await self._handle_disconnect()
                
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                if self.error_handler:
                    await self.error_handler(e)
    
    async def _process_message(self, message: Union[Dict, List], receive_time: float) -> None:
        """Process received message."""
        # Handle message array
        if isinstance(message, list):
            for msg in message:
                await self._process_single_message(msg, receive_time)
        else:
            await self._process_single_message(message, receive_time)
    
    async def _process_single_message(self, message: Dict[str, Any], receive_time: float) -> None:
        """Process a single message."""
        msg_type = message.get("T", message.get("msg"))
        
        # Calculate latency if timestamp is available
        if "t" in message:
            try:
                # Parse timestamp
                msg_time = datetime.fromisoformat(message["t"].replace("Z", "+00:00"))
                latency = (receive_time - msg_time.timestamp()) * 1000  # ms
                self.message_latencies.append(latency)
                
                # Keep only recent latencies
                if len(self.message_latencies) > 1000:
                    self.message_latencies.pop(0)
            except Exception:
                pass
        
        # Handle authentication response
        if msg_type == "success" and message.get("msg") == "authenticated":
            self.authenticated = True
            logger.info("Authentication successful")
            return
        elif isinstance(message, list):
            # Handle message arrays
            for msg in message:
                if msg.get("T") == "success" and msg.get("msg") == "authenticated":
                    self.authenticated = True
                    logger.info("Authentication successful") 
                    return
        
        # Handle errors
        if msg_type == "error":
            error_msg = message.get("msg", "Unknown error")
            logger.error(f"Server error: {error_msg}")
            if self.error_handler:
                await self.error_handler(Exception(error_msg))
            return
        
        # Route to appropriate handlers
        handlers = self.message_handlers.get(msg_type, [])
        for handler in handlers:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Handler error for {msg_type}: {e}")
    
    async def _resubscribe(self) -> None:
        """Resubscribe to previous subscriptions after reconnection."""
        if any(self.subscriptions.values()):
            logger.info("Resubscribing to previous subscriptions")
            
            # Create subscription message with all previous subscriptions
            subscription = {}
            for key, symbols in self.subscriptions.items():
                if symbols:
                    subscription[key] = list(set(symbols))  # Remove duplicates
            
            if subscription:
                message = {
                    "action": self.SUBSCRIBE_MESSAGE,
                    **subscription
                }
                await self._send_message(message)
    
    async def _heartbeat_monitor(self) -> None:
        """Monitor connection health and trigger reconnection if needed."""
        while self.connected:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Check if we've received messages recently
                time_since_last_message = time.time() - self.last_message_time
                
                if time_since_last_message > self.heartbeat_interval * 2:
                    logger.warning(f"No messages received for {time_since_last_message:.1f} seconds")
                    
                    # Try to send a ping
                    if self.ws:
                        try:
                            pong = await self.ws.ping()
                            await asyncio.wait_for(pong, timeout=5.0)
                            logger.debug("Ping successful")
                        except Exception:
                            logger.error("Ping failed, triggering reconnection")
                            self.connected = False
                            await self._handle_disconnect()
                
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
    
    async def _handle_disconnect(self) -> None:
        """Handle disconnection and trigger reconnection."""
        logger.info("Handling disconnection")
        
        # Cancel receive task
        if self.receive_task:
            self.receive_task.cancel()
        
        # Close WebSocket
        if self.ws:
            await self.ws.close()
            self.ws = None
        
        # Trigger reconnection
        await self.connect()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get connection and performance metrics."""
        avg_latency = sum(self.message_latencies) / len(self.message_latencies) if self.message_latencies else 0
        
        return {
            "connected": self.connected,
            "authenticated": self.authenticated,
            "reconnect_count": self.reconnect_count,
            "avg_latency_ms": round(avg_latency, 2),
            "min_latency_ms": round(min(self.message_latencies), 2) if self.message_latencies else 0,
            "max_latency_ms": round(max(self.message_latencies), 2) if self.message_latencies else 0,
            "subscriptions": {k: len(v) for k, v in self.subscriptions.items()},
            "total_subscriptions": sum(len(v) for v in self.subscriptions.values())
        }
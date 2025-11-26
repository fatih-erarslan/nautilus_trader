"""
Finnhub WebSocket client for real-time market data
Free tier includes real-time US stocks via WebSocket
"""
import asyncio
import aiohttp
import websockets
import json
import time
import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from collections import defaultdict

from .realtime_manager import DataSource, DataPoint

logger = logging.getLogger(__name__)


class FinnhubClient(DataSource):
    """Finnhub WebSocket client for real-time data"""
    
    WEBSOCKET_URL = "wss://ws.finnhub.io"
    REST_API_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: str, use_websocket: bool = True):
        super().__init__("finnhub")
        self.api_key = api_key
        self.use_websocket = use_websocket
        
        # WebSocket connection
        self.ws_connection = None
        self.ws_task = None
        self.reconnect_delay = 1  # Start with 1 second
        self.max_reconnect_delay = 60
        
        # REST session
        self.rest_session: Optional[aiohttp.ClientSession] = None
        
        # Subscription management
        self.subscribed_symbols: Set[str] = set()
        self.pending_subscriptions: Set[str] = set()
        
        # Callbacks
        self.data_callback = None
        
        # Metrics
        self.messages_received = 0
        self.connection_start_time = None
        self.last_message_time = None
        self.error_count = 0
        
        # Heartbeat
        self.heartbeat_task = None
        self.last_pong = time.time()
    
    async def connect(self) -> bool:
        """Connect to Finnhub"""
        try:
            self.connection_start_time = time.time()
            
            # Create REST session
            self.rest_session = aiohttp.ClientSession()
            
            # Connect via WebSocket if enabled
            if self.use_websocket:
                success = await self._connect_websocket()
                if success:
                    self.connection_type = "WebSocket"
                    self.is_connected = True
                    return True
                else:
                    logger.warning("WebSocket connection failed, falling back to REST polling")
            
            # Fallback to REST polling
            self.connection_type = "REST"
            self.is_connected = True
            
            # Start REST polling task
            self.ws_task = asyncio.create_task(self._rest_polling_loop())
            
            logger.info(f"Connected to Finnhub via {self.connection_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Finnhub: {e}")
            self.is_connected = False
            return False
    
    async def _connect_websocket(self) -> bool:
        """Establish WebSocket connection"""
        try:
            # Connect with authentication
            ws_url = f"{self.WEBSOCKET_URL}?token={self.api_key}"
            self.ws_connection = await websockets.connect(ws_url)
            
            # Start message handler
            self.ws_task = asyncio.create_task(self._websocket_handler())
            
            # Start heartbeat
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Resubscribe to any pending symbols
            if self.pending_subscriptions:
                for symbol in self.pending_subscriptions:
                    await self._send_subscription(symbol, True)
                self.pending_subscriptions.clear()
            
            self.reconnect_delay = 1  # Reset reconnect delay on success
            logger.info("WebSocket connection established")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def _websocket_handler(self) -> None:
        """Handle WebSocket messages"""
        try:
            async for message in self.ws_connection:
                try:
                    data = json.loads(message)
                    await self._process_websocket_message(data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON: {message}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            await self._handle_websocket_disconnect()
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
            await self._handle_websocket_disconnect()
    
    async def _process_websocket_message(self, data: Dict[str, Any]) -> None:
        """Process WebSocket message"""
        msg_type = data.get('type')
        
        if msg_type == 'trade':
            # Real-time trade data
            await self._process_trades(data.get('data', []))
            
        elif msg_type == 'ping':
            # Respond to ping
            self.last_pong = time.time()
            
        elif msg_type == 'error':
            logger.error(f"Finnhub error: {data.get('msg')}")
            self.error_count += 1
            
        self.messages_received += 1
        self.last_message_time = time.time()
    
    async def _process_trades(self, trades: List[Dict[str, Any]]) -> None:
        """Process trade data"""
        for trade in trades:
            try:
                # Calculate latency
                trade_time = trade.get('t', 0) / 1000  # Convert ms to seconds
                current_time = time.time()
                latency_ms = (current_time - trade_time) * 1000 if trade_time > 0 else None
                
                # Create DataPoint
                data_point = DataPoint(
                    source=self.name,
                    symbol=trade.get('s', ''),
                    timestamp=datetime.fromtimestamp(trade_time) if trade_time > 0 else datetime.now(),
                    price=float(trade.get('p', 0)),
                    volume=int(trade.get('v', 0)),
                    latency_ms=latency_ms,
                    sequence_id=f"finnhub_{trade.get('s')}_{trade.get('t')}",
                    metadata={
                        'conditions': trade.get('c', []),
                        'trade_timestamp_ms': trade.get('t')
                    }
                )
                
                # Deliver to callback
                if self.data_callback:
                    await self.data_callback(data_point)
                    
            except Exception as e:
                logger.error(f"Error processing trade: {e}")
    
    async def _handle_websocket_disconnect(self) -> None:
        """Handle WebSocket disconnection with reconnect"""
        self.is_connected = False
        
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
        
        # Cancel heartbeat
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        # Attempt reconnection with exponential backoff
        for attempt in range(self.max_reconnect_attempts):
            logger.info(f"Reconnecting to WebSocket (attempt {attempt + 1})")
            await asyncio.sleep(self.reconnect_delay)
            
            if await self._connect_websocket():
                return
            
            # Exponential backoff
            self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
        
        logger.error("Failed to reconnect to WebSocket, falling back to REST")
        self.connection_type = "REST"
        self.ws_task = asyncio.create_task(self._rest_polling_loop())
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to keep connection alive"""
        while self.is_connected and self.ws_connection:
            try:
                # Check if we've received a pong recently
                if time.time() - self.last_pong > 60:
                    logger.warning("No pong received for 60 seconds, reconnecting")
                    await self._handle_websocket_disconnect()
                    break
                
                # Send ping
                if self.ws_connection and not self.ws_connection.closed:
                    await self.ws_connection.ping()
                
                await asyncio.sleep(30)  # Ping every 30 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                break
    
    async def _rest_polling_loop(self) -> None:
        """Poll REST API for updates (fallback mode)"""
        while self.is_connected and self.connection_type == "REST":
            try:
                # Poll each subscribed symbol
                for symbol in list(self.subscribed_symbols):
                    quote = await self._fetch_rest_quote(symbol)
                    if quote:
                        await self._process_rest_quote(symbol, quote)
                    
                    # Small delay between requests
                    await asyncio.sleep(0.1)
                
                # Wait before next polling cycle
                await asyncio.sleep(1)  # Poll every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"REST polling error: {e}")
                await asyncio.sleep(5)
    
    async def _fetch_rest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch quote via REST API"""
        if not self.rest_session:
            return None
        
        try:
            url = f"{self.REST_API_URL}/quote"
            params = {"symbol": symbol, "token": self.api_key}
            
            async with self.rest_session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"REST API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
        
        return None
    
    async def _process_rest_quote(self, symbol: str, quote: Dict[str, Any]) -> None:
        """Process REST API quote"""
        try:
            # Create DataPoint from quote
            data_point = DataPoint(
                source=self.name,
                symbol=symbol,
                timestamp=datetime.now(),
                price=float(quote.get('c', 0)),  # Current price
                volume=int(quote.get('v', 0)),   # Volume
                bid=float(quote.get('b', 0)) if quote.get('b') else None,
                ask=float(quote.get('a', 0)) if quote.get('a') else None,
                latency_ms=100.0,  # Estimated REST latency
                metadata={
                    'open': quote.get('o'),
                    'high': quote.get('h'),
                    'low': quote.get('l'),
                    'previous_close': quote.get('pc'),
                    'change': quote.get('d'),
                    'change_percent': quote.get('dp')
                }
            )
            
            # Deliver to callback
            if self.data_callback:
                await self.data_callback(data_point)
                
        except Exception as e:
            logger.error(f"Error processing REST quote: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from Finnhub"""
        try:
            self.is_connected = False
            
            # Cancel tasks
            if self.ws_task:
                self.ws_task.cancel()
                try:
                    await self.ws_task
                except asyncio.CancelledError:
                    pass
            
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
                try:
                    await self.heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # Close connections
            if self.ws_connection:
                await self.ws_connection.close()
                self.ws_connection = None
            
            if self.rest_session:
                await self.rest_session.close()
                self.rest_session = None
            
            logger.info("Disconnected from Finnhub")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols"""
        if not self.is_connected:
            # Store for later subscription
            self.pending_subscriptions.update(symbols)
            return
        
        for symbol in symbols:
            if symbol not in self.subscribed_symbols:
                self.subscribed_symbols.add(symbol)
                
                if self.connection_type == "WebSocket" and self.ws_connection:
                    await self._send_subscription(symbol, True)
        
        logger.info(f"Subscribed to {len(symbols)} symbols")
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols"""
        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
                
                if self.connection_type == "WebSocket" and self.ws_connection:
                    await self._send_subscription(symbol, False)
    
    async def _send_subscription(self, symbol: str, subscribe: bool) -> None:
        """Send subscription message via WebSocket"""
        try:
            msg = {
                "type": "subscribe" if subscribe else "unsubscribe",
                "symbol": symbol
            }
            
            if self.ws_connection and not self.ws_connection.closed:
                await self.ws_connection.send(json.dumps(msg))
                logger.debug(f"{'Subscribed to' if subscribe else 'Unsubscribed from'} {symbol}")
                
        except Exception as e:
            logger.error(f"Error sending subscription: {e}")
    
    def set_data_callback(self, callback) -> None:
        """Set callback for data delivery"""
        self.data_callback = callback
    
    async def fetch_quote(self, symbol: str) -> Optional[DataPoint]:
        """Fetch a single quote on demand"""
        quote = await self._fetch_rest_quote(symbol)
        if quote:
            await self._process_rest_quote(symbol, quote)
            # Return the last processed data point
            # In a real implementation, we'd return the DataPoint directly
        return None
    
    async def get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company profile information"""
        if not self.rest_session:
            return None
        
        try:
            url = f"{self.REST_API_URL}/stock/profile2"
            params = {"symbol": symbol, "token": self.api_key}
            
            async with self.rest_session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                    
        except Exception as e:
            logger.error(f"Error fetching company profile: {e}")
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics"""
        uptime = time.time() - self.connection_start_time if self.connection_start_time else 0
        
        return {
            'source': self.name,
            'connection_type': self.connection_type,
            'is_connected': self.is_connected,
            'symbols_subscribed': len(self.subscribed_symbols),
            'messages_received': self.messages_received,
            'error_count': self.error_count,
            'uptime_seconds': uptime,
            'last_message_time': self.last_message_time,
            'reconnect_delay': self.reconnect_delay
        }
    
    def is_market_open(self) -> bool:
        """Check if US market is open"""
        from datetime import time
        now = datetime.now()
        
        # Simple check (doesn't account for holidays)
        if now.weekday() >= 5:  # Weekend
            return False
        
        # Market hours 9:30 AM - 4:00 PM ET
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        # This is simplified - should account for timezone
        current_time = now.time()
        return market_open <= current_time <= market_close
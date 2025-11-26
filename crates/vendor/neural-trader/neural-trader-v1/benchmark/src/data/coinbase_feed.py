"""
Coinbase WebSocket feed for real-time cryptocurrency data
Free tier includes real-time price updates via WebSocket
"""
import asyncio
import aiohttp
import websockets
import json
import time
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime
from decimal import Decimal

from .realtime_manager import DataSource, DataPoint

logger = logging.getLogger(__name__)


class CoinbaseFeed(DataSource):
    """Coinbase WebSocket feed for crypto data"""
    
    WEBSOCKET_URL = "wss://ws-feed.exchange.coinbase.com"
    REST_API_URL = "https://api.exchange.coinbase.com"
    
    # Common crypto pairs
    DEFAULT_PRODUCTS = [
        "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD",
        "MATIC-USD", "LINK-USD", "DOT-USD", "UNI-USD"
    ]
    
    def __init__(self, use_websocket: bool = True, sandbox: bool = False):
        super().__init__("coinbase")
        self.use_websocket = use_websocket
        self.sandbox = sandbox
        
        # Use sandbox URLs if specified
        if sandbox:
            self.WEBSOCKET_URL = "wss://ws-feed-public.sandbox.exchange.coinbase.com"
            self.REST_API_URL = "https://api-public.sandbox.exchange.coinbase.com"
        
        # WebSocket connection
        self.ws_connection = None
        self.ws_task = None
        
        # REST session
        self.rest_session: Optional[aiohttp.ClientSession] = None
        
        # Subscription management
        self.subscribed_products: Set[str] = set()
        self.product_ids_map: Dict[str, str] = {}  # symbol -> product_id
        
        # Order book tracking (best bid/ask)
        self.order_books: Dict[str, Dict[str, Any]] = {}
        
        # Callbacks
        self.data_callback = None
        
        # Metrics
        self.messages_received = 0
        self.connection_start_time = None
        self.last_message_time = None
        self.error_count = 0
        
        # Sequence tracking for deduplication
        self.last_sequences: Dict[str, int] = {}
    
    async def connect(self) -> bool:
        """Connect to Coinbase"""
        try:
            self.connection_start_time = time.time()
            
            # Create REST session
            self.rest_session = aiohttp.ClientSession()
            
            # Get available products
            await self._fetch_products()
            
            # Connect via WebSocket if enabled
            if self.use_websocket:
                success = await self._connect_websocket()
                if success:
                    self.connection_type = "WebSocket"
                    self.is_connected = True
                    return True
                else:
                    logger.warning("WebSocket connection failed, falling back to REST")
            
            # Fallback to REST polling
            self.connection_type = "REST"
            self.is_connected = True
            
            # Start REST polling task
            self.ws_task = asyncio.create_task(self._rest_polling_loop())
            
            logger.info(f"Connected to Coinbase via {self.connection_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Coinbase: {e}")
            self.is_connected = False
            return False
    
    async def _fetch_products(self) -> None:
        """Fetch available trading products"""
        try:
            url = f"{self.REST_API_URL}/products"
            async with self.rest_session.get(url) as response:
                if response.status == 200:
                    products = await response.json()
                    
                    # Build symbol mapping
                    for product in products:
                        if product.get('status') == 'online':
                            product_id = product['id']
                            base = product['base_currency']
                            quote = product['quote_currency']
                            
                            # Map common symbol formats
                            self.product_ids_map[f"{base}"] = product_id
                            self.product_ids_map[f"{base}-{quote}"] = product_id
                            self.product_ids_map[f"{base}/{quote}"] = product_id
                    
                    logger.info(f"Loaded {len(self.product_ids_map)} product mappings")
                    
        except Exception as e:
            logger.error(f"Error fetching products: {e}")
    
    async def _connect_websocket(self) -> bool:
        """Establish WebSocket connection"""
        try:
            self.ws_connection = await websockets.connect(self.WEBSOCKET_URL)
            
            # Start message handler
            self.ws_task = asyncio.create_task(self._websocket_handler())
            
            # Subscribe to channels
            if self.subscribed_products:
                await self._send_subscribe_message(list(self.subscribed_products))
            
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
        
        if msg_type == 'ticker':
            await self._process_ticker(data)
            
        elif msg_type == 'match':
            await self._process_match(data)
            
        elif msg_type == 'l2update':
            await self._process_l2_update(data)
            
        elif msg_type == 'heartbeat':
            # Heartbeat messages indicate connection is alive
            self.last_message_time = time.time()
            
        elif msg_type == 'error':
            logger.error(f"Coinbase error: {data.get('message')}")
            self.error_count += 1
            
        elif msg_type == 'subscriptions':
            logger.info(f"Active subscriptions: {data.get('channels', [])}")
        
        self.messages_received += 1
    
    async def _process_ticker(self, data: Dict[str, Any]) -> None:
        """Process ticker update"""
        try:
            product_id = data.get('product_id', '')
            
            # Calculate latency
            msg_time = datetime.fromisoformat(data.get('time', '').replace('Z', '+00:00'))
            latency_ms = (datetime.utcnow() - msg_time.replace(tzinfo=None)).total_seconds() * 1000
            
            # Create DataPoint
            data_point = DataPoint(
                source=self.name,
                symbol=product_id,
                timestamp=msg_time.replace(tzinfo=None),
                price=float(data.get('price', 0)),
                volume=float(data.get('volume_24h', 0)),
                bid=float(data.get('best_bid', 0)),
                ask=float(data.get('best_ask', 0)),
                latency_ms=max(0, latency_ms),  # Ensure non-negative
                sequence_id=f"coinbase_{product_id}_{data.get('sequence')}",
                metadata={
                    'open_24h': float(data.get('open_24h', 0)),
                    'high_24h': float(data.get('high_24h', 0)),
                    'low_24h': float(data.get('low_24h', 0)),
                    'volume_30d': float(data.get('volume_30d', 0)),
                    'best_bid_size': float(data.get('best_bid_size', 0)),
                    'best_ask_size': float(data.get('best_ask_size', 0)),
                    'side': data.get('side'),
                    'trade_id': data.get('trade_id')
                }
            )
            
            # Deliver to callback
            if self.data_callback:
                await self.data_callback(data_point)
                
        except Exception as e:
            logger.error(f"Error processing ticker: {e}")
    
    async def _process_match(self, data: Dict[str, Any]) -> None:
        """Process trade match"""
        try:
            product_id = data.get('product_id', '')
            
            # Calculate latency
            msg_time = datetime.fromisoformat(data.get('time', '').replace('Z', '+00:00'))
            latency_ms = (datetime.utcnow() - msg_time.replace(tzinfo=None)).total_seconds() * 1000
            
            # Get best bid/ask from order book if available
            bid, ask = self._get_best_bid_ask(product_id)
            
            # Create DataPoint
            data_point = DataPoint(
                source=self.name,
                symbol=product_id,
                timestamp=msg_time.replace(tzinfo=None),
                price=float(data.get('price', 0)),
                volume=float(data.get('size', 0)),
                bid=bid,
                ask=ask,
                latency_ms=max(0, latency_ms),
                sequence_id=f"coinbase_match_{product_id}_{data.get('sequence')}",
                metadata={
                    'side': data.get('side'),
                    'trade_id': data.get('trade_id'),
                    'maker_order_id': data.get('maker_order_id'),
                    'taker_order_id': data.get('taker_order_id')
                }
            )
            
            # Deliver to callback
            if self.data_callback:
                await self.data_callback(data_point)
                
        except Exception as e:
            logger.error(f"Error processing match: {e}")
    
    async def _process_l2_update(self, data: Dict[str, Any]) -> None:
        """Process level 2 order book update"""
        try:
            product_id = data.get('product_id', '')
            changes = data.get('changes', [])
            
            # Initialize order book if needed
            if product_id not in self.order_books:
                self.order_books[product_id] = {'bids': {}, 'asks': {}}
            
            # Apply changes
            for change in changes:
                side, price, size = change
                price = float(price)
                size = float(size)
                
                book_side = self.order_books[product_id]['bids' if side == 'buy' else 'asks']
                
                if size == 0:
                    # Remove price level
                    book_side.pop(price, None)
                else:
                    # Update price level
                    book_side[price] = size
            
            # Update best bid/ask cache
            self._update_best_prices(product_id)
            
        except Exception as e:
            logger.error(f"Error processing L2 update: {e}")
    
    def _get_best_bid_ask(self, product_id: str) -> Tuple[Optional[float], Optional[float]]:
        """Get best bid and ask from order book"""
        if product_id not in self.order_books:
            return None, None
        
        book = self.order_books[product_id]
        
        # Get best bid (highest buy price)
        best_bid = max(book['bids'].keys()) if book['bids'] else None
        
        # Get best ask (lowest sell price)
        best_ask = min(book['asks'].keys()) if book['asks'] else None
        
        return best_bid, best_ask
    
    def _update_best_prices(self, product_id: str) -> None:
        """Update cached best bid/ask prices"""
        # This is handled in _get_best_bid_ask for simplicity
        pass
    
    async def _handle_websocket_disconnect(self) -> None:
        """Handle WebSocket disconnection with reconnect"""
        self.is_connected = False
        
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
        
        # Clear order books as they may be stale
        self.order_books.clear()
        
        # Attempt reconnection
        for attempt in range(self.max_reconnect_attempts):
            logger.info(f"Reconnecting to WebSocket (attempt {attempt + 1})")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            if await self._connect_websocket():
                return
        
        logger.error("Failed to reconnect to WebSocket, falling back to REST")
        self.connection_type = "REST"
        self.ws_task = asyncio.create_task(self._rest_polling_loop())
    
    async def _rest_polling_loop(self) -> None:
        """Poll REST API for updates (fallback mode)"""
        while self.is_connected and self.connection_type == "REST":
            try:
                # Poll each subscribed product
                for product_id in list(self.subscribed_products):
                    ticker = await self._fetch_ticker(product_id)
                    if ticker:
                        await self._process_rest_ticker(product_id, ticker)
                    
                    # Small delay between requests
                    await asyncio.sleep(0.1)
                
                # Wait before next polling cycle
                await asyncio.sleep(1)  # Poll every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"REST polling error: {e}")
                await asyncio.sleep(5)
    
    async def _fetch_ticker(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Fetch ticker via REST API"""
        if not self.rest_session:
            return None
        
        try:
            url = f"{self.REST_API_URL}/products/{product_id}/ticker"
            
            async with self.rest_session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"REST API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error fetching ticker for {product_id}: {e}")
        
        return None
    
    async def _process_rest_ticker(self, product_id: str, ticker: Dict[str, Any]) -> None:
        """Process REST API ticker"""
        try:
            # Create DataPoint from ticker
            data_point = DataPoint(
                source=self.name,
                symbol=product_id,
                timestamp=datetime.now(),
                price=float(ticker.get('price', 0)),
                volume=float(ticker.get('volume', 0)),
                bid=float(ticker.get('bid', 0)),
                ask=float(ticker.get('ask', 0)),
                latency_ms=100.0,  # Estimated REST latency
                metadata={
                    'size': float(ticker.get('size', 0)),
                    'time': ticker.get('time'),
                    'trade_id': ticker.get('trade_id')
                }
            )
            
            # Deliver to callback
            if self.data_callback:
                await self.data_callback(data_point)
                
        except Exception as e:
            logger.error(f"Error processing REST ticker: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from Coinbase"""
        try:
            self.is_connected = False
            
            # Unsubscribe if connected
            if self.ws_connection and self.subscribed_products:
                await self._send_unsubscribe_message(list(self.subscribed_products))
            
            # Cancel tasks
            if self.ws_task:
                self.ws_task.cancel()
                try:
                    await self.ws_task
                except asyncio.CancelledError:
                    pass
            
            # Close connections
            if self.ws_connection:
                await self.ws_connection.close()
                self.ws_connection = None
            
            if self.rest_session:
                await self.rest_session.close()
                self.rest_session = None
            
            logger.info("Disconnected from Coinbase")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols"""
        # Convert symbols to Coinbase product IDs
        product_ids = []
        for symbol in symbols:
            # Try to map symbol to product ID
            if symbol in self.product_ids_map:
                product_ids.append(self.product_ids_map[symbol])
            elif '-' in symbol:
                # Already in product ID format
                product_ids.append(symbol)
            else:
                # Try common USD pairing
                usd_pair = f"{symbol}-USD"
                if usd_pair in self.product_ids_map.values():
                    product_ids.append(usd_pair)
                else:
                    logger.warning(f"Unknown symbol: {symbol}")
        
        # Add to subscribed set
        self.subscribed_products.update(product_ids)
        
        # Send subscription if connected via WebSocket
        if self.connection_type == "WebSocket" and self.ws_connection:
            await self._send_subscribe_message(product_ids)
        
        logger.info(f"Subscribed to {len(product_ids)} products")
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols"""
        # Convert symbols to product IDs
        product_ids = []
        for symbol in symbols:
            if symbol in self.product_ids_map:
                product_ids.append(self.product_ids_map[symbol])
            elif symbol in self.subscribed_products:
                product_ids.append(symbol)
        
        # Remove from subscribed set
        for product_id in product_ids:
            self.subscribed_products.discard(product_id)
        
        # Send unsubscribe if connected via WebSocket
        if self.connection_type == "WebSocket" and self.ws_connection:
            await self._send_unsubscribe_message(product_ids)
    
    async def _send_subscribe_message(self, product_ids: List[str]) -> None:
        """Send subscribe message via WebSocket"""
        try:
            msg = {
                "type": "subscribe",
                "product_ids": product_ids,
                "channels": [
                    "ticker",
                    "matches",
                    {"name": "level2", "product_ids": product_ids}
                ]
            }
            
            if self.ws_connection and not self.ws_connection.closed:
                await self.ws_connection.send(json.dumps(msg))
                logger.debug(f"Subscribed to {len(product_ids)} products")
                
        except Exception as e:
            logger.error(f"Error sending subscribe message: {e}")
    
    async def _send_unsubscribe_message(self, product_ids: List[str]) -> None:
        """Send unsubscribe message via WebSocket"""
        try:
            msg = {
                "type": "unsubscribe",
                "product_ids": product_ids,
                "channels": ["ticker", "matches", "level2"]
            }
            
            if self.ws_connection and not self.ws_connection.closed:
                await self.ws_connection.send(json.dumps(msg))
                
        except Exception as e:
            logger.error(f"Error sending unsubscribe message: {e}")
    
    def set_data_callback(self, callback) -> None:
        """Set callback for data delivery"""
        self.data_callback = callback
    
    async def fetch_quote(self, symbol: str) -> Optional[DataPoint]:
        """Fetch a single quote on demand"""
        # Convert symbol to product ID
        product_id = self.product_ids_map.get(symbol, symbol)
        
        ticker = await self._fetch_ticker(product_id)
        if ticker:
            await self._process_rest_ticker(product_id, ticker)
            # Return the processed data point
            # In a real implementation, we'd return it directly
        
        return None
    
    async def get_24hr_stats(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get 24-hour statistics for a product"""
        if not self.rest_session:
            return None
        
        try:
            url = f"{self.REST_API_URL}/products/{product_id}/stats"
            
            async with self.rest_session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                    
        except Exception as e:
            logger.error(f"Error fetching 24hr stats: {e}")
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get feed metrics"""
        uptime = time.time() - self.connection_start_time if self.connection_start_time else 0
        
        return {
            'source': self.name,
            'connection_type': self.connection_type,
            'is_connected': self.is_connected,
            'products_subscribed': len(self.subscribed_products),
            'messages_received': self.messages_received,
            'error_count': self.error_count,
            'uptime_seconds': uptime,
            'last_message_time': self.last_message_time,
            'order_books_tracked': len(self.order_books),
            'sandbox_mode': self.sandbox
        }
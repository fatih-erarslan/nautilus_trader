"""CryptoCompare data adapter for cryptocurrency market data.

Provides real-time cryptocurrency prices, volumes, and market data
using the CryptoCompare free API.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
import logging
import aiohttp
import websockets
import json

from ..realtime_feed import DataUpdate, DataSource

logger = logging.getLogger(__name__)


@dataclass
class CryptoCompareConfig:
    """Configuration for CryptoCompare adapter."""
    api_key: Optional[str] = None  # Optional for free tier
    symbols: List[str] = field(default_factory=list)  # Crypto symbols (BTC, ETH, etc.)
    vs_currencies: List[str] = field(default_factory=lambda: ["USD"])  # Quote currencies
    update_interval: float = 1.0  # For REST polling
    enable_websocket: bool = True  # CryptoCompare has WebSocket support
    include_volume: bool = True
    include_market_cap: bool = True
    exchanges: List[str] = field(default_factory=list)  # Specific exchanges or empty for aggregate


class CryptoCompareAdapter:
    """CryptoCompare data adapter for crypto market data."""
    
    # API endpoints
    REST_BASE_URL = "https://min-api.cryptocompare.com/data"
    WS_URL = "wss://streamer.cryptocompare.com/v2"
    
    # Common crypto symbols
    POPULAR_CRYPTOS = ["BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "MATIC", "SOL", "DOT", "AVAX"]
    
    def __init__(self, config: CryptoCompareConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._update_task = None
        self._ws_task = None
        self._callbacks: List[Callable] = []
        
        # WebSocket subscriptions
        self._subscriptions: Set[str] = set()
        
        # Metrics
        self._total_updates = 0
        self._ws_reconnects = 0
        self._last_update_time = 0
        
        # API headers
        self._headers = {}
        if config.api_key:
            self._headers["Authorization"] = f"Apikey {config.api_key}"
    
    async def start(self):
        """Start the adapter."""
        if self._running:
            return
        
        self._session = aiohttp.ClientSession(headers=self._headers)
        self._running = True
        
        if self.config.enable_websocket:
            self._ws_task = asyncio.create_task(self._websocket_handler())
        else:
            self._update_task = asyncio.create_task(self._polling_loop())
        
        logger.info(f"CryptoCompare adapter started for {len(self.config.symbols)} symbols")
    
    async def stop(self):
        """Stop the adapter."""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
        if self._ws_task:
            self._ws_task.cancel()
        
        if self._ws:
            await self._ws.close()
        
        if self._session:
            await self._session.close()
        
        logger.info("CryptoCompare adapter stopped")
    
    def add_callback(self, callback: Callable[[DataUpdate], None]):
        """Add callback for data updates."""
        self._callbacks.append(callback)
    
    async def _websocket_handler(self):
        """Handle WebSocket connection and messages."""
        while self._running:
            try:
                # Connect to WebSocket
                async with websockets.connect(self.WS_URL) as ws:
                    self._ws = ws
                    logger.info("Connected to CryptoCompare WebSocket")
                    
                    # Subscribe to symbols
                    await self._subscribe_all()
                    
                    # Handle messages
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            await self._process_ws_message(data)
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON in WebSocket message: {message}")
                        except Exception as e:
                            logger.error(f"Error processing WebSocket message: {e}")
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self._ws_reconnects += 1
                
                # Wait before reconnecting
                await asyncio.sleep(5)
    
    async def _subscribe_all(self):
        """Subscribe to all configured symbols via WebSocket."""
        if not self._ws:
            return
        
        # Build subscription list
        subs = []
        for symbol in self.config.symbols:
            for vs_currency in self.config.vs_currencies:
                # Format: 5~CCCAGG~BTC~USD for aggregate price
                sub = f"5~CCCAGG~{symbol}~{vs_currency}"
                subs.append(sub)
                self._subscriptions.add(sub)
                
                # Also subscribe to volume/OHLC data
                if self.config.include_volume:
                    vol_sub = f"24~CCCAGG~{symbol}~{vs_currency}"
                    subs.append(vol_sub)
                    self._subscriptions.add(vol_sub)
        
        # Send subscription message
        sub_message = {
            "action": "SubAdd",
            "subs": subs
        }
        
        await self._ws.send(json.dumps(sub_message))
        logger.info(f"Subscribed to {len(subs)} data streams")
    
    async def _process_ws_message(self, data: Dict[str, Any]):
        """Process WebSocket message."""
        try:
            msg_type = data.get("TYPE")
            
            # Price update (TYPE 5)
            if msg_type == "5" and data.get("PRICE"):
                await self._process_price_update(data)
            
            # OHLC/Volume update (TYPE 24)
            elif msg_type == "24":
                await self._process_ohlc_update(data)
            
            # Heartbeat
            elif msg_type == "999":
                logger.debug("Received heartbeat")
            
            # Error or info messages
            elif "MESSAGE" in data:
                logger.info(f"CryptoCompare message: {data['MESSAGE']}")
                
        except Exception as e:
            logger.error(f"Error processing WebSocket data: {e}")
    
    async def _process_price_update(self, data: Dict[str, Any]):
        """Process real-time price update."""
        try:
            symbol = data.get("FROMSYMBOL")
            to_symbol = data.get("TOSYMBOL")
            price = data.get("PRICE")
            timestamp = data.get("LASTUPDATE", time.time())
            
            if not symbol or price is None:
                return
            
            # Extract additional data
            metadata = {
                "source": "cryptocompare_ws",
                "to_currency": to_symbol,
                "exchange": data.get("MARKET", "CCCAGG"),
                "volume_24h": data.get("VOLUME24HOUR"),
                "volume_24h_to": data.get("VOLUME24HOURTO"),
                "open_24h": data.get("OPEN24HOUR"),
                "high_24h": data.get("HIGH24HOUR"),
                "low_24h": data.get("LOW24HOUR"),
                "change_24h": data.get("CHANGE24HOUR"),
                "change_pct_24h": data.get("CHANGEPCT24HOUR"),
                "last_volume": data.get("LASTVOLUME"),
                "last_volume_to": data.get("LASTVOLUMETO"),
                "last_trade_id": data.get("LASTTRADEID"),
                "volume_hour": data.get("VOLUMEHOUR"),
                "volume_hour_to": data.get("VOLUMEHOURTO"),
            }
            
            # Create data update
            update = DataUpdate(
                symbol=f"{symbol}-{to_symbol}",
                price=float(price),
                timestamp=float(timestamp),
                source=DataSource.WEBSOCKET,
                metadata=metadata
            )
            
            self._total_updates += 1
            self._last_update_time = time.time()
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    await callback(update)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing price update: {e}")
    
    async def _process_ohlc_update(self, data: Dict[str, Any]):
        """Process OHLC/volume update."""
        # Similar to price update but with OHLC data
        pass
    
    async def _polling_loop(self):
        """REST API polling loop (fallback when WebSocket not used)."""
        while self._running:
            try:
                # Fetch prices for all symbol pairs
                await self._fetch_multi_prices()
                
                # Sleep for update interval
                await asyncio.sleep(self.config.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Polling loop error: {e}")
                await asyncio.sleep(5)
    
    async def _fetch_multi_prices(self):
        """Fetch multiple prices via REST API."""
        if not self.config.symbols:
            return
        
        # CryptoCompare allows multiple symbols in one request
        fsyms = ",".join(self.config.symbols[:50])  # Max 50 symbols
        tsyms = ",".join(self.config.vs_currencies)
        
        params = {
            "fsyms": fsyms,
            "tsyms": tsyms,
        }
        
        if self.config.exchanges:
            params["e"] = ",".join(self.config.exchanges)
        
        url = f"{self.REST_BASE_URL}/pricemultifull"
        
        try:
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if "RAW" in data:
                        raw_data = data["RAW"]
                        
                        for from_sym, to_data in raw_data.items():
                            for to_sym, price_data in to_data.items():
                                await self._process_rest_price(from_sym, to_sym, price_data)
                else:
                    logger.error(f"HTTP error {response.status}")
                    
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
    
    async def _process_rest_price(self, from_symbol: str, to_symbol: str, data: Dict[str, Any]):
        """Process price data from REST API."""
        try:
            price = data.get("PRICE")
            if price is None:
                return
            
            # Create metadata
            metadata = {
                "source": "cryptocompare_rest",
                "to_currency": to_symbol,
                "exchange": data.get("MARKET", "CCCAGG"),
                "volume_24h": data.get("VOLUME24HOUR"),
                "volume_24h_to": data.get("VOLUME24HOURTO"),
                "open_24h": data.get("OPEN24HOUR"),
                "high_24h": data.get("HIGH24HOUR"),
                "low_24h": data.get("LOW24HOUR"),
                "change_24h": data.get("CHANGE24HOUR"),
                "change_pct_24h": data.get("CHANGEPCT24HOUR"),
                "market_cap": data.get("MKTCAP"),
                "total_volume_24h": data.get("TOTALVOLUME24H"),
                "total_volume_24h_to": data.get("TOTALVOLUME24HTO"),
                "supply": data.get("SUPPLY"),
                "last_update": data.get("LASTUPDATE"),
            }
            
            # Create data update
            update = DataUpdate(
                symbol=f"{from_symbol}-{to_symbol}",
                price=float(price),
                timestamp=float(data.get("LASTUPDATE", time.time())),
                source=DataSource.REST,
                metadata=metadata
            )
            
            self._total_updates += 1
            self._last_update_time = time.time()
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    await callback(update)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing REST price: {e}")
    
    async def get_historical_data(self, symbol: str, vs_currency: str = "USD",
                                limit: int = 100, aggregate: int = 1) -> Optional[List[Dict]]:
        """Get historical OHLCV data."""
        params = {
            "fsym": symbol,
            "tsym": vs_currency,
            "limit": limit,
            "aggregate": aggregate,
        }
        
        url = f"{self.REST_BASE_URL}/histoday"
        
        try:
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("Data", [])
                    
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
        
        return None
    
    async def get_exchanges(self) -> Optional[Dict[str, Any]]:
        """Get list of available exchanges."""
        url = f"{self.REST_BASE_URL}/exchanges/general"
        
        try:
            async with self._session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("Data", {})
                    
        except Exception as e:
            logger.error(f"Error getting exchanges: {e}")
        
        return None
    
    async def get_coin_list(self) -> Optional[Dict[str, Any]]:
        """Get list of all available cryptocurrencies."""
        url = f"{self.REST_BASE_URL}/blockchain/list"
        
        try:
            async with self._session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("Data", {})
                    
        except Exception as e:
            logger.error(f"Error getting coin list: {e}")
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter metrics."""
        return {
            "total_updates": self._total_updates,
            "websocket_reconnects": self._ws_reconnects,
            "symbols_tracked": len(self.config.symbols),
            "vs_currencies": len(self.config.vs_currencies),
            "total_pairs": len(self.config.symbols) * len(self.config.vs_currencies),
            "websocket_enabled": self.config.enable_websocket,
            "websocket_connected": self._ws is not None and not self._ws.closed if self._ws else False,
            "subscriptions": len(self._subscriptions),
            "last_update": self._last_update_time,
        }
    
    async def add_symbol_pair(self, symbol: str, vs_currency: str = "USD"):
        """Add a symbol pair to track."""
        if symbol not in self.config.symbols:
            self.config.symbols.append(symbol)
        
        if vs_currency not in self.config.vs_currencies:
            self.config.vs_currencies.append(vs_currency)
        
        # If WebSocket is active, subscribe to new pair
        if self._ws and not self._ws.closed:
            sub = f"5~CCCAGG~{symbol}~{vs_currency}"
            await self._ws.send(json.dumps({
                "action": "SubAdd",
                "subs": [sub]
            }))
            self._subscriptions.add(sub)
        
        logger.info(f"Added symbol pair: {symbol}/{vs_currency}")
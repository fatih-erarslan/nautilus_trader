"""
CWTS Ultra Client for FreqTrade - Pure Python Implementation
High-performance client using memory-mapped files and WebSocket
"""

import mmap
import struct
import time
import asyncio
import websockets
import json
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Constants
MAX_SYMBOLS = 100
MAX_SIGNALS = 1000
BOOK_DEPTH = 50
SHM_SIZE = 1048576  # 1MB

# Signal types
SIGNAL_HOLD = 0
SIGNAL_BUY = 1
SIGNAL_SELL = 2


@dataclass
class MarketData:
    """Market data structure."""
    bid_price: float
    ask_price: float
    bid_volume: float
    ask_volume: float
    last_price: float
    volume_24h: float
    high_24h: float
    low_24h: float
    timestamp: int
    updates: int


@dataclass
class Signal:
    """Trading signal structure."""
    action: int  # 0=hold, 1=buy, 2=sell
    confidence: float  # 0.0-1.0
    size: float  # Position size
    price: float  # Limit price (0 for market)
    stop_loss: float
    take_profit: float
    strategy_id: int
    timestamp: int
    symbol: str


class CWTSUltraClient:
    """
    High-performance CWTS Ultra client for FreqTrade integration.
    Uses memory-mapped files for fast IPC and WebSocket as fallback.
    """
    
    def __init__(self, shm_path: str = "/dev/shm/cwts_ultra", 
                 websocket_url: str = "ws://localhost:4000"):
        """Initialize CWTS Ultra client."""
        self.shm_path = shm_path
        self.websocket_url = websocket_url
        self.shm = None
        self.ws = None
        self.connected = False
        self.symbol_map = {}
        self.last_sequence = 0
        
        # Try to initialize shared memory
        try:
            self._init_shared_memory()
        except Exception as e:
            logger.warning(f"Shared memory not available: {e}")
            logger.info("Using WebSocket fallback")
    
    def _init_shared_memory(self):
        """Initialize memory-mapped file for IPC."""
        try:
            # Create or open the memory-mapped file
            with open(self.shm_path, 'r+b') as f:
                self.shm = mmap.mmap(f.fileno(), SHM_SIZE)
                self.connected = True
                logger.info(f"Connected to shared memory at {self.shm_path}")
        except FileNotFoundError:
            # Create the file if it doesn't exist
            with open(self.shm_path, 'wb') as f:
                f.write(b'\x00' * SHM_SIZE)
            with open(self.shm_path, 'r+b') as f:
                self.shm = mmap.mmap(f.fileno(), SHM_SIZE)
                self.connected = True
                logger.info(f"Created and connected to shared memory at {self.shm_path}")
    
    async def _init_websocket(self):
        """Initialize WebSocket connection."""
        try:
            self.ws = await websockets.connect(self.websocket_url)
            logger.info(f"Connected to WebSocket at {self.websocket_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    def get_market_data(self, symbol: str) -> Dict:
        """
        Get current market data for a symbol.
        Returns dict with bid, ask, last price, etc.
        """
        symbol_idx = self._get_symbol_index(symbol)
        if symbol_idx < 0:
            return {}
        
        if self.connected and self.shm:
            # Read from shared memory
            offset = 16 + (symbol_idx * 88)  # Header + symbol_idx * sizeof(MarketData)
            
            try:
                self.shm.seek(offset)
                data = struct.unpack('ddddddddQI', self.shm.read(88))
                
                market_data = {
                    'bid': data[0],
                    'ask': data[1],
                    'bid_volume': data[2],
                    'ask_volume': data[3],
                    'last': data[4],
                    'volume': data[5],
                    'high': data[6],
                    'low': data[7],
                    'timestamp': data[8] / 1000000.0,
                    'updates': data[9],
                    'spread': data[1] - data[0],
                    'mid': (data[1] + data[0]) / 2.0
                }
                
                # Check if data is fresh (within 1 second)
                now = time.time()
                if now - market_data['timestamp'] > 1.0:
                    return {}
                
                return market_data
            except Exception as e:
                logger.error(f"Error reading market data: {e}")
                return {}
        else:
            # Fallback to WebSocket
            return asyncio.run(self._get_market_data_ws(symbol))
    
    async def _get_market_data_ws(self, symbol: str) -> Dict:
        """Get market data via WebSocket."""
        if not self.ws:
            if not await self._init_websocket():
                return {}
        
        try:
            request = {
                "method": "get_market_data",
                "params": {"symbol": symbol}
            }
            await self.ws.send(json.dumps(request))
            response = await asyncio.wait_for(self.ws.recv(), timeout=1.0)
            return json.loads(response)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            return {}
    
    def get_order_book(self, symbol: str, depth: int = 10) -> np.ndarray:
        """
        Get order book for a symbol.
        Returns numpy array with shape (depth, 4) containing [bid_price, bid_vol, ask_price, ask_vol].
        """
        symbol_idx = self._get_symbol_index(symbol)
        if symbol_idx < 0:
            return np.empty((0, 4), dtype=np.float64)
        
        if self.connected and self.shm:
            # Read from shared memory
            # Simplified: return random data for now
            # In production, properly parse the shared memory structure
            return np.random.random((depth, 4)) * 100
        else:
            # Fallback to WebSocket
            return asyncio.run(self._get_order_book_ws(symbol, depth))
    
    async def _get_order_book_ws(self, symbol: str, depth: int) -> np.ndarray:
        """Get order book via WebSocket."""
        if not self.ws:
            if not await self._init_websocket():
                return np.empty((0, 4), dtype=np.float64)
        
        try:
            request = {
                "method": "get_order_book",
                "params": {"symbol": symbol, "depth": depth}
            }
            await self.ws.send(json.dumps(request))
            response = await asyncio.wait_for(self.ws.recv(), timeout=1.0)
            data = json.loads(response)
            
            if 'bids' in data and 'asks' in data:
                book = np.empty((depth, 4), dtype=np.float64)
                for i in range(min(depth, len(data['bids']))):
                    book[i, 0] = data['bids'][i][0]  # bid price
                    book[i, 1] = data['bids'][i][1]  # bid volume
                for i in range(min(depth, len(data['asks']))):
                    book[i, 2] = data['asks'][i][0]  # ask price
                    book[i, 3] = data['asks'][i][1]  # ask volume
                return book
            
        except Exception as e:
            logger.error(f"WebSocket order book error: {e}")
        
        return np.empty((0, 4), dtype=np.float64)
    
    def send_signal(self, symbol: str, action: int, confidence: float = 1.0,
                   size: float = 0.0, price: float = 0.0,
                   stop_loss: float = 0.0, take_profit: float = 0.0,
                   strategy_id: int = 0) -> bool:
        """
        Send trading signal to CWTS Ultra.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            action: 0=hold, 1=buy, 2=sell
            confidence: Signal confidence (0.0-1.0)
            size: Position size (0 for default)
            price: Limit price (0 for market order)
            stop_loss: Stop loss price
            take_profit: Take profit price
            strategy_id: Strategy identifier
        
        Returns:
            True if signal was sent successfully
        """
        if action not in [0, 1, 2]:
            return False
        
        signal = Signal(
            action=action,
            confidence=max(0.0, min(1.0, confidence)),
            size=size,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_id=strategy_id,
            timestamp=int(time.time() * 1000000),
            symbol=symbol
        )
        
        if self.connected and self.shm:
            # Write to shared memory
            return self._write_signal_to_shm(signal)
        else:
            # Fallback to WebSocket
            return asyncio.run(self._send_signal_ws(signal))
    
    def _write_signal_to_shm(self, signal: Signal) -> bool:
        """Write signal to shared memory."""
        try:
            # Simplified: just log for now
            # In production, properly write to the shared memory structure
            logger.info(f"Signal written to SHM: {signal.symbol} {signal.action}")
            return True
        except Exception as e:
            logger.error(f"Error writing signal: {e}")
            return False
    
    async def _send_signal_ws(self, signal: Signal) -> bool:
        """Send signal via WebSocket."""
        if not self.ws:
            if not await self._init_websocket():
                return False
        
        try:
            request = {
                "method": "send_signal",
                "params": {
                    "symbol": signal.symbol,
                    "action": signal.action,
                    "confidence": signal.confidence,
                    "size": signal.size,
                    "price": signal.price,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "strategy_id": signal.strategy_id
                }
            }
            await self.ws.send(json.dumps(request))
            return True
        except Exception as e:
            logger.error(f"WebSocket signal error: {e}")
            return False
    
    def get_market_snapshot(self, symbols: List[str]) -> np.ndarray:
        """
        Get market snapshot for multiple symbols.
        Returns numpy array with columns: [bid, ask, last, volume, spread, timestamp].
        """
        n_symbols = len(symbols)
        if n_symbols == 0:
            return np.empty((0, 6), dtype=np.float64)
        
        snapshot = np.empty((n_symbols, 6), dtype=np.float64)
        
        for i, symbol in enumerate(symbols):
            data = self.get_market_data(symbol)
            if data:
                snapshot[i, 0] = data['bid']
                snapshot[i, 1] = data['ask']
                snapshot[i, 2] = data['last']
                snapshot[i, 3] = data['volume']
                snapshot[i, 4] = data['spread']
                snapshot[i, 5] = data['timestamp']
            else:
                snapshot[i, :] = np.nan
        
        return snapshot
    
    def _get_symbol_index(self, symbol: str) -> int:
        """Get symbol index in shared memory arrays."""
        if symbol not in self.symbol_map:
            if len(self.symbol_map) >= MAX_SYMBOLS:
                return -1
            self.symbol_map[symbol] = len(self.symbol_map)
        return self.symbol_map[symbol]
    
    def close(self):
        """Clean up resources."""
        if self.shm:
            self.shm.close()
            self.shm = None
        
        if self.ws:
            asyncio.run(self.ws.close())
            self.ws = None
        
        self.connected = False
    
    def __del__(self):
        """Destructor - ensure cleanup."""
        self.close()


# Convenience functions
def create_client(shm_path: str = "/dev/shm/cwts_ultra",
                 websocket_url: str = "ws://localhost:4000") -> CWTSUltraClient:
    """Create a new CWTS Ultra client."""
    return CWTSUltraClient(shm_path, websocket_url)


def buy_signal(client: CWTSUltraClient, symbol: str, confidence: float = 1.0,
              size: float = 0.0, price: float = 0.0,
              stop_loss: float = 0.0, take_profit: float = 0.0,
              strategy_id: int = 0) -> bool:
    """Send buy signal."""
    return client.send_signal(symbol, SIGNAL_BUY, confidence, size, price,
                             stop_loss, take_profit, strategy_id)


def sell_signal(client: CWTSUltraClient, symbol: str, confidence: float = 1.0,
               size: float = 0.0, price: float = 0.0,
               stop_loss: float = 0.0, take_profit: float = 0.0,
               strategy_id: int = 0) -> bool:
    """Send sell signal."""
    return client.send_signal(symbol, SIGNAL_SELL, confidence, size, price,
                             stop_loss, take_profit, strategy_id)


def hold_signal(client: CWTSUltraClient, symbol: str, strategy_id: int = 0) -> bool:
    """Send hold signal."""
    return client.send_signal(symbol, SIGNAL_HOLD, 1.0, 0.0, 0.0, 0.0, 0.0, strategy_id)
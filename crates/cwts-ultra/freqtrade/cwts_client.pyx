# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: profile=False

"""
CWTS Ultra High-Performance Client for FreqTrade
Cython implementation for ultra-low latency communication
"""

from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free
from posix.mman cimport mmap, munmap, PROT_READ, PROT_WRITE, MAP_SHARED
from posix.fcntl cimport open, O_RDWR, O_CREAT
from posix.unistd cimport close, ftruncate
import numpy as np
cimport numpy as cnp
import asyncio
import time

# Constants
DEF MAX_SYMBOLS = 100
DEF MAX_SIGNALS = 1000
DEF BOOK_DEPTH = 50
DEF SHM_SIZE = 1048576  # 1MB shared memory

# Signal types
DEF SIGNAL_HOLD = 0
DEF SIGNAL_BUY = 1
DEF SIGNAL_SELL = 2

# Shared memory structure
cdef struct MarketData:
    double bid_price
    double ask_price
    double bid_volume
    double ask_volume
    double last_price
    double volume_24h
    double high_24h
    double low_24h
    uint64_t timestamp
    uint32_t updates

cdef struct Signal:
    uint8_t action          # 0=hold, 1=buy, 2=sell
    double confidence       # 0.0-1.0
    double size            # Position size
    double price           # Limit price (0 for market)
    double stop_loss       # Stop loss price
    double take_profit     # Take profit price
    uint64_t strategy_id   # Strategy identifier
    uint64_t timestamp     # Signal timestamp
    char symbol[16]        # Trading symbol

cdef struct OrderBookLevel:
    double price
    double volume

cdef struct SharedMemoryLayout:
    # Header
    uint64_t timestamp
    uint32_t sequence
    uint32_t flags
    uint32_t active_symbols
    
    # Market data array
    MarketData market_data[MAX_SYMBOLS]
    
    # Signal queue
    uint32_t signal_write_idx
    uint32_t signal_read_idx
    Signal signals[MAX_SIGNALS]
    
    # Order book data
    OrderBookLevel bid_levels[MAX_SYMBOLS][BOOK_DEPTH]
    OrderBookLevel ask_levels[MAX_SYMBOLS][BOOK_DEPTH]
    uint32_t book_depth[MAX_SYMBOLS]

cdef class CWTSUltraClient:
    """
    High-performance CWTS Ultra client for FreqTrade integration.
    Uses shared memory for ultra-low latency communication.
    """
    
    cdef:
        SharedMemoryLayout* shm_ptr
        int shm_fd
        object ws_client
        dict symbol_map
        uint64_t last_sequence
        bint connected
        object loop
        
    def __cinit__(self):
        self.shm_ptr = NULL
        self.shm_fd = -1
        self.connected = False
        self.symbol_map = {}
        self.last_sequence = 0
        
    def __init__(self, shm_path="/dev/shm/cwts_ultra", websocket_url="ws://localhost:4000"):
        """Initialize CWTS Ultra client with shared memory and WebSocket fallback."""
        self.loop = asyncio.get_event_loop()
        self._init_shared_memory(shm_path)
        self._init_websocket(websocket_url)
        
    cdef _init_shared_memory(self, str shm_path):
        """Initialize shared memory connection."""
        cdef bytes path_bytes = shm_path.encode('utf-8')
        
        # Open or create shared memory
        self.shm_fd = open(path_bytes, O_RDWR | O_CREAT, 0o666)
        if self.shm_fd < 0:
            raise RuntimeError(f"Failed to open shared memory at {shm_path}")
        
        # Set size
        if ftruncate(self.shm_fd, SHM_SIZE) < 0:
            close(self.shm_fd)
            raise RuntimeError("Failed to set shared memory size")
        
        # Map memory
        self.shm_ptr = <SharedMemoryLayout*>mmap(
            NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, self.shm_fd, 0
        )
        
        if self.shm_ptr == <SharedMemoryLayout*>-1:
            close(self.shm_fd)
            raise RuntimeError("Failed to map shared memory")
        
        self.connected = True
        
    def _init_websocket(self, url):
        """Initialize WebSocket client as fallback."""
        # This will be initialized lazily when needed
        self.ws_client = None
        
    cpdef dict get_market_data(self, str symbol):
        """
        Get current market data for a symbol.
        Returns dict with bid, ask, last price, etc.
        """
        cdef:
            int symbol_idx = self._get_symbol_index(symbol)
            MarketData* data
            dict result = {}
            
        if symbol_idx < 0 or not self.connected:
            return {}
            
        data = &self.shm_ptr.market_data[symbol_idx]
        
        # Check if data is fresh (within 1 second)
        cdef uint64_t now = <uint64_t>(time.time() * 1000000)
        if now - data.timestamp > 1000000:  # 1 second
            return {}
        
        result['bid'] = data.bid_price
        result['ask'] = data.ask_price
        result['bid_volume'] = data.bid_volume
        result['ask_volume'] = data.ask_volume
        result['last'] = data.last_price
        result['volume'] = data.volume_24h
        result['high'] = data.high_24h
        result['low'] = data.low_24h
        result['timestamp'] = data.timestamp / 1000000.0  # Convert to seconds
        result['spread'] = data.ask_price - data.bid_price
        result['mid'] = (data.ask_price + data.bid_price) / 2.0
        
        return result
    
    cpdef cnp.ndarray get_order_book(self, str symbol, int depth=10):
        """
        Get order book for a symbol.
        Returns numpy array with shape (depth, 4) containing [bid_price, bid_vol, ask_price, ask_vol].
        """
        cdef:
            int symbol_idx = self._get_symbol_index(symbol)
            int actual_depth
            cnp.ndarray[double, ndim=2] book
            int i
            
        if symbol_idx < 0 or not self.connected:
            return np.empty((0, 4), dtype=np.float64)
        
        actual_depth = min(depth, self.shm_ptr.book_depth[symbol_idx])
        if actual_depth == 0:
            return np.empty((0, 4), dtype=np.float64)
        
        book = np.empty((actual_depth, 4), dtype=np.float64)
        
        for i in range(actual_depth):
            book[i, 0] = self.shm_ptr.bid_levels[symbol_idx][i].price
            book[i, 1] = self.shm_ptr.bid_levels[symbol_idx][i].volume
            book[i, 2] = self.shm_ptr.ask_levels[symbol_idx][i].price
            book[i, 3] = self.shm_ptr.ask_levels[symbol_idx][i].volume
        
        return book
    
    cpdef bint send_signal(self, str symbol, int action, double confidence=1.0, 
                          double size=0.0, double price=0.0, 
                          double stop_loss=0.0, double take_profit=0.0,
                          uint64_t strategy_id=0):
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
        cdef:
            uint32_t write_idx
            uint32_t next_idx
            Signal* signal
            bytes symbol_bytes
            
        if not self.connected or action not in [0, 1, 2]:
            return False
        
        # Get next write position (atomic operation would be better)
        write_idx = self.shm_ptr.signal_write_idx
        next_idx = (write_idx + 1) % MAX_SIGNALS
        
        # Check if queue is full
        if next_idx == self.shm_ptr.signal_read_idx:
            return False
        
        # Write signal
        signal = &self.shm_ptr.signals[write_idx]
        signal.action = action
        signal.confidence = max(0.0, min(1.0, confidence))
        signal.size = size
        signal.price = price
        signal.stop_loss = stop_loss
        signal.take_profit = take_profit
        signal.strategy_id = strategy_id
        signal.timestamp = <uint64_t>(time.time() * 1000000)
        
        # Copy symbol (truncate if necessary)
        symbol_bytes = symbol.encode('utf-8')[:15]
        memset(signal.symbol, 0, 16)
        memcpy(signal.symbol, <char*>symbol_bytes, len(symbol_bytes))
        
        # Update write index (atomic operation would be better)
        self.shm_ptr.signal_write_idx = next_idx
        
        return True
    
    cpdef cnp.ndarray get_market_snapshot(self, list symbols):
        """
        Get market snapshot for multiple symbols.
        Returns numpy array with columns: [bid, ask, last, volume, spread, timestamp].
        """
        cdef:
            int n_symbols = len(symbols)
            cnp.ndarray[double, ndim=2] snapshot
            int i, idx
            MarketData* data
            
        if not self.connected or n_symbols == 0:
            return np.empty((0, 6), dtype=np.float64)
        
        snapshot = np.empty((n_symbols, 6), dtype=np.float64)
        
        for i in range(n_symbols):
            idx = self._get_symbol_index(symbols[i])
            if idx < 0:
                snapshot[i, :] = np.nan
                continue
            
            data = &self.shm_ptr.market_data[idx]
            snapshot[i, 0] = data.bid_price
            snapshot[i, 1] = data.ask_price
            snapshot[i, 2] = data.last_price
            snapshot[i, 3] = data.volume_24h
            snapshot[i, 4] = data.ask_price - data.bid_price  # spread
            snapshot[i, 5] = data.timestamp / 1000000.0
        
        return snapshot
    
    cdef int _get_symbol_index(self, str symbol):
        """Get symbol index in shared memory arrays."""
        # In production, this would use a proper symbol mapping
        # For now, use a simple hash
        if symbol not in self.symbol_map:
            if len(self.symbol_map) >= MAX_SYMBOLS:
                return -1
            self.symbol_map[symbol] = len(self.symbol_map)
        return self.symbol_map[symbol]
    
    def close(self):
        """Clean up resources."""
        if self.shm_ptr != NULL:
            munmap(self.shm_ptr, SHM_SIZE)
            self.shm_ptr = NULL
        
        if self.shm_fd >= 0:
            close(self.shm_fd)
            self.shm_fd = -1
        
        self.connected = False
    
    def __dealloc__(self):
        """Destructor - ensure cleanup."""
        self.close()

# Python-friendly wrapper functions
def create_client(shm_path="/dev/shm/cwts_ultra", websocket_url="ws://localhost:4000"):
    """Create a new CWTS Ultra client."""
    return CWTSUltraClient(shm_path, websocket_url)

def buy_signal(client, symbol, confidence=1.0, size=0.0, price=0.0, 
               stop_loss=0.0, take_profit=0.0, strategy_id=0):
    """Send buy signal."""
    return client.send_signal(symbol, SIGNAL_BUY, confidence, size, price, 
                             stop_loss, take_profit, strategy_id)

def sell_signal(client, symbol, confidence=1.0, size=0.0, price=0.0,
                stop_loss=0.0, take_profit=0.0, strategy_id=0):
    """Send sell signal."""
    return client.send_signal(symbol, SIGNAL_SELL, confidence, size, price,
                             stop_loss, take_profit, strategy_id)

def hold_signal(client, symbol, strategy_id=0):
    """Send hold signal."""
    return client.send_signal(symbol, SIGNAL_HOLD, 1.0, 0.0, 0.0, 0.0, 0.0, strategy_id)
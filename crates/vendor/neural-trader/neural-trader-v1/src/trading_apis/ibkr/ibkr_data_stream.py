"""
Interactive Brokers Real-Time Data Stream Handler

Provides ultra-low latency streaming market data with optimized message processing,
batching, and filtering capabilities for high-frequency trading.
"""

import asyncio
import time
import struct
from typing import Dict, List, Optional, Callable, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import numpy as np
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Types of market data available"""
    TRADES = "trades"
    QUOTES = "quotes"
    DEPTH = "depth"
    BARS = "bars"
    NEWS = "news"
    SCANNER = "scanner"
    FUNDAMENTAL = "fundamental"
    OPTION_CHAIN = "option_chain"


class TickType(Enum):
    """IB tick types for real-time data"""
    BID_SIZE = 0
    BID = 1
    ASK = 2
    ASK_SIZE = 3
    LAST = 4
    LAST_SIZE = 5
    HIGH = 6
    LOW = 7
    VOLUME = 8
    CLOSE = 9
    BID_OPTION_COMPUTATION = 10
    ASK_OPTION_COMPUTATION = 11
    LAST_OPTION_COMPUTATION = 12
    MODEL_OPTION = 13
    OPEN = 14
    LOW_13_WEEK = 15
    HIGH_13_WEEK = 16
    LOW_26_WEEK = 17
    HIGH_26_WEEK = 18
    LOW_52_WEEK = 19
    HIGH_52_WEEK = 20
    AVG_VOLUME = 21
    OPEN_INTEREST = 22
    OPTION_HISTORICAL_VOL = 23
    OPTION_IMPLIED_VOL = 24
    LAST_TIMESTAMP = 45
    DELAYED_BID = 66
    DELAYED_ASK = 67
    DELAYED_LAST = 68
    DELAYED_HIGH = 72
    DELAYED_LOW = 73
    DELAYED_CLOSE = 75
    MARK_PRICE = 84


@dataclass
class StreamConfig:
    """Configuration for data streaming"""
    buffer_size: int = 10000
    batch_size: int = 100
    batch_timeout_ms: float = 10.0
    max_symbols: int = 100
    tick_filtering: bool = True
    conflation_ms: float = 0  # 0 = no conflation
    compression: bool = True
    snapshot_interval_ms: float = 1000.0
    use_native_parsing: bool = True  # Use optimized C parsing if available
    pre_allocate_buffers: bool = True


@dataclass
class MarketSnapshot:
    """Point-in-time market snapshot"""
    symbol: str
    timestamp: float
    bid: float = 0.0
    bid_size: int = 0
    ask: float = 0.0
    ask_size: int = 0
    last: float = 0.0
    last_size: int = 0
    volume: int = 0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    vwap: float = 0.0
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0
    
    @property
    def mid(self) -> float:
        """Calculate mid price"""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last


@dataclass
class DepthLevel:
    """Single level of market depth"""
    price: float
    size: int
    market_maker: str = ""
    
    
@dataclass
class MarketDepth:
    """Full market depth book"""
    symbol: str
    timestamp: float
    bids: List[DepthLevel] = field(default_factory=list)
    asks: List[DepthLevel] = field(default_factory=list)
    
    def get_best_bid(self) -> Optional[DepthLevel]:
        """Get best bid"""
        return self.bids[0] if self.bids else None
    
    def get_best_ask(self) -> Optional[DepthLevel]:
        """Get best ask"""
        return self.asks[0] if self.asks else None
    
    def get_total_bid_size(self, levels: int = 5) -> int:
        """Get total bid size for top N levels"""
        return sum(level.size for level in self.bids[:levels])
    
    def get_total_ask_size(self, levels: int = 5) -> int:
        """Get total ask size for top N levels"""
        return sum(level.size for level in self.asks[:levels])


class StreamBuffer:
    """High-performance circular buffer for market data"""
    
    def __init__(self, size: int):
        self.size = size
        self.buffer = deque(maxlen=size)
        self.write_index = 0
        self.read_index = 0
        self._lock = asyncio.Lock()
    
    async def write(self, data: Any):
        """Write data to buffer"""
        async with self._lock:
            self.buffer.append(data)
            self.write_index += 1
    
    async def read_batch(self, max_items: int) -> List[Any]:
        """Read batch of items"""
        async with self._lock:
            items = []
            while len(items) < max_items and self.buffer:
                items.append(self.buffer.popleft())
            return items
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self.buffer) == 0
    
    def size_bytes(self) -> int:
        """Estimate buffer size in bytes"""
        # Rough estimate - improve based on actual data types
        return len(self.buffer) * 100


class MessageParser:
    """Optimized message parser for IB protocol"""
    
    @staticmethod
    def parse_tick_price(data: bytes) -> Tuple[int, int, float, int]:
        """
        Parse tick price message
        Returns: (req_id, tick_type, price, size)
        """
        # Assuming binary protocol for speed
        # Format: <req_id:4><tick_type:2><price:8><size:4>
        if len(data) < 18:
            return None
        
        req_id = struct.unpack('>I', data[0:4])[0]
        tick_type = struct.unpack('>H', data[4:6])[0]
        price = struct.unpack('>d', data[6:14])[0]
        size = struct.unpack('>I', data[14:18])[0]
        
        return req_id, tick_type, price, size
    
    @staticmethod
    def parse_tick_size(data: bytes) -> Tuple[int, int, int]:
        """
        Parse tick size message
        Returns: (req_id, tick_type, size)
        """
        if len(data) < 10:
            return None
        
        req_id = struct.unpack('>I', data[0:4])[0]
        tick_type = struct.unpack('>H', data[4:6])[0]
        size = struct.unpack('>I', data[6:10])[0]
        
        return req_id, tick_type, size
    
    @staticmethod
    def parse_market_depth(data: bytes) -> Tuple[int, int, int, int, float, int, str]:
        """
        Parse market depth update
        Returns: (req_id, position, market_maker_id, operation, side, price, size)
        """
        # Implementation depends on actual protocol
        pass


class IBKRDataStream:
    """
    High-performance market data streaming handler
    
    Features:
    - Sub-millisecond tick processing
    - Automatic batching and conflation
    - Memory-efficient circular buffers
    - Real-time filtering and aggregation
    - Multiple data type support
    """
    
    def __init__(self, client, config: Optional[StreamConfig] = None):
        self.client = client  # IBKRClient instance
        self.config = config or StreamConfig()
        self.subscriptions: Dict[str, Set[DataType]] = defaultdict(set)
        self.snapshots: Dict[str, MarketSnapshot] = {}
        self.depth_books: Dict[str, MarketDepth] = {}
        self.buffers: Dict[str, StreamBuffer] = {}
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.req_id_map: Dict[int, str] = {}  # req_id -> symbol mapping
        self.next_req_id = 1000
        self._processor_task = None
        self._stats = {
            'ticks_received': 0,
            'ticks_processed': 0,
            'ticks_dropped': 0,
            'batches_processed': 0,
            'avg_batch_size': 0,
            'processing_time_ms': deque(maxlen=1000)
        }
        self._last_snapshot_time = {}
        self.parser = MessageParser()
    
    async def subscribe(self, 
                       symbol: str, 
                       data_types: List[DataType],
                       callback: Optional[Callable] = None) -> bool:
        """
        Subscribe to market data for a symbol
        
        Args:
            symbol: Stock symbol
            data_types: List of data types to subscribe to
            callback: Optional callback for data updates
            
        Returns:
            True if subscription successful
        """
        if len(self.subscriptions) >= self.config.max_symbols:
            logger.error(f"Maximum symbols ({self.config.max_symbols}) reached")
            return False
        
        try:
            # Initialize buffer for symbol
            if symbol not in self.buffers:
                self.buffers[symbol] = StreamBuffer(self.config.buffer_size)
            
            # Initialize snapshot
            if symbol not in self.snapshots:
                self.snapshots[symbol] = MarketSnapshot(symbol=symbol, timestamp=time.time())
            
            # Register callback
            if callback:
                self.callbacks[symbol].append(callback)
            
            # Subscribe to each data type
            for data_type in data_types:
                if data_type == DataType.TRADES:
                    req_id = await self._subscribe_trades(symbol)
                elif data_type == DataType.QUOTES:
                    req_id = await self._subscribe_quotes(symbol)
                elif data_type == DataType.DEPTH:
                    req_id = await self._subscribe_depth(symbol)
                else:
                    logger.warning(f"Unsupported data type: {data_type}")
                    continue
                
                if req_id:
                    self.subscriptions[symbol].add(data_type)
                    self.req_id_map[req_id] = symbol
            
            # Start processor if not running
            if not self._processor_task:
                self._processor_task = asyncio.create_task(self._process_loop())
            
            logger.info(f"Subscribed to {symbol} for {data_types}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
            return False
    
    async def _subscribe_trades(self, symbol: str) -> Optional[int]:
        """Subscribe to trade data"""
        if not self.client or not self.client._connected:
            return None
        
        req_id = self._get_next_req_id()
        
        # In real implementation, send subscription to IB
        # For now, simulate with callback registration
        self.client.register_callback('tick_price', 
                                    lambda data: asyncio.create_task(self._on_tick_price(data)))
        self.client.register_callback('tick_size', 
                                    lambda data: asyncio.create_task(self._on_tick_size(data)))
        
        return req_id
    
    async def _subscribe_quotes(self, symbol: str) -> Optional[int]:
        """Subscribe to quote data"""
        if not self.client or not self.client._connected:
            return None
        
        req_id = self._get_next_req_id()
        
        # Register for bid/ask updates
        self.client.register_callback('tick_price', 
                                    lambda data: asyncio.create_task(self._on_tick_price(data)))
        
        return req_id
    
    async def _subscribe_depth(self, symbol: str) -> Optional[int]:
        """Subscribe to market depth"""
        if not self.client or not self.client._connected:
            return None
        
        req_id = self._get_next_req_id()
        
        # Initialize depth book
        if symbol not in self.depth_books:
            self.depth_books[symbol] = MarketDepth(symbol=symbol, timestamp=time.time())
        
        # Register for depth updates
        self.client.register_callback('market_depth', 
                                    lambda data: asyncio.create_task(self._on_market_depth(data)))
        
        return req_id
    
    def _get_next_req_id(self) -> int:
        """Get next request ID"""
        req_id = self.next_req_id
        self.next_req_id += 1
        return req_id
    
    async def _on_tick_price(self, data: Any):
        """Handle tick price update"""
        try:
            # Parse message
            if self.config.use_native_parsing:
                parsed = self.parser.parse_tick_price(data)
                if not parsed:
                    return
                req_id, tick_type, price, size = parsed
            else:
                # Fallback to standard parsing
                req_id = data.get('req_id')
                tick_type = data.get('tick_type')
                price = data.get('price')
                size = data.get('size', 0)
            
            # Get symbol from request ID
            symbol = self.req_id_map.get(req_id)
            if not symbol:
                return
            
            # Update snapshot
            snapshot = self.snapshots.get(symbol)
            if snapshot:
                timestamp = time.time()
                
                if tick_type == TickType.BID.value:
                    snapshot.bid = price
                    snapshot.bid_size = size
                elif tick_type == TickType.ASK.value:
                    snapshot.ask = price
                    snapshot.ask_size = size
                elif tick_type == TickType.LAST.value:
                    snapshot.last = price
                    snapshot.last_size = size
                
                snapshot.timestamp = timestamp
                
                # Add to buffer
                await self.buffers[symbol].write({
                    'type': 'tick',
                    'tick_type': tick_type,
                    'price': price,
                    'size': size,
                    'timestamp': timestamp
                })
            
            self._stats['ticks_received'] += 1
            
        except Exception as e:
            logger.error(f"Error processing tick price: {e}")
            self._stats['ticks_dropped'] += 1
    
    async def _on_tick_size(self, data: Any):
        """Handle tick size update"""
        try:
            # Similar to tick price handling
            req_id = data.get('req_id')
            tick_type = data.get('tick_type')
            size = data.get('size')
            
            symbol = self.req_id_map.get(req_id)
            if not symbol:
                return
            
            snapshot = self.snapshots.get(symbol)
            if snapshot:
                if tick_type == TickType.BID_SIZE.value:
                    snapshot.bid_size = size
                elif tick_type == TickType.ASK_SIZE.value:
                    snapshot.ask_size = size
                elif tick_type == TickType.VOLUME.value:
                    snapshot.volume = size
                
                snapshot.timestamp = time.time()
            
            self._stats['ticks_received'] += 1
            
        except Exception as e:
            logger.error(f"Error processing tick size: {e}")
            self._stats['ticks_dropped'] += 1
    
    async def _on_market_depth(self, data: Any):
        """Handle market depth update"""
        try:
            req_id = data.get('req_id')
            position = data.get('position')
            market_maker = data.get('market_maker', '')
            operation = data.get('operation')  # 0=insert, 1=update, 2=delete
            side = data.get('side')  # 0=ask, 1=bid
            price = data.get('price')
            size = data.get('size')
            
            symbol = self.req_id_map.get(req_id)
            if not symbol:
                return
            
            depth = self.depth_books.get(symbol)
            if not depth:
                return
            
            level = DepthLevel(price=price, size=size, market_maker=market_maker)
            
            # Update depth book based on operation
            if side == 1:  # Bid
                if operation == 0:  # Insert
                    depth.bids.insert(position, level)
                elif operation == 1:  # Update
                    if position < len(depth.bids):
                        depth.bids[position] = level
                elif operation == 2:  # Delete
                    if position < len(depth.bids):
                        depth.bids.pop(position)
            else:  # Ask
                if operation == 0:  # Insert
                    depth.asks.insert(position, level)
                elif operation == 1:  # Update
                    if position < len(depth.asks):
                        depth.asks[position] = level
                elif operation == 2:  # Delete
                    if position < len(depth.asks):
                        depth.asks.pop(position)
            
            depth.timestamp = time.time()
            
            # Add to buffer
            await self.buffers[symbol].write({
                'type': 'depth',
                'depth': depth,
                'timestamp': depth.timestamp
            })
            
        except Exception as e:
            logger.error(f"Error processing market depth: {e}")
    
    async def _process_loop(self):
        """Main processing loop for batching and callbacks"""
        batch_interval = self.config.batch_timeout_ms / 1000.0
        
        while True:
            try:
                start_time = time.time()
                total_processed = 0
                
                # Process each symbol's buffer
                for symbol, buffer in self.buffers.items():
                    if buffer.is_empty():
                        continue
                    
                    # Read batch
                    batch = await buffer.read_batch(self.config.batch_size)
                    if not batch:
                        continue
                    
                    # Apply conflation if configured
                    if self.config.conflation_ms > 0:
                        batch = self._conflate_batch(batch, self.config.conflation_ms)
                    
                    # Process batch
                    await self._process_batch(symbol, batch)
                    total_processed += len(batch)
                    
                    # Check if snapshot update needed
                    await self._check_snapshot_update(symbol)
                
                # Update statistics
                if total_processed > 0:
                    self._stats['batches_processed'] += 1
                    self._stats['ticks_processed'] += total_processed
                    
                    # Update average batch size
                    avg = self._stats['avg_batch_size']
                    self._stats['avg_batch_size'] = (avg * 0.9) + (total_processed * 0.1)
                
                # Track processing time
                process_time = (time.time() - start_time) * 1000
                self._stats['processing_time_ms'].append(process_time)
                
                # Sleep for remainder of batch interval
                sleep_time = max(0, batch_interval - (time.time() - start_time))
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in process loop: {e}")
                await asyncio.sleep(1)
    
    def _conflate_batch(self, batch: List[Dict], conflation_ms: float) -> List[Dict]:
        """Apply conflation to reduce data rate"""
        if not batch:
            return batch
        
        conflated = []
        last_timestamp = 0
        
        for item in batch:
            timestamp = item.get('timestamp', 0)
            if (timestamp - last_timestamp) * 1000 >= conflation_ms:
                conflated.append(item)
                last_timestamp = timestamp
        
        return conflated
    
    async def _process_batch(self, symbol: str, batch: List[Dict]):
        """Process a batch of updates"""
        if not batch:
            return
        
        # Get callbacks for symbol
        callbacks = self.callbacks.get(symbol, [])
        
        # Prepare batch data
        batch_data = {
            'symbol': symbol,
            'updates': batch,
            'snapshot': self.snapshots.get(symbol),
            'depth': self.depth_books.get(symbol),
            'timestamp': time.time()
        }
        
        # Call callbacks
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(batch_data)
                else:
                    callback(batch_data)
            except Exception as e:
                logger.error(f"Error in callback for {symbol}: {e}")
    
    async def _check_snapshot_update(self, symbol: str):
        """Check if snapshot update should be sent"""
        last_update = self._last_snapshot_time.get(symbol, 0)
        current_time = time.time()
        
        if (current_time - last_update) * 1000 >= self.config.snapshot_interval_ms:
            self._last_snapshot_time[symbol] = current_time
            
            # Send snapshot update
            snapshot = self.snapshots.get(symbol)
            if snapshot:
                for callback in self.callbacks.get(symbol, []):
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback({'type': 'snapshot', 'snapshot': snapshot})
                        else:
                            callback({'type': 'snapshot', 'snapshot': snapshot})
                    except Exception as e:
                        logger.error(f"Error in snapshot callback: {e}")
    
    async def unsubscribe(self, symbol: str, data_types: Optional[List[DataType]] = None):
        """Unsubscribe from market data"""
        if symbol not in self.subscriptions:
            return
        
        if data_types:
            for data_type in data_types:
                self.subscriptions[symbol].discard(data_type)
            
            if not self.subscriptions[symbol]:
                del self.subscriptions[symbol]
        else:
            del self.subscriptions[symbol]
        
        # Clean up if no more subscriptions
        if symbol not in self.subscriptions:
            if symbol in self.buffers:
                del self.buffers[symbol]
            if symbol in self.snapshots:
                del self.snapshots[symbol]
            if symbol in self.depth_books:
                del self.depth_books[symbol]
            if symbol in self.callbacks:
                del self.callbacks[symbol]
        
        logger.info(f"Unsubscribed from {symbol}")
    
    def get_snapshot(self, symbol: str) -> Optional[MarketSnapshot]:
        """Get current market snapshot for symbol"""
        return self.snapshots.get(symbol)
    
    def get_depth(self, symbol: str) -> Optional[MarketDepth]:
        """Get current market depth for symbol"""
        return self.depth_books.get(symbol)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        stats = self._stats.copy()
        
        # Calculate processing latency stats
        if stats['processing_time_ms']:
            times = list(stats['processing_time_ms'])
            stats['processing_latency'] = {
                'avg': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'p95': sorted(times)[int(len(times)*0.95)] if len(times) > 20 else max(times)
            }
        
        # Calculate drop rate
        total = stats['ticks_received']
        if total > 0:
            stats['drop_rate'] = stats['ticks_dropped'] / total
        
        # Buffer usage
        stats['buffer_usage'] = {}
        for symbol, buffer in self.buffers.items():
            stats['buffer_usage'][symbol] = {
                'size': len(buffer.buffer),
                'capacity': buffer.size,
                'usage_pct': (len(buffer.buffer) / buffer.size) * 100
            }
        
        return stats
    
    async def stop(self):
        """Stop the data stream"""
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        # Clear all subscriptions
        for symbol in list(self.subscriptions.keys()):
            await self.unsubscribe(symbol)
        
        logger.info("Data stream stopped")
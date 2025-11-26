"""Real-time data feed integration with WebSocket and REST support.

Provides high-performance data streaming with <10ms latency,
automatic failover, circuit breakers, and backpressure handling.
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Callable
import logging

import aiohttp
import websockets
from websockets.exceptions import WebSocketException

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state enumeration."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    FAILED = auto()


class DataSource(Enum):
    """Data source types with priority."""
    WEBSOCKET = (1, True)  # (priority, is_realtime)
    REST = (2, False)
    
    @property
    def priority(self) -> int:
        return self.value[0]
    
    @property
    def is_realtime(self) -> bool:
        return self.value[1]


@dataclass
class DataUpdate:
    """Data update message."""
    symbol: str
    price: float
    timestamp: float
    source: DataSource
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        """Make updates hashable for deduplication."""
        return hash((self.symbol, self.price, self.timestamp))


@dataclass
class FeedConfig:
    """Configuration for real-time feed."""
    websocket_url: str = ""
    rest_url: str = ""
    max_latency_ms: float = 10
    max_updates_per_second: int = 10000
    enable_failover: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 30
    backpressure_buffer_size: int = 50000
    sources: List[Dict[str, str]] = field(default_factory=list)
    aggregation_strategy: str = "merge_dedupe"
    reconnect_delay: float = 1.0
    max_reconnect_attempts: int = 5


class BackpressureHandler:
    """Handle backpressure when buffer fills up."""
    
    def __init__(self, buffer_size: int = 50000, 
                 high_watermark: float = 0.8,
                 low_watermark: float = 0.6):
        self.max_buffer_size = buffer_size
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self.buffer: deque = deque(maxlen=buffer_size)
        self.is_active = False
        self.dropped_count = 0
        self._lock = asyncio.Lock()
    
    async def add_item(self, item: Any) -> bool:
        """Add item to buffer, return False if dropped."""
        async with self._lock:
            if len(self.buffer) >= self.max_buffer_size:
                self.dropped_count += 1
                return False
            
            self.buffer.append(item)
            
            # Check if we should activate backpressure
            if not self.is_active and self.buffer_utilization > self.high_watermark:
                self.is_active = True
                logger.warning(f"Backpressure activated at {self.buffer_utilization:.1%} utilization")
            
            return True
    
    async def remove_item(self) -> Optional[Any]:
        """Remove and return item from buffer."""
        async with self._lock:
            if not self.buffer:
                return None
            
            item = self.buffer.popleft()
            
            # Check if we should deactivate backpressure
            if self.is_active and self.buffer_utilization < self.low_watermark:
                self.is_active = False
                logger.info(f"Backpressure deactivated at {self.buffer_utilization:.1%} utilization")
            
            return item
    
    @property
    def buffer_size(self) -> int:
        """Current buffer size."""
        return len(self.buffer)
    
    @property
    def buffer_utilization(self) -> float:
        """Buffer utilization percentage."""
        return len(self.buffer) / self.max_buffer_size
    
    def add_item_sync(self, item: Any) -> bool:
        """Synchronous version for compatibility."""
        if len(self.buffer) >= self.max_buffer_size:
            self.dropped_count += 1
            return False
        
        self.buffer.append(item)
        
        if not self.is_active and self.buffer_utilization > self.high_watermark:
            self.is_active = True
        
        return True
    
    def remove_item_sync(self) -> Optional[Any]:
        """Synchronous version for compatibility."""
        if not self.buffer:
            return None
        
        item = self.buffer.popleft()
        
        if self.is_active and self.buffer_utilization < self.low_watermark:
            self.is_active = False
        
        return item


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5,
                 timeout_seconds: int = 30,
                 half_open_max_calls: int = 1):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls
        
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        self._state = "CLOSED"
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        if self._state == "OPEN" and self._time_since_open() > self.timeout_seconds:
            return "HALF_OPEN"
        return self._state
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == "OPEN"
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open."""
        return self.state == "HALF_OPEN"
    
    async def can_execute(self) -> bool:
        """Check if execution is allowed."""
        async with self._lock:
            state = self.state
            
            if state == "CLOSED":
                return True
            elif state == "OPEN":
                return False
            else:  # HALF_OPEN
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
    
    async def record_success(self):
        """Record successful execution."""
        async with self._lock:
            if self.state == "HALF_OPEN":
                self._state = "CLOSED"
                self._failure_count = 0
                self._half_open_calls = 0
                logger.info("Circuit breaker closed after successful execution")
    
    async def record_failure(self):
        """Record failed execution."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                self._state = "OPEN"
                self._half_open_calls = 0
                logger.warning(f"Circuit breaker opened after {self._failure_count} failures")
            elif self.state == "HALF_OPEN":
                self._state = "OPEN"
                logger.warning("Circuit breaker reopened after failure in half-open state")
    
    def _time_since_open(self) -> float:
        """Time since circuit was opened."""
        if self._last_failure_time is None:
            return 0
        return time.time() - self._last_failure_time


class RealtimeFeed:
    """High-performance real-time data feed."""
    
    def __init__(self, config: FeedConfig):
        self.config = config
        self.connection_state = ConnectionState.DISCONNECTED
        self.is_websocket_active = False
        
        # WebSocket and HTTP session
        self._ws = None
        self._session = None
        
        # Data handling
        self.backpressure_handler = BackpressureHandler(
            buffer_size=config.backpressure_buffer_size
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            timeout_seconds=config.circuit_breaker_timeout
        )
        
        # Performance tracking
        self._processed_updates: List[DataUpdate] = []
        self._update_cache: Set[DataUpdate] = set()
        self._dropped_count = 0
        self._latency_samples: deque = deque(maxlen=1000)
        self._last_update_time = time.time()
        self._update_count = 0
        
        # Tasks
        self._tasks: List[asyncio.Task] = []
    
    async def connect(self):
        """Connect to data feed (WebSocket with REST fallback)."""
        self.connection_state = ConnectionState.CONNECTING
        
        try:
            if self.config.websocket_url:
                await self._connect_websocket()
            else:
                await self._connect_rest()
            
            self.connection_state = ConnectionState.CONNECTED
            logger.info(f"Connected to data feed (WebSocket: {self.is_websocket_active})")
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.connection_state = ConnectionState.FAILED
            
            if self.config.enable_failover and self.is_websocket_active:
                logger.info("Attempting REST fallback")
                await self._connect_rest()
    
    async def connect_all_sources(self):
        """Connect to all configured sources."""
        tasks = []
        for source in self.config.sources:
            if source["type"] == "websocket":
                task = asyncio.create_task(self._connect_websocket_source(source["url"]))
            else:
                task = asyncio.create_task(self._connect_rest_source(source["url"]))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _connect_websocket(self):
        """Connect via WebSocket."""
        try:
            self._ws = await websockets.connect(self.config.websocket_url)
            self.is_websocket_active = True
            
            # Start message handler
            task = asyncio.create_task(self._websocket_handler())
            self._tasks.append(task)
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    async def _connect_rest(self):
        """Connect via REST API."""
        if not self._session:
            self._session = aiohttp.ClientSession()
        
        self.is_websocket_active = False
        
        # Start polling task
        task = asyncio.create_task(self._rest_polling_handler())
        self._tasks.append(task)
    
    async def _websocket_handler(self):
        """Handle WebSocket messages."""
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    await self._handle_websocket_message(data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {message}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
            await self._handle_connection_failure()
    
    async def _rest_polling_handler(self):
        """Poll REST API for updates."""
        while self.connection_state == ConnectionState.CONNECTED:
            try:
                async with self._session.get(self.config.rest_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        await self._handle_rest_response(data)
                    else:
                        logger.warning(f"REST API returned status {response.status}")
                
                # Adaptive polling rate based on update frequency
                await asyncio.sleep(0.01)  # 100Hz polling for low latency
                
            except Exception as e:
                logger.error(f"REST polling error: {e}")
                await asyncio.sleep(1)
    
    async def _handle_websocket_message(self, data: Dict[str, Any]):
        """Process WebSocket message with latency tracking."""
        start_time = time.perf_counter()
        
        update = DataUpdate(
            symbol=data.get("symbol"),
            price=data.get("price"),
            timestamp=data.get("timestamp", time.time()),
            source=DataSource.WEBSOCKET,
            metadata=data
        )
        
        await self.handle_update(update)
        
        # Track latency
        latency = (time.perf_counter() - start_time) * 1000
        self._latency_samples.append(latency)
    
    async def _handle_rest_response(self, data: Any):
        """Process REST API response."""
        if isinstance(data, list):
            for item in data:
                update = DataUpdate(
                    symbol=item.get("symbol"),
                    price=item.get("price"),
                    timestamp=item.get("timestamp", time.time()),
                    source=DataSource.REST,
                    metadata=item
                )
                await self.handle_update(update)
        else:
            update = DataUpdate(
                symbol=data.get("symbol"),
                price=data.get("price"),
                timestamp=data.get("timestamp", time.time()),
                source=DataSource.REST,
                metadata=data
            )
            await self.handle_update(update)
    
    async def handle_update(self, update: DataUpdate):
        """Handle data update with deduplication and backpressure."""
        # Deduplication
        if update in self._update_cache:
            return
        
        self._update_cache.add(update)
        
        # Backpressure handling
        if not await self.backpressure_handler.add_item(update):
            self._dropped_count += 1
            return
        
        # Process update
        await self._process_update(update)
        
        # Update metrics
        self._update_count += 1
    
    async def _process_update(self, update: DataUpdate):
        """Process a single update."""
        self._processed_updates.append(update)
        
        # Trim cache to prevent memory growth
        if len(self._update_cache) > 10000:
            self._update_cache.clear()
    
    async def _check_connection_health(self):
        """Check connection health and trigger failover if needed."""
        if self.is_websocket_active and not self._websocket_healthy():
            logger.warning("WebSocket unhealthy, triggering failover")
            self.is_websocket_active = False
            await self._connect_rest()
    
    def _websocket_healthy(self) -> bool:
        """Check if WebSocket connection is healthy."""
        return self._ws is not None and not self._ws.closed
    
    async def _handle_connection_failure(self):
        """Handle connection failure with circuit breaker."""
        await self.circuit_breaker.record_failure()
        
        if self.config.enable_failover and await self.circuit_breaker.can_execute():
            self.connection_state = ConnectionState.RECONNECTING
            await asyncio.sleep(self.config.reconnect_delay)
            await self.connect()
    
    def get_buffered_updates(self) -> List[DataUpdate]:
        """Get updates from buffer."""
        updates = []
        while True:
            update = self.backpressure_handler.remove_item_sync()
            if update is None:
                break
            updates.append(update)
        return updates
    
    def get_processed_updates(self) -> List[DataUpdate]:
        """Get processed updates."""
        return self._processed_updates.copy()
    
    def get_dropped_count(self) -> int:
        """Get number of dropped updates."""
        return self._dropped_count + self.backpressure_handler.dropped_count
    
    async def get_aggregated_data(self) -> Dict[str, DataUpdate]:
        """Get aggregated data by symbol."""
        aggregated = {}
        for update in self._processed_updates:
            if update.symbol not in aggregated or update.timestamp > aggregated[update.symbol].timestamp:
                aggregated[update.symbol] = update
        return aggregated
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        now = time.time()
        duration = now - self._last_update_time
        
        avg_latency = sum(self._latency_samples) / len(self._latency_samples) if self._latency_samples else 0
        updates_per_second = self._update_count / duration if duration > 0 else 0
        
        return {
            "avg_latency_ms": avg_latency,
            "max_latency_ms": max(self._latency_samples) if self._latency_samples else 0,
            "min_latency_ms": min(self._latency_samples) if self._latency_samples else 0,
            "updates_per_second": updates_per_second,
            "buffer_utilization": self.backpressure_handler.buffer_utilization,
            "dropped_updates": self.get_dropped_count(),
            "processed_updates": len(self._processed_updates),
        }
    
    async def close(self):
        """Close all connections and clean up."""
        self.connection_state = ConnectionState.DISCONNECTED
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Close connections
        if self._ws:
            await self._ws.close()
        
        if self._session:
            await self._session.close()
        
        logger.info("Real-time feed closed")


# Stub implementations for source-specific connections
async def _connect_websocket_source(self, url: str):
    """Connect to a specific WebSocket source."""
    # Implementation would be similar to _connect_websocket
    pass


async def _connect_rest_source(self, url: str):
    """Connect to a specific REST source."""
    # Implementation would be similar to _connect_rest
    pass
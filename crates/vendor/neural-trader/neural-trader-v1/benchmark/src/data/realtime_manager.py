"""
Real-time data manager for unified data source integration
"""
import asyncio
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from collections import defaultdict, deque
import time
import statistics
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Connection status for data sources"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class DataPoint:
    """Individual data point from a source"""
    source: str
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    latency_ms: Optional[float] = None
    sequence_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedData:
    """Aggregated data from multiple sources"""
    symbol: str
    timestamp: datetime
    price: float  # Weighted average
    volume: int   # Total volume
    bid: Optional[float] = None
    ask: Optional[float] = None
    sources: List[str] = field(default_factory=list)
    spread: Optional[float] = None
    confidence: float = 1.0


@dataclass
class LatencyStats:
    """Latency statistics for a data source"""
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    count: int


@dataclass
class SystemStats:
    """Overall system statistics"""
    total_symbols: int
    total_data_points: int
    active_connections: int
    cache_size_mb: float
    uptime_seconds: float
    avg_latency_ms: float


class DataFeed(ABC):
    """Abstract base class for data feeds"""
    
    @abstractmethod
    async def start(self) -> None:
        """Start the data feed"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the data feed"""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols"""
        pass


class DataSource(ABC):
    """Abstract base class for data sources"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        self.connection_type = "WebSocket"
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the data source"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the data source"""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols"""
        pass
    
    async def fetch_quote(self, symbol: str) -> Optional[DataPoint]:
        """Fetch a single quote (for REST API)"""
        return None


class RealtimeManager:
    """Unified manager for all real-time data sources"""
    
    def __init__(self, max_connections: int = 10, cache_size_mb: int = 100):
        self.sources: Dict[str, DataSource] = {}
        self.active_connections: Set[str] = set()
        self.max_connections = max_connections
        self.cache_size_mb = cache_size_mb
        
        # Data storage
        self.data_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.aggregated_cache: Dict[str, AggregatedData] = {}
        self.seen_sequences: Set[str] = set()  # For deduplication
        
        # Latency tracking
        self.latency_samples: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Status tracking
        self.connection_status: Dict[str, ConnectionStatus] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # System state
        self.is_running = False
        self.start_time = time.time()
        self.total_data_points = 0
        
        # Callbacks
        self.data_callbacks: List[Callable[[DataPoint], None]] = []
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
    
    def add_source(self, source: DataSource) -> None:
        """Add a data source to the manager"""
        self.sources[source.name] = source
        self.connection_status[source.name] = ConnectionStatus.DISCONNECTED
    
    async def connect_source(self, source_name: str) -> bool:
        """Connect to a specific data source"""
        if source_name not in self.sources:
            logger.error(f"Source {source_name} not found")
            return False
        
        source = self.sources[source_name]
        self.connection_status[source_name] = ConnectionStatus.CONNECTING
        
        try:
            # Try WebSocket first if available
            if hasattr(source, 'connect_websocket'):
                try:
                    await source.connect_websocket()
                    source.connection_type = "WebSocket"
                except Exception as e:
                    logger.warning(f"WebSocket connection failed for {source_name}: {e}")
                    # Fallback to REST
                    source.connection_type = "REST"
                    result = await source.connect()
            else:
                result = await source.connect()
            
            if result:
                source.is_connected = True
                self.active_connections.add(source_name)
                self.connection_status[source_name] = ConnectionStatus.CONNECTED
                logger.info(f"Connected to {source_name} via {source.connection_type}")
                return True
            else:
                self.connection_status[source_name] = ConnectionStatus.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to {source_name}: {e}")
            self.connection_status[source_name] = ConnectionStatus.ERROR
            return False
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all data sources"""
        results = {}
        
        # Connect concurrently but respect max connections
        tasks = []
        for source_name in self.sources:
            if len(self.active_connections) >= self.max_connections:
                break
            tasks.append(self.connect_source(source_name))
        
        if tasks:
            connection_results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, source_name in enumerate(list(self.sources.keys())[:len(tasks)]):
                if isinstance(connection_results[i], Exception):
                    results[source_name] = False
                else:
                    results[source_name] = connection_results[i]
        
        self.is_running = True
        return results
    
    async def subscribe(self, source_name: str, symbols: List[str]) -> None:
        """Subscribe to symbols on a specific source"""
        if source_name not in self.sources:
            raise ValueError(f"Source {source_name} not found")
        
        source = self.sources[source_name]
        await source.subscribe(symbols)
        self.subscriptions[source_name].update(symbols)
    
    async def add_data_point(self, data_point: DataPoint) -> None:
        """Add a data point from a source"""
        # Deduplication
        if data_point.sequence_id and data_point.sequence_id in self.seen_sequences:
            return
        
        if data_point.sequence_id:
            self.seen_sequences.add(data_point.sequence_id)
            # Keep set size manageable
            if len(self.seen_sequences) > 10000:
                self.seen_sequences.clear()
        
        # Track latency
        if data_point.latency_ms is not None:
            self.latency_samples[data_point.source].append(data_point.latency_ms)
        
        # Store data
        self.data_cache[data_point.symbol].append(data_point)
        self.total_data_points += 1
        
        # Trigger callbacks
        for callback in self.data_callbacks:
            try:
                callback(data_point)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        # Update aggregated data
        await self._update_aggregated_data(data_point.symbol)
    
    async def _update_aggregated_data(self, symbol: str) -> None:
        """Update aggregated data for a symbol"""
        if symbol not in self.data_cache:
            return
        
        recent_data = list(self.data_cache[symbol])
        if not recent_data:
            return
        
        # Get latest data from each source
        source_data = {}
        for dp in reversed(recent_data):
            if dp.source not in source_data:
                source_data[dp.source] = dp
        
        if not source_data:
            return
        
        # Calculate aggregated values
        prices = [dp.price for dp in source_data.values()]
        volumes = [dp.volume for dp in source_data.values()]
        
        avg_price = statistics.mean(prices)
        total_volume = sum(volumes)
        
        # Calculate bid/ask if available
        bids = [dp.bid for dp in source_data.values() if dp.bid is not None]
        asks = [dp.ask for dp in source_data.values() if dp.ask is not None]
        
        avg_bid = statistics.mean(bids) if bids else None
        avg_ask = statistics.mean(asks) if asks else None
        
        # Create aggregated data
        self.aggregated_cache[symbol] = AggregatedData(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            price=avg_price,
            volume=total_volume,
            bid=avg_bid,
            ask=avg_ask,
            sources=list(source_data.keys()),
            spread=avg_ask - avg_bid if avg_ask and avg_bid else None,
            confidence=len(source_data) / len(self.active_connections) if self.active_connections else 0
        )
    
    async def get_aggregated_data(self, symbol: str) -> Optional[AggregatedData]:
        """Get aggregated data for a symbol"""
        return self.aggregated_cache.get(symbol)
    
    async def get_latency_stats(self, source_name: str) -> Optional[LatencyStats]:
        """Get latency statistics for a source"""
        if source_name not in self.latency_samples:
            return None
        
        samples = list(self.latency_samples[source_name])
        if not samples:
            return None
        
        samples_sorted = sorted(samples)
        return LatencyStats(
            avg_latency_ms=statistics.mean(samples),
            min_latency_ms=min(samples),
            max_latency_ms=max(samples),
            p95_latency_ms=samples_sorted[int(len(samples) * 0.95)] if len(samples) > 1 else samples[0],
            p99_latency_ms=samples_sorted[int(len(samples) * 0.99)] if len(samples) > 1 else samples[0],
            count=len(samples)
        )
    
    async def get_stats(self) -> SystemStats:
        """Get overall system statistics"""
        # Calculate cache size (rough estimate)
        cache_size = 0
        for symbol_data in self.data_cache.values():
            cache_size += len(symbol_data) * 200  # Approx bytes per data point
        
        # Get average latency across all sources
        all_latencies = []
        for samples in self.latency_samples.values():
            all_latencies.extend(samples)
        
        avg_latency = statistics.mean(all_latencies) if all_latencies else 0
        
        return SystemStats(
            total_symbols=len(self.data_cache),
            total_data_points=self.total_data_points,
            active_connections=len(self.active_connections),
            cache_size_mb=cache_size / (1024 * 1024),
            uptime_seconds=time.time() - self.start_time,
            avg_latency_ms=avg_latency
        )
    
    def get_connection_status(self, source_name: str) -> ConnectionStatus:
        """Get connection status for a source"""
        return self.connection_status.get(source_name, ConnectionStatus.DISCONNECTED)
    
    async def handle_disconnect(self, source_name: str) -> None:
        """Handle source disconnection"""
        if source_name not in self.sources:
            return
        
        source = self.sources[source_name]
        self.connection_status[source_name] = ConnectionStatus.RECONNECTING
        self.active_connections.discard(source_name)
        
        # Attempt reconnection
        for attempt in range(source.max_reconnect_attempts):
            logger.info(f"Reconnecting to {source_name}, attempt {attempt + 1}")
            source.reconnect_attempts = attempt + 1
            
            if await self.connect_source(source_name):
                logger.info(f"Reconnected to {source_name}")
                # Resubscribe to symbols
                if source_name in self.subscriptions:
                    await source.subscribe(list(self.subscriptions[source_name]))
                return
            
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"Failed to reconnect to {source_name}")
        self.connection_status[source_name] = ConnectionStatus.ERROR
    
    async def get_data_count(self, symbol: str) -> int:
        """Get the number of unique data points for a symbol"""
        if symbol not in self.data_cache:
            return 0
        
        # Count unique data points (by sequence_id or timestamp)
        unique_points = set()
        for dp in self.data_cache[symbol]:
            if dp.sequence_id:
                unique_points.add(dp.sequence_id)
            else:
                unique_points.add((dp.source, dp.timestamp, dp.price))
        
        return len(unique_points)
    
    async def shutdown(self) -> None:
        """Shutdown the manager and disconnect all sources"""
        logger.info("Shutting down RealtimeManager")
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Disconnect all sources
        tasks = []
        for source_name, source in self.sources.items():
            if source.is_connected:
                tasks.append(source.disconnect())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.active_connections.clear()
        logger.info("RealtimeManager shutdown complete")


class RealtimeDataManager:
    """Simple wrapper for RealtimeManager for integration testing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize realtime data manager."""
        self.config = config
        self.manager = RealtimeManager()
        self.connected = False
    
    def start(self) -> bool:
        """Start the real-time data manager."""
        try:
            # Mock successful start
            self.connected = True
            return True
        except Exception:
            return False
    
    def get_latest_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get latest data for symbols."""
        # Mock data response
        data = {}
        for symbol in symbols:
            data[symbol] = {
                'price': 100.0 + hash(symbol) % 100,
                'volume': 1000000,
                'timestamp': datetime.now(),
                'bid': 99.95,
                'ask': 100.05
            }
        return data
"""Multi-source data aggregation with deduplication and conflict resolution.

Aggregates real-time data from multiple sources with intelligent
deduplication, conflict resolution, and quality-based weighting.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import statistics
import logging
import hashlib

from .realtime_feed import DataUpdate, DataSource

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Data aggregation strategies."""
    MERGE_DEDUPE = auto()  # Merge with deduplication
    LATEST_ONLY = auto()   # Keep only latest update
    AVERAGE = auto()       # Average prices from multiple sources
    WEIGHTED = auto()      # Weighted average by source quality
    VOTING = auto()        # Majority voting for consensus


@dataclass
class SourceQuality:
    """Track quality metrics for data sources."""
    source_id: str
    total_updates: int = 0
    valid_updates: int = 0
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_count: int = 0
    last_update_time: float = 0
    
    @property
    def validity_rate(self) -> float:
        """Calculate validity rate."""
        if self.total_updates == 0:
            return 0
        return self.valid_updates / self.total_updates
    
    @property
    def avg_latency(self) -> float:
        """Calculate average latency."""
        if not self.latency_samples:
            return float('inf')
        return statistics.mean(self.latency_samples)
    
    @property
    def quality_score(self) -> float:
        """Calculate overall quality score (0-1)."""
        # Combine validity rate and latency into quality score
        validity_weight = 0.7
        latency_weight = 0.3
        
        # Normalize latency (assume 100ms is bad, 1ms is good)
        normalized_latency = max(0, 1 - (self.avg_latency / 100))
        
        return (validity_weight * self.validity_rate + 
                latency_weight * normalized_latency)


@dataclass
class AggregatedUpdate:
    """Aggregated data update from multiple sources."""
    symbol: str
    price: float
    timestamp: float
    source_count: int
    sources: List[str]
    confidence: float  # 0-1 confidence score
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        """Make updates hashable."""
        return hash((self.symbol, self.price, self.timestamp))


@dataclass
class AggregatorConfig:
    """Configuration for data aggregator."""
    strategy: AggregationStrategy = AggregationStrategy.MERGE_DEDUPE
    deduplication_window_ms: int = 100
    max_price_deviation_percent: float = 5.0
    min_sources_for_consensus: int = 2
    quality_weight_threshold: float = 0.5
    enable_outlier_detection: bool = True
    outlier_z_score: float = 3.0
    conflict_resolution_strategy: str = "quality_weighted"


class ConflictResolver:
    """Resolve conflicts between different data sources."""
    
    def __init__(self, config: AggregatorConfig):
        self.config = config
        self._resolution_strategies = {
            "quality_weighted": self._quality_weighted_resolution,
            "majority_vote": self._majority_vote_resolution,
            "latest_wins": self._latest_wins_resolution,
            "average": self._average_resolution,
        }
    
    def resolve(self, updates: List[Tuple[DataUpdate, SourceQuality]]) -> Optional[AggregatedUpdate]:
        """Resolve conflicts between updates."""
        if not updates:
            return None
        
        if len(updates) == 1:
            update, quality = updates[0]
            return AggregatedUpdate(
                symbol=update.symbol,
                price=update.price,
                timestamp=update.timestamp,
                source_count=1,
                sources=[quality.source_id],
                confidence=quality.quality_score
            )
        
        strategy = self._resolution_strategies.get(
            self.config.conflict_resolution_strategy,
            self._quality_weighted_resolution
        )
        
        return strategy(updates)
    
    def _quality_weighted_resolution(self, updates: List[Tuple[DataUpdate, SourceQuality]]) -> AggregatedUpdate:
        """Resolve using quality-weighted average."""
        total_weight = 0
        weighted_price = 0
        latest_timestamp = 0
        sources = []
        
        for update, quality in updates:
            weight = quality.quality_score
            if weight < self.config.quality_weight_threshold:
                continue
            
            weighted_price += update.price * weight
            total_weight += weight
            latest_timestamp = max(latest_timestamp, update.timestamp)
            sources.append(quality.source_id)
        
        if total_weight == 0:
            # Fallback to simple average
            return self._average_resolution(updates)
        
        avg_price = weighted_price / total_weight
        confidence = min(1.0, total_weight / len(updates))
        
        return AggregatedUpdate(
            symbol=updates[0][0].symbol,
            price=avg_price,
            timestamp=latest_timestamp,
            source_count=len(sources),
            sources=sources,
            confidence=confidence
        )
    
    def _majority_vote_resolution(self, updates: List[Tuple[DataUpdate, SourceQuality]]) -> AggregatedUpdate:
        """Resolve using majority voting."""
        price_buckets = defaultdict(list)
        
        # Group prices into buckets
        for update, quality in updates:
            bucket = round(update.price, 2)  # Round to cents
            price_buckets[bucket].append((update, quality))
        
        # Find majority bucket
        majority_bucket = max(price_buckets.items(), key=lambda x: len(x[1]))
        majority_price = majority_bucket[0]
        majority_updates = majority_bucket[1]
        
        sources = [q.source_id for _, q in majority_updates]
        confidence = len(majority_updates) / len(updates)
        
        return AggregatedUpdate(
            symbol=updates[0][0].symbol,
            price=majority_price,
            timestamp=max(u.timestamp for u, _ in majority_updates),
            source_count=len(sources),
            sources=sources,
            confidence=confidence
        )
    
    def _latest_wins_resolution(self, updates: List[Tuple[DataUpdate, SourceQuality]]) -> AggregatedUpdate:
        """Resolve by taking the latest update."""
        latest_update, latest_quality = max(updates, key=lambda x: x[0].timestamp)
        
        return AggregatedUpdate(
            symbol=latest_update.symbol,
            price=latest_update.price,
            timestamp=latest_update.timestamp,
            source_count=1,
            sources=[latest_quality.source_id],
            confidence=latest_quality.quality_score
        )
    
    def _average_resolution(self, updates: List[Tuple[DataUpdate, SourceQuality]]) -> AggregatedUpdate:
        """Resolve by simple average."""
        prices = [u.price for u, _ in updates]
        avg_price = statistics.mean(prices)
        
        sources = [q.source_id for _, q in updates]
        
        return AggregatedUpdate(
            symbol=updates[0][0].symbol,
            price=avg_price,
            timestamp=max(u.timestamp for u, _ in updates),
            source_count=len(sources),
            sources=sources,
            confidence=0.5  # Medium confidence for simple average
        )


class DataAggregator:
    """Aggregate data from multiple sources with deduplication."""
    
    def __init__(self, config: AggregatorConfig):
        self.config = config
        self.conflict_resolver = ConflictResolver(config)
        
        # Source tracking
        self._source_quality: Dict[str, SourceQuality] = {}
        self._source_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Deduplication
        self._seen_updates: Set[str] = set()
        self._update_window: deque = deque(maxlen=10000)
        
        # Aggregated data
        self._aggregated_data: Dict[str, AggregatedUpdate] = {}
        self._aggregation_history: deque = deque(maxlen=100000)
        
        # Metrics
        self._total_updates = 0
        self._duplicates_filtered = 0
        self._conflicts_resolved = 0
        
        # Tasks
        self._aggregation_task = None
        self._cleanup_task = None
    
    async def start(self):
        """Start aggregation tasks."""
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Stop aggregation tasks."""
        if self._aggregation_task:
            self._aggregation_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
    
    def add_source(self, source_id: str, quality_hint: float = 1.0):
        """Register a data source."""
        if source_id not in self._source_quality:
            self._source_quality[source_id] = SourceQuality(source_id)
            logger.info(f"Added data source: {source_id}")
    
    async def add_update(self, update: DataUpdate, source_id: str):
        """Add update from a source."""
        self._total_updates += 1
        
        # Check for duplicates
        update_hash = self._get_update_hash(update)
        if update_hash in self._seen_updates:
            self._duplicates_filtered += 1
            return
        
        self._seen_updates.add(update_hash)
        self._update_window.append((update_hash, time.time()))
        
        # Add to source buffer
        self._source_buffers[source_id].append(update)
        
        # Update source quality
        if source_id in self._source_quality:
            quality = self._source_quality[source_id]
            quality.total_updates += 1
            quality.last_update_time = time.time()
            
            # Calculate latency
            latency = (time.time() - update.timestamp) * 1000
            quality.latency_samples.append(latency)
    
    def _get_update_hash(self, update: DataUpdate) -> str:
        """Generate hash for deduplication."""
        # Hash based on symbol, price, and time window
        time_bucket = int(update.timestamp * 1000 / self.config.deduplication_window_ms)
        hash_str = f"{update.symbol}:{update.price}:{time_bucket}"
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    async def _aggregation_loop(self):
        """Main aggregation loop."""
        while True:
            try:
                await self._aggregate_updates()
                await asyncio.sleep(0.001)  # 1ms aggregation interval
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Aggregation error: {e}")
                await asyncio.sleep(0.1)
    
    async def _aggregate_updates(self):
        """Aggregate updates from all sources."""
        # Group updates by symbol
        symbol_updates: Dict[str, List[Tuple[DataUpdate, SourceQuality]]] = defaultdict(list)
        
        for source_id, buffer in self._source_buffers.items():
            if not buffer:
                continue
            
            quality = self._source_quality.get(source_id)
            if not quality:
                continue
            
            # Process updates in buffer
            processed = 0
            while buffer and processed < 100:  # Process max 100 per source per cycle
                update = buffer.popleft()
                symbol_updates[update.symbol].append((update, quality))
                processed += 1
        
        # Aggregate by symbol
        for symbol, updates in symbol_updates.items():
            if len(updates) == 1:
                # Single source, no conflict
                update, quality = updates[0]
                aggregated = AggregatedUpdate(
                    symbol=symbol,
                    price=update.price,
                    timestamp=update.timestamp,
                    source_count=1,
                    sources=[quality.source_id],
                    confidence=quality.quality_score
                )
            else:
                # Multiple sources, resolve conflict
                self._conflicts_resolved += 1
                aggregated = self.conflict_resolver.resolve(updates)
            
            if aggregated:
                self._aggregated_data[symbol] = aggregated
                self._aggregation_history.append(aggregated)
    
    async def _cleanup_loop(self):
        """Clean up old data periodically."""
        while True:
            try:
                await asyncio.sleep(10)  # Cleanup every 10 seconds
                
                # Clean up old hashes
                current_time = time.time()
                cutoff_time = current_time - (self.config.deduplication_window_ms / 1000 * 10)
                
                while self._update_window and self._update_window[0][1] < cutoff_time:
                    old_hash, _ = self._update_window.popleft()
                    self._seen_updates.discard(old_hash)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    def get_aggregated_data(self, symbol: Optional[str] = None) -> Dict[str, AggregatedUpdate]:
        """Get aggregated data."""
        if symbol:
            return {symbol: self._aggregated_data[symbol]} if symbol in self._aggregated_data else {}
        return self._aggregated_data.copy()
    
    def get_latest_update(self, symbol: str) -> Optional[AggregatedUpdate]:
        """Get latest aggregated update for symbol."""
        return self._aggregated_data.get(symbol)
    
    def get_source_quality(self, source_id: str) -> Optional[SourceQuality]:
        """Get quality metrics for a source."""
        return self._source_quality.get(source_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregation metrics."""
        return {
            "total_updates": self._total_updates,
            "duplicates_filtered": self._duplicates_filtered,
            "conflicts_resolved": self._conflicts_resolved,
            "active_symbols": len(self._aggregated_data),
            "active_sources": len(self._source_quality),
            "deduplication_efficiency": self._duplicates_filtered / self._total_updates if self._total_updates > 0 else 0,
            "source_quality": {
                source_id: {
                    "validity_rate": quality.validity_rate,
                    "avg_latency_ms": quality.avg_latency,
                    "quality_score": quality.quality_score,
                    "total_updates": quality.total_updates
                }
                for source_id, quality in self._source_quality.items()
            }
        }
    
    def detect_outliers(self, symbol: str) -> List[str]:
        """Detect outlier sources for a symbol."""
        if not self.config.enable_outlier_detection:
            return []
        
        # Get recent updates for symbol
        recent_updates = []
        for source_id, buffer in self._source_buffers.items():
            for update in buffer:
                if update.symbol == symbol:
                    recent_updates.append((update.price, source_id))
        
        if len(recent_updates) < 3:
            return []  # Need at least 3 sources
        
        prices = [price for price, _ in recent_updates]
        mean_price = statistics.mean(prices)
        stdev_price = statistics.stdev(prices)
        
        if stdev_price == 0:
            return []
        
        # Find outliers
        outlier_sources = []
        for price, source_id in recent_updates:
            z_score = abs(price - mean_price) / stdev_price
            if z_score > self.config.outlier_z_score:
                outlier_sources.append(source_id)
        
        return outlier_sources
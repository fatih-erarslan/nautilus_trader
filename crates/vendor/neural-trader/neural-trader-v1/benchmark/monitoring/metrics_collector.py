"""
Metrics Collector - Centralized metrics collection and aggregation.

This module collects metrics from all system components and provides
unified access to performance data.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import threading
import psutil
import sqlite3


class MetricCategory(Enum):
    """Categories of metrics."""
    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    PERFORMANCE = "performance"
    ERROR = "error"


class AggregationType(Enum):
    """Types of metric aggregation."""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    RATE = "rate"
    PERCENTILE = "percentile"


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    category: MetricCategory
    description: str
    unit: str
    aggregation: AggregationType
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedMetric:
    """Aggregated metric data."""
    name: str
    value: float
    count: int
    min_value: float
    max_value: float
    sum_value: float
    avg_value: float
    timestamp_start: float
    timestamp_end: float
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Centralized metrics collection and aggregation system.
    
    Features:
    - Real-time metric collection
    - Multi-source aggregation
    - Time-series storage
    - Query interface
    - Export capabilities
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.logger = self._setup_logging()
        
        # Storage
        self.storage_path = Path(storage_path) if storage_path else Path("monitoring_data/metrics.db")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Metric definitions
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # In-memory storage
        self.raw_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.aggregated_metrics: Dict[str, Dict[str, AggregatedMetric]] = defaultdict(dict)
        
        # Collection state
        self.is_collecting = False
        self.collection_interval = 1.0  # seconds
        self.aggregation_interval = 60.0  # seconds
        
        # Collectors
        self.metric_collectors: List[Callable[[], Dict[str, float]]] = []
        self.collection_tasks: List[asyncio.Task] = []
        
        # Database connection
        self.db_connection: Optional[sqlite3.Connection] = None
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Register built-in metrics
        self._register_builtin_metrics()
        
        self.logger.info("Metrics Collector initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for metrics collector."""
        logger = logging.getLogger('metrics_collector')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _register_builtin_metrics(self):
        """Register built-in system metrics."""
        builtin_metrics = [
            # System metrics
            MetricDefinition(
                name="system.cpu.usage",
                category=MetricCategory.SYSTEM,
                description="CPU usage percentage",
                unit="percent",
                aggregation=AggregationType.AVERAGE
            ),
            MetricDefinition(
                name="system.memory.usage",
                category=MetricCategory.SYSTEM,
                description="Memory usage percentage",
                unit="percent",
                aggregation=AggregationType.AVERAGE
            ),
            MetricDefinition(
                name="system.disk.usage",
                category=MetricCategory.SYSTEM,
                description="Disk usage percentage",
                unit="percent",
                aggregation=AggregationType.AVERAGE
            ),
            MetricDefinition(
                name="system.network.bytes_sent",
                category=MetricCategory.SYSTEM,
                description="Network bytes sent",
                unit="bytes",
                aggregation=AggregationType.RATE
            ),
            MetricDefinition(
                name="system.network.bytes_recv",
                category=MetricCategory.SYSTEM,
                description="Network bytes received",
                unit="bytes",
                aggregation=AggregationType.RATE
            ),
            
            # Application metrics
            MetricDefinition(
                name="app.requests.total",
                category=MetricCategory.APPLICATION,
                description="Total number of requests",
                unit="count",
                aggregation=AggregationType.RATE
            ),
            MetricDefinition(
                name="app.requests.duration",
                category=MetricCategory.APPLICATION,
                description="Request duration",
                unit="milliseconds",
                aggregation=AggregationType.PERCENTILE
            ),
            MetricDefinition(
                name="app.errors.total",
                category=MetricCategory.ERROR,
                description="Total number of errors",
                unit="count",
                aggregation=AggregationType.RATE
            ),
            
            # Performance metrics
            MetricDefinition(
                name="perf.latency.p95",
                category=MetricCategory.PERFORMANCE,
                description="95th percentile latency",
                unit="milliseconds",
                aggregation=AggregationType.PERCENTILE
            ),
            MetricDefinition(
                name="perf.throughput",
                category=MetricCategory.PERFORMANCE,
                description="Operations per second",
                unit="ops/sec",
                aggregation=AggregationType.AVERAGE
            ),
            
            # Business metrics
            MetricDefinition(
                name="business.trades.executed",
                category=MetricCategory.BUSINESS,
                description="Number of trades executed",
                unit="count",
                aggregation=AggregationType.SUM
            ),
            MetricDefinition(
                name="business.pnl.realized",
                category=MetricCategory.BUSINESS,
                description="Realized profit and loss",
                unit="currency",
                aggregation=AggregationType.SUM
            )
        ]
        
        for metric_def in builtin_metrics:
            self.register_metric(metric_def)
    
    def register_metric(self, metric_def: MetricDefinition):
        """Register a metric definition."""
        with self.lock:
            self.metric_definitions[metric_def.name] = metric_def
        self.logger.debug(f"Registered metric: {metric_def.name}")
    
    def add_collector(self, collector: Callable[[], Dict[str, float]]):
        """Add a metric collector function."""
        self.metric_collectors.append(collector)
        self.logger.debug(f"Added metric collector: {collector.__name__}")
    
    async def start(self) -> bool:
        """Start metrics collection."""
        if self.is_collecting:
            self.logger.warning("Metrics collection already running")
            return True
        
        try:
            self.logger.info("Starting metrics collection...")
            
            # Initialize database
            await self._init_database()
            
            # Start collection tasks
            self.collection_tasks = [
                asyncio.create_task(self._system_metrics_collector()),
                asyncio.create_task(self._custom_metrics_collector()),
                asyncio.create_task(self._aggregation_processor()),
                asyncio.create_task(self._persistence_worker())
            ]
            
            self.is_collecting = True
            self.logger.info("Metrics collection started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start metrics collection: {e}")
            await self.stop()
            return False
    
    async def stop(self) -> bool:
        """Stop metrics collection."""
        if not self.is_collecting:
            return True
        
        self.logger.info("Stopping metrics collection...")
        
        try:
            self.is_collecting = False
            
            # Cancel collection tasks
            for task in self.collection_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.collection_tasks:
                await asyncio.gather(*self.collection_tasks, return_exceptions=True)
            
            # Close database connection
            if self.db_connection:
                self.db_connection.close()
                self.db_connection = None
            
            self.logger.info("Metrics collection stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping metrics collection: {e}")
            return False
    
    async def _init_database(self):
        """Initialize SQLite database for metrics storage."""
        self.db_connection = sqlite3.connect(self.storage_path, check_same_thread=False)
        
        # Create tables
        self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS metric_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp REAL NOT NULL,
                tags TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS aggregated_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                count INTEGER NOT NULL,
                min_value REAL NOT NULL,
                max_value REAL NOT NULL,
                sum_value REAL NOT NULL,
                avg_value REAL NOT NULL,
                timestamp_start REAL NOT NULL,
                timestamp_end REAL NOT NULL,
                tags TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        self.db_connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_metric_points_name_timestamp 
            ON metric_points(name, timestamp)
        """)
        
        self.db_connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_aggregated_metrics_name_timestamp 
            ON aggregated_metrics(name, timestamp_start)
        """)
        
        self.db_connection.commit()
        self.logger.debug("Database initialized")
    
    async def _system_metrics_collector(self):
        """Collect system metrics."""
        self.logger.info("System metrics collector started")
        
        while self.is_collecting:
            try:
                timestamp = time.time()
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                await self.record_metric("system.cpu.usage", cpu_percent, timestamp)
                
                # Memory usage
                memory = psutil.virtual_memory()
                await self.record_metric("system.memory.usage", memory.percent, timestamp)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                await self.record_metric("system.disk.usage", disk_percent, timestamp)
                
                # Network I/O
                network = psutil.net_io_counters()
                await self.record_metric("system.network.bytes_sent", network.bytes_sent, timestamp)
                await self.record_metric("system.network.bytes_recv", network.bytes_recv, timestamp)
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _custom_metrics_collector(self):
        """Collect metrics from registered collectors."""
        self.logger.info("Custom metrics collector started")
        
        while self.is_collecting:
            try:
                timestamp = time.time()
                
                # Collect from all registered collectors
                for collector in self.metric_collectors:
                    try:
                        metrics = collector()
                        for name, value in metrics.items():
                            await self.record_metric(name, value, timestamp)
                    except Exception as e:
                        self.logger.error(f"Collector {collector.__name__} error: {e}")
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Custom metrics collection error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _aggregation_processor(self):
        """Process metric aggregations."""
        self.logger.info("Aggregation processor started")
        
        while self.is_collecting:
            try:
                await self._process_aggregations()
                await asyncio.sleep(self.aggregation_interval)
                
            except Exception as e:
                self.logger.error(f"Aggregation processing error: {e}")
                await asyncio.sleep(self.aggregation_interval)
    
    async def _process_aggregations(self):
        """Process metric aggregations for the current interval."""
        current_time = time.time()
        interval_start = current_time - self.aggregation_interval
        
        with self.lock:
            for metric_name, points in self.raw_metrics.items():
                if not points:
                    continue
                
                # Filter points in the current interval
                interval_points = [
                    p for p in points 
                    if p.timestamp >= interval_start and p.timestamp <= current_time
                ]
                
                if not interval_points:
                    continue
                
                # Calculate aggregations
                values = [p.value for p in interval_points]
                
                aggregated = AggregatedMetric(
                    name=metric_name,
                    value=sum(values) / len(values),  # Average as default
                    count=len(values),
                    min_value=min(values),
                    max_value=max(values),
                    sum_value=sum(values),
                    avg_value=sum(values) / len(values),
                    timestamp_start=interval_start,
                    timestamp_end=current_time,
                    tags=interval_points[0].tags if interval_points else {}
                )
                
                # Store aggregation
                interval_key = f"{int(interval_start)}-{int(current_time)}"
                self.aggregated_metrics[metric_name][interval_key] = aggregated
                
                # Limit aggregated metrics storage
                if len(self.aggregated_metrics[metric_name]) > 1000:
                    # Remove oldest entries
                    sorted_keys = sorted(self.aggregated_metrics[metric_name].keys())
                    for key in sorted_keys[:100]:
                        del self.aggregated_metrics[metric_name][key]
    
    async def _persistence_worker(self):
        """Persist metrics to database."""
        self.logger.info("Persistence worker started")
        
        while self.is_collecting:
            try:
                await self._persist_metrics()
                await asyncio.sleep(10)  # Persist every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Persistence error: {e}")
                await asyncio.sleep(10)
    
    async def _persist_metrics(self):
        """Persist current metrics to database."""
        if not self.db_connection:
            return
        
        with self.lock:
            # Persist raw metrics
            for metric_name, points in self.raw_metrics.items():
                points_to_persist = list(points)[-100:]  # Persist last 100 points
                
                for point in points_to_persist:
                    self.db_connection.execute("""
                        INSERT INTO metric_points (name, value, timestamp, tags, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        point.name,
                        point.value,
                        point.timestamp,
                        json.dumps(point.tags),
                        json.dumps(point.metadata)
                    ))
            
            # Persist aggregated metrics
            for metric_name, intervals in self.aggregated_metrics.items():
                for interval_key, aggregated in intervals.items():
                    # Check if already persisted
                    cursor = self.db_connection.execute("""
                        SELECT COUNT(*) FROM aggregated_metrics 
                        WHERE name = ? AND timestamp_start = ? AND timestamp_end = ?
                    """, (metric_name, aggregated.timestamp_start, aggregated.timestamp_end))
                    
                    if cursor.fetchone()[0] == 0:
                        self.db_connection.execute("""
                            INSERT INTO aggregated_metrics (
                                name, value, count, min_value, max_value, sum_value, avg_value,
                                timestamp_start, timestamp_end, tags
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            aggregated.name,
                            aggregated.value,
                            aggregated.count,
                            aggregated.min_value,
                            aggregated.max_value,
                            aggregated.sum_value,
                            aggregated.avg_value,
                            aggregated.timestamp_start,
                            aggregated.timestamp_end,
                            json.dumps(aggregated.tags)
                        ))
            
            self.db_connection.commit()
    
    async def record_metric(self, name: str, value: float, timestamp: Optional[float] = None, 
                           tags: Optional[Dict[str, str]] = None, 
                           metadata: Optional[Dict[str, Any]] = None):
        """Record a metric point."""
        if timestamp is None:
            timestamp = time.time()
        
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=timestamp,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        with self.lock:
            self.raw_metrics[name].append(point)
    
    def get_metric_history(self, name: str, hours: int = 24) -> List[MetricPoint]:
        """Get metric history for the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.lock:
            if name not in self.raw_metrics:
                return []
            
            return [
                point for point in self.raw_metrics[name]
                if point.timestamp >= cutoff_time
            ]
    
    def get_aggregated_metrics(self, name: str, hours: int = 24) -> List[AggregatedMetric]:
        """Get aggregated metrics for the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.lock:
            if name not in self.aggregated_metrics:
                return []
            
            return [
                metric for metric in self.aggregated_metrics[name].values()
                if metric.timestamp_start >= cutoff_time
            ]
    
    def get_current_values(self) -> Dict[str, float]:
        """Get current values for all metrics."""
        current_values = {}
        
        with self.lock:
            for name, points in self.raw_metrics.items():
                if points:
                    current_values[name] = points[-1].value
        
        return current_values
    
    def get_metric_definitions(self) -> Dict[str, MetricDefinition]:
        """Get all metric definitions."""
        with self.lock:
            return self.metric_definitions.copy()
    
    def export_metrics(self, output_path: str, format: str = "json", hours: int = 24):
        """Export metrics to file."""
        cutoff_time = time.time() - (hours * 3600)
        
        export_data = {
            'export_time': time.time(),
            'hours': hours,
            'metrics': {}
        }
        
        with self.lock:
            for name, points in self.raw_metrics.items():
                recent_points = [
                    {
                        'value': p.value,
                        'timestamp': p.timestamp,
                        'tags': p.tags,
                        'metadata': p.metadata
                    }
                    for p in points if p.timestamp >= cutoff_time
                ]
                
                if recent_points:
                    export_data['metrics'][name] = recent_points
        
        # Save to file
        output_path = Path(output_path)
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format.lower() == "csv":
            import pandas as pd
            
            # Flatten data for CSV
            rows = []
            for name, points in export_data['metrics'].items():
                for point in points:
                    row = {
                        'metric_name': name,
                        'value': point['value'],
                        'timestamp': point['timestamp'],
                        **point['tags']
                    }
                    rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        
        self.logger.info(f"Metrics exported to: {output_path}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get collector status."""
        with self.lock:
            return {
                'collecting': self.is_collecting,
                'collection_interval': self.collection_interval,
                'aggregation_interval': self.aggregation_interval,
                'metric_count': len(self.raw_metrics),
                'total_points': sum(len(points) for points in self.raw_metrics.values()),
                'registered_metrics': len(self.metric_definitions),
                'collectors': len(self.metric_collectors),
                'storage_path': str(self.storage_path)
            }
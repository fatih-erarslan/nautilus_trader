"""
Connection Pool Manager for Trading APIs

Manages persistent connections with automatic failover, health monitoring,
and load balancing for ultra-low latency trading operations.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Type, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
from collections import deque
import random
from enum import Enum

from .api_interface import TradingAPIInterface


class ConnectionState(Enum):
    """Connection states"""
    IDLE = "idle"
    CONNECTING = "connecting"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    DISCONNECTED = "disconnected"


@dataclass
class ConnectionMetrics:
    """Metrics for a single connection"""
    connection_id: str
    state: ConnectionState
    requests_handled: int = 0
    errors_count: int = 0
    total_latency_ms: float = 0.0
    last_used: datetime = field(default_factory=datetime.now)
    last_error: Optional[str] = None
    health_score: float = 100.0
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency"""
        if self.requests_handled == 0:
            return 0.0
        return self.total_latency_ms / self.requests_handled
    
    def update_health_score(self) -> None:
        """Update health score based on errors and latency"""
        # Start with base score
        score = 100.0
        
        # Penalize for errors (each error reduces score by 10)
        score -= min(self.errors_count * 10, 50)
        
        # Penalize for high latency
        if self.avg_latency_ms > 100:  # Over 100ms is concerning
            score -= min((self.avg_latency_ms - 100) / 10, 30)
        
        # Penalize for inactivity
        idle_time = datetime.now() - self.last_used
        if idle_time > timedelta(minutes=5):
            score -= min(idle_time.total_seconds() / 60, 20)
        
        self.health_score = max(score, 0.0)


@dataclass
class PooledConnection:
    """Wrapper for a pooled API connection"""
    connection_id: str
    api_instance: TradingAPIInterface
    metrics: ConnectionMetrics
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    async def acquire(self) -> bool:
        """Try to acquire this connection"""
        return await self.lock.acquire()
    
    def release(self) -> None:
        """Release this connection"""
        self.lock.release()
        self.metrics.last_used = datetime.now()


class ConnectionPool:
    """
    High-performance connection pool for trading APIs.
    
    Features:
    - Persistent connection management
    - Automatic failover and recovery
    - Load balancing with health-based routing
    - Circuit breaker pattern
    - Connection warming and pre-allocation
    """
    
    def __init__(self, 
                 api_class: Type[TradingAPIInterface],
                 config: Dict[str, Any],
                 min_connections: int = 2,
                 max_connections: int = 10,
                 health_check_interval: int = 30):
        """
        Initialize connection pool.
        
        Args:
            api_class: Class implementing TradingAPIInterface
            config: Configuration for API connections
            min_connections: Minimum number of connections to maintain
            max_connections: Maximum number of connections allowed
            health_check_interval: Seconds between health checks
        """
        self.api_class = api_class
        self.config = config
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.health_check_interval = health_check_interval
        
        self.connections: Dict[str, PooledConnection] = {}
        self.connection_queue: deque = deque()
        self._lock = asyncio.Lock()
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._warmup_task: Optional[asyncio.Task] = None
        
        # Circuit breaker settings
        self.circuit_breaker_threshold = 5  # Errors before opening circuit
        self.circuit_breaker_timeout = 60  # Seconds before retry
        self._circuit_open = False
        self._circuit_opened_at: Optional[datetime] = None
        
        # Callback handlers
        self._error_handler: Optional[Callable] = None
        self._metric_handler: Optional[Callable] = None
        
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> None:
        """Initialize the connection pool with minimum connections."""
        self._running = True
        
        # Create initial connections
        for i in range(self.min_connections):
            await self._create_connection(f"conn_{i}")
        
        # Start background tasks
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._warmup_task = asyncio.create_task(self._connection_warmup_loop())
        
        self.logger.info(f"Connection pool initialized with {len(self.connections)} connections")
    
    async def _create_connection(self, connection_id: str) -> Optional[PooledConnection]:
        """Create a new connection."""
        try:
            # Create API instance
            api_instance = self.api_class(self.config)
            
            # Connect with timeout
            connected = await asyncio.wait_for(
                api_instance.connect(),
                timeout=10.0
            )
            
            if not connected:
                self.logger.error(f"Failed to connect {connection_id}")
                return None
            
            # Create pooled connection
            metrics = ConnectionMetrics(
                connection_id=connection_id,
                state=ConnectionState.IDLE
            )
            
            pooled_conn = PooledConnection(
                connection_id=connection_id,
                api_instance=api_instance,
                metrics=metrics
            )
            
            async with self._lock:
                self.connections[connection_id] = pooled_conn
                self.connection_queue.append(connection_id)
            
            self.logger.info(f"Created connection {connection_id}")
            return pooled_conn
            
        except Exception as e:
            self.logger.error(f"Error creating connection {connection_id}: {e}")
            return None
    
    async def acquire_connection(self, 
                               preferred_strategy: str = "health") -> Optional[PooledConnection]:
        """
        Acquire a connection from the pool.
        
        Args:
            preferred_strategy: Strategy for connection selection
                - "health": Choose healthiest connection
                - "random": Random selection
                - "round_robin": Round-robin selection
                
        Returns:
            PooledConnection if available, None otherwise
        """
        # Check circuit breaker
        if self._circuit_open:
            if datetime.now() - self._circuit_opened_at > timedelta(seconds=self.circuit_breaker_timeout):
                self._circuit_open = False
                self._circuit_opened_at = None
            else:
                self.logger.warning("Circuit breaker is open")
                return None
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            async with self._lock:
                available_connections = [
                    conn_id for conn_id, conn in self.connections.items()
                    if conn.metrics.state in [ConnectionState.IDLE, ConnectionState.ACTIVE]
                    and not conn.lock.locked()
                ]
                
                if not available_connections:
                    # Try to create new connection if under limit
                    if len(self.connections) < self.max_connections:
                        new_conn_id = f"conn_{len(self.connections)}"
                        new_conn = await self._create_connection(new_conn_id)
                        if new_conn:
                            available_connections = [new_conn_id]
                    else:
                        self.logger.warning("No available connections and at max limit")
                        await asyncio.sleep(0.1)
                        retry_count += 1
                        continue
                
                # Select connection based on strategy
                if preferred_strategy == "health":
                    # Sort by health score
                    available_connections.sort(
                        key=lambda cid: self.connections[cid].metrics.health_score,
                        reverse=True
                    )
                    selected_id = available_connections[0]
                elif preferred_strategy == "random":
                    selected_id = random.choice(available_connections)
                else:  # round_robin
                    # Move selected to end of queue
                    selected_id = available_connections[0]
                    self.connection_queue.rotate(-1)
                
                connection = self.connections[selected_id]
                
                # Try to acquire the lock
                acquired = await connection.acquire()
                if acquired:
                    connection.metrics.state = ConnectionState.BUSY
                    return connection
                
            retry_count += 1
            await asyncio.sleep(0.05)  # Brief wait before retry
        
        self.logger.error("Failed to acquire connection after retries")
        return None
    
    def release_connection(self, connection: PooledConnection) -> None:
        """
        Release a connection back to the pool.
        
        Args:
            connection: Connection to release
        """
        connection.release()
        connection.metrics.state = ConnectionState.IDLE
        connection.metrics.update_health_score()
        
        # Trigger metric callback if set
        if self._metric_handler:
            self._metric_handler(connection.metrics)
    
    async def execute_with_connection(self, 
                                    operation: Callable,
                                    *args,
                                    **kwargs) -> Any:
        """
        Execute an operation using a pooled connection.
        
        Args:
            operation: Async function to execute
            *args, **kwargs: Arguments for the operation
            
        Returns:
            Result of the operation
        """
        connection = await self.acquire_connection()
        if not connection:
            raise RuntimeError("Failed to acquire connection from pool")
        
        try:
            start_time = time.perf_counter()
            
            # Execute operation
            result = await operation(connection.api_instance, *args, **kwargs)
            
            # Update metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            connection.metrics.requests_handled += 1
            connection.metrics.total_latency_ms += latency_ms
            
            return result
            
        except Exception as e:
            # Update error metrics
            connection.metrics.errors_count += 1
            connection.metrics.last_error = str(e)
            
            # Check circuit breaker
            total_errors = sum(
                conn.metrics.errors_count 
                for conn in self.connections.values()
            )
            
            if total_errors >= self.circuit_breaker_threshold:
                self._circuit_open = True
                self._circuit_opened_at = datetime.now()
                self.logger.error("Circuit breaker opened due to excessive errors")
            
            # Trigger error handler if set
            if self._error_handler:
                self._error_handler(e, connection.metrics)
            
            raise
            
        finally:
            self.release_connection(connection)
    
    async def _health_check_loop(self) -> None:
        """Background task to monitor connection health."""
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Check each connection
                for conn_id, connection in list(self.connections.items()):
                    try:
                        health = await connection.api_instance.health_check()
                        
                        if health['status'] != 'healthy':
                            connection.metrics.state = ConnectionState.ERROR
                            self.logger.warning(f"Connection {conn_id} unhealthy: {health}")
                            
                            # Try to reconnect
                            await connection.api_instance.disconnect()
                            connected = await connection.api_instance.connect()
                            
                            if connected:
                                connection.metrics.state = ConnectionState.IDLE
                                connection.metrics.errors_count = 0
                            else:
                                # Remove failed connection
                                async with self._lock:
                                    del self.connections[conn_id]
                                    
                    except Exception as e:
                        self.logger.error(f"Health check failed for {conn_id}: {e}")
                        connection.metrics.state = ConnectionState.ERROR
                
                # Ensure minimum connections
                if len(self.connections) < self.min_connections:
                    for i in range(self.min_connections - len(self.connections)):
                        await self._create_connection(f"conn_recovery_{int(time.time())}_{i}")
                        
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
    
    async def _connection_warmup_loop(self) -> None:
        """Background task to keep connections warm."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Warmup every minute
                
                # Send lightweight requests to idle connections
                for connection in self.connections.values():
                    if (connection.metrics.state == ConnectionState.IDLE and 
                        datetime.now() - connection.metrics.last_used > timedelta(minutes=2)):
                        
                        try:
                            # Simple health check to keep connection alive
                            await connection.api_instance.health_check()
                        except Exception as e:
                            self.logger.warning(f"Warmup failed for {connection.connection_id}: {e}")
                            
            except Exception as e:
                self.logger.error(f"Warmup loop error: {e}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get current pool statistics."""
        total_connections = len(self.connections)
        
        state_counts = {state: 0 for state in ConnectionState}
        for conn in self.connections.values():
            state_counts[conn.metrics.state] += 1
        
        # Calculate aggregate metrics
        total_requests = sum(c.metrics.requests_handled for c in self.connections.values())
        total_errors = sum(c.metrics.errors_count for c in self.connections.values())
        avg_health = sum(c.metrics.health_score for c in self.connections.values()) / total_connections if total_connections > 0 else 0
        
        return {
            'total_connections': total_connections,
            'state_distribution': {s.value: c for s, c in state_counts.items()},
            'total_requests': total_requests,
            'total_errors': total_errors,
            'average_health_score': avg_health,
            'circuit_breaker_open': self._circuit_open,
            'connection_metrics': [
                {
                    'id': conn.connection_id,
                    'state': conn.metrics.state.value,
                    'requests': conn.metrics.requests_handled,
                    'errors': conn.metrics.errors_count,
                    'avg_latency_ms': conn.metrics.avg_latency_ms,
                    'health_score': conn.metrics.health_score
                }
                for conn in self.connections.values()
            ]
        }
    
    def set_error_handler(self, handler: Callable) -> None:
        """Set callback for error handling."""
        self._error_handler = handler
    
    def set_metric_handler(self, handler: Callable) -> None:
        """Set callback for metric updates."""
        self._metric_handler = handler
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the connection pool."""
        self._running = False
        
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._warmup_task:
            self._warmup_task.cancel()
        
        # Disconnect all connections
        for connection in self.connections.values():
            try:
                await connection.api_instance.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting {connection.connection_id}: {e}")
        
        self.connections.clear()
        self.logger.info("Connection pool shutdown complete")
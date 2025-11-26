"""Connection Pool for managing multiple Alpaca WebSocket connections."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import random

from .alpaca_client import AlpacaWebSocketClient

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection states."""
    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass
class ConnectionInfo:
    """Information about a pooled connection."""
    client: AlpacaWebSocketClient
    state: ConnectionState = ConnectionState.IDLE
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    subscription_count: int = 0
    error_count: int = 0
    reconnect_count: int = 0
    
    @property
    def uptime(self) -> float:
        """Get connection uptime in seconds."""
        return time.time() - self.created_at
    
    @property
    def idle_time(self) -> float:
        """Get idle time in seconds."""
        return time.time() - self.last_used


class ConnectionPool:
    """Manages multiple WebSocket connections with health checks and failover.
    
    Features:
    - Multiple connection management
    - Health checks and heartbeat monitoring
    - Connection rotation for load balancing
    - Automatic failover handling
    - Connection lifecycle management
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        pool_size: int = 3,
        max_subscriptions_per_connection: int = 200,
        health_check_interval: float = 30.0,
        connection_timeout: float = 10.0,
        idle_timeout: float = 300.0,
        rotation_interval: float = 3600.0
    ):
        """Initialize connection pool.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            pool_size: Number of connections to maintain
            max_subscriptions_per_connection: Max subscriptions per connection
            health_check_interval: Interval for health checks
            connection_timeout: Timeout for connection attempts
            idle_timeout: Timeout for idle connections
            rotation_interval: Interval for connection rotation
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.pool_size = pool_size
        self.max_subscriptions_per_connection = max_subscriptions_per_connection
        self.health_check_interval = health_check_interval
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        self.rotation_interval = rotation_interval
        
        # Connection pool
        self.connections: Dict[str, ConnectionInfo] = {}
        self.connection_counter = 0
        
        # Symbol routing
        self.symbol_connections: Dict[str, str] = {}  # symbol -> connection_id
        
        # Pool state
        self.running = False
        self.maintenance_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.total_connections_created = 0
        self.total_failovers = 0
        self.total_rotations = 0
        
    async def start(self) -> None:
        """Start the connection pool."""
        if not self.running:
            self.running = True
            
            # Create initial connections
            await self._create_connections()
            
            # Start maintenance task
            self.maintenance_task = asyncio.create_task(
                self._maintenance_loop(),
                name="pool-maintenance"
            )
            
            logger.info(f"Started connection pool with {self.pool_size} connections")
    
    async def stop(self) -> None:
        """Stop the connection pool."""
        self.running = False
        
        # Cancel maintenance task
        if self.maintenance_task:
            self.maintenance_task.cancel()
        
        # Close all connections
        for conn_info in self.connections.values():
            try:
                await conn_info.client.disconnect()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        
        self.connections.clear()
        self.symbol_connections.clear()
        
        logger.info("Stopped connection pool")
    
    async def get_connection(self, symbols: Optional[List[str]] = None) -> AlpacaWebSocketClient:
        """Get a connection for the given symbols.
        
        Args:
            symbols: Optional list of symbols
            
        Returns:
            WebSocket client connection
        """
        # Find best connection
        conn_id = await self._find_best_connection(symbols)
        
        if conn_id and conn_id in self.connections:
            conn_info = self.connections[conn_id]
            conn_info.last_used = time.time()
            
            # Update symbol routing
            if symbols:
                for symbol in symbols:
                    self.symbol_connections[symbol] = conn_id
                conn_info.subscription_count += len(symbols)
            
            return conn_info.client
        
        # No suitable connection found, create new one if pool not full
        if len(self.connections) < self.pool_size:
            conn_id = await self._create_connection()
            if conn_id:
                conn_info = self.connections[conn_id]
                
                # Update symbol routing
                if symbols:
                    for symbol in symbols:
                        self.symbol_connections[symbol] = conn_id
                    conn_info.subscription_count += len(symbols)
                
                return conn_info.client
        
        raise Exception("No available connections in pool")
    
    async def release_symbols(self, symbols: List[str]) -> None:
        """Release symbols from their connections.
        
        Args:
            symbols: List of symbols to release
        """
        # Update connection counts
        connection_updates: Dict[str, int] = {}
        
        for symbol in symbols:
            if symbol in self.symbol_connections:
                conn_id = self.symbol_connections[symbol]
                connection_updates[conn_id] = connection_updates.get(conn_id, 0) + 1
                del self.symbol_connections[symbol]
        
        # Update subscription counts
        for conn_id, count in connection_updates.items():
            if conn_id in self.connections:
                self.connections[conn_id].subscription_count -= count
    
    async def _create_connections(self) -> None:
        """Create initial pool connections."""
        tasks = []
        for _ in range(self.pool_size):
            tasks.append(self._create_connection())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        created = sum(1 for r in results if r and not isinstance(r, Exception))
        logger.info(f"Created {created}/{self.pool_size} initial connections")
    
    async def _create_connection(self) -> Optional[str]:
        """Create a new connection.
        
        Returns:
            Connection ID if successful
        """
        try:
            # Generate connection ID
            self.connection_counter += 1
            conn_id = f"conn-{self.connection_counter}"
            
            # Create client
            client = AlpacaWebSocketClient(
                api_key=self.api_key,
                api_secret=self.api_secret,
                stream_type="data",
                feed="sip"
            )
            
            # Create connection info
            conn_info = ConnectionInfo(
                client=client,
                state=ConnectionState.CONNECTING
            )
            self.connections[conn_id] = conn_info
            
            # Connect with timeout
            await asyncio.wait_for(
                client.connect(),
                timeout=self.connection_timeout
            )
            
            conn_info.state = ConnectionState.AUTHENTICATED
            self.total_connections_created += 1
            
            logger.info(f"Created connection {conn_id}")
            return conn_id
            
        except asyncio.TimeoutError:
            logger.error(f"Connection timeout for {conn_id}")
            conn_info.state = ConnectionState.FAILED
            conn_info.error_count += 1
            
        except Exception as e:
            logger.error(f"Failed to create connection {conn_id}: {e}")
            if conn_id in self.connections:
                self.connections[conn_id].state = ConnectionState.FAILED
                self.connections[conn_id].error_count += 1
        
        return None
    
    async def _find_best_connection(self, symbols: Optional[List[str]] = None) -> Optional[str]:
        """Find the best connection for the given symbols.
        
        Args:
            symbols: Optional list of symbols
            
        Returns:
            Connection ID or None
        """
        # Get healthy connections
        healthy_connections = [
            (conn_id, conn_info)
            for conn_id, conn_info in self.connections.items()
            if conn_info.state == ConnectionState.AUTHENTICATED
            and conn_info.client.connected
            and conn_info.client.authenticated
        ]
        
        if not healthy_connections:
            return None
        
        # If symbols provided, check for existing connections
        if symbols:
            # Check if any symbols already have connections
            existing_connections = set()
            for symbol in symbols:
                if symbol in self.symbol_connections:
                    existing_connections.add(self.symbol_connections[symbol])
            
            # Prefer existing connections if they have capacity
            for conn_id in existing_connections:
                if conn_id in self.connections:
                    conn_info = self.connections[conn_id]
                    if (conn_info.subscription_count + len(symbols) <= 
                        self.max_subscriptions_per_connection):
                        return conn_id
        
        # Find connection with lowest load
        best_conn_id = None
        min_load = float('inf')
        
        for conn_id, conn_info in healthy_connections:
            # Calculate load factor
            load = conn_info.subscription_count / self.max_subscriptions_per_connection
            
            # Prefer connections with lower load
            if load < min_load:
                # Check if connection has capacity
                new_symbols_count = len(symbols) if symbols else 0
                if (conn_info.subscription_count + new_symbols_count <= 
                    self.max_subscriptions_per_connection):
                    min_load = load
                    best_conn_id = conn_id
        
        return best_conn_id
    
    async def _maintenance_loop(self) -> None:
        """Perform periodic maintenance on the connection pool."""
        while self.running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Perform health checks
                await self._health_check()
                
                # Handle idle connections
                await self._handle_idle_connections()
                
                # Rotate connections if needed
                await self._rotate_connections()
                
                # Ensure minimum pool size
                await self._ensure_pool_size()
                
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
    
    async def _health_check(self) -> None:
        """Check health of all connections."""
        for conn_id, conn_info in list(self.connections.items()):
            try:
                client = conn_info.client
                
                # Check connection state
                if not client.connected or not client.authenticated:
                    logger.warning(f"Connection {conn_id} is unhealthy")
                    
                    # Try to reconnect
                    if conn_info.error_count < 3:
                        conn_info.state = ConnectionState.CONNECTING
                        await asyncio.wait_for(
                            client.connect(),
                            timeout=self.connection_timeout
                        )
                        conn_info.state = ConnectionState.AUTHENTICATED
                        conn_info.reconnect_count += 1
                        logger.info(f"Reconnected {conn_id}")
                    else:
                        # Too many errors, remove connection
                        await self._remove_connection(conn_id)
                        
                else:
                    # Connection is healthy
                    conn_info.state = ConnectionState.AUTHENTICATED
                    
            except Exception as e:
                logger.error(f"Health check failed for {conn_id}: {e}")
                conn_info.error_count += 1
                conn_info.state = ConnectionState.FAILED
    
    async def _handle_idle_connections(self) -> None:
        """Handle connections that have been idle too long."""
        current_time = time.time()
        
        for conn_id, conn_info in list(self.connections.items()):
            # Check if connection is idle
            if (conn_info.idle_time > self.idle_timeout and
                conn_info.subscription_count == 0 and
                len(self.connections) > 1):  # Keep at least one connection
                
                logger.info(f"Removing idle connection {conn_id}")
                await self._remove_connection(conn_id)
    
    async def _rotate_connections(self) -> None:
        """Rotate old connections to prevent staleness."""
        current_time = time.time()
        
        for conn_id, conn_info in list(self.connections.items()):
            # Check if connection should be rotated
            if conn_info.uptime > self.rotation_interval:
                logger.info(f"Rotating connection {conn_id} (uptime: {conn_info.uptime:.1f}s)")
                
                # Create new connection first
                new_conn_id = await self._create_connection()
                
                if new_conn_id:
                    # Migrate subscriptions
                    await self._migrate_subscriptions(conn_id, new_conn_id)
                    
                    # Remove old connection
                    await self._remove_connection(conn_id)
                    
                    self.total_rotations += 1
    
    async def _ensure_pool_size(self) -> None:
        """Ensure pool maintains minimum size."""
        current_size = len(self.connections)
        
        if current_size < self.pool_size:
            # Create missing connections
            tasks = []
            for _ in range(self.pool_size - current_size):
                tasks.append(self._create_connection())
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _remove_connection(self, conn_id: str) -> None:
        """Remove a connection from the pool.
        
        Args:
            conn_id: Connection ID to remove
        """
        if conn_id not in self.connections:
            return
        
        conn_info = self.connections[conn_id]
        
        try:
            # Disconnect client
            await conn_info.client.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting {conn_id}: {e}")
        
        # Remove from pool
        del self.connections[conn_id]
        
        # Update symbol routing
        symbols_to_update = [
            symbol for symbol, cid in self.symbol_connections.items()
            if cid == conn_id
        ]
        
        for symbol in symbols_to_update:
            del self.symbol_connections[symbol]
        
        logger.info(f"Removed connection {conn_id}")
    
    async def _migrate_subscriptions(self, old_conn_id: str, new_conn_id: str) -> None:
        """Migrate subscriptions from old connection to new.
        
        Args:
            old_conn_id: Old connection ID
            new_conn_id: New connection ID
        """
        if old_conn_id not in self.connections or new_conn_id not in self.connections:
            return
        
        # Find symbols on old connection
        symbols_to_migrate = [
            symbol for symbol, conn_id in self.symbol_connections.items()
            if conn_id == old_conn_id
        ]
        
        if symbols_to_migrate:
            logger.info(f"Migrating {len(symbols_to_migrate)} symbols from {old_conn_id} to {new_conn_id}")
            
            # Update routing
            for symbol in symbols_to_migrate:
                self.symbol_connections[symbol] = new_conn_id
            
            # Update subscription counts
            self.connections[old_conn_id].subscription_count -= len(symbols_to_migrate)
            self.connections[new_conn_id].subscription_count += len(symbols_to_migrate)
            
            self.total_failovers += 1
    
    def get_pool_metrics(self) -> Dict[str, Any]:
        """Get connection pool metrics."""
        healthy_count = sum(
            1 for conn_info in self.connections.values()
            if conn_info.state == ConnectionState.AUTHENTICATED
        )
        
        total_subscriptions = sum(
            conn_info.subscription_count
            for conn_info in self.connections.values()
        )
        
        avg_uptime = (
            sum(conn_info.uptime for conn_info in self.connections.values()) /
            len(self.connections)
        ) if self.connections else 0
        
        return {
            "pool_size": len(self.connections),
            "healthy_connections": healthy_count,
            "total_subscriptions": total_subscriptions,
            "avg_uptime_seconds": round(avg_uptime, 2),
            "total_connections_created": self.total_connections_created,
            "total_failovers": self.total_failovers,
            "total_rotations": self.total_rotations,
            "connections": {
                conn_id: {
                    "state": conn_info.state.value,
                    "subscriptions": conn_info.subscription_count,
                    "uptime": round(conn_info.uptime, 2),
                    "idle_time": round(conn_info.idle_time, 2),
                    "errors": conn_info.error_count,
                    "reconnects": conn_info.reconnect_count
                }
                for conn_id, conn_info in self.connections.items()
            }
        }
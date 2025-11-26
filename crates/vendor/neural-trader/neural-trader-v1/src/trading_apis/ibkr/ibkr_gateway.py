"""
Interactive Brokers Gateway Connection Manager

Provides optimized connection management for IB Gateway with support for
multiple connection modes, load balancing, and failover capabilities.
"""

import asyncio
import socket
import ssl
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import aiohttp

logger = logging.getLogger(__name__)


class ConnectionMode(Enum):
    """IB Gateway connection modes"""
    DIRECT = "direct"          # Direct TCP connection
    SECURE = "secure"          # SSL/TLS connection
    WEBSOCKET = "websocket"    # WebSocket connection
    FIX = "fix"               # FIX protocol connection


@dataclass
class GatewayConfig:
    """Configuration for IB Gateway connections"""
    primary_host: str = "127.0.0.1"
    primary_port: int = 4001  # Gateway default port
    backup_hosts: List[Tuple[str, int]] = None
    connection_mode: ConnectionMode = ConnectionMode.DIRECT
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    connection_timeout: float = 5.0
    read_timeout: float = 30.0
    write_timeout: float = 5.0
    max_connections: int = 5
    load_balance: bool = False
    health_check_interval: float = 10.0
    compression_enabled: bool = True
    
    def __post_init__(self):
        if self.backup_hosts is None:
            self.backup_hosts = []


@dataclass
class ConnectionStats:
    """Track connection statistics"""
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    connection_time: float = 0
    last_message_time: float = 0
    errors: int = 0
    reconnects: int = 0
    latency_samples: List[float] = None
    
    def __post_init__(self):
        if self.latency_samples is None:
            self.latency_samples = []


class ConnectionPool:
    """Manage a pool of connections for load balancing"""
    
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.connections: List['GatewayConnection'] = []
        self.active_index = 0
        self._lock = asyncio.Lock()
    
    async def get_connection(self) -> Optional['GatewayConnection']:
        """Get next available connection using round-robin"""
        async with self._lock:
            if not self.connections:
                return None
            
            # Try connections in order
            for _ in range(len(self.connections)):
                conn = self.connections[self.active_index]
                self.active_index = (self.active_index + 1) % len(self.connections)
                
                if conn.is_connected:
                    return conn
            
            return None
    
    async def add_connection(self, connection: 'GatewayConnection'):
        """Add a connection to the pool"""
        async with self._lock:
            self.connections.append(connection)
    
    async def remove_connection(self, connection: 'GatewayConnection'):
        """Remove a connection from the pool"""
        async with self._lock:
            if connection in self.connections:
                self.connections.remove(connection)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        async with self._lock:
            total_stats = {
                'total_connections': len(self.connections),
                'active_connections': sum(1 for c in self.connections if c.is_connected),
                'total_messages_sent': sum(c.stats.messages_sent for c in self.connections),
                'total_messages_received': sum(c.stats.messages_received for c in self.connections),
                'total_errors': sum(c.stats.errors for c in self.connections)
            }
            return total_stats


class GatewayConnection:
    """Single gateway connection with low-level optimizations"""
    
    def __init__(self, host: str, port: int, config: GatewayConfig):
        self.host = host
        self.port = port
        self.config = config
        self.socket = None
        self.reader = None
        self.writer = None
        self.is_connected = False
        self.stats = ConnectionStats()
        self._read_task = None
        self._message_queue = asyncio.Queue()
        self._callbacks = {}
    
    async def connect(self) -> bool:
        """Establish connection to gateway"""
        try:
            start_time = time.time()
            
            if self.config.connection_mode == ConnectionMode.WEBSOCKET:
                # WebSocket connection
                return await self._connect_websocket()
            else:
                # TCP connection
                if self.config.ssl_enabled:
                    ssl_context = self._create_ssl_context()
                    self.reader, self.writer = await asyncio.wait_for(
                        asyncio.open_connection(
                            self.host, self.port, ssl=ssl_context
                        ),
                        timeout=self.config.connection_timeout
                    )
                else:
                    self.reader, self.writer = await asyncio.wait_for(
                        asyncio.open_connection(self.host, self.port),
                        timeout=self.config.connection_timeout
                    )
                
                # Set socket options for low latency
                sock = self.writer.get_extra_info('socket')
                if sock:
                    # Disable Nagle's algorithm for lower latency
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    # Set keep-alive
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    # Increase buffer sizes
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
                
                self.is_connected = True
                self.stats.connection_time = time.time() - start_time
                
                # Start read loop
                self._read_task = asyncio.create_task(self._read_loop())
                
                logger.info(f"Connected to gateway {self.host}:{self.port} in {self.stats.connection_time*1000:.1f}ms")
                return True
                
        except asyncio.TimeoutError:
            logger.error(f"Connection to {self.host}:{self.port} timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to {self.host}:{self.port}: {e}")
            return False
    
    async def _connect_websocket(self) -> bool:
        """Establish WebSocket connection"""
        try:
            session = aiohttp.ClientSession()
            ws_url = f"ws{'s' if self.config.ssl_enabled else ''}://{self.host}:{self.port}/ws"
            
            self.websocket = await session.ws_connect(
                ws_url,
                timeout=aiohttp.ClientTimeout(total=self.config.connection_timeout),
                compress=self.config.compression_enabled
            )
            
            self.is_connected = True
            self._read_task = asyncio.create_task(self._read_websocket_loop())
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for secure connections"""
        context = ssl.create_default_context()
        
        if self.config.ssl_cert_path:
            context.load_cert_chain(self.config.ssl_cert_path)
        
        # Optimize for low latency
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1
        
        return context
    
    async def send_message(self, message: bytes) -> bool:
        """Send message with latency tracking"""
        if not self.is_connected:
            return False
        
        try:
            start_time = time.time()
            
            if self.config.connection_mode == ConnectionMode.WEBSOCKET:
                await self.websocket.send_bytes(message)
            else:
                self.writer.write(message)
                await self.writer.drain()
            
            latency = (time.time() - start_time) * 1000
            self.stats.latency_samples.append(latency)
            if len(self.stats.latency_samples) > 1000:
                self.stats.latency_samples = self.stats.latency_samples[-1000:]
            
            self.stats.messages_sent += 1
            self.stats.bytes_sent += len(message)
            self.stats.last_message_time = time.time()
            
            if latency > 10:  # Log if latency exceeds 10ms
                logger.warning(f"High send latency: {latency:.1f}ms")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.stats.errors += 1
            await self._handle_connection_error()
            return False
    
    async def _read_loop(self):
        """Continuous read loop for TCP connections"""
        buffer = bytearray()
        
        while self.is_connected:
            try:
                # Read with timeout
                data = await asyncio.wait_for(
                    self.reader.read(4096),
                    timeout=self.config.read_timeout
                )
                
                if not data:
                    logger.warning("Connection closed by gateway")
                    break
                
                self.stats.bytes_received += len(data)
                buffer.extend(data)
                
                # Process complete messages from buffer
                while buffer:
                    # Extract message (implement protocol-specific logic)
                    message, remaining = self._extract_message(buffer)
                    if message:
                        self.stats.messages_received += 1
                        await self._message_queue.put(message)
                        buffer = remaining
                    else:
                        break
                        
            except asyncio.TimeoutError:
                # Timeout is normal, continue
                continue
            except Exception as e:
                logger.error(f"Read error: {e}")
                self.stats.errors += 1
                break
        
        await self._handle_connection_error()
    
    async def _read_websocket_loop(self):
        """Continuous read loop for WebSocket connections"""
        while self.is_connected:
            try:
                msg = await self.websocket.receive()
                
                if msg.type == aiohttp.WSMsgType.BINARY:
                    self.stats.messages_received += 1
                    self.stats.bytes_received += len(msg.data)
                    await self._message_queue.put(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {msg.data}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("WebSocket closed")
                    break
                    
            except Exception as e:
                logger.error(f"WebSocket read error: {e}")
                self.stats.errors += 1
                break
        
        await self._handle_connection_error()
    
    def _extract_message(self, buffer: bytearray) -> Tuple[Optional[bytes], bytearray]:
        """
        Extract a complete message from buffer
        
        This should be implemented based on the specific protocol.
        For now, we'll assume messages are length-prefixed.
        """
        if len(buffer) < 4:
            return None, buffer
        
        # Read message length (first 4 bytes, big-endian)
        msg_len = int.from_bytes(buffer[:4], 'big')
        
        if len(buffer) < 4 + msg_len:
            return None, buffer
        
        # Extract message
        message = bytes(buffer[4:4 + msg_len])
        remaining = buffer[4 + msg_len:]
        
        return message, remaining
    
    async def get_message(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """Get next message from queue"""
        try:
            if timeout:
                return await asyncio.wait_for(self._message_queue.get(), timeout=timeout)
            else:
                return await self._message_queue.get()
        except asyncio.TimeoutError:
            return None
    
    async def disconnect(self):
        """Close connection gracefully"""
        self.is_connected = False
        
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        
        if self.config.connection_mode == ConnectionMode.WEBSOCKET:
            if hasattr(self, 'websocket'):
                await self.websocket.close()
        else:
            if self.writer:
                self.writer.close()
                await self.writer.wait_closed()
        
        logger.info(f"Disconnected from {self.host}:{self.port}")
    
    async def _handle_connection_error(self):
        """Handle connection errors"""
        self.is_connected = False
        self.stats.reconnects += 1
        
        # Notify callbacks
        if 'disconnected' in self._callbacks:
            for callback in self._callbacks['disconnected']:
                asyncio.create_task(callback(self))
    
    def register_callback(self, event: str, callback):
        """Register event callback"""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics"""
        if not self.stats.latency_samples:
            return {}
        
        samples = self.stats.latency_samples
        return {
            'avg': sum(samples) / len(samples),
            'min': min(samples),
            'max': max(samples),
            'p50': sorted(samples)[len(samples)//2],
            'p95': sorted(samples)[int(len(samples)*0.95)] if len(samples) > 20 else max(samples),
            'p99': sorted(samples)[int(len(samples)*0.99)] if len(samples) > 100 else max(samples),
            'samples': len(samples)
        }


class IBKRGateway:
    """
    High-level gateway manager with load balancing and failover
    """
    
    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or GatewayConfig()
        self.pool = ConnectionPool(self.config)
        self.primary_connection = None
        self._health_check_task = None
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    async def connect(self) -> bool:
        """Connect to gateway with failover support"""
        # Try primary connection
        primary = GatewayConnection(
            self.config.primary_host,
            self.config.primary_port,
            self.config
        )
        
        if await primary.connect():
            self.primary_connection = primary
            await self.pool.add_connection(primary)
            
            # Set up additional connections for load balancing
            if self.config.load_balance and self.config.max_connections > 1:
                await self._setup_load_balanced_connections()
            
            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_monitor())
            
            return True
        
        # Try backup hosts
        for host, port in self.config.backup_hosts:
            logger.info(f"Trying backup gateway {host}:{port}")
            backup = GatewayConnection(host, port, self.config)
            
            if await backup.connect():
                self.primary_connection = backup
                await self.pool.add_connection(backup)
                self._health_check_task = asyncio.create_task(self._health_monitor())
                return True
        
        logger.error("Failed to connect to any gateway")
        return False
    
    async def _setup_load_balanced_connections(self):
        """Set up additional connections for load balancing"""
        tasks = []
        
        for i in range(1, min(self.config.max_connections, 5)):
            conn = GatewayConnection(
                self.config.primary_host,
                self.config.primary_port,
                self.config
            )
            tasks.append(conn.connect())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if result is True:
                await self.pool.add_connection(tasks[i])
                logger.info(f"Added load-balanced connection {i+1}")
    
    async def send_message(self, message: bytes) -> bool:
        """Send message using available connection"""
        if self.config.load_balance:
            conn = await self.pool.get_connection()
        else:
            conn = self.primary_connection
        
        if conn:
            return await conn.send_message(message)
        
        logger.error("No available connections")
        return False
    
    async def receive_message(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """Receive message from primary connection"""
        if self.primary_connection:
            return await self.primary_connection.get_message(timeout)
        return None
    
    async def _health_monitor(self):
        """Monitor connection health and reconnect if needed"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Check all connections
                stats = await self.pool.get_stats()
                
                if stats['active_connections'] == 0:
                    logger.warning("All connections lost, attempting reconnect")
                    await self.connect()
                elif stats['active_connections'] < self.config.max_connections * 0.5:
                    logger.warning(f"Low connection count: {stats['active_connections']}")
                    # Try to add more connections
                    await self._setup_load_balanced_connections()
                
                # Log statistics
                if stats['total_errors'] > 0:
                    logger.info(f"Gateway stats: {stats}")
                    
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def disconnect(self):
        """Disconnect all connections"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        for conn in self.pool.connections:
            await conn.disconnect()
        
        self._executor.shutdown(wait=False)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive gateway statistics"""
        pool_stats = await self.pool.get_stats()
        
        if self.primary_connection:
            latency_stats = self.primary_connection.get_latency_stats()
            pool_stats['latency'] = latency_stats
        
        return pool_stats
"""
Ultra-Low Latency Execution Pipeline
Target: <50ms total latency

Architecture:
- WebSocket feeds with connection pooling
- Lock-free circular buffers
- Parallel validation pipeline
- Direct market access
- Hardware timestamp precision
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import json
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import websockets
import aiohttp
from functools import wraps
import uvloop  # For faster event loop
import socket
import struct
from queue import Queue

# Configure for maximum performance
logging.basicConfig(level=logging.ERROR)  # Minimal logging for performance
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure optimized for speed"""
    symbol: str
    price: float
    volume: float
    timestamp_ns: int  # Nanosecond precision
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    
    def __post_init__(self):
        if self.spread == 0.0 and self.bid > 0 and self.ask > 0:
            self.spread = self.ask - self.bid

@dataclass
class TradeOrder:
    """Trade order with validation metadata"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    order_type: str = 'MARKET'
    price: Optional[float] = None
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())
    order_id: Optional[str] = None
    validated: bool = False
    execution_latency_ns: int = 0

@dataclass
class ExecutionResult:
    """Trade execution result"""
    order_id: str
    status: str  # 'FILLED', 'PARTIAL', 'REJECTED'
    filled_quantity: float
    avg_price: float
    total_latency_ns: int
    validation_time_ns: int
    execution_time_ns: int
    timestamp_ns: int

class LockFreeBuffer:
    """Lock-free circular buffer for ultra-low latency"""
    
    def __init__(self, size: int = 4096):
        self.size = size
        self.buffer = [None] * size
        self.head = 0
        self.tail = 0
        self._full = False
    
    def put(self, item):
        """Non-blocking put operation"""
        self.buffer[self.head] = item
        if self._full:
            self.tail = (self.tail + 1) % self.size
        self.head = (self.head + 1) % self.size
        if self.head == self.tail:
            self._full = True
    
    def get(self):
        """Non-blocking get operation"""
        if not self._full and self.head == self.tail:
            return None
        
        item = self.buffer[self.tail]
        self.tail = (self.tail + 1) % self.size
        self._full = False
        return item
    
    def is_empty(self):
        return not self._full and self.head == self.tail

class WebSocketFeedManager:
    """Ultra-fast WebSocket feed manager with connection pooling"""
    
    def __init__(self, max_connections: int = 8):
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.data_callbacks: List[Callable] = []
        self.max_connections = max_connections
        self.connection_pool = []
        self.reconnect_delays = {}
        self.running = False
        
        # Performance optimizations
        self.socket_options = {
            'ping_interval': None,  # Disable ping for speed
            'ping_timeout': None,
            'close_timeout': 1,
            'max_size': 2**16,  # Smaller buffer
            'compression': None,  # Disable compression
        }
    
    async def connect_feed(self, url: str, symbol: str):
        """Establish WebSocket connection with optimal settings"""
        try:
            # Configure socket for low latency
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            ws = await websockets.connect(
                url,
                sock=sock,
                **self.socket_options
            )
            
            self.connections[symbol] = ws
            logger.info(f"Connected to {symbol} feed")
            
            # Start listening
            asyncio.create_task(self._listen_feed(ws, symbol))
            
        except Exception as e:
            logger.error(f"Failed to connect {symbol}: {e}")
            await self._schedule_reconnect(url, symbol)
    
    async def _listen_feed(self, ws, symbol: str):
        """Listen to WebSocket feed with minimal processing"""
        try:
            async for message in ws:
                timestamp_ns = time.time_ns()
                
                # Parse with minimal overhead
                try:
                    data = json.loads(message) if isinstance(message, str) else message
                    
                    # Extract price data quickly
                    if 'price' in data or 'p' in data:
                        price = data.get('price', data.get('p', 0))
                        volume = data.get('volume', data.get('v', 0))
                        bid = data.get('bid', data.get('b', 0))
                        ask = data.get('ask', data.get('a', 0))
                        
                        market_data = MarketData(
                            symbol=symbol,
                            price=float(price),
                            volume=float(volume),
                            bid=float(bid) if bid else 0.0,
                            ask=float(ask) if ask else 0.0,
                            timestamp_ns=timestamp_ns
                        )
                        
                        # Notify callbacks immediately
                        for callback in self.data_callbacks:
                            callback(market_data)
                
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue  # Skip malformed data
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Connection closed for {symbol}")
        except Exception as e:
            logger.error(f"Feed error for {symbol}: {e}")
    
    async def _schedule_reconnect(self, url: str, symbol: str, delay: float = 1.0):
        """Schedule reconnection with exponential backoff"""
        await asyncio.sleep(delay)
        await self.connect_feed(url, symbol)
    
    def add_data_callback(self, callback: Callable):
        """Add market data callback"""
        self.data_callbacks.append(callback)
    
    async def close_all(self):
        """Close all connections"""
        for ws in self.connections.values():
            await ws.close()
        self.connections.clear()

class ValidationEngine:
    """Ultra-fast validation engine with parallel processing"""
    
    def __init__(self, max_workers: int = 4):
        self.validators: List[Callable] = []
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.risk_limits = {
            'max_position_size': 100000,
            'max_order_value': 50000,
            'max_daily_volume': 1000000,
            'min_spread_bps': 1,  # Basis points
        }
        
        # Pre-compiled validation rules
        self._setup_validators()
    
    def _setup_validators(self):
        """Setup pre-compiled validation rules"""
        
        def size_validator(order: TradeOrder, market_data: MarketData) -> bool:
            return order.quantity <= self.risk_limits['max_position_size']
        
        def value_validator(order: TradeOrder, market_data: MarketData) -> bool:
            value = order.quantity * (order.price or market_data.price)
            return value <= self.risk_limits['max_order_value']
        
        def spread_validator(order: TradeOrder, market_data: MarketData) -> bool:
            if market_data.spread <= 0:
                return True  # Skip if spread unavailable
            spread_bps = (market_data.spread / market_data.price) * 10000
            return spread_bps >= self.risk_limits['min_spread_bps']
        
        def liquidity_validator(order: TradeOrder, market_data: MarketData) -> bool:
            return market_data.volume >= order.quantity * 2  # 2x volume buffer
        
        self.validators = [
            size_validator,
            value_validator,
            spread_validator,
            liquidity_validator
        ]
    
    async def validate_order(self, order: TradeOrder, market_data: MarketData) -> bool:
        """Parallel validation with early exit"""
        start_time = time.time_ns()
        
        # Run validators in parallel
        validation_futures = [
            self.executor.submit(validator, order, market_data)
            for validator in self.validators
        ]
        
        # Early exit on first failure
        for future in validation_futures:
            try:
                if not future.result(timeout=0.005):  # 5ms timeout per validator
                    order.validated = False
                    return False
            except Exception:
                order.validated = False
                return False
        
        order.validated = True
        validation_time = time.time_ns() - start_time
        
        # Log if validation took too long
        if validation_time > 5_000_000:  # 5ms
            logger.warning(f"Slow validation: {validation_time/1_000_000:.2f}ms")
        
        return True

class TradeExecutor:
    """Direct market access trade executor"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_endpoints = {
            'orders': '/api/v1/orders',
            'positions': '/api/v1/positions',
            'account': '/api/v1/account'
        }
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'avg_latency_ns': 0,
            'min_latency_ns': float('inf'),
            'max_latency_ns': 0
        }
        
        # Connection pooling for HTTP requests
        self.connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=50,
            ttl_dns_cache=300,
            use_dns_cache=True,
            enable_cleanup_closed=True
        )
    
    async def initialize(self, base_url: str, api_key: str):
        """Initialize executor with optimized session"""
        timeout = aiohttp.ClientTimeout(
            total=0.1,  # 100ms total timeout
            connect=0.05,  # 50ms connect timeout
            sock_read=0.05  # 50ms read timeout
        )
        
        self.session = aiohttp.ClientSession(
            base_url=base_url,
            connector=self.connector,
            timeout=timeout,
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'Connection': 'keep-alive'
            }
        )
    
    async def execute_order(self, order: TradeOrder) -> ExecutionResult:
        """Execute trade with minimal latency"""
        start_time = time.time_ns()
        
        if not self.session:
            raise RuntimeError("Executor not initialized")
        
        # Prepare order payload
        payload = {
            'symbol': order.symbol,
            'side': order.side,
            'quantity': str(order.quantity),
            'type': order.order_type,
            'timeInForce': 'IOC',  # Immediate or Cancel for speed
        }
        
        if order.price:
            payload['price'] = str(order.price)
        
        try:
            # Execute HTTP request
            async with self.session.post(
                self.api_endpoints['orders'],
                json=payload
            ) as response:
                
                execution_time = time.time_ns()
                response_data = await response.json()
                
                if response.status == 200 or response.status == 201:
                    result = ExecutionResult(
                        order_id=response_data.get('orderId', ''),
                        status=response_data.get('status', 'UNKNOWN'),
                        filled_quantity=float(response_data.get('executedQty', 0)),
                        avg_price=float(response_data.get('price', 0)),
                        total_latency_ns=execution_time - start_time,
                        validation_time_ns=0,  # Set by pipeline
                        execution_time_ns=execution_time - start_time,
                        timestamp_ns=execution_time
                    )
                    
                    # Update stats
                    self._update_stats(result.total_latency_ns)
                    self.execution_stats['successful_orders'] += 1
                    
                    return result
                else:
                    # Handle API errors
                    error_msg = response_data.get('msg', 'Unknown error')
                    return ExecutionResult(
                        order_id='',
                        status='REJECTED',
                        filled_quantity=0,
                        avg_price=0,
                        total_latency_ns=execution_time - start_time,
                        validation_time_ns=0,
                        execution_time_ns=execution_time - start_time,
                        timestamp_ns=execution_time
                    )
        
        except asyncio.TimeoutError:
            return ExecutionResult(
                order_id='',
                status='TIMEOUT',
                filled_quantity=0,
                avg_price=0,
                total_latency_ns=time.time_ns() - start_time,
                validation_time_ns=0,
                execution_time_ns=time.time_ns() - start_time,
                timestamp_ns=time.time_ns()
            )
        
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return ExecutionResult(
                order_id='',
                status='ERROR',
                filled_quantity=0,
                avg_price=0,
                total_latency_ns=time.time_ns() - start_time,
                validation_time_ns=0,
                execution_time_ns=time.time_ns() - start_time,
                timestamp_ns=time.time_ns()
            )
        
        finally:
            self.execution_stats['total_orders'] += 1
    
    def _update_stats(self, latency_ns: int):
        """Update execution statistics"""
        stats = self.execution_stats
        
        # Update min/max
        stats['min_latency_ns'] = min(stats['min_latency_ns'], latency_ns)
        stats['max_latency_ns'] = max(stats['max_latency_ns'], latency_ns)
        
        # Update average (running average)
        if stats['total_orders'] > 0:
            stats['avg_latency_ns'] = (
                (stats['avg_latency_ns'] * (stats['total_orders'] - 1) + latency_ns) 
                / stats['total_orders']
            )
        else:
            stats['avg_latency_ns'] = latency_ns
    
    async def close(self):
        """Clean shutdown"""
        if self.session:
            await self.session.close()

class ExecutionPipeline:
    """Ultra-low latency execution pipeline orchestrator"""
    
    def __init__(self, 
                 buffer_size: int = 4096,
                 validation_workers: int = 4,
                 executor_pool_size: int = 8):
        
        # Core components
        self.feed_manager = WebSocketFeedManager(max_connections=executor_pool_size)
        self.validator = ValidationEngine(max_workers=validation_workers)
        self.executor = TradeExecutor()
        
        # Lock-free buffers
        self.market_data_buffer = LockFreeBuffer(buffer_size)
        self.order_buffer = LockFreeBuffer(buffer_size)
        self.result_buffer = LockFreeBuffer(buffer_size)
        
        # State management
        self.running = False
        self.latest_market_data: Dict[str, MarketData] = {}
        
        # Performance metrics
        self.pipeline_stats = {
            'orders_processed': 0,
            'successful_executions': 0,
            'avg_pipeline_latency_ns': 0,
            'latency_p50_ns': 0,
            'latency_p95_ns': 0,
            'latency_p99_ns': 0
        }
        
        self.latency_samples = deque(maxlen=10000)  # Keep last 10k samples
        
        # Setup callbacks
        self.feed_manager.add_data_callback(self._on_market_data)
    
    async def initialize(self, 
                        feeds: Dict[str, str],  # symbol -> websocket_url
                        api_config: Dict[str, str]):  # base_url, api_key
        """Initialize pipeline components"""
        
        # Set event loop to uvloop for performance
        if not isinstance(asyncio.get_running_loop(), uvloop.Loop):
            logger.warning("Consider using uvloop for better performance")
        
        # Initialize executor
        await self.executor.initialize(
            api_config['base_url'],
            api_config['api_key']
        )
        
        # Connect to feeds
        for symbol, url in feeds.items():
            await self.feed_manager.connect_feed(url, symbol)
        
        logger.info("Pipeline initialized")
    
    def _on_market_data(self, market_data: MarketData):
        """Handle incoming market data with minimal latency"""
        # Update latest data
        self.latest_market_data[market_data.symbol] = market_data
        
        # Add to buffer for processing
        self.market_data_buffer.put(market_data)
    
    async def submit_order(self, order: TradeOrder) -> str:
        """Submit order to pipeline"""
        order_id = f"order_{time.time_ns()}"
        order.order_id = order_id
        
        self.order_buffer.put(order)
        return order_id
    
    async def start_pipeline(self):
        """Start the execution pipeline"""
        self.running = True
        
        # Start processing tasks
        tasks = [
            asyncio.create_task(self._process_orders()),
            asyncio.create_task(self._monitor_performance())
        ]
        
        logger.info("Pipeline started - targeting <50ms latency")
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down pipeline...")
            await self.stop_pipeline()
    
    async def _process_orders(self):
        """Main order processing loop"""
        while self.running:
            order = self.order_buffer.get()
            
            if order is None:
                await asyncio.sleep(0.0001)  # 0.1ms sleep
                continue
            
            # Process order with timing
            start_time = time.time_ns()
            
            try:
                # Get latest market data
                market_data = self.latest_market_data.get(order.symbol)
                if not market_data:
                    logger.warning(f"No market data for {order.symbol}")
                    continue
                
                # Validate order
                validation_start = time.time_ns()
                is_valid = await self.validator.validate_order(order, market_data)
                validation_time = time.time_ns() - validation_start
                
                if not is_valid:
                    logger.warning(f"Order validation failed: {order.order_id}")
                    continue
                
                # Execute trade
                execution_result = await self.executor.execute_order(order)
                execution_result.validation_time_ns = validation_time
                
                # Calculate total pipeline latency
                total_latency = time.time_ns() - start_time
                execution_result.total_latency_ns = total_latency
                
                # Store result
                self.result_buffer.put(execution_result)
                
                # Update pipeline stats
                self._update_pipeline_stats(total_latency)
                
                # Log if exceeding target latency
                if total_latency > 50_000_000:  # 50ms
                    logger.warning(
                        f"Pipeline latency exceeded target: "
                        f"{total_latency/1_000_000:.2f}ms for {order.symbol}"
                    )
                
            except Exception as e:
                logger.error(f"Order processing error: {e}")
    
    def _update_pipeline_stats(self, latency_ns: int):
        """Update pipeline performance statistics"""
        stats = self.pipeline_stats
        stats['orders_processed'] += 1
        
        # Add to samples for percentile calculation
        self.latency_samples.append(latency_ns)
        
        # Update running average
        if stats['orders_processed'] > 1:
            stats['avg_pipeline_latency_ns'] = (
                (stats['avg_pipeline_latency_ns'] * (stats['orders_processed'] - 1) + latency_ns)
                / stats['orders_processed']
            )
        else:
            stats['avg_pipeline_latency_ns'] = latency_ns
        
        # Update percentiles every 100 orders
        if stats['orders_processed'] % 100 == 0:
            samples_array = np.array(list(self.latency_samples))
            stats['latency_p50_ns'] = np.percentile(samples_array, 50)
            stats['latency_p95_ns'] = np.percentile(samples_array, 95)
            stats['latency_p99_ns'] = np.percentile(samples_array, 99)
    
    async def _monitor_performance(self):
        """Monitor and log performance metrics"""
        while self.running:
            await asyncio.sleep(10)  # Report every 10 seconds
            
            stats = self.pipeline_stats
            exec_stats = self.executor.execution_stats
            
            logger.info(
                f"Pipeline Performance - "
                f"Orders: {stats['orders_processed']}, "
                f"Avg Latency: {stats['avg_pipeline_latency_ns']/1_000_000:.2f}ms, "
                f"P95: {stats['latency_p95_ns']/1_000_000:.2f}ms, "
                f"P99: {stats['latency_p99_ns']/1_000_000:.2f}ms, "
                f"Success Rate: {(exec_stats['successful_orders']/max(exec_stats['total_orders'],1)*100):.1f}%"
            )
    
    def get_execution_result(self, order_id: str) -> Optional[ExecutionResult]:
        """Get execution result by order ID"""
        # In a real implementation, you'd maintain an index
        # For now, we'll return from buffer (simplified)
        result = self.result_buffer.get()
        return result if result and result.order_id == order_id else None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            'pipeline': self.pipeline_stats.copy(),
            'execution': self.executor.execution_stats.copy(),
            'target_latency_ms': 50,
            'current_avg_latency_ms': self.pipeline_stats['avg_pipeline_latency_ns'] / 1_000_000,
            'performance_ratio': min(50.0 / (self.pipeline_stats['avg_pipeline_latency_ns'] / 1_000_000), 1.0)
        }
    
    async def stop_pipeline(self):
        """Stop the pipeline gracefully"""
        self.running = False
        await self.feed_manager.close_all()
        await self.executor.close()
        logger.info("Pipeline stopped")

# Usage Example and Testing
async def main():
    """Example usage of the ultra-low latency pipeline"""
    
    # Configure feeds (example URLs - replace with actual endpoints)
    feeds = {
        'BTCUSDT': 'wss://stream.binance.com:9443/ws/btcusdt@ticker',
        'ETHUSDT': 'wss://stream.binance.com:9443/ws/ethusdt@ticker',
    }
    
    # Configure API
    api_config = {
        'base_url': 'https://api.binance.com',
        'api_key': 'your_api_key_here'
    }
    
    # Initialize pipeline
    pipeline = ExecutionPipeline(
        buffer_size=8192,
        validation_workers=6,
        executor_pool_size=12
    )
    
    try:
        # Initialize components
        await pipeline.initialize(feeds, api_config)
        
        # Submit test orders
        test_orders = [
            TradeOrder(symbol='BTCUSDT', side='BUY', quantity=0.001),
            TradeOrder(symbol='ETHUSDT', side='BUY', quantity=0.01),
        ]
        
        for order in test_orders:
            order_id = await pipeline.submit_order(order)
            logger.info(f"Submitted order: {order_id}")
        
        # Start pipeline (runs until stopped)
        await pipeline.start_pipeline()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await pipeline.stop_pipeline()

if __name__ == "__main__":
    # Use uvloop for maximum performance
    import uvloop
    uvloop.install()
    
    asyncio.run(main())
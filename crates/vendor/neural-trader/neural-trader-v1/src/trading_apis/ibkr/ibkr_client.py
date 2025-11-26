"""
Interactive Brokers TWS API Client

High-performance async wrapper for the IB TWS API with connection resilience,
automatic reconnection, and optimized for low latency trading.
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import logging

# Note: In production, install with: pip install ib_insync
try:
    from ib_insync import IB, Stock, Option, Future, Forex, Order, MarketOrder, LimitOrder, StopOrder
    from ib_insync import util
except ImportError:
    # Mock for development
    IB = Stock = Option = Future = Forex = Order = MarketOrder = LimitOrder = StopOrder = object
    util = None

logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """Configuration for IB connection"""
    host: str = "127.0.0.1"
    port: int = 7497  # TWS paper trading port (7496 for live)
    client_id: int = 1
    account: str = ""
    timeout: float = 10.0
    readonly: bool = False
    auto_reconnect: bool = True
    reconnect_interval: float = 5.0
    max_reconnect_attempts: int = 10


@dataclass
class LatencyMetrics:
    """Track latency metrics for performance monitoring"""
    order_submission: List[float]
    order_fill: List[float]
    data_reception: List[float]
    heartbeat: List[float]
    
    def __init__(self):
        self.order_submission = []
        self.order_fill = []
        self.data_reception = []
        self.heartbeat = []
    
    def add_metric(self, metric_type: str, latency_ms: float):
        """Add a latency measurement"""
        if hasattr(self, metric_type):
            getattr(self, metric_type).append(latency_ms)
            # Keep only last 1000 measurements
            if len(getattr(self, metric_type)) > 1000:
                setattr(self, metric_type, getattr(self, metric_type)[-1000:])
    
    def get_stats(self, metric_type: str) -> Dict[str, float]:
        """Get statistics for a metric type"""
        if not hasattr(self, metric_type):
            return {}
        
        values = getattr(self, metric_type)
        if not values:
            return {}
        
        return {
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'p50': sorted(values)[len(values)//2],
            'p95': sorted(values)[int(len(values)*0.95)] if len(values) > 20 else max(values),
            'p99': sorted(values)[int(len(values)*0.99)] if len(values) > 100 else max(values)
        }


class IBKRClient:
    """
    High-performance Interactive Brokers client with async support
    
    Features:
    - Async/await interface for all operations
    - Automatic reconnection on connection loss
    - Sub-100ms latency for order submission
    - Real-time performance metrics
    - Thread-safe operation
    """
    
    def __init__(self, config: Optional[ConnectionConfig] = None):
        self.config = config or ConnectionConfig()
        self.ib = None
        self._connected = False
        self._reconnecting = False
        self._callbacks = defaultdict(list)
        self._pending_orders = {}
        self._positions = {}
        self._account_values = {}
        self.metrics = LatencyMetrics()
        self._event_loop = None
        self._connection_lock = asyncio.Lock()
        self._last_heartbeat = time.time()
        
    async def connect(self) -> bool:
        """
        Connect to IB TWS/Gateway with automatic retry
        
        Returns:
            bool: True if connected successfully
        """
        async with self._connection_lock:
            if self._connected:
                return True
            
            attempt = 0
            while attempt < self.config.max_reconnect_attempts:
                try:
                    start_time = time.time()
                    
                    if not self.ib:
                        self.ib = IB()
                    
                    # Connect with timeout
                    await asyncio.wait_for(
                        self._do_connect(),
                        timeout=self.config.timeout
                    )
                    
                    # Set up event handlers
                    self._setup_event_handlers()
                    
                    # Start heartbeat monitoring
                    asyncio.create_task(self._heartbeat_monitor())
                    
                    self._connected = True
                    connect_time = (time.time() - start_time) * 1000
                    logger.info(f"Connected to IB in {connect_time:.1f}ms")
                    
                    # Request initial data
                    await self._initialize_data()
                    
                    return True
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Connection attempt {attempt + 1} timed out")
                except Exception as e:
                    logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                
                attempt += 1
                if attempt < self.config.max_reconnect_attempts:
                    await asyncio.sleep(self.config.reconnect_interval)
            
            return False
    
    async def _do_connect(self):
        """Perform the actual connection"""
        if util:
            await self.ib.connectAsync(
                self.config.host,
                self.config.port,
                clientId=self.config.client_id,
                readonly=self.config.readonly
            )
        else:
            # Mock connection for development
            await asyncio.sleep(0.1)
    
    def _setup_event_handlers(self):
        """Set up IB event handlers"""
        if not self.ib or not util:
            return
        
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_execution
        self.ib.errorEvent += self._on_error
        self.ib.positionEvent += self._on_position
        self.ib.accountValueEvent += self._on_account_value
        self.ib.disconnectedEvent += self._on_disconnected
    
    async def _initialize_data(self):
        """Initialize account data after connection"""
        if not self.ib or not util:
            return
        
        try:
            # Request account summary
            self.ib.reqAccountSummary()
            
            # Request positions
            self.ib.reqPositions()
            
            # Wait for initial data
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Failed to initialize data: {e}")
    
    async def disconnect(self):
        """Disconnect from IB"""
        async with self._connection_lock:
            if self.ib and util:
                self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")
    
    async def place_order(self, 
                         symbol: str,
                         quantity: int,
                         order_type: str = "MKT",
                         side: str = "BUY",
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         **kwargs) -> Optional[str]:
        """
        Place an order with low latency
        
        Args:
            symbol: Stock symbol
            quantity: Order quantity
            order_type: MKT, LMT, STP, etc.
            side: BUY or SELL
            price: Limit price (for LMT orders)
            stop_price: Stop price (for STP orders)
            **kwargs: Additional order parameters
            
        Returns:
            Order ID if successful, None otherwise
        """
        if not self._connected:
            logger.error("Not connected to IB")
            return None
        
        start_time = time.time()
        
        try:
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Create order based on type
            if order_type == "MKT":
                order = MarketOrder(side, quantity)
            elif order_type == "LMT":
                order = LimitOrder(side, quantity, price or 0)
            elif order_type == "STP":
                order = StopOrder(side, quantity, stop_price or 0)
            else:
                logger.error(f"Unsupported order type: {order_type}")
                return None
            
            # Apply additional parameters
            for key, value in kwargs.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            
            # Place order asynchronously
            if self.ib and util:
                trade = self.ib.placeOrder(contract, order)
                order_id = trade.order.orderId
                self._pending_orders[order_id] = {
                    'trade': trade,
                    'submit_time': time.time(),
                    'symbol': symbol,
                    'quantity': quantity,
                    'side': side
                }
            else:
                # Mock for development
                order_id = f"DEV{int(time.time()*1000)}"
                self._pending_orders[order_id] = {
                    'submit_time': time.time(),
                    'symbol': symbol,
                    'quantity': quantity,
                    'side': side
                }
            
            # Track submission latency
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.add_metric('order_submission', latency_ms)
            
            if latency_ms > 100:
                logger.warning(f"Order submission latency: {latency_ms:.1f}ms (> 100ms target)")
            else:
                logger.info(f"Order placed in {latency_ms:.1f}ms - ID: {order_id}")
            
            return str(order_id)
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancel request sent successfully
        """
        if not self._connected:
            return False
        
        try:
            if order_id in self._pending_orders:
                if self.ib and util:
                    trade = self._pending_orders[order_id].get('trade')
                    if trade:
                        self.ib.cancelOrder(trade.order)
                logger.info(f"Cancel request sent for order {order_id}")
                return True
            else:
                logger.warning(f"Order {order_id} not found in pending orders")
                return False
                
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions"""
        return self._positions.copy()
    
    async def get_account_values(self) -> Dict[str, Any]:
        """Get account values"""
        return self._account_values.copy()
    
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific order"""
        if order_id in self._pending_orders:
            return self._pending_orders[order_id].copy()
        return None
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for specific events"""
        self._callbacks[event].append(callback)
    
    async def _heartbeat_monitor(self):
        """Monitor connection health"""
        while self._connected:
            try:
                # Send heartbeat
                if self.ib and util:
                    start_time = time.time()
                    self.ib.reqCurrentTime()
                    heartbeat_ms = (time.time() - start_time) * 1000
                    self.metrics.add_metric('heartbeat', heartbeat_ms)
                
                self._last_heartbeat = time.time()
                
                # Check if we need to reconnect
                if (time.time() - self._last_heartbeat > 30 and 
                    self.config.auto_reconnect and 
                    not self._reconnecting):
                    logger.warning("Heartbeat timeout, reconnecting...")
                    asyncio.create_task(self._auto_reconnect())
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def _auto_reconnect(self):
        """Automatically reconnect on connection loss"""
        if self._reconnecting:
            return
        
        self._reconnecting = True
        self._connected = False
        
        try:
            logger.info("Starting auto-reconnection...")
            if await self.connect():
                logger.info("Auto-reconnection successful")
                # Notify callbacks
                for callback in self._callbacks.get('reconnected', []):
                    asyncio.create_task(callback())
            else:
                logger.error("Auto-reconnection failed")
                
        finally:
            self._reconnecting = False
    
    # Event handlers
    def _on_order_status(self, trade):
        """Handle order status updates"""
        if not util:
            return
            
        try:
            order_id = trade.order.orderId
            if order_id in self._pending_orders:
                # Calculate fill latency if filled
                if trade.orderStatus.status == 'Filled':
                    submit_time = self._pending_orders[order_id]['submit_time']
                    fill_latency_ms = (time.time() - submit_time) * 1000
                    self.metrics.add_metric('order_fill', fill_latency_ms)
                    logger.info(f"Order {order_id} filled in {fill_latency_ms:.1f}ms")
                
                # Update order info
                self._pending_orders[order_id].update({
                    'status': trade.orderStatus.status,
                    'filled': trade.orderStatus.filled,
                    'remaining': trade.orderStatus.remaining,
                    'avgFillPrice': trade.orderStatus.avgFillPrice,
                    'lastUpdate': time.time()
                })
                
                # Notify callbacks
                for callback in self._callbacks.get('order_status', []):
                    asyncio.create_task(callback(order_id, trade.orderStatus))
                    
        except Exception as e:
            logger.error(f"Error handling order status: {e}")
    
    def _on_execution(self, trade, fill):
        """Handle execution details"""
        try:
            # Notify callbacks
            for callback in self._callbacks.get('execution', []):
                asyncio.create_task(callback(trade, fill))
        except Exception as e:
            logger.error(f"Error handling execution: {e}")
    
    def _on_position(self, position):
        """Handle position updates"""
        try:
            key = f"{position.contract.symbol}_{position.contract.secType}"
            self._positions[key] = {
                'symbol': position.contract.symbol,
                'position': position.position,
                'avgCost': position.avgCost,
                'marketValue': position.position * position.marketPrice,
                'marketPrice': position.marketPrice,
                'unrealizedPNL': position.unrealizedPNL,
                'realizedPNL': position.realizedPNL
            }
            
            # Notify callbacks
            for callback in self._callbacks.get('position', []):
                asyncio.create_task(callback(position))
                
        except Exception as e:
            logger.error(f"Error handling position: {e}")
    
    def _on_account_value(self, value):
        """Handle account value updates"""
        try:
            self._account_values[value.tag] = {
                'value': value.value,
                'currency': value.currency,
                'account': value.account
            }
            
            # Notify callbacks
            for callback in self._callbacks.get('account_value', []):
                asyncio.create_task(callback(value))
                
        except Exception as e:
            logger.error(f"Error handling account value: {e}")
    
    def _on_error(self, reqId, errorCode, errorString, contract):
        """Handle errors"""
        logger.error(f"IB Error - ReqId: {reqId}, Code: {errorCode}, Message: {errorString}")
        
        # Notify callbacks
        for callback in self._callbacks.get('error', []):
            asyncio.create_task(callback(reqId, errorCode, errorString, contract))
    
    def _on_disconnected(self):
        """Handle disconnection"""
        logger.warning("Disconnected from IB")
        self._connected = False
        
        if self.config.auto_reconnect and not self._reconnecting:
            asyncio.create_task(self._auto_reconnect())
    
    def get_latency_report(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive latency statistics"""
        report = {}
        for metric_type in ['order_submission', 'order_fill', 'data_reception', 'heartbeat']:
            stats = self.metrics.get_stats(metric_type)
            if stats:
                report[metric_type] = stats
        return report
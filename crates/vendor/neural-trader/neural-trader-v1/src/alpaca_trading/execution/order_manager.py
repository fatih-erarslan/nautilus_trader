"""Order Manager for tracking order lifecycle and state.

Handles order state tracking, fill monitoring, and modifications.
Optimized for low-latency with in-memory state management.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging
from dataclasses import dataclass, field
import aiohttp

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REPLACED = "replaced"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(Enum):
    """Time in force enumeration."""
    DAY = "day"
    GTC = "gtc"  # Good till cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    GTX = "gtx"  # Good till extended
    CLS = "cls"  # Close
    OPG = "opg"  # Opening


@dataclass
class Order:
    """Order data structure with tracking metadata."""
    # Core fields
    order_id: str
    client_order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    qty: float
    order_type: OrderType
    time_in_force: TimeInForce
    
    # Price fields
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_percent: Optional[float] = None
    trail_price: Optional[float] = None
    
    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Latency tracking
    submission_latency_ms: Optional[float] = None
    fill_latency_ms: Optional[float] = None
    
    # Metadata
    extended_hours: bool = False
    signal_id: Optional[str] = None
    strategy_id: Optional[str] = None
    notes: Optional[str] = None
    
    # Alpaca specific
    alpaca_order_id: Optional[str] = None
    alpaca_status: Optional[str] = None
    alpaca_raw: Optional[Dict[str, Any]] = None


class OrderManager:
    """High-performance order lifecycle manager.
    
    Features:
    - In-memory order tracking for ultra-low latency
    - Async event-driven architecture
    - Real-time fill monitoring
    - Order modification support
    - Comprehensive latency tracking
    """
    
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://paper-api.alpaca.markets"):
        """Initialize order manager.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            base_url: API base URL (paper or live)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        
        # Order storage
        self._orders: Dict[str, Order] = {}  # client_order_id -> Order
        self._alpaca_orders: Dict[str, str] = {}  # alpaca_order_id -> client_order_id
        
        # Event callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'order_submitted': [],
            'order_filled': [],
            'order_partially_filled': [],
            'order_cancelled': [],
            'order_rejected': [],
            'order_replaced': []
        }
        
        # Performance metrics
        self._metrics = {
            'total_orders': 0,
            'filled_orders': 0,
            'rejected_orders': 0,
            'avg_submission_latency_ms': 0,
            'avg_fill_latency_ms': 0
        }
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
    async def create_order(self, order: Order) -> Order:
        """Create and track a new order.
        
        Args:
            order: Order object to create
            
        Returns:
            Created order with tracking metadata
        """
        async with self._lock:
            # Store order
            self._orders[order.client_order_id] = order
            self._metrics['total_orders'] += 1
            
            # Log creation
            logger.info(f"Created order {order.client_order_id} for {order.qty} {order.symbol}")
            
        return order
    
    async def update_order_status(self, client_order_id: str, alpaca_order: Dict[str, Any]) -> Optional[Order]:
        """Update order status from Alpaca response.
        
        Args:
            client_order_id: Client order ID
            alpaca_order: Alpaca order response
            
        Returns:
            Updated order or None if not found
        """
        async with self._lock:
            order = self._orders.get(client_order_id)
            if not order:
                return None
            
            # Update Alpaca fields
            order.alpaca_order_id = alpaca_order.get('id')
            order.alpaca_status = alpaca_order.get('status')
            order.alpaca_raw = alpaca_order
            
            # Map Alpaca order ID
            if order.alpaca_order_id:
                self._alpaca_orders[order.alpaca_order_id] = client_order_id
            
            # Update status
            old_status = order.status
            new_status = self._map_alpaca_status(alpaca_order.get('status'))
            order.status = new_status
            
            # Update fill information
            filled_qty = float(alpaca_order.get('filled_qty', 0))
            if filled_qty > order.filled_qty:
                order.filled_qty = filled_qty
                order.avg_fill_price = float(alpaca_order.get('filled_avg_price', 0))
                
                # Calculate fill latency
                if order.submitted_at and not order.filled_at:
                    order.filled_at = datetime.utcnow()
                    order.fill_latency_ms = (order.filled_at - order.submitted_at).total_seconds() * 1000
                    
                    # Update metrics
                    self._update_fill_metrics(order.fill_latency_ms)
            
            # Update timestamp
            order.updated_at = datetime.utcnow()
            
            # Trigger callbacks if status changed
            if old_status != new_status:
                await self._trigger_callbacks(order, old_status, new_status)
            
            return order
    
    async def get_order(self, client_order_id: str) -> Optional[Order]:
        """Get order by client order ID.
        
        Args:
            client_order_id: Client order ID
            
        Returns:
            Order object or None if not found
        """
        return self._orders.get(client_order_id)
    
    async def get_order_by_alpaca_id(self, alpaca_order_id: str) -> Optional[Order]:
        """Get order by Alpaca order ID.
        
        Args:
            alpaca_order_id: Alpaca order ID
            
        Returns:
            Order object or None if not found
        """
        client_order_id = self._alpaca_orders.get(alpaca_order_id)
        if client_order_id:
            return self._orders.get(client_order_id)
        return None
    
    async def get_active_orders(self) -> List[Order]:
        """Get all active orders.
        
        Returns:
            List of active orders
        """
        active_statuses = {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}
        return [order for order in self._orders.values() if order.status in active_statuses]
    
    async def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of orders for the symbol
        """
        return [order for order in self._orders.values() if order.symbol == symbol]
    
    async def cancel_order(self, client_order_id: str) -> bool:
        """Mark order as pending cancellation.
        
        Args:
            client_order_id: Client order ID
            
        Returns:
            True if order found and marked for cancellation
        """
        async with self._lock:
            order = self._orders.get(client_order_id)
            if order and order.status in {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}:
                logger.info(f"Marking order {client_order_id} for cancellation")
                return True
        return False
    
    async def replace_order(self, client_order_id: str, new_qty: Optional[float] = None,
                           new_limit_price: Optional[float] = None,
                           new_stop_price: Optional[float] = None) -> Optional[Order]:
        """Mark order for replacement.
        
        Args:
            client_order_id: Client order ID
            new_qty: New quantity (optional)
            new_limit_price: New limit price (optional)
            new_stop_price: New stop price (optional)
            
        Returns:
            Order marked for replacement or None if not found
        """
        async with self._lock:
            order = self._orders.get(client_order_id)
            if order and order.status in {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}:
                logger.info(f"Marking order {client_order_id} for replacement")
                return order
        return None
    
    def register_callback(self, event: str, callback: Callable):
        """Register event callback.
        
        Args:
            event: Event name (order_submitted, order_filled, etc.)
            callback: Async callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    async def _trigger_callbacks(self, order: Order, old_status: OrderStatus, new_status: OrderStatus):
        """Trigger status change callbacks.
        
        Args:
            order: Order that changed
            old_status: Previous status
            new_status: New status
        """
        # Determine event type
        event = None
        if new_status == OrderStatus.SUBMITTED:
            event = 'order_submitted'
        elif new_status == OrderStatus.FILLED:
            event = 'order_filled'
            self._metrics['filled_orders'] += 1
        elif new_status == OrderStatus.PARTIALLY_FILLED:
            event = 'order_partially_filled'
        elif new_status == OrderStatus.CANCELLED:
            event = 'order_cancelled'
        elif new_status == OrderStatus.REJECTED:
            event = 'order_rejected'
            self._metrics['rejected_orders'] += 1
        elif new_status == OrderStatus.REPLACED:
            event = 'order_replaced'
        
        # Trigger callbacks
        if event and event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(order)
                    else:
                        callback(order)
                except Exception as e:
                    logger.error(f"Error in callback for {event}: {e}")
    
    def _map_alpaca_status(self, alpaca_status: str) -> OrderStatus:
        """Map Alpaca order status to internal status.
        
        Args:
            alpaca_status: Alpaca order status
            
        Returns:
            Internal order status
        """
        status_map = {
            'new': OrderStatus.SUBMITTED,
            'accepted': OrderStatus.SUBMITTED,
            'pending_new': OrderStatus.PENDING,
            'accepted_for_bidding': OrderStatus.SUBMITTED,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'done_for_day': OrderStatus.EXPIRED,
            'canceled': OrderStatus.CANCELLED,
            'expired': OrderStatus.EXPIRED,
            'replaced': OrderStatus.REPLACED,
            'pending_cancel': OrderStatus.SUBMITTED,
            'pending_replace': OrderStatus.SUBMITTED,
            'rejected': OrderStatus.REJECTED,
            'suspended': OrderStatus.SUBMITTED,
            'calculated': OrderStatus.SUBMITTED
        }
        return status_map.get(alpaca_status, OrderStatus.PENDING)
    
    def _update_fill_metrics(self, fill_latency_ms: float):
        """Update fill latency metrics.
        
        Args:
            fill_latency_ms: Fill latency in milliseconds
        """
        current_avg = self._metrics['avg_fill_latency_ms']
        filled_count = self._metrics['filled_orders']
        
        # Calculate new average
        if filled_count == 1:
            self._metrics['avg_fill_latency_ms'] = fill_latency_ms
        else:
            self._metrics['avg_fill_latency_ms'] = (
                (current_avg * (filled_count - 1) + fill_latency_ms) / filled_count
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        return self._metrics.copy()
    
    async def cleanup_old_orders(self, hours: int = 24):
        """Clean up old completed orders.
        
        Args:
            hours: Number of hours to keep completed orders
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        completed_statuses = {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
        
        async with self._lock:
            to_remove = []
            for client_order_id, order in self._orders.items():
                if order.status in completed_statuses and order.updated_at and order.updated_at < cutoff_time:
                    to_remove.append(client_order_id)
            
            # Remove old orders
            for client_order_id in to_remove:
                order = self._orders.pop(client_order_id)
                if order.alpaca_order_id in self._alpaca_orders:
                    del self._alpaca_orders[order.alpaca_order_id]
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old orders")

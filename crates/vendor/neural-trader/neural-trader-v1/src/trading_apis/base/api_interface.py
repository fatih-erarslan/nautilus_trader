"""
Abstract Base Class for Trading APIs

Provides a standardized interface for all trading API implementations with
focus on ultra-low latency operations and consistent error handling.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
from dataclasses import dataclass
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np


@dataclass
class OrderRequest:
    """Standardized order request structure"""
    symbol: str
    quantity: float
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop', etc.
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'DAY'
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class OrderResponse:
    """Standardized order response structure"""
    order_id: str
    status: str
    symbol: str
    quantity: float
    filled_quantity: float
    side: str
    order_type: str
    price: Optional[float]
    avg_fill_price: Optional[float]
    timestamp: datetime
    latency_ms: float
    raw_response: Dict[str, Any]


@dataclass
class MarketData:
    """Standardized market data structure"""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime
    latency_ms: float
    raw_data: Dict[str, Any]


@dataclass
class AccountBalance:
    """Standardized account balance structure"""
    cash: float
    portfolio_value: float
    buying_power: float
    margin_used: float
    positions: List[Dict[str, Any]]
    timestamp: datetime
    raw_data: Dict[str, Any]


class TradingAPIInterface(ABC):
    """
    Abstract base class for all trading API implementations.
    Designed for ultra-low latency with async operations and connection pooling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trading API interface.
        
        Args:
            config: Configuration dictionary with API credentials and settings
        """
        self.config = config
        self._connected = False
        self._latency_buffer = []  # Circular buffer for latency tracking
        self._max_latency_samples = 1000
        self._callbacks: Dict[str, List[Callable]] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the trading API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the trading API.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """
        Place a trading order with microsecond precision timing.
        
        Args:
            order: OrderRequest object with order details
            
        Returns:
            OrderResponse object with order status and timing info
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            order_id: Unique identifier of the order to cancel
            
        Returns:
            Dictionary with cancellation status
        """
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderResponse:
        """
        Get current status of an order.
        
        Args:
            order_id: Unique identifier of the order
            
        Returns:
            OrderResponse object with current order status
        """
        pass
    
    @abstractmethod
    async def get_market_data(self, symbols: List[str]) -> List[MarketData]:
        """
        Get real-time market data for specified symbols.
        
        Args:
            symbols: List of symbol tickers
            
        Returns:
            List of MarketData objects
        """
        pass
    
    @abstractmethod
    async def get_account_balance(self) -> AccountBalance:
        """
        Get current account balance and positions.
        
        Returns:
            AccountBalance object with account details
        """
        pass
    
    @abstractmethod
    async def subscribe_market_data(self, symbols: List[str], 
                                   callback: Callable[[MarketData], None]) -> bool:
        """
        Subscribe to real-time market data updates.
        
        Args:
            symbols: List of symbols to subscribe to
            callback: Function to call with market data updates
            
        Returns:
            bool: True if subscription successful
        """
        pass
    
    @abstractmethod
    async def unsubscribe_market_data(self, symbols: List[str]) -> bool:
        """
        Unsubscribe from market data updates.
        
        Args:
            symbols: List of symbols to unsubscribe from
            
        Returns:
            bool: True if unsubscription successful
        """
        pass
    
    # Non-abstract utility methods
    
    def measure_latency(self, start_time: float) -> float:
        """
        Measure and record latency for an operation.
        
        Args:
            start_time: Start time from time.perf_counter()
            
        Returns:
            float: Latency in milliseconds
        """
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Add to circular buffer
        if len(self._latency_buffer) >= self._max_latency_samples:
            self._latency_buffer.pop(0)
        self._latency_buffer.append(latency_ms)
        
        return latency_ms
    
    def get_latency_stats(self) -> Dict[str, float]:
        """
        Get latency statistics from recent operations.
        
        Returns:
            Dictionary with latency statistics (mean, median, p95, p99, max)
        """
        if not self._latency_buffer:
            return {
                'mean': 0.0,
                'median': 0.0,
                'p95': 0.0,
                'p99': 0.0,
                'max': 0.0,
                'min': 0.0
            }
        
        latencies = np.array(self._latency_buffer)
        return {
            'mean': np.mean(latencies),
            'median': np.median(latencies),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'max': np.max(latencies),
            'min': np.min(latencies)
        }
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """
        Register a callback for specific events.
        
        Args:
            event_type: Type of event ('order_fill', 'error', etc.)
            callback: Function to call when event occurs
        """
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
    
    def trigger_callbacks(self, event_type: str, data: Any) -> None:
        """
        Trigger all callbacks for a specific event type.
        
        Args:
            event_type: Type of event
            data: Data to pass to callbacks
        """
        if event_type in self._callbacks:
            for callback in self._callbacks[event_type]:
                # Run callbacks in thread pool to avoid blocking
                self._executor.submit(callback, data)
    
    @property
    def is_connected(self) -> bool:
        """Check if API is currently connected."""
        return self._connected
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the API connection.
        
        Returns:
            Dictionary with health status and metrics
        """
        try:
            start_time = time.perf_counter()
            
            # Try to get account balance as a health check
            balance = await self.get_account_balance()
            
            latency_ms = self.measure_latency(start_time)
            
            return {
                'status': 'healthy',
                'connected': self.is_connected,
                'latency_ms': latency_ms,
                'latency_stats': self.get_latency_stats(),
                'timestamp': datetime.now()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'connected': self.is_connected,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def validate_order(self, order: OrderRequest) -> List[str]:
        """
        Validate an order request before submission.
        
        Args:
            order: OrderRequest to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not order.symbol:
            errors.append("Symbol is required")
            
        if order.quantity <= 0:
            errors.append("Quantity must be positive")
            
        if order.side not in ['buy', 'sell']:
            errors.append("Side must be 'buy' or 'sell'")
            
        if order.order_type == 'limit' and order.price is None:
            errors.append("Price is required for limit orders")
            
        if order.order_type == 'stop' and order.stop_price is None:
            errors.append("Stop price is required for stop orders")
            
        return errors
    
    def __enter__(self):
        """Context manager entry."""
        asyncio.run(self.connect())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.run(self.disconnect())
        self._executor.shutdown(wait=True)
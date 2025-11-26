"""
Lime Trading API Integration with Ultra-Low Latency Optimizations

This module provides the main interface for Lime Trading with:
- FIX protocol connectivity
- Risk management
- Order management
- Memory optimization
- Performance monitoring
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
import logging
from contextlib import asynccontextmanager

from .fix.lime_client import LowLatencyFIXClient, OrderLatencyMetrics
from .core.lime_order_manager import LimeOrderManager, Order, OrderStatus
from .risk.lime_risk_engine import LimeRiskEngine, RiskCheckResult, RiskLimits
from .memory.memory_pool import memory_manager
from .monitoring.performance_monitor import PerformanceMonitor


@dataclass
class LimeConfig:
    """Configuration for Lime Trading API"""
    # FIX connection
    fix_config_file: str
    sender_comp_id: str
    target_comp_id: str
    host: str
    port: int
    
    # Performance settings
    cpu_core: int = -1  # CPU core to pin to (-1 for no pinning)
    use_hardware_timestamps: bool = True
    enable_memory_pools: bool = True
    
    # Risk limits
    risk_limits: Optional[RiskLimits] = None
    
    # Order management
    max_orders: int = 100000
    order_pool_size: int = 10000
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_interval: int = 1000  # milliseconds
    

@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    orders_sent: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    orders_canceled: int = 0
    
    # Latency metrics (microseconds)
    avg_order_latency: float = 0.0
    p99_order_latency: float = 0.0
    avg_fill_latency: float = 0.0
    
    # Volume metrics
    total_volume: int = 0
    total_notional: float = 0.0
    
    # Risk metrics
    risk_checks: int = 0
    risk_rejections: int = 0
    
    # System metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gc_collections: int = 0


class LimeTradingAPI:
    """
    Main Lime Trading API with ultra-low latency optimizations
    """
    
    def __init__(self, config: LimeConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.fix_client = None
        self.order_manager = None
        self.risk_engine = None
        self.performance_monitor = None
        
        # Event handlers
        self.execution_handler: Optional[Callable] = None
        self.order_update_handler: Optional[Callable] = None
        self.risk_event_handler: Optional[Callable] = None
        
        # Internal state
        self.is_connected = False
        self.is_running = False
        self.start_time = 0
        
        # Metrics
        self.metrics = TradingMetrics()
        self.metrics_lock = threading.Lock()
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all trading components"""
        # Risk engine
        self.risk_engine = LimeRiskEngine(self.config.risk_limits)
        
        # Order manager
        self.order_manager = LimeOrderManager(self.config.max_orders)
        
        # Performance monitor
        if self.config.enable_monitoring:
            self.performance_monitor = PerformanceMonitor(
                interval_ms=self.config.metrics_interval
            )
            
        # FIX client
        self.fix_client = LowLatencyFIXClient(
            config_file=self.config.fix_config_file,
            cpu_core=self.config.cpu_core
        )
        
        # Set up FIX client callbacks
        self.fix_client.execution_handler = self._handle_execution_report
        self.fix_client.order_handler = self._handle_order_update
        
    def start(self):
        """Start the trading API"""
        if self.is_running:
            return
            
        self.logger.info("Starting Lime Trading API...")
        self.start_time = time.time()
        
        # Start FIX client
        self.fix_client.start()
        
        # Start performance monitor
        if self.performance_monitor:
            self.performance_monitor.start()
            
        self.is_running = True
        self.logger.info("Lime Trading API started successfully")
        
    def stop(self):
        """Stop the trading API"""
        if not self.is_running:
            return
            
        self.logger.info("Stopping Lime Trading API...")
        
        # Stop FIX client
        if self.fix_client:
            self.fix_client.stop()
            
        # Stop performance monitor
        if self.performance_monitor:
            self.performance_monitor.stop()
            
        # Cleanup memory pools
        memory_manager.cleanup()
        
        self.is_running = False
        self.is_connected = False
        self.logger.info("Lime Trading API stopped")
        
    def send_order(self,
                   symbol: str,
                   side: str,
                   quantity: int,
                   order_type: str = 'MARKET',
                   price: Optional[float] = None,
                   time_in_force: str = 'DAY',
                   account: str = 'DEFAULT') -> Tuple[bool, str, Optional[str]]:
        """
        Send order with pre-trade risk checks
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            side: 'BUY' or 'SELL'
            quantity: Number of shares
            order_type: 'MARKET' or 'LIMIT'
            price: Price for limit orders
            time_in_force: 'DAY', 'IOC', 'FOK'
            account: Trading account
            
        Returns:
            Tuple of (success, order_id, error_message)
        """
        if not self.is_connected:
            return False, "", "Not connected to exchange"
            
        # Validate inputs
        if not symbol or not side or quantity <= 0:
            return False, "", "Invalid order parameters"
            
        if order_type == 'LIMIT' and price is None:
            return False, "", "Price required for limit orders"
            
        # Pre-trade risk check
        risk_result, risk_reason = self.risk_engine.check_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price or 0.0
        )
        
        if risk_result != RiskCheckResult.PASSED:
            with self.metrics_lock:
                self.metrics.risk_rejections += 1
            return False, "", f"Risk check failed: {risk_reason}"
            
        # Create order
        try:
            order_id, order = self.order_manager.create_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                account=account
            )
            
            # Send to exchange
            fix_order_id = self.fix_client.send_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                tif=time_in_force,
                account=account
            )
            
            # Update order status
            self.order_manager.update_order(order_id, 'SUBMITTED')
            
            # Update metrics
            with self.metrics_lock:
                self.metrics.orders_sent += 1
                
            return True, order_id, None
            
        except Exception as e:
            self.logger.error(f"Error sending order: {e}")
            return False, "", str(e)
            
    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """
        Cancel order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Tuple of (success, error_message)
        """
        if not self.is_connected:
            return False, "Not connected to exchange"
            
        # Get order
        order = self.order_manager.get_order(order_id)
        if not order:
            return False, "Order not found"
            
        if order.status not in [OrderStatus.SUBMITTED, OrderStatus.ACKNOWLEDGED]:
            return False, "Order cannot be canceled"
            
        try:
            # Generate cancel order ID
            cancel_id = f"CANCEL{int(time.time() * 1000000)}"
            
            # Send cancel request
            self.fix_client.cancel_order(cancel_id, order_id)
            
            # Update metrics
            with self.metrics_lock:
                self.metrics.orders_canceled += 1
                
            return True, ""
            
        except Exception as e:
            self.logger.error(f"Error canceling order: {e}")
            return False, str(e)
            
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.order_manager.get_order(order_id)
        
    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        return self.order_manager.get_active_orders()
        
    def get_position(self, symbol: str) -> int:
        """Get position for symbol"""
        return self.order_manager.get_position(symbol)
        
    def get_metrics(self) -> TradingMetrics:
        """Get current trading metrics"""
        with self.metrics_lock:
            # Update latency metrics from FIX client
            latency_stats = self.fix_client.get_latency_stats()
            if latency_stats:
                self.metrics.avg_order_latency = latency_stats.get('ack_mean_us', 0)
                self.metrics.p99_order_latency = latency_stats.get('ack_p99_us', 0)
                self.metrics.avg_fill_latency = latency_stats.get('fill_mean_us', 0)
                
            # Update risk metrics
            risk_metrics = self.risk_engine.get_risk_metrics()
            self.metrics.risk_checks = risk_metrics.get('check_count', 0)
            self.metrics.risk_rejections = risk_metrics.get('rejection_count', 0)
            
            # Update performance metrics
            if self.performance_monitor:
                perf_stats = self.performance_monitor.get_current_stats()
                self.metrics.cpu_usage = perf_stats.get('cpu_percent', 0)
                self.metrics.memory_usage = perf_stats.get('memory_percent', 0)
                
            return self.metrics
            
    def _handle_execution_report(self, message, order_id: str, exec_type: str, order_status: str):
        """Handle FIX execution report"""
        # Update order status
        self.order_manager.update_order(order_id, order_status)
        
        # Update metrics
        with self.metrics_lock:
            if exec_type == 'FILL':
                self.metrics.orders_filled += 1
                
                # Update volume metrics
                order = self.order_manager.get_order(order_id)
                if order:
                    self.metrics.total_volume += order.quantity
                    self.metrics.total_notional += order.quantity * order.price
                    
                    # Update risk engine position
                    self.risk_engine.update_position(
                        symbol=order.symbol,
                        quantity_delta=order.quantity if order.side == 'BUY' else -order.quantity,
                        price=order.price
                    )
                    
            elif exec_type == 'REJECTED':
                self.metrics.orders_rejected += 1
                
        # Call user handler
        if self.execution_handler:
            self.execution_handler(message, order_id, exec_type, order_status)
            
    def _handle_order_update(self, order_id: str, status: str):
        """Handle order update"""
        # Update order status
        self.order_manager.update_order(order_id, status)
        
        # Call user handler
        if self.order_update_handler:
            self.order_update_handler(order_id, status)
            
    def set_execution_handler(self, handler: Callable):
        """Set execution report handler"""
        self.execution_handler = handler
        
    def set_order_update_handler(self, handler: Callable):
        """Set order update handler"""
        self.order_update_handler = handler
        
    def set_risk_event_handler(self, handler: Callable):
        """Set risk event handler"""
        self.risk_event_handler = handler
        
    @asynccontextmanager
    async def trading_session(self):
        """Async context manager for trading session"""
        try:
            self.start()
            yield self
        finally:
            self.stop()
            
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status"""
        return {
            'connected': self.is_connected,
            'running': self.is_running,
            'uptime': time.time() - self.start_time if self.start_time > 0 else 0,
            'session_id': str(self.fix_client.session_id) if self.fix_client.session_id else None
        }


# Example usage and configuration
def create_lime_config() -> LimeConfig:
    """Create default Lime Trading configuration"""
    return LimeConfig(
        fix_config_file="lime_fix.cfg",
        sender_comp_id="TRADERCLIENT",
        target_comp_id="LIME",
        host="fix.lime.com",
        port=4001,
        cpu_core=1,  # Pin to CPU core 1
        use_hardware_timestamps=True,
        enable_memory_pools=True,
        risk_limits=RiskLimits(
            max_position_size=100000,
            max_position_value=10_000_000,
            max_single_order_size=10000,
            max_daily_loss=500_000,
            max_orders_per_second=100
        ),
        max_orders=100000,
        order_pool_size=10000,
        enable_monitoring=True,
        metrics_interval=1000
    )


async def example_usage():
    """Example usage of Lime Trading API"""
    config = create_lime_config()
    
    async with LimeTradingAPI(config).trading_session() as api:
        # Set up event handlers
        def on_execution(message, order_id, exec_type, status):
            print(f"Execution: {order_id} {exec_type} {status}")
            
        def on_order_update(order_id, status):
            print(f"Order update: {order_id} {status}")
            
        api.set_execution_handler(on_execution)
        api.set_order_update_handler(on_order_update)
        
        # Send market order
        success, order_id, error = api.send_order(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            order_type="MARKET"
        )
        
        if success:
            print(f"Order sent: {order_id}")
            
            # Wait for fill
            await asyncio.sleep(1)
            
            # Check order status
            order = api.get_order(order_id)
            if order:
                print(f"Order status: {order.status}")
                
        # Get metrics
        metrics = api.get_metrics()
        print(f"Orders sent: {metrics.orders_sent}")
        print(f"Orders filled: {metrics.orders_filled}")
        print(f"Avg latency: {metrics.avg_order_latency:.2f}μs")


if __name__ == "__main__":
    asyncio.run(example_usage())
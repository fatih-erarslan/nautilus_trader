"""
Example usage of the Trading API Connection Manager infrastructure.

This demonstrates how to use the ultra-low latency connection management
system with proper error handling and monitoring.
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, Any

# Import the base infrastructure components
from api_interface import TradingAPIInterface, OrderRequest, OrderResponse, MarketData
from connection_pool import ConnectionPool
from latency_monitor import LatencyMonitor, LatencyAlert
from config_loader import ConfigLoader


# Example implementation of a trading API
class ExampleTradingAPI(TradingAPIInterface):
    """Example implementation for demonstration purposes"""
    
    async def connect(self) -> bool:
        """Simulate connection to trading API"""
        await asyncio.sleep(0.01)  # Simulate network delay
        self._connected = True
        return True
    
    async def disconnect(self) -> bool:
        """Simulate disconnection"""
        self._connected = False
        return True
    
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Simulate order placement"""
        start_time = time.perf_counter()
        
        # Simulate API call with varying latency
        await asyncio.sleep(0.002 + random.random() * 0.003)  # 2-5ms
        
        latency_ms = self.measure_latency(start_time)
        
        return OrderResponse(
            order_id=f"ORD-{int(time.time() * 1000)}",
            status="submitted",
            symbol=order.symbol,
            quantity=order.quantity,
            filled_quantity=0,
            side=order.side,
            order_type=order.order_type,
            price=order.price,
            avg_fill_price=None,
            timestamp=datetime.now(),
            latency_ms=latency_ms,
            raw_response={"status": "ok"}
        )
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Simulate order cancellation"""
        await asyncio.sleep(0.002)
        return {"status": "cancelled", "order_id": order_id}
    
    async def get_order_status(self, order_id: str) -> OrderResponse:
        """Simulate getting order status"""
        # Implementation would go here
        pass
    
    async def get_market_data(self, symbols: List[str]) -> List[MarketData]:
        """Simulate getting market data"""
        start_time = time.perf_counter()
        await asyncio.sleep(0.001)  # 1ms simulated latency
        
        latency_ms = self.measure_latency(start_time)
        
        return [
            MarketData(
                symbol=symbol,
                bid=100.00 + random.random(),
                ask=100.02 + random.random(),
                last=100.01 + random.random(),
                volume=1000000 * random.random(),
                timestamp=datetime.now(),
                latency_ms=latency_ms,
                raw_data={}
            )
            for symbol in symbols
        ]
    
    async def get_account_balance(self) -> AccountBalance:
        """Simulate getting account balance"""
        # Implementation would go here
        pass
    
    async def subscribe_market_data(self, symbols: List[str], 
                                   callback: Callable[[MarketData], None]) -> bool:
        """Simulate market data subscription"""
        # Implementation would go here
        return True
    
    async def unsubscribe_market_data(self, symbols: List[str]) -> bool:
        """Simulate market data unsubscription"""
        # Implementation would go here
        return True


async def latency_alert_handler(alert: LatencyAlert):
    """Handle latency alerts"""
    print(f"⚠️  LATENCY ALERT: {alert.operation} - {alert.latency_ms:.2f}ms "
          f"(threshold: {alert.threshold_ms}ms) - Level: {alert.level.value}")
    
    # In production, this could:
    # - Send to monitoring system
    # - Trigger circuit breakers
    # - Notify operations team
    # - Switch to backup connections


async def demonstrate_connection_manager():
    """Demonstrate the connection manager in action"""
    
    print("🚀 Trading API Connection Manager Demo\n")
    
    # 1. Load configuration
    print("1️⃣  Loading configuration...")
    config_path = Path("/workspaces/ai-news-trader/config/trading_apis.yaml")
    config_loader = ConfigLoader(config_path)
    config = await config_loader.load_config()
    
    # Validate configuration
    errors = config_loader.validate_config()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        return
    
    print(f"✅ Loaded {len(config.apis)} API configurations")
    print(f"   Enabled APIs: {[name for name, api in config.apis.items() if api.enabled]}")
    
    # 2. Initialize latency monitor
    print("\n2️⃣  Initializing latency monitor...")
    latency_monitor = LatencyMonitor(
        alert_thresholds={
            'place_order': 5.0,  # 5ms threshold
            'market_data': 2.0   # 2ms threshold
        }
    )
    latency_monitor.add_alert_callback(latency_alert_handler)
    await latency_monitor.start_monitoring()
    print("✅ Latency monitor active")
    
    # 3. Create connection pool
    print("\n3️⃣  Creating connection pool...")
    # For demo, use example API with first config
    api_config = list(config.apis.values())[0]
    
    connection_pool = ConnectionPool(
        api_class=ExampleTradingAPI,
        config=api_config.dict(),
        min_connections=config.connection_pool['min_connections'],
        max_connections=config.connection_pool['max_connections'],
        health_check_interval=config.connection_pool['health_check_interval']
    )
    
    await connection_pool.initialize()
    print(f"✅ Connection pool initialized with {config.connection_pool['min_connections']} connections")
    
    # 4. Demonstrate order placement with connection pool
    print("\n4️⃣  Testing order placement...")
    
    async def place_order_with_monitoring(order: OrderRequest):
        """Place order with full monitoring"""
        measurement = latency_monitor.measure('place_order')
        
        try:
            # Execute with connection from pool
            response = await connection_pool.execute_with_connection(
                lambda api, o: api.place_order(o),
                order
            )
            
            measurement.stop()
            latency_monitor.record(measurement)
            
            return response
            
        except Exception as e:
            measurement.stop()
            latency_monitor.record(measurement)
            raise
    
    # Place multiple orders to demonstrate pooling and monitoring
    orders = [
        OrderRequest(
            symbol="AAPL",
            quantity=100,
            side="buy",
            order_type="limit",
            price=150.00
        ),
        OrderRequest(
            symbol="GOOGL",
            quantity=50,
            side="sell",
            order_type="market"
        ),
        OrderRequest(
            symbol="MSFT",
            quantity=75,
            side="buy",
            order_type="limit",
            price=300.00
        )
    ]
    
    # Place orders concurrently
    tasks = [place_order_with_monitoring(order) for order in orders]
    responses = await asyncio.gather(*tasks)
    
    print(f"✅ Placed {len(responses)} orders")
    for i, response in enumerate(responses):
        print(f"   Order {i+1}: {response.order_id} - "
              f"Latency: {response.latency_ms:.2f}ms")
    
    # 5. Test market data with monitoring
    print("\n5️⃣  Testing market data retrieval...")
    
    async def get_market_data_with_monitoring(symbols: List[str]):
        """Get market data with monitoring"""
        measurement = latency_monitor.measure('market_data')
        
        try:
            data = await connection_pool.execute_with_connection(
                lambda api, s: api.get_market_data(s),
                symbols
            )
            
            measurement.stop()
            latency_monitor.record(measurement)
            
            return data
            
        except Exception as e:
            measurement.stop()
            latency_monitor.record(measurement)
            raise
    
    # Get market data for multiple symbols
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    market_data = await get_market_data_with_monitoring(symbols)
    
    print(f"✅ Retrieved market data for {len(market_data)} symbols")
    avg_latency = sum(d.latency_ms for d in market_data) / len(market_data)
    print(f"   Average latency: {avg_latency:.2f}ms")
    
    # 6. Show pool statistics
    print("\n6️⃣  Connection pool statistics:")
    pool_stats = connection_pool.get_pool_stats()
    print(f"   Total connections: {pool_stats['total_connections']}")
    print(f"   Total requests: {pool_stats['total_requests']}")
    print(f"   Average health score: {pool_stats['average_health_score']:.1f}")
    print("   Connection states:")
    for state, count in pool_stats['state_distribution'].items():
        if count > 0:
            print(f"     - {state}: {count}")
    
    # 7. Show latency statistics
    print("\n7️⃣  Latency statistics:")
    latency_stats = latency_monitor.get_profile_stats()
    
    for operation, stats in latency_stats.items():
        if stats['count'] > 0:
            print(f"\n   {operation}:")
            print(f"     Count: {stats['count']}")
            print(f"     Mean: {stats['mean_ms']:.2f}ms")
            print(f"     Median: {stats['median_ms']:.2f}ms")
            print(f"     P95: {stats['p95_ms']:.2f}ms")
            print(f"     P99: {stats['p99_ms']:.2f}ms")
            print(f"     Max: {stats['max_ms']:.2f}ms")
    
    # 8. Generate summary report
    print("\n8️⃣  Summary report:")
    summary = latency_monitor.get_summary_report()
    print(f"   Total measurements: {summary['total_measurements']}")
    print(f"   Operations tracked: {summary['operations_tracked']}")
    print(f"   Recent alerts: {summary['recent_alerts']}")
    
    if summary['overall_stats']:
        print(f"   Overall latency:")
        print(f"     Mean: {summary['overall_stats']['mean_ms']:.2f}ms")
        print(f"     P95: {summary['overall_stats']['p95_ms']:.2f}ms")
        print(f"     P99: {summary['overall_stats']['p99_ms']:.2f}ms")
    
    # 9. Export metrics
    print("\n9️⃣  Exporting metrics...")
    metrics_file = "/workspaces/ai-news-trader/logs/latency_metrics.json"
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    latency_monitor.export_metrics(metrics_file)
    print(f"✅ Metrics exported to {metrics_file}")
    
    # 10. Cleanup
    print("\n🔟 Cleaning up...")
    await connection_pool.shutdown()
    await latency_monitor.stop_monitoring()
    print("✅ Shutdown complete")
    
    print("\n✨ Demo completed successfully!")


async def demonstrate_hot_reload():
    """Demonstrate configuration hot reload"""
    print("\n🔄 Configuration Hot Reload Demo\n")
    
    config_path = Path("/workspaces/ai-news-trader/config/trading_apis.yaml")
    config_loader = ConfigLoader(config_path, enable_hot_reload=True)
    
    # Add reload callback
    async def on_config_reload(new_config):
        print(f"📢 Configuration reloaded at {new_config.loaded_at}")
        print(f"   Active APIs: {[name for name, api in new_config.apis.items() if api.enabled]}")
    
    config_loader.add_reload_callback(on_config_reload)
    
    # Load initial config
    config = await config_loader.load_config()
    print(f"✅ Initial config loaded with {len(config.apis)} APIs")
    
    print("\n⏳ Waiting for config changes...")
    print("   (Try modifying the trading_apis.yaml file)")
    
    # In a real application, this would run indefinitely
    await asyncio.sleep(30)
    
    print("✅ Hot reload demo complete")


# Additional imports needed
import time
import random
from datetime import datetime
from typing import List, Callable
from api_interface import AccountBalance


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_connection_manager())
    
    # Uncomment to test hot reload
    # asyncio.run(demonstrate_hot_reload())
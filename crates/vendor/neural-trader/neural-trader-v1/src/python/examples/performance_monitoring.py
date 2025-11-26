"""
Performance Monitoring Example
=============================

This example demonstrates comprehensive performance monitoring and alerting
capabilities of the Python Supabase client.
"""

import asyncio
import os
import random
from uuid import uuid4
from datetime import datetime, timedelta

from supabase_client import NeuralTradingClient
from supabase_client.monitoring.performance_monitor import (
    PerformanceMonitor,
    MetricData,
    MetricType,
    AlertSeverity,
    PerformanceThreshold
)

class TradingSystemMonitor:
    """Comprehensive monitoring system for trading operations."""
    
    def __init__(self, client: NeuralTradingClient):
        self.client = client
        self.monitor = client.performance
        self.running = False
        
    async def setup_monitoring(self):
        """Set up performance monitoring with thresholds and alerts."""
        print("🔧 Setting up performance monitoring...")
        
        # Define performance thresholds
        thresholds = [
            PerformanceThreshold(
                metric_name="api_response_time",
                warning_threshold=200.0,
                error_threshold=500.0,
                critical_threshold=1000.0,
                operator=">"
            ),
            PerformanceThreshold(
                metric_name="order_execution_time",
                warning_threshold=100.0,
                error_threshold=300.0,
                critical_threshold=1000.0,
                operator=">"
            ),
            PerformanceThreshold(
                metric_name="model_prediction_accuracy",
                warning_threshold=0.6,
                error_threshold=0.5,
                critical_threshold=0.4,
                operator="<"
            ),
            PerformanceThreshold(
                metric_name="bot_daily_pnl",
                warning_threshold=-100.0,
                error_threshold=-500.0,
                critical_threshold=-1000.0,
                operator="<"
            ),
            PerformanceThreshold(
                metric_name="system_cpu_usage",
                warning_threshold=70.0,
                error_threshold=85.0,
                critical_threshold=95.0,
                operator=">"
            ),
            PerformanceThreshold(
                metric_name="memory_usage_percent",
                warning_threshold=80.0,
                error_threshold=90.0,
                critical_threshold=95.0,
                operator=">"
            )
        ]
        
        # Set thresholds
        for threshold in thresholds:
            success, error = await self.monitor.set_threshold(threshold)
            if error:
                print(f"❌ Error setting threshold for {threshold.metric_name}: {error}")
            else:
                print(f"✅ Set threshold for {threshold.metric_name}")
        
        # Start monitoring
        await self.monitor.start_monitoring()
        print("✅ Performance monitoring started")
    
    async def simulate_trading_metrics(self):
        """Simulate various trading system metrics."""
        print("📊 Starting metric simulation...")
        
        metric_generators = {
            "api_response_time": lambda: random.uniform(50, 800),
            "order_execution_time": lambda: random.uniform(20, 500),
            "database_query_time": lambda: random.uniform(10, 200),
            "model_prediction_time": lambda: random.uniform(100, 2000),
            "model_prediction_accuracy": lambda: random.uniform(0.3, 0.9),
            "system_cpu_usage": lambda: random.uniform(20, 90),
            "memory_usage_percent": lambda: random.uniform(40, 95),
            "disk_io_wait": lambda: random.uniform(0, 50),
            "network_latency": lambda: random.uniform(5, 100),
            "bot_daily_pnl": lambda: random.uniform(-800, 600),
            "active_connections": lambda: random.randint(10, 100),
            "queue_size": lambda: random.randint(0, 50),
            "error_rate": lambda: random.uniform(0, 0.1),
            "throughput_requests_per_second": lambda: random.uniform(50, 500)
        }
        
        components = ["api", "database", "ml_engine", "trading_engine", "risk_manager"]
        strategies = ["momentum", "mean_reversion", "neural_sentiment", "arbitrage"]
        
        while self.running:
            try:
                # Generate batch of metrics
                metrics_batch = []
                
                for metric_name, generator in metric_generators.items():
                    for component in components:
                        # Skip some combinations for realism
                        if metric_name == "model_prediction_accuracy" and component != "ml_engine":
                            continue
                        if metric_name == "bot_daily_pnl" and component != "trading_engine":
                            continue
                        
                        value = generator()
                        
                        # Add some strategy-specific metrics
                        if metric_name in ["bot_daily_pnl", "model_prediction_accuracy"]:
                            for strategy in strategies:
                                metric = MetricData(
                                    name=metric_name,
                                    value=value + random.uniform(-0.1, 0.1) * value,
                                    metric_type=MetricType.GAUGE,
                                    tags={
                                        "component": component,
                                        "strategy": strategy,
                                        "environment": "demo"
                                    }
                                )
                                metrics_batch.append(metric)
                        else:
                            metric = MetricData(
                                name=metric_name,
                                value=value,
                                metric_type=MetricType.GAUGE,
                                tags={
                                    "component": component,
                                    "environment": "demo"
                                }
                            )
                            metrics_batch.append(metric)
                
                # Record batch of metrics
                count, error = await self.monitor.record_batch_metrics(metrics_batch)
                if error:
                    print(f"❌ Error recording metrics: {error}")
                else:
                    print(f"📈 Recorded {count} metrics")
                
                await asyncio.sleep(5)  # Generate metrics every 5 seconds
                
            except Exception as e:
                print(f"❌ Error in metric simulation: {e}")
                await asyncio.sleep(5)
    
    async def generate_trading_events(self):
        """Generate simulated trading events and metrics."""
        print("🤖 Starting trading event simulation...")
        
        while self.running:
            try:
                # Simulate order processing
                order_processing_time = random.uniform(20, 300)
                await self.monitor.record_metric(MetricData(
                    name="order_processing_time",
                    value=order_processing_time,
                    metric_type=MetricType.TIMER,
                    tags={"order_type": random.choice(["market", "limit", "stop"])}
                ))
                
                # Simulate trade execution
                if random.random() > 0.7:  # 30% chance
                    execution_latency = random.uniform(10, 150)
                    await self.monitor.record_metric(MetricData(
                        name="trade_execution_latency",
                        value=execution_latency,
                        metric_type=MetricType.TIMER,
                        tags={"venue": random.choice(["venue_a", "venue_b", "venue_c"])}
                    ))
                
                # Simulate risk checks
                risk_check_time = random.uniform(1, 50)
                await self.monitor.record_metric(MetricData(
                    name="risk_check_time",
                    value=risk_check_time,
                    metric_type=MetricType.TIMER,
                    tags={"check_type": "pre_trade"}
                ))
                
                # Simulate portfolio updates
                portfolio_update_time = random.uniform(5, 100)
                await self.monitor.record_metric(MetricData(
                    name="portfolio_update_time",
                    value=portfolio_update_time,
                    metric_type=MetricType.TIMER,
                    tags={"update_type": "position"}
                ))
                
                await asyncio.sleep(3)  # Generate events every 3 seconds
                
            except Exception as e:
                print(f"❌ Error in event simulation: {e}")
                await asyncio.sleep(3)
    
    async def monitor_system_health(self):
        """Monitor overall system health and generate reports."""
        print("🏥 Starting system health monitoring...")
        
        while self.running:
            try:
                # Get system health
                health, error = await self.monitor.get_system_health()
                if error:
                    print(f"❌ Error getting system health: {error}")
                else:
                    print(f"\n🏥 System Health Report:")
                    print(f"Overall Status: {health.overall_status}")
                    print(f"Active Alerts: {len(health.active_alerts)}")
                    
                    # Display component health
                    print("Component Health:")
                    for component, status in health.component_statuses.items():
                        emoji = {"healthy": "✅", "warning": "⚠️", "error": "❌", "critical": "🚨"}.get(status, "❓")
                        print(f"  {emoji} {component}: {status}")
                    
                    # Display performance summary
                    summary = health.performance_summary
                    if summary:
                        print("Performance Summary:")
                        if "avg_latency_ms" in summary:
                            print(f"  Avg Latency: {summary['avg_latency_ms']:.1f}ms")
                        if "total_errors" in summary:
                            print(f"  Total Errors: {summary['total_errors']}")
                        if "error_rate" in summary:
                            print(f"  Error Rate: {summary['error_rate']:.2%}")
                
                # Generate aggregated metrics report
                aggregates = {}
                key_metrics = ["api_response_time", "order_execution_time", "model_prediction_accuracy"]
                
                for metric in key_metrics:
                    agg_result, error = await self.monitor.calculate_aggregates(
                        metric_name=metric,
                        aggregation="avg",
                        window_minutes=5
                    )
                    if not error and agg_result:
                        aggregates[metric] = agg_result
                
                if aggregates:
                    print("\n📊 5-Minute Aggregates:")
                    for metric, result in aggregates.items():
                        for group, data in result.items():
                            print(f"  {metric}: {data['value']:.2f} (count: {data['count']})")
                
                await asyncio.sleep(30)  # Health check every 30 seconds
                
            except Exception as e:
                print(f"❌ Error in health monitoring: {e}")
                await asyncio.sleep(30)
    
    async def handle_alerts(self):
        """Monitor and handle performance alerts."""
        print("🚨 Starting alert monitoring...")
        
        while self.running:
            try:
                # In a real system, you might subscribe to alert notifications
                # For this example, we'll poll for active alerts
                
                # Get recent alerts
                recent_metrics = await self.monitor.supabase.select(
                    "performance_alerts",
                    filter_dict={"status": "active"},
                    order_by="-created_at",
                    limit=10
                )
                
                for alert in recent_metrics:
                    severity = alert.get("severity", "info")
                    message = alert.get("message", "")
                    metric_name = alert.get("metric_name", "")
                    
                    emoji = {
                        "info": "💡",
                        "warning": "⚠️",
                        "error": "❌",
                        "critical": "🚨"
                    }.get(severity, "📢")
                    
                    print(f"{emoji} ALERT [{severity.upper()}] {metric_name}: {message}")
                    
                    # Auto-resolve info alerts after displaying
                    if severity == "info":
                        await self.monitor.resolve_alert(
                            alert["id"],
                            "Auto-resolved info alert"
                        )
                
                await asyncio.sleep(10)  # Check alerts every 10 seconds
                
            except Exception as e:
                print(f"❌ Error in alert handling: {e}")
                await asyncio.sleep(10)
    
    async def start_monitoring(self):
        """Start all monitoring tasks."""
        await self.setup_monitoring()
        
        self.running = True
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self.simulate_trading_metrics()),
            asyncio.create_task(self.generate_trading_events()),
            asyncio.create_task(self.monitor_system_health()),
            asyncio.create_task(self.handle_alerts())
        ]
        
        print("🎯 Performance monitoring system is now active!")
        return tasks
    
    async def stop_monitoring(self):
        """Stop all monitoring tasks."""
        print("🛑 Stopping performance monitoring...")
        
        self.running = False
        await self.monitor.stop_monitoring()
        
        print("✅ Performance monitoring stopped")

async def main():
    """Main function demonstrating performance monitoring."""
    
    # Initialize client
    client = NeuralTradingClient(
        url=os.getenv("SUPABASE_URL", "https://your-project.supabase.co"),
        key=os.getenv("SUPABASE_ANON_KEY", "your-anon-key"),
        service_key=os.getenv("SUPABASE_SERVICE_KEY")
    )
    
    try:
        await client.connect()
        print("✅ Connected to Supabase")
        
        # Create monitoring system
        monitoring_system = TradingSystemMonitor(client)
        
        # Start monitoring
        monitoring_tasks = await monitoring_system.start_monitoring()
        
        print("📊 Monitoring system running... Press Ctrl+C to stop")
        print("Watch for metrics, alerts, and system health reports!")
        print()
        
        try:
            # Run for demonstration period
            await asyncio.sleep(180)  # Run for 3 minutes
        except KeyboardInterrupt:
            print("\n⚠️ Received interrupt signal")
        
        # Stop monitoring
        await monitoring_system.stop_monitoring()
        
        # Cancel tasks
        for task in monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to cleanup
        await asyncio.gather(*monitoring_tasks, return_exceptions=True)
        
        # Generate final report
        print("\n📊 Final Performance Report:")
        health, error = await client.performance.get_system_health()
        if not error:
            print(f"Final System Status: {health.overall_status}")
            print(f"Total Alerts Generated: {len(health.active_alerts)}")
        
        # Get metrics summary
        metrics = await client.supabase.select(
            "performance_metrics",
            order_by="-timestamp",
            limit=1000
        )
        
        if metrics:
            print(f"Total Metrics Recorded: {len(metrics)}")
            
            # Calculate some basic statistics
            api_metrics = [m for m in metrics if m["name"] == "api_response_time"]
            if api_metrics:
                avg_response_time = sum(float(m["value"]) for m in api_metrics) / len(api_metrics)
                print(f"Average API Response Time: {avg_response_time:.1f}ms")
        
        print("🎉 Performance monitoring example completed!")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await client.disconnect()
        print("👋 Disconnected from Supabase")

if __name__ == "__main__":
    print("📊 Starting performance monitoring example...")
    print("This example demonstrates comprehensive system monitoring,")
    print("metrics collection, alerting, and health reporting.")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Example stopped by user")
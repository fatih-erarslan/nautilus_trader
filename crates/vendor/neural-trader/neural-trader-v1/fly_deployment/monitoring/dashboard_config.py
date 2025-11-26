"""
Monitoring Dashboard Configuration for GPU Trading Platform
Provides metrics collection and dashboard setup for fly.io deployment
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import psutil
import GPUtil
import asyncio
import aiohttp
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: str
    value: float
    labels: Dict[str, str]


class MetricsCollector:
    """Collects system and application metrics for monitoring"""
    
    def __init__(self):
        # Create custom registry for clean metrics
        self.registry = CollectorRegistry()
        
        # System metrics
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage', registry=self.registry)
        self.memory_usage = Gauge('memory_usage_percent', 'Memory usage percentage', registry=self.registry)
        self.disk_usage = Gauge('disk_usage_percent', 'Disk usage percentage', registry=self.registry)
        
        # GPU metrics
        self.gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization percentage', ['gpu_id', 'gpu_name'], registry=self.registry)
        self.gpu_memory_usage = Gauge('gpu_memory_usage_percent', 'GPU memory usage percentage', ['gpu_id', 'gpu_name'], registry=self.registry)
        self.gpu_temperature = Gauge('gpu_temperature_celsius', 'GPU temperature in Celsius', ['gpu_id', 'gpu_name'], registry=self.registry)
        self.gpu_power_usage = Gauge('gpu_power_usage_watts', 'GPU power usage in watts', ['gpu_id', 'gpu_name'], registry=self.registry)
        
        # Application metrics
        self.http_requests_total = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'], registry=self.registry)
        self.http_request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'], registry=self.registry)
        
        # Trading metrics
        self.trades_total = Counter('trades_total', 'Total number of trades', ['status', 'symbol'], registry=self.registry)
        self.trade_duration = Histogram('trade_duration_seconds', 'Trade execution duration', ['symbol'], registry=self.registry)
        self.portfolio_value = Gauge('portfolio_value_usd', 'Current portfolio value in USD', registry=self.registry)
        self.active_positions = Gauge('active_positions_count', 'Number of active trading positions', registry=self.registry)
        
        # Health metrics
        self.health_check_status = Gauge('health_check_status', 'Health check status (1=healthy, 0=unhealthy)', ['check_name'], registry=self.registry)
        self.health_check_duration = Histogram('health_check_duration_seconds', 'Health check duration', ['check_name'], registry=self.registry)
        
        # Performance metrics
        self.model_inference_duration = Histogram('model_inference_duration_seconds', 'Model inference duration', ['model_name'], registry=self.registry)
        self.model_accuracy = Gauge('model_accuracy_percent', 'Model prediction accuracy', ['model_name'], registry=self.registry)
        
        # Error metrics
        self.errors_total = Counter('errors_total', 'Total number of errors', ['error_type', 'component'], registry=self.registry)
        
        # Start background collection
        self._collection_task = None
        self._running = False
    
    async def start_collection(self, interval: int = 30):
        """Start background metrics collection"""
        self._running = True
        self._collection_task = asyncio.create_task(self._collect_loop(interval))
    
    async def stop_collection(self):
        """Stop background metrics collection"""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
    
    async def _collect_loop(self, interval: int):
        """Background loop to collect metrics"""
        while self._running:
            try:
                await self.collect_system_metrics()
                await self.collect_gpu_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error collecting metrics: {e}")
                await asyncio.sleep(interval)
    
    async def collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.disk_usage.set(disk_percent)
            
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
    
    async def collect_gpu_metrics(self):
        """Collect GPU metrics"""
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                labels = [str(i), gpu.name]
                
                self.gpu_utilization.labels(*labels).set(gpu.load * 100)
                self.gpu_memory_usage.labels(*labels).set((gpu.memoryUsed / gpu.memoryTotal) * 100)
                self.gpu_temperature.labels(*labels).set(gpu.temperature)
                
                # Power usage if available
                if hasattr(gpu, 'powerDraw'):
                    self.gpu_power_usage.labels(*labels).set(gpu.powerDraw)
                
        except Exception as e:
            print(f"Error collecting GPU metrics: {e}")
    
    def record_http_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        self.http_requests_total.labels(method, endpoint, str(status)).inc()
        self.http_request_duration.labels(method, endpoint).observe(duration)
    
    def record_trade(self, symbol: str, status: str, duration: float):
        """Record trading metrics"""
        self.trades_total.labels(status, symbol).inc()
        self.trade_duration.labels(symbol).observe(duration)
    
    def update_portfolio_value(self, value: float):
        """Update portfolio value metric"""
        self.portfolio_value.set(value)
    
    def update_active_positions(self, count: int):
        """Update active positions count"""
        self.active_positions.set(count)
    
    def record_health_check(self, check_name: str, status: bool, duration: float):
        """Record health check metrics"""
        self.health_check_status.labels(check_name).set(1 if status else 0)
        self.health_check_duration.labels(check_name).observe(duration)
    
    def record_model_inference(self, model_name: str, duration: float, accuracy: Optional[float] = None):
        """Record model inference metrics"""
        self.model_inference_duration.labels(model_name).observe(duration)
        if accuracy is not None:
            self.model_accuracy.labels(model_name).set(accuracy)
    
    def record_error(self, error_type: str, component: str):
        """Record error metrics"""
        self.errors_total.labels(error_type, component).inc()
    
    def get_metrics(self) -> str:
        """Get Prometheus-formatted metrics"""
        return generate_latest(self.registry).decode('utf-8')


class DashboardConfig:
    """Configuration for monitoring dashboards"""
    
    def __init__(self):
        self.dashboards = {
            "system_overview": self.get_system_overview_config(),
            "gpu_monitoring": self.get_gpu_monitoring_config(),
            "trading_performance": self.get_trading_performance_config(),
            "application_health": self.get_application_health_config(),
            "error_tracking": self.get_error_tracking_config()
        }
    
    def get_system_overview_config(self) -> Dict[str, Any]:
        """System resource monitoring dashboard"""
        return {
            "title": "System Overview",
            "panels": [
                {
                    "title": "CPU Usage",
                    "type": "stat",
                    "targets": [{"expr": "cpu_usage_percent"}],
                    "thresholds": [{"value": 80, "color": "yellow"}, {"value": 90, "color": "red"}]
                },
                {
                    "title": "Memory Usage",
                    "type": "stat", 
                    "targets": [{"expr": "memory_usage_percent"}],
                    "thresholds": [{"value": 80, "color": "yellow"}, {"value": 90, "color": "red"}]
                },
                {
                    "title": "Disk Usage",
                    "type": "stat",
                    "targets": [{"expr": "disk_usage_percent"}],
                    "thresholds": [{"value": 80, "color": "yellow"}, {"value": 90, "color": "red"}]
                },
                {
                    "title": "System Resources Over Time",
                    "type": "graph",
                    "targets": [
                        {"expr": "cpu_usage_percent", "legendFormat": "CPU %"},
                        {"expr": "memory_usage_percent", "legendFormat": "Memory %"},
                        {"expr": "disk_usage_percent", "legendFormat": "Disk %"}
                    ]
                }
            ]
        }
    
    def get_gpu_monitoring_config(self) -> Dict[str, Any]:
        """GPU monitoring dashboard"""
        return {
            "title": "GPU Monitoring",
            "panels": [
                {
                    "title": "GPU Utilization",
                    "type": "stat",
                    "targets": [{"expr": "gpu_utilization_percent"}],
                    "thresholds": [{"value": 80, "color": "yellow"}, {"value": 95, "color": "red"}]
                },
                {
                    "title": "GPU Memory Usage",
                    "type": "stat",
                    "targets": [{"expr": "gpu_memory_usage_percent"}],
                    "thresholds": [{"value": 80, "color": "yellow"}, {"value": 90, "color": "red"}]
                },
                {
                    "title": "GPU Temperature",
                    "type": "stat",
                    "targets": [{"expr": "gpu_temperature_celsius"}],
                    "thresholds": [{"value": 80, "color": "yellow"}, {"value": 90, "color": "red"}]
                },
                {
                    "title": "GPU Metrics Over Time",
                    "type": "graph",
                    "targets": [
                        {"expr": "gpu_utilization_percent", "legendFormat": "GPU Utilization %"},
                        {"expr": "gpu_memory_usage_percent", "legendFormat": "GPU Memory %"},
                        {"expr": "gpu_temperature_celsius", "legendFormat": "GPU Temperature °C"}
                    ]
                },
                {
                    "title": "GPU Power Usage",
                    "type": "graph",
                    "targets": [{"expr": "gpu_power_usage_watts", "legendFormat": "Power (W)"}]
                }
            ]
        }
    
    def get_trading_performance_config(self) -> Dict[str, Any]:
        """Trading performance dashboard"""
        return {
            "title": "Trading Performance",
            "panels": [
                {
                    "title": "Total Trades",
                    "type": "stat",
                    "targets": [{"expr": "sum(trades_total)"}]
                },
                {
                    "title": "Portfolio Value",
                    "type": "stat",
                    "targets": [{"expr": "portfolio_value_usd"}],
                    "unit": "currencyUSD"
                },
                {
                    "title": "Active Positions",
                    "type": "stat",
                    "targets": [{"expr": "active_positions_count"}]
                },
                {
                    "title": "Trade Success Rate",
                    "type": "stat",
                    "targets": [{"expr": "rate(trades_total{status=\"success\"}[5m]) / rate(trades_total[5m]) * 100"}],
                    "unit": "percent"
                },
                {
                    "title": "Portfolio Value Over Time",
                    "type": "graph",
                    "targets": [{"expr": "portfolio_value_usd", "legendFormat": "Portfolio Value"}]
                },
                {
                    "title": "Trade Volume by Symbol",
                    "type": "graph",
                    "targets": [{"expr": "sum by (symbol) (rate(trades_total[5m]))", "legendFormat": "{{symbol}}"}]
                },
                {
                    "title": "Trade Duration",
                    "type": "graph",
                    "targets": [{"expr": "histogram_quantile(0.95, rate(trade_duration_seconds_bucket[5m]))", "legendFormat": "95th percentile"}]
                }
            ]
        }
    
    def get_application_health_config(self) -> Dict[str, Any]:
        """Application health monitoring dashboard"""
        return {
            "title": "Application Health",
            "panels": [
                {
                    "title": "Overall Health Status",
                    "type": "stat",
                    "targets": [{"expr": "min(health_check_status)"}],
                    "thresholds": [{"value": 0.5, "color": "red"}, {"value": 1, "color": "green"}]
                },
                {
                    "title": "Health Checks",
                    "type": "table",
                    "targets": [{"expr": "health_check_status", "format": "table"}]
                },
                {
                    "title": "HTTP Request Rate",
                    "type": "graph",
                    "targets": [{"expr": "sum(rate(http_requests_total[5m]))", "legendFormat": "Requests/sec"}]
                },
                {
                    "title": "HTTP Response Times",
                    "type": "graph",
                    "targets": [
                        {"expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))", "legendFormat": "50th percentile"},
                        {"expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))", "legendFormat": "95th percentile"},
                        {"expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))", "legendFormat": "99th percentile"}
                    ]
                },
                {
                    "title": "Model Inference Performance",
                    "type": "graph",
                    "targets": [{"expr": "histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[5m]))", "legendFormat": "{{model_name}}"}]
                }
            ]
        }
    
    def get_error_tracking_config(self) -> Dict[str, Any]:
        """Error tracking dashboard"""
        return {
            "title": "Error Tracking",
            "panels": [
                {
                    "title": "Error Rate",
                    "type": "stat",
                    "targets": [{"expr": "sum(rate(errors_total[5m]))"}],
                    "thresholds": [{"value": 0.1, "color": "yellow"}, {"value": 1, "color": "red"}]
                },
                {
                    "title": "Errors by Type",
                    "type": "pie",
                    "targets": [{"expr": "sum by (error_type) (rate(errors_total[5m]))", "legendFormat": "{{error_type}}"}]
                },
                {
                    "title": "Errors by Component",
                    "type": "pie", 
                    "targets": [{"expr": "sum by (component) (rate(errors_total[5m]))", "legendFormat": "{{component}}"}]
                },
                {
                    "title": "Error Rate Over Time",
                    "type": "graph",
                    "targets": [{"expr": "sum by (error_type) (rate(errors_total[5m]))", "legendFormat": "{{error_type}}"}]
                },
                {
                    "title": "HTTP Error Rates",
                    "type": "graph",
                    "targets": [
                        {"expr": "sum(rate(http_requests_total{status=~\"4..\"}[5m]))", "legendFormat": "4xx errors"},
                        {"expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m]))", "legendFormat": "5xx errors"}
                    ]
                }
            ]
        }
    
    def export_grafana_dashboard(self, dashboard_name: str, output_dir: str = "/app/monitoring/dashboards"):
        """Export dashboard configuration for Grafana"""
        if dashboard_name not in self.dashboards:
            raise ValueError(f"Dashboard {dashboard_name} not found")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        dashboard = self.dashboards[dashboard_name]
        grafana_dashboard = {
            "dashboard": {
                "id": None,
                "title": dashboard["title"],
                "tags": ["ruvtrade", "gpu", "trading"],
                "timezone": "UTC",
                "panels": self._convert_panels_to_grafana(dashboard["panels"]),
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }
        
        output_file = os.path.join(output_dir, f"{dashboard_name}.json")
        with open(output_file, 'w') as f:
            json.dump(grafana_dashboard, f, indent=2)
        
        return output_file
    
    def _convert_panels_to_grafana(self, panels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert internal panel format to Grafana format"""
        grafana_panels = []
        
        for i, panel in enumerate(panels):
            grafana_panel = {
                "id": i + 1,
                "title": panel["title"],
                "type": panel["type"],
                "gridPos": {"h": 8, "w": 12, "x": (i % 2) * 12, "y": (i // 2) * 8},
                "targets": [
                    {
                        "expr": target["expr"],
                        "legendFormat": target.get("legendFormat", ""),
                        "refId": chr(65 + j)  # A, B, C, etc.
                    }
                    for j, target in enumerate(panel["targets"])
                ]
            }
            
            # Add panel-specific configuration
            if panel["type"] == "stat":
                grafana_panel["fieldConfig"] = {
                    "defaults": {
                        "thresholds": {
                            "steps": [{"color": "green", "value": None}] + 
                                    [{"color": t["color"], "value": t["value"]} for t in panel.get("thresholds", [])]
                        }
                    }
                }
            
            grafana_panels.append(grafana_panel)
        
        return grafana_panels
    
    def export_all_dashboards(self, output_dir: str = "/app/monitoring/dashboards"):
        """Export all dashboards to Grafana format"""
        exported_files = []
        for dashboard_name in self.dashboards.keys():
            file_path = self.export_grafana_dashboard(dashboard_name, output_dir)
            exported_files.append(file_path)
        
        return exported_files


class AlertManager:
    """Manages alerts and notifications for monitoring"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv('ALERT_WEBHOOK_URL')
        self.alert_rules = self._get_default_alert_rules()
    
    def _get_default_alert_rules(self) -> List[Dict[str, Any]]:
        """Get default alert rules"""
        return [
            {
                "name": "HighCPUUsage",
                "condition": "cpu_usage_percent > 90",
                "duration": "5m",
                "severity": "warning",
                "message": "CPU usage is above 90% for 5 minutes"
            },
            {
                "name": "HighMemoryUsage", 
                "condition": "memory_usage_percent > 90",
                "duration": "5m",
                "severity": "warning",
                "message": "Memory usage is above 90% for 5 minutes"
            },
            {
                "name": "GPUOverheating",
                "condition": "gpu_temperature_celsius > 85",
                "duration": "2m",
                "severity": "critical",
                "message": "GPU temperature is above 85°C"
            },
            {
                "name": "HealthCheckFailed",
                "condition": "health_check_status == 0",
                "duration": "1m",
                "severity": "critical",
                "message": "Health check failed"
            },
            {
                "name": "HighErrorRate",
                "condition": "sum(rate(errors_total[5m])) > 10",
                "duration": "2m",
                "severity": "warning",
                "message": "Error rate is above 10 errors per minute"
            }
        ]
    
    async def send_alert(self, alert_name: str, message: str, severity: str = "warning"):
        """Send alert notification"""
        if not self.webhook_url:
            print(f"ALERT [{severity.upper()}] {alert_name}: {message}")
            return
        
        payload = {
            "alert_name": alert_name,
            "message": message,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "ruvtrade-gpu-platform"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        print(f"Alert sent successfully: {alert_name}")
                    else:
                        print(f"Failed to send alert: {response.status}")
        except Exception as e:
            print(f"Error sending alert: {e}")


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Initialize metrics collector
        collector = MetricsCollector()
        await collector.start_collection(interval=10)
        
        # Initialize dashboard config
        dashboard_config = DashboardConfig()
        
        # Export dashboards
        exported = dashboard_config.export_all_dashboards("./dashboards")
        print(f"Exported dashboards: {exported}")
        
        # Run for a short time to collect some metrics
        await asyncio.sleep(30)
        
        # Get metrics
        metrics = collector.get_metrics()
        print("Sample metrics:")
        print(metrics[:500] + "..." if len(metrics) > 500 else metrics)
        
        await collector.stop_collection()
    
    asyncio.run(main())
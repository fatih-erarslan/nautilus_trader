"""
Performance Monitoring Dashboard for Trading APIs

Real-time monitoring and visualization of:
- API latency metrics
- Throughput performance
- Error rates and availability
- System health indicators
- Historical performance trends
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import threading
from collections import deque, defaultdict
import numpy as np
import logging

# Try to import optional visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.dates import DateFormatter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric measurement point"""
    timestamp: datetime
    value: float
    api_name: str
    metric_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIStatus:
    """Current status of an API"""
    name: str
    is_healthy: bool
    avg_latency_us: float
    p95_latency_us: float
    p99_latency_us: float
    success_rate: float
    error_count: int
    requests_per_second: float
    last_updated: datetime


class PerformanceMonitor:
    """
    Real-time performance monitoring for trading APIs
    """
    
    def __init__(self, 
                 apis: Dict[str, Any],
                 history_minutes: int = 60,
                 update_interval: float = 1.0):
        """
        Initialize performance monitor
        
        Args:
            apis: Dictionary of API instances
            history_minutes: Minutes of history to keep
            update_interval: Update interval in seconds
        """
        self.apis = apis
        self.history_minutes = history_minutes
        self.update_interval = update_interval
        
        # Metrics storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=int(history_minutes * 60 / update_interval)))
        self.current_status: Dict[str, APIStatus] = {}
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Performance thresholds
        self.latency_threshold_us = 5000  # 5ms
        self.error_rate_threshold = 0.05  # 5%
        self.success_rate_threshold = 0.95  # 95%
        
        # Alerts
        self.alerts: List[Dict[str, Any]] = []
        self.alert_callbacks: List[callable] = []
        
        # Initialize status
        for api_name in self.apis:
            self.current_status[api_name] = APIStatus(
                name=api_name,
                is_healthy=True,
                avg_latency_us=0.0,
                p95_latency_us=0.0,
                p99_latency_us=0.0,
                success_rate=1.0,
                error_count=0,
                requests_per_second=0.0,
                last_updated=datetime.now()
            )
    
    async def start_monitoring(self):
        """Start real-time monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect metrics from all APIs
                await self._collect_metrics()
                
                # Update status
                self._update_status()
                
                # Check for alerts
                self._check_alerts()
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)
    
    async def _collect_metrics(self):
        """Collect performance metrics from all APIs"""
        tasks = []
        
        for api_name, api_instance in self.apis.items():
            task = self._collect_single_api_metrics(api_name, api_instance)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _collect_single_api_metrics(self, api_name: str, api_instance: Any):
        """Collect metrics from a single API"""
        timestamp = datetime.now()
        
        # Measure latency
        start_time = time.perf_counter()
        success = True
        
        try:
            # Perform lightweight health check
            if hasattr(api_instance, 'health_check'):
                await api_instance.health_check()
            elif hasattr(api_instance, 'get_server_time'):
                await api_instance.get_server_time()
            else:
                # Fallback to account info
                await api_instance.get_account_info()
                
        except Exception as e:
            success = False
            logger.debug(f"Health check failed for {api_name}: {e}")
        
        latency_us = (time.perf_counter() - start_time) * 1_000_000
        
        # Store latency metric
        self.metrics_history[f"{api_name}_latency"].append(
            MetricPoint(
                timestamp=timestamp,
                value=latency_us,
                api_name=api_name,
                metric_type="latency"
            )
        )
        
        # Store success metric
        self.metrics_history[f"{api_name}_success"].append(
            MetricPoint(
                timestamp=timestamp,
                value=1.0 if success else 0.0,
                api_name=api_name,
                metric_type="success"
            )
        )
    
    def _update_status(self):
        """Update current status for all APIs"""
        for api_name in self.apis:
            # Get recent metrics
            latency_metrics = [
                m for m in self.metrics_history[f"{api_name}_latency"]
                if (datetime.now() - m.timestamp).total_seconds() < 60
            ]
            
            success_metrics = [
                m for m in self.metrics_history[f"{api_name}_success"]
                if (datetime.now() - m.timestamp).total_seconds() < 60
            ]
            
            if latency_metrics:
                latencies = [m.value for m in latency_metrics]
                avg_latency = np.mean(latencies)
                p95_latency = np.percentile(latencies, 95)
                p99_latency = np.percentile(latencies, 99)
            else:
                avg_latency = p95_latency = p99_latency = 0.0
            
            if success_metrics:
                success_rate = np.mean([m.value for m in success_metrics])
                error_count = sum(1 for m in success_metrics if m.value == 0)
                requests_per_second = len(success_metrics) / 60.0
            else:
                success_rate = 1.0
                error_count = 0
                requests_per_second = 0.0
            
            # Determine health status
            is_healthy = (
                avg_latency < self.latency_threshold_us and
                success_rate > self.success_rate_threshold and
                error_count < 5
            )
            
            # Update status
            self.current_status[api_name] = APIStatus(
                name=api_name,
                is_healthy=is_healthy,
                avg_latency_us=avg_latency,
                p95_latency_us=p95_latency,
                p99_latency_us=p99_latency,
                success_rate=success_rate,
                error_count=error_count,
                requests_per_second=requests_per_second,
                last_updated=datetime.now()
            )
    
    def _check_alerts(self):
        """Check for alert conditions"""
        for api_name, status in self.current_status.items():
            # High latency alert
            if status.avg_latency_us > self.latency_threshold_us:
                self._create_alert(
                    api_name,
                    "high_latency",
                    f"Average latency {status.avg_latency_us:.0f}μs exceeds threshold",
                    "warning"
                )
            
            # Low success rate alert
            if status.success_rate < self.success_rate_threshold:
                self._create_alert(
                    api_name,
                    "low_success_rate",
                    f"Success rate {status.success_rate:.1%} below threshold",
                    "error"
                )
            
            # API unhealthy alert
            if not status.is_healthy:
                self._create_alert(
                    api_name,
                    "unhealthy",
                    f"API marked as unhealthy",
                    "error"
                )
    
    def _create_alert(self, api_name: str, alert_type: str, message: str, severity: str):
        """Create and process an alert"""
        alert = {
            'api_name': api_name,
            'alert_type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now(),
            'acknowledged': False
        }
        
        # Check if this alert already exists recently
        recent_alerts = [
            a for a in self.alerts
            if (datetime.now() - a['timestamp']).total_seconds() < 300 and
            a['api_name'] == api_name and a['alert_type'] == alert_type
        ]
        
        if not recent_alerts:
            self.alerts.append(alert)
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
            
            logger.warning(f"Alert: {api_name} - {message}")
    
    def get_current_status(self) -> Dict[str, APIStatus]:
        """Get current status of all APIs"""
        return self.current_status.copy()
    
    def get_metrics_history(self, 
                          api_name: str, 
                          metric_type: str,
                          minutes: int = 60) -> List[MetricPoint]:
        """Get historical metrics for an API"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        metrics = self.metrics_history[f"{api_name}_{metric_type}"]
        
        return [m for m in metrics if m.timestamp > cutoff_time]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all APIs"""
        summary = {
            'timestamp': datetime.now(),
            'total_apis': len(self.apis),
            'healthy_apis': sum(1 for s in self.current_status.values() if s.is_healthy),
            'avg_latency_us': np.mean([s.avg_latency_us for s in self.current_status.values()]),
            'min_latency_us': min([s.avg_latency_us for s in self.current_status.values()]),
            'max_latency_us': max([s.avg_latency_us for s in self.current_status.values()]),
            'avg_success_rate': np.mean([s.success_rate for s in self.current_status.values()]),
            'total_requests_per_second': sum([s.requests_per_second for s in self.current_status.values()]),
            'active_alerts': len([a for a in self.alerts if not a['acknowledged']]),
            'api_status': {name: status for name, status in self.current_status.items()}
        }
        
        return summary
    
    def acknowledge_alert(self, alert_index: int):
        """Acknowledge an alert"""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index]['acknowledged'] = True
    
    def add_alert_callback(self, callback: callable):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'current_status': {
                name: {
                    'name': status.name,
                    'is_healthy': status.is_healthy,
                    'avg_latency_us': status.avg_latency_us,
                    'p95_latency_us': status.p95_latency_us,
                    'p99_latency_us': status.p99_latency_us,
                    'success_rate': status.success_rate,
                    'error_count': status.error_count,
                    'requests_per_second': status.requests_per_second,
                    'last_updated': status.last_updated.isoformat()
                }
                for name, status in self.current_status.items()
            },
            'metrics_history': {
                key: [
                    {
                        'timestamp': point.timestamp.isoformat(),
                        'value': point.value,
                        'api_name': point.api_name,
                        'metric_type': point.metric_type,
                        'metadata': point.metadata
                    }
                    for point in deque_data
                ]
                for key, deque_data in self.metrics_history.items()
            },
            'alerts': [
                {
                    'api_name': alert['api_name'],
                    'alert_type': alert['alert_type'],
                    'message': alert['message'],
                    'severity': alert['severity'],
                    'timestamp': alert['timestamp'].isoformat(),
                    'acknowledged': alert['acknowledged']
                }
                for alert in self.alerts
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")


class DashboardVisualizer:
    """
    Visualization component for the monitoring dashboard
    """
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.fig = None
        self.axes = None
    
    def create_matplotlib_dashboard(self, figsize: tuple = (15, 10)):
        """Create matplotlib-based dashboard"""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for this visualization")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Trading APIs Performance Dashboard', fontsize=16)
        
        self.fig = fig
        self.axes = axes.flatten()
        
        return fig, axes
    
    def update_matplotlib_dashboard(self):
        """Update matplotlib dashboard with current data"""
        if not self.fig or not self.axes:
            return
        
        # Clear previous plots
        for ax in self.axes:
            ax.clear()
        
        # Plot 1: Latency over time
        ax1 = self.axes[0]
        for api_name in self.monitor.apis:
            metrics = self.monitor.get_metrics_history(api_name, "latency", 30)
            if metrics:
                timestamps = [m.timestamp for m in metrics]
                latencies = [m.value / 1000 for m in metrics]  # Convert to ms
                ax1.plot(timestamps, latencies, label=api_name, marker='o', markersize=2)
        
        ax1.set_title('API Latency (Last 30 minutes)')
        ax1.set_ylabel('Latency (ms)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Success rate over time
        ax2 = self.axes[1]
        for api_name in self.monitor.apis:
            metrics = self.monitor.get_metrics_history(api_name, "success", 30)
            if metrics:
                timestamps = [m.timestamp for m in metrics]
                success_rates = [m.value * 100 for m in metrics]  # Convert to percentage
                ax2.plot(timestamps, success_rates, label=api_name, marker='o', markersize=2)
        
        ax2.set_title('Success Rate (Last 30 minutes)')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_ylim(0, 105)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Current status bar chart
        ax3 = self.axes[2]
        api_names = list(self.monitor.current_status.keys())
        latencies = [status.avg_latency_us / 1000 for status in self.monitor.current_status.values()]
        colors = ['green' if status.is_healthy else 'red' for status in self.monitor.current_status.values()]
        
        bars = ax3.bar(api_names, latencies, color=colors, alpha=0.7)
        ax3.set_title('Current Average Latency')
        ax3.set_ylabel('Latency (ms)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, latency in zip(bars, latencies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{latency:.1f}ms', ha='center', va='bottom')
        
        # Plot 4: Alert summary
        ax4 = self.axes[3]
        if self.monitor.alerts:
            alert_counts = defaultdict(int)
            for alert in self.monitor.alerts:
                if not alert['acknowledged']:
                    alert_counts[alert['severity']] += 1
            
            if alert_counts:
                severities = list(alert_counts.keys())
                counts = list(alert_counts.values())
                colors = {'error': 'red', 'warning': 'orange', 'info': 'blue'}
                bar_colors = [colors.get(s, 'gray') for s in severities]
                
                ax4.bar(severities, counts, color=bar_colors, alpha=0.7)
                ax4.set_title('Active Alerts by Severity')
                ax4.set_ylabel('Count')
            else:
                ax4.text(0.5, 0.5, 'No Active Alerts', ha='center', va='center', 
                        transform=ax4.transAxes, fontsize=14, color='green')
                ax4.set_title('Alert Status')
        else:
            ax4.text(0.5, 0.5, 'No Alerts', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14, color='green')
            ax4.set_title('Alert Status')
        
        # Format time axes
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        return self.fig
    
    def create_plotly_dashboard(self):
        """Create Plotly-based interactive dashboard"""
        if not HAS_PLOTLY:
            raise ImportError("plotly is required for this visualization")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('API Latency', 'Success Rate', 'Current Status', 'Alert Summary'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Latency over time
        for api_name in self.monitor.apis:
            metrics = self.monitor.get_metrics_history(api_name, "latency", 60)
            if metrics:
                timestamps = [m.timestamp for m in metrics]
                latencies = [m.value / 1000 for m in metrics]  # Convert to ms
                
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=latencies,
                        mode='lines+markers',
                        name=f'{api_name} Latency',
                        line=dict(width=2),
                        marker=dict(size=4)
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Success rate over time
        for api_name in self.monitor.apis:
            metrics = self.monitor.get_metrics_history(api_name, "success", 60)
            if metrics:
                timestamps = [m.timestamp for m in metrics]
                success_rates = [m.value * 100 for m in metrics]
                
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=success_rates,
                        mode='lines+markers',
                        name=f'{api_name} Success Rate',
                        line=dict(width=2),
                        marker=dict(size=4)
                    ),
                    row=1, col=2
                )
        
        # Plot 3: Current status
        api_names = list(self.monitor.current_status.keys())
        latencies = [status.avg_latency_us / 1000 for status in self.monitor.current_status.values()]
        colors = ['green' if status.is_healthy else 'red' for status in self.monitor.current_status.values()]
        
        fig.add_trace(
            go.Bar(
                x=api_names,
                y=latencies,
                name='Current Latency',
                marker_color=colors,
                text=[f'{l:.1f}ms' for l in latencies],
                textposition='auto',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Plot 4: Alert summary
        if self.monitor.alerts:
            alert_counts = defaultdict(int)
            for alert in self.monitor.alerts:
                if not alert['acknowledged']:
                    alert_counts[alert['severity']] += 1
            
            if alert_counts:
                severities = list(alert_counts.keys())
                counts = list(alert_counts.values())
                colors = {'error': 'red', 'warning': 'orange', 'info': 'blue'}
                bar_colors = [colors.get(s, 'gray') for s in severities]
                
                fig.add_trace(
                    go.Bar(
                        x=severities,
                        y=counts,
                        name='Active Alerts',
                        marker_color=bar_colors,
                        text=counts,
                        textposition='auto',
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title='Trading APIs Performance Dashboard',
            showlegend=True,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Success Rate (%)", row=1, col=2)
        fig.update_xaxes(title_text="API", row=2, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=2, col=1)
        fig.update_xaxes(title_text="Severity", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        return fig
    
    def generate_html_report(self, output_file: str):
        """Generate HTML report with dashboard"""
        if not HAS_PLOTLY:
            raise ImportError("plotly is required for HTML report generation")
        
        fig = self.create_plotly_dashboard()
        
        # Get current summary
        summary = self.monitor.get_performance_summary()
        
        # Generate HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading APIs Performance Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f0f0f0; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: white; border-radius: 3px; }}
                .healthy {{ color: green; }}
                .unhealthy {{ color: red; }}
                .warning {{ color: orange; }}
            </style>
        </head>
        <body>
            <h1>Trading APIs Performance Dashboard</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Performance Summary</h2>
                <div class="metric">
                    <strong>Total APIs:</strong> {summary['total_apis']}
                </div>
                <div class="metric">
                    <strong>Healthy APIs:</strong> 
                    <span class="{'healthy' if summary['healthy_apis'] == summary['total_apis'] else 'warning'}">
                        {summary['healthy_apis']}/{summary['total_apis']}
                    </span>
                </div>
                <div class="metric">
                    <strong>Average Latency:</strong> {summary['avg_latency_us']:.0f}μs
                </div>
                <div class="metric">
                    <strong>Success Rate:</strong> {summary['avg_success_rate']:.1%}
                </div>
                <div class="metric">
                    <strong>Active Alerts:</strong> 
                    <span class="{'healthy' if summary['active_alerts'] == 0 else 'warning'}">
                        {summary['active_alerts']}
                    </span>
                </div>
            </div>
            
            <div id="dashboard">
                {fig.to_html(include_plotlyjs='cdn', div_id='dashboard')}
            </div>
            
            <h2>API Status Details</h2>
            <table border="1" style="width: 100%; border-collapse: collapse;">
                <tr>
                    <th>API Name</th>
                    <th>Status</th>
                    <th>Avg Latency</th>
                    <th>P95 Latency</th>
                    <th>P99 Latency</th>
                    <th>Success Rate</th>
                    <th>RPS</th>
                    <th>Last Updated</th>
                </tr>
        """
        
        for api_name, status in summary['api_status'].items():
            health_class = 'healthy' if status.is_healthy else 'unhealthy'
            html_content += f"""
                <tr>
                    <td>{status.name}</td>
                    <td class="{health_class}">{'Healthy' if status.is_healthy else 'Unhealthy'}</td>
                    <td>{status.avg_latency_us:.0f}μs</td>
                    <td>{status.p95_latency_us:.0f}μs</td>
                    <td>{status.p99_latency_us:.0f}μs</td>
                    <td>{status.success_rate:.1%}</td>
                    <td>{status.requests_per_second:.1f}</td>
                    <td>{status.last_updated.strftime('%H:%M:%S')}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_file}")


class ConsoleMonitor:
    """
    Console-based monitoring interface
    """
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.running = False
    
    def start_console_monitoring(self, refresh_interval: int = 5):
        """Start console monitoring with periodic updates"""
        self.running = True
        
        def console_loop():
            while self.running:
                try:
                    self.print_dashboard()
                    time.sleep(refresh_interval)
                except KeyboardInterrupt:
                    self.running = False
                    break
        
        console_thread = threading.Thread(target=console_loop)
        console_thread.daemon = True
        console_thread.start()
        
        logger.info("Console monitoring started")
    
    def stop_console_monitoring(self):
        """Stop console monitoring"""
        self.running = False
    
    def print_dashboard(self):
        """Print dashboard to console"""
        # Clear screen (works on most terminals)
        print("\033[2J\033[H")
        
        # Header
        print("=" * 80)
        print("TRADING APIs PERFORMANCE DASHBOARD")
        print("=" * 80)
        print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Summary
        summary = self.monitor.get_performance_summary()
        print(f"APIs: {summary['healthy_apis']}/{summary['total_apis']} healthy | "
              f"Avg Latency: {summary['avg_latency_us']:.0f}μs | "
              f"Success Rate: {summary['avg_success_rate']:.1%} | "
              f"Alerts: {summary['active_alerts']}")
        print()
        
        # API Status Table
        print("API STATUS:")
        print("-" * 80)
        print(f"{'API Name':<15} {'Status':<10} {'Latency':<12} {'P95':<10} {'Success':<10} {'RPS':<8}")
        print("-" * 80)
        
        for api_name, status in self.monitor.current_status.items():
            status_str = "HEALTHY" if status.is_healthy else "UNHEALTHY"
            print(f"{api_name:<15} {status_str:<10} {status.avg_latency_us:>8.0f}μs "
                  f"{status.p95_latency_us:>8.0f}μs {status.success_rate:>7.1%} "
                  f"{status.requests_per_second:>6.1f}")
        
        print()
        
        # Recent Alerts
        recent_alerts = [a for a in self.monitor.alerts if not a['acknowledged']][-5:]
        if recent_alerts:
            print("RECENT ALERTS:")
            print("-" * 80)
            for alert in recent_alerts:
                severity_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(alert['severity'], "")
                print(f"{severity_icon} {alert['timestamp'].strftime('%H:%M:%S')} "
                      f"{alert['api_name']}: {alert['message']}")
        else:
            print("No active alerts ✅")
        
        print()
        print("Press Ctrl+C to stop monitoring")


# Example usage
async def main():
    # Mock APIs for demonstration
    class MockAPI:
        def __init__(self, name: str, latency_ms: float = 1.0, fail_rate: float = 0.0):
            self.name = name
            self.latency_ms = latency_ms
            self.fail_rate = fail_rate
        
        async def health_check(self):
            await asyncio.sleep(self.latency_ms / 1000)
            if np.random.random() < self.fail_rate:
                raise Exception(f"API {self.name} failed")
            return {"status": "ok"}
    
    # Create test APIs
    test_apis = {
        "primary": MockAPI("primary", 1.0, 0.02),
        "secondary": MockAPI("secondary", 3.0, 0.05),
        "backup": MockAPI("backup", 10.0, 0.1)
    }
    
    # Create monitor
    monitor = PerformanceMonitor(test_apis, update_interval=2.0)
    
    # Start monitoring
    await monitor.start_monitoring()
    
    # Start console monitoring
    console_monitor = ConsoleMonitor(monitor)
    console_monitor.start_console_monitoring()
    
    try:
        # Let it run for a while
        await asyncio.sleep(60)
        
        # Create visualizations
        visualizer = DashboardVisualizer(monitor)
        
        if HAS_PLOTLY:
            visualizer.generate_html_report("dashboard.html")
        
        if HAS_MATPLOTLIB:
            fig, axes = visualizer.create_matplotlib_dashboard()
            visualizer.update_matplotlib_dashboard()
            plt.savefig("dashboard.png")
            plt.show()
        
    finally:
        # Stop monitoring
        console_monitor.stop_console_monitoring()
        await monitor.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
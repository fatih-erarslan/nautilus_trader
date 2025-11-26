#!/usr/bin/env python3
"""
Interactive Performance Dashboard for AI News Trading Platform.

This module provides a real-time interactive dashboard for monitoring
performance metrics, trends, and system health.

Features:
- Real-time performance monitoring
- Historical trend analysis
- Component performance breakdown
- Optimization recommendations
- Interactive charts and visualizations
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import sys

# Dashboard framework imports (would use streamlit, dash, or similar in production)
try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit and Plotly not available. Dashboard will run in console mode.")

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from validation import PerformanceValidator, ValidationStatus
from reports import PerformanceSummaryGenerator, OptimizationRecommendationEngine


@dataclass
class DashboardMetric:
    """Dashboard metric data point"""
    name: str
    value: float
    unit: str
    target: float
    status: str
    timestamp: datetime
    category: str
    trend: Optional[str] = None


@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    refresh_interval: int = 30  # seconds
    max_history_points: int = 100
    enable_real_time: bool = True
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = None
    

class PerformanceDashboard:
    """Interactive performance monitoring dashboard"""
    
    def __init__(self, config: Optional[DashboardConfig] = None, output_dir: str = None):
        """Initialize performance dashboard
        
        Args:
            config: Dashboard configuration
            output_dir: Output directory for data storage
        """
        self.config = config or DashboardConfig()
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.performance_validator = PerformanceValidator(output_dir=str(self.output_dir))
        self.summary_generator = PerformanceSummaryGenerator()
        self.recommendation_engine = OptimizationRecommendationEngine()
        
        # Data storage
        self.metrics_history: Dict[str, List[DashboardMetric]] = {}
        self.current_metrics: Dict[str, DashboardMetric] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.recommendations: Dict[str, Any] = {}
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.last_update = None
        
        # Dashboard state
        self.dashboard_data = {
            'last_updated': None,
            'system_status': 'UNKNOWN',
            'total_metrics': 0,
            'healthy_metrics': 0,
            'warning_metrics': 0,
            'critical_metrics': 0
        }
    
    def start_real_time_monitoring(self):
        """Start real-time performance monitoring"""
        if self.config.enable_real_time and not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("Real-time monitoring started")
    
    def stop_real_time_monitoring(self):
        """Stop real-time performance monitoring"""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            self.logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Real-time monitoring loop"""
        while self.monitoring_active:
            try:
                # Run quick validation
                asyncio.run(self._collect_metrics())
                time.sleep(self.config.refresh_interval)
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.config.refresh_interval)
    
    async def _collect_metrics(self):
        """Collect current performance metrics"""
        try:
            # Run quick validation to get current metrics
            summary = await self.performance_validator.validate_all(quick_mode=True)
            
            # Convert results to dashboard metrics
            current_time = datetime.now()
            new_metrics = {}
            
            for result in self.performance_validator.results:
                metric = DashboardMetric(
                    name=result.test_name,
                    value=result.measured_value if result.measured_value is not None else 0,
                    unit=result.target.unit,
                    target=result.target.target_value,
                    status=result.status.value,
                    timestamp=current_time,
                    category=result.category
                )
                
                new_metrics[result.test_name] = metric
                
                # Add to history
                if result.test_name not in self.metrics_history:
                    self.metrics_history[result.test_name] = []
                
                self.metrics_history[result.test_name].append(metric)
                
                # Limit history size
                if len(self.metrics_history[result.test_name]) > self.config.max_history_points:
                    self.metrics_history[result.test_name] = self.metrics_history[result.test_name][-self.config.max_history_points:]
            
            self.current_metrics = new_metrics
            self.last_update = current_time
            
            # Update dashboard data
            self._update_dashboard_data(summary)
            
            # Check for alerts
            if self.config.enable_alerts:
                self._check_alerts()
            
        except Exception as e:
            self.logger.error(f"Metrics collection failed: {e}")
    
    def _update_dashboard_data(self, summary):
        """Update dashboard summary data"""
        self.dashboard_data = {
            'last_updated': datetime.now(),
            'system_status': summary.overall_status.value,
            'total_metrics': summary.total_tests,
            'healthy_metrics': summary.passed_tests,
            'warning_metrics': summary.warning_tests,
            'critical_metrics': summary.failed_tests + summary.error_tests
        }
    
    def _check_alerts(self):
        """Check for performance alerts"""
        current_time = datetime.now()
        
        for metric_name, metric in self.current_metrics.items():
            if metric.status in ['FAIL', 'ERROR']:
                alert = {
                    'timestamp': current_time,
                    'severity': 'HIGH' if metric.status == 'FAIL' else 'CRITICAL',
                    'metric': metric_name,
                    'message': f"{metric_name} failed: {metric.value} {metric.unit} (target: {metric.target} {metric.unit})",
                    'category': metric.category
                }
                
                # Avoid duplicate alerts
                if not any(a['metric'] == metric_name and 
                          (current_time - a['timestamp']).seconds < 300 
                          for a in self.alerts):
                    self.alerts.append(alert)
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = current_time - timedelta(hours=24)
        self.alerts = [a for a in self.alerts if a['timestamp'] > cutoff_time]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        if not self.current_metrics:
            return {'error': 'No metrics available'}
        
        summary = {
            'last_updated': self.last_update.isoformat() if self.last_update else None,
            'total_metrics': len(self.current_metrics),
            'metrics_by_status': {},
            'metrics_by_category': {},
            'system_health_score': 0
        }
        
        # Count by status
        for metric in self.current_metrics.values():
            status = metric.status
            summary['metrics_by_status'][status] = summary['metrics_by_status'].get(status, 0) + 1
        
        # Count by category
        for metric in self.current_metrics.values():
            category = metric.category
            summary['metrics_by_category'][category] = summary['metrics_by_category'].get(category, 0) + 1
        
        # Calculate health score
        total_metrics = len(self.current_metrics)
        healthy_metrics = summary['metrics_by_status'].get('PASS', 0)
        summary['system_health_score'] = (healthy_metrics / total_metrics * 100) if total_metrics > 0 else 0
        
        return summary
    
    def get_trend_data(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get trend data for a specific metric"""
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history[metric_name] 
            if m.timestamp > cutoff_time
        ]
        
        return [
            {
                'timestamp': m.timestamp.isoformat(),
                'value': m.value,
                'target': m.target,
                'status': m.status
            }
            for m in recent_metrics
        ]
    
    def get_performance_heatmap(self) -> Dict[str, Any]:
        """Generate performance heatmap data"""
        if not self.current_metrics:
            return {}
        
        categories = list(set(m.category for m in self.current_metrics.values()))
        heatmap_data = []
        
        for category in categories:
            category_metrics = [m for m in self.current_metrics.values() if m.category == category]
            
            for metric in category_metrics:
                # Calculate performance score (0-100)
                if metric.status == 'PASS':
                    score = 100
                elif metric.status == 'WARNING':
                    score = 60
                elif metric.status == 'FAIL':
                    score = 20
                else:
                    score = 0
                
                heatmap_data.append({
                    'category': category,
                    'metric': metric.name,
                    'score': score,
                    'value': metric.value,
                    'target': metric.target,
                    'status': metric.status
                })
        
        return {
            'data': heatmap_data,
            'categories': categories
        }
    
    async def run_console_dashboard(self):
        """Run dashboard in console mode"""
        print("="*80)
        print("AI NEWS TRADING PLATFORM - PERFORMANCE DASHBOARD")
        print("="*80)
        print("Running in console mode. Press Ctrl+C to exit.")
        print()
        
        try:
            self.start_real_time_monitoring()
            
            while True:
                # Clear screen (works on most terminals)
                print("\033[2J\033[H")
                
                # Header
                print("="*80)
                print("AI NEWS TRADING PLATFORM - PERFORMANCE DASHBOARD")
                print("="*80)
                
                if self.last_update:
                    print(f"Last Updated: {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    print("Last Updated: Never")
                
                print(f"System Status: {self.dashboard_data['system_status']}")
                print()
                
                # Metrics summary
                summary = self.get_metrics_summary()
                if 'error' not in summary:
                    print("METRICS SUMMARY:")
                    print(f"  Total Metrics: {summary['total_metrics']}")
                    print(f"  System Health: {summary['system_health_score']:.1f}%")
                    print()
                    
                    # Status breakdown
                    print("STATUS BREAKDOWN:")
                    for status, count in summary['metrics_by_status'].items():
                        icon = {'PASS': '✅', 'FAIL': '❌', 'WARNING': '⚠️', 'ERROR': '🔴'}.get(status, '❓')
                        print(f"  {icon} {status}: {count}")
                    print()
                    
                    # Category breakdown
                    print("CATEGORY BREAKDOWN:")
                    for category, count in summary['metrics_by_category'].items():
                        print(f"  {category.capitalize()}: {count} metrics")
                    print()
                
                # Current metrics
                if self.current_metrics:
                    print("CURRENT METRICS:")
                    for metric in sorted(self.current_metrics.values(), key=lambda x: x.category):
                        status_icon = {'PASS': '✅', 'FAIL': '❌', 'WARNING': '⚠️', 'ERROR': '🔴'}.get(metric.status, '❓')
                        print(f"  {status_icon} {metric.name}: {metric.value:.2f} {metric.unit} (target: {metric.target} {metric.unit})")
                    print()
                
                # Recent alerts
                if self.alerts:
                    recent_alerts = sorted(self.alerts, key=lambda x: x['timestamp'], reverse=True)[:5]
                    print("RECENT ALERTS:")
                    for alert in recent_alerts:
                        severity_icon = {'HIGH': '⚠️', 'CRITICAL': '🔴'}.get(alert['severity'], '❓')
                        time_str = alert['timestamp'].strftime('%H:%M:%S')
                        print(f"  {severity_icon} [{time_str}] {alert['message']}")
                    print()
                
                print("="*80)
                print("Press Ctrl+C to exit")
                
                # Wait for next update
                time.sleep(self.config.refresh_interval)
                
        except KeyboardInterrupt:
            print("\nDashboard stopped by user")
        finally:
            self.stop_real_time_monitoring()
    
    def create_streamlit_dashboard(self):
        """Create Streamlit dashboard"""
        if not STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit and Plotly are required for web dashboard")
        
        st.set_page_config(
            page_title="AI News Trading - Performance Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Dashboard title
        st.title("🏆 AI News Trading Platform - Performance Dashboard")
        st.markdown("Real-time performance monitoring and analytics")
        
        # Sidebar controls
        st.sidebar.header("Dashboard Controls")
        
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=self.config.enable_real_time)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 300, self.config.refresh_interval)
        
        if st.sidebar.button("Refresh Now"):
            asyncio.run(self._collect_metrics())
        
        if st.sidebar.button("Start Monitoring"):
            self.start_real_time_monitoring()
            st.sidebar.success("Monitoring started")
        
        if st.sidebar.button("Stop Monitoring"):
            self.stop_real_time_monitoring()
            st.sidebar.info("Monitoring stopped")
        
        # Main dashboard content
        if not self.current_metrics:
            st.warning("No performance data available. Click 'Refresh Now' to collect metrics.")
            return
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        summary = self.get_metrics_summary()
        
        with col1:
            st.metric(
                label="System Health",
                value=f"{summary['system_health_score']:.1f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Total Metrics",
                value=summary['total_metrics'],
                delta=None
            )
        
        with col3:
            healthy = summary['metrics_by_status'].get('PASS', 0)
            st.metric(
                label="Healthy Metrics",
                value=healthy,
                delta=None
            )
        
        with col4:
            failed = summary['metrics_by_status'].get('FAIL', 0) + summary['metrics_by_status'].get('ERROR', 0)
            st.metric(
                label="Failed Metrics",
                value=failed,
                delta=None
            )
        
        # Performance overview charts
        st.header("Performance Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Status pie chart
            status_data = summary['metrics_by_status']
            if status_data:
                fig_pie = px.pie(
                    values=list(status_data.values()),
                    names=list(status_data.keys()),
                    title="Metrics by Status",
                    color_discrete_map={
                        'PASS': '#28a745',
                        'FAIL': '#dc3545',
                        'WARNING': '#ffc107',
                        'ERROR': '#6c757d'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Category bar chart
            category_data = summary['metrics_by_category']
            if category_data:
                fig_bar = px.bar(
                    x=list(category_data.keys()),
                    y=list(category_data.values()),
                    title="Metrics by Category",
                    labels={'x': 'Category', 'y': 'Count'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Performance heatmap
        st.header("Performance Heatmap")
        heatmap_data = self.get_performance_heatmap()
        
        if heatmap_data and heatmap_data['data']:
            df = pd.DataFrame(heatmap_data['data'])
            
            # Create heatmap
            fig_heatmap = px.imshow(
                df.pivot(index='category', columns='metric', values='score'),
                title="Performance Heatmap (Higher is Better)",
                color_continuous_scale='RdYlGn',
                aspect='auto'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Detailed metrics table
        st.header("Detailed Metrics")
        
        metrics_data = []
        for metric in self.current_metrics.values():
            metrics_data.append({
                'Metric': metric.name,
                'Category': metric.category,
                'Current Value': f"{metric.value:.2f} {metric.unit}",
                'Target': f"{metric.target} {metric.unit}",
                'Status': metric.status,
                'Last Updated': metric.timestamp.strftime('%H:%M:%S')
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Color code the status column
        def color_status(val):
            color_map = {
                'PASS': 'background-color: #d4edda',
                'FAIL': 'background-color: #f8d7da',
                'WARNING': 'background-color: #fff3cd',
                'ERROR': 'background-color: #f5c6cb'
            }
            return color_map.get(val, '')
        
        styled_df = df_metrics.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Alerts section
        if self.alerts:
            st.header("Recent Alerts")
            
            alert_data = []
            for alert in sorted(self.alerts, key=lambda x: x['timestamp'], reverse=True)[:10]:
                alert_data.append({
                    'Time': alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Severity': alert['severity'],
                    'Metric': alert['metric'],
                    'Message': alert['message'],
                    'Category': alert['category']
                })
            
            df_alerts = pd.DataFrame(alert_data)
            st.dataframe(df_alerts, use_container_width=True)
        
        # Auto-refresh
        if auto_refresh and self.config.enable_real_time:
            time.sleep(refresh_interval)
            st.rerun()


def main():
    """Main entry point for performance dashboard"""
    parser = argparse.ArgumentParser(description="AI News Trading Performance Dashboard")
    parser.add_argument('--mode', choices=['console', 'web'], default='console',
                       help='Dashboard mode (console or web)')
    parser.add_argument('--refresh-interval', type=int, default=30,
                       help='Refresh interval in seconds')
    parser.add_argument('--output-dir', help='Output directory for data storage')
    parser.add_argument('--enable-alerts', action='store_true',
                       help='Enable performance alerts')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Configure dashboard
    config = DashboardConfig(
        refresh_interval=args.refresh_interval,
        enable_real_time=True,
        enable_alerts=args.enable_alerts
    )
    
    # Initialize dashboard
    dashboard = PerformanceDashboard(config=config, output_dir=args.output_dir)
    
    try:
        if args.mode == 'web' and STREAMLIT_AVAILABLE:
            print("Starting web dashboard...")
            print("Access the dashboard at: http://localhost:8501")
            dashboard.create_streamlit_dashboard()
        else:
            if args.mode == 'web' and not STREAMLIT_AVAILABLE:
                print("Web mode requires streamlit and plotly. Falling back to console mode.")
            print("Starting console dashboard...")
            asyncio.run(dashboard.run_console_dashboard())
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Dashboard failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    import argparse
    main()
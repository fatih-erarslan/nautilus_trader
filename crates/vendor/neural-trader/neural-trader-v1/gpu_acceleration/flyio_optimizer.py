"""
Fly.io GPU Optimization System
Specialized optimizations for fly.io GPU instances and infrastructure.
"""

import os
import json
import time
import logging
import threading
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import requests
import psutil
import cupy as cp
import numpy as np

from .flyio_gpu_config import get_gpu_config_manager, GPUType
from .gpu_monitor import get_gpu_monitor
from .gpu_memory_manager import get_gpu_memory_manager
from .cpu_fallback import get_fallback_manager
from .batch_optimizer import create_batch_processor
from .mixed_precision import create_mixed_precision_manager

logger = logging.getLogger(__name__)


class FlyRegion(Enum):
    """Fly.io regions with GPU support."""
    ORD = "ord"  # Chicago
    IAD = "iad"  # Washington DC
    LAX = "lax"  # Los Angeles
    SJC = "sjc"  # San Jose
    DFW = "dfw"  # Dallas
    YYZ = "yyz"  # Toronto
    LHR = "lhr"  # London
    AMS = "ams"  # Amsterdam
    NRT = "nrt"  # Tokyo
    SYD = "syd"  # Sydney


@dataclass
class FlyInstanceMetrics:
    """Fly.io instance-specific metrics."""
    instance_id: str
    region: str
    machine_type: str
    gpu_type: str
    uptime_seconds: float
    network_latency_ms: float
    disk_io_rate: float
    cpu_utilization: float
    memory_utilization: float
    network_rx_bytes: int
    network_tx_bytes: int
    cost_per_hour: float
    performance_score: float
    timestamp: float


@dataclass
class FlyOptimizationConfig:
    """Configuration for fly.io optimizations."""
    # Instance management
    target_region: FlyRegion = FlyRegion.ORD
    preferred_gpu_types: List[GPUType] = field(default_factory=lambda: [
        GPUType.A100_40GB, GPUType.V100_32GB, GPUType.RTX_A6000
    ])
    
    # Auto-scaling
    enable_auto_scaling: bool = True
    min_instances: int = 1
    max_instances: int = 5
    scale_up_threshold: float = 0.8  # GPU utilization
    scale_down_threshold: float = 0.3
    scale_cooldown_minutes: int = 5
    
    # Cost optimization
    enable_cost_optimization: bool = True
    max_cost_per_hour: float = 10.0
    prefer_spot_instances: bool = True
    auto_shutdown_idle_minutes: int = 30
    
    # Performance optimization
    enable_performance_tuning: bool = True
    optimize_for_latency: bool = False
    enable_multi_region: bool = False
    
    # Health monitoring
    health_check_interval: int = 30
    restart_unhealthy_instances: bool = True
    alert_on_performance_degradation: bool = True


class FlyMetadataService:
    """Interface to fly.io metadata service."""
    
    def __init__(self):
        """Initialize metadata service."""
        self.metadata_url = "http://169.254.169.254"
        self.instance_info = None
        
    def get_instance_info(self) -> Dict[str, Any]:
        """Get current instance information."""
        if self.instance_info is None:
            self.instance_info = self._fetch_instance_info()
        return self.instance_info
        
    def _fetch_instance_info(self) -> Dict[str, Any]:
        """Fetch instance information from metadata service."""
        try:
            # Try to get fly.io metadata
            response = requests.get(f"{self.metadata_url}/metadata/v1/instance", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
            
        # Fallback to environment variables
        return {
            'id': os.environ.get('FLY_MACHINE_ID', 'unknown'),
            'region': os.environ.get('FLY_REGION', 'ord'),
            'app_name': os.environ.get('FLY_APP_NAME', 'ai-news-trader'),
            'machine_type': 'performance-8x',  # Default assumption
            'gpu_type': 'a100-40gb'  # Default assumption
        }
        
    def get_region_latency(self, target_region: str) -> float:
        """Get network latency to target region."""
        try:
            # Simple ping test (would be more sophisticated in practice)
            result = subprocess.run([
                'ping', '-c', '3', f'{target_region}.fly.dev'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Parse ping output for average latency
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'avg' in line:
                        # Extract average from: min/avg/max/mdev = ...
                        parts = line.split('=')[1].strip().split('/')
                        return float(parts[1])
                        
        except:
            pass
            
        return 100.0  # Default high latency
        
    def get_cost_info(self) -> Dict[str, float]:
        """Get current instance cost information."""
        instance_info = self.get_instance_info()
        
        # Fly.io pricing (as of 2024 - would need API for real-time)
        pricing_map = {
            'a100-40gb': 7.50,  # Per hour
            'a100-80gb': 10.00,
            'v100-32gb': 4.00,
            'v100-16gb': 3.00,
            'rtx-a6000': 2.50,
            'rtx-4090': 2.00,
            'rtx-3090': 1.50
        }
        
        gpu_type = instance_info.get('gpu_type', 'a100-40gb')
        base_cost = pricing_map.get(gpu_type, 5.0)
        
        return {
            'base_cost_per_hour': base_cost,
            'current_cost_per_hour': base_cost,  # Would include usage-based costs
            'estimated_monthly': base_cost * 24 * 30
        }


class FlyHealthChecker:
    """Monitors fly.io specific health metrics."""
    
    def __init__(self, config: FlyOptimizationConfig):
        """Initialize health checker."""
        self.config = config
        self.metadata_service = FlyMetadataService()
        self.last_health_check = 0
        
    def get_instance_health(self) -> Dict[str, Any]:
        """Get comprehensive instance health."""
        current_time = time.time()
        
        # Rate limit health checks
        if current_time - self.last_health_check < self.config.health_check_interval:
            return {'status': 'cached'}
            
        self.last_health_check = current_time
        
        # Get basic system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get network stats
        network = psutil.net_io_counters()
        
        # Get GPU metrics
        gpu_monitor = get_gpu_monitor()
        gpu_metrics = gpu_monitor.get_current_metrics()
        
        # Get instance info
        instance_info = self.metadata_service.get_instance_info()
        cost_info = self.metadata_service.get_cost_info()
        
        health = {
            'timestamp': current_time,
            'instance_id': instance_info.get('id', 'unknown'),
            'region': instance_info.get('region', 'unknown'),
            'uptime': self._get_uptime(),
            'cpu_utilization': cpu_percent,
            'memory_utilization': memory.percent,
            'disk_utilization': (disk.used / disk.total) * 100,
            'network_rx_bytes': network.bytes_recv,
            'network_tx_bytes': network.bytes_sent,
            'gpu_healthy': gpu_metrics is not None,
            'gpu_utilization': gpu_metrics.gpu_utilization if gpu_metrics else 0.0,
            'gpu_memory_utilization': gpu_metrics.memory_utilization if gpu_metrics else 0.0,
            'gpu_temperature': gpu_metrics.temperature_c if gpu_metrics else 0.0,
            'cost_per_hour': cost_info['current_cost_per_hour'],
            'performance_score': self._calculate_performance_score(),
            'health_status': 'healthy'  # Will be updated based on checks
        }
        
        # Determine health status
        health['health_status'] = self._determine_health_status(health)
        
        return health
        
    def _get_uptime(self) -> float:
        """Get instance uptime in seconds."""
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.readline().split()[0])
                return uptime_seconds
        except:
            return 0.0
            
    def _calculate_performance_score(self) -> float:
        """Calculate performance score (0-100)."""
        gpu_monitor = get_gpu_monitor()
        gpu_metrics = gpu_monitor.get_current_metrics()
        
        if not gpu_metrics:
            return 50.0  # Neutral score if no GPU metrics
            
        # Score based on utilization and temperature
        utilization_score = min(gpu_metrics.gpu_utilization, 90) / 90 * 50
        
        # Temperature score (lower is better)
        temp_score = max(0, (85 - gpu_metrics.temperature_c) / 85) * 30
        
        # Throughput score
        throughput_score = min(gpu_metrics.throughput_ops_sec / 10000, 1.0) * 20
        
        return utilization_score + temp_score + throughput_score
        
    def _determine_health_status(self, health: Dict[str, Any]) -> str:
        """Determine overall health status."""
        if health['gpu_temperature'] > 80:
            return 'unhealthy'
        elif health['gpu_utilization'] > 95:
            return 'degraded'
        elif health['memory_utilization'] > 90:
            return 'degraded'
        elif health['performance_score'] < 30:
            return 'degraded'
        else:
            return 'healthy'


class FlyAutoScaler:
    """Automatic scaling for fly.io instances."""
    
    def __init__(self, config: FlyOptimizationConfig):
        """Initialize auto scaler."""
        self.config = config
        self.last_scale_action = 0
        self.current_instances = 1
        self.scaling_lock = threading.Lock()
        
    def should_scale_up(self, metrics: Dict[str, Any]) -> bool:
        """Determine if scaling up is needed."""
        if not self.config.enable_auto_scaling:
            return False
            
        if self.current_instances >= self.config.max_instances:
            return False
            
        # Check cooldown
        cooldown_seconds = self.config.scale_cooldown_minutes * 60
        if time.time() - self.last_scale_action < cooldown_seconds:
            return False
            
        # Check utilization threshold
        return metrics.get('gpu_utilization', 0) > self.config.scale_up_threshold * 100
        
    def should_scale_down(self, metrics: Dict[str, Any]) -> bool:
        """Determine if scaling down is needed."""
        if not self.config.enable_auto_scaling:
            return False
            
        if self.current_instances <= self.config.min_instances:
            return False
            
        # Check cooldown
        cooldown_seconds = self.config.scale_cooldown_minutes * 60
        if time.time() - self.last_scale_action < cooldown_seconds:
            return False
            
        # Check utilization threshold
        return metrics.get('gpu_utilization', 0) < self.config.scale_down_threshold * 100
        
    def scale_up(self) -> bool:
        """Scale up instances."""
        with self.scaling_lock:
            if not self.should_scale_up({'gpu_utilization': 90}):  # Force check
                return False
                
            logger.info(f"Scaling up from {self.current_instances} to {self.current_instances + 1} instances")
            
            # In practice, would call fly.io API
            success = self._call_fly_api('scale_up')
            
            if success:
                self.current_instances += 1
                self.last_scale_action = time.time()
                
            return success
            
    def scale_down(self) -> bool:
        """Scale down instances."""
        with self.scaling_lock:
            if not self.should_scale_down({'gpu_utilization': 20}):  # Force check
                return False
                
            logger.info(f"Scaling down from {self.current_instances} to {self.current_instances - 1} instances")
            
            # In practice, would call fly.io API
            success = self._call_fly_api('scale_down')
            
            if success:
                self.current_instances -= 1
                self.last_scale_action = time.time()
                
            return success
            
    def _call_fly_api(self, action: str) -> bool:
        """Call fly.io API for scaling (mock implementation)."""
        # In practice, would use flyctl or fly.io API
        logger.info(f"Mock fly.io API call: {action}")
        return True


class FlyCostOptimizer:
    """Cost optimization for fly.io deployment."""
    
    def __init__(self, config: FlyOptimizationConfig):
        """Initialize cost optimizer."""
        self.config = config
        self.cost_history = []
        self.idle_start_time = None
        
    def track_costs(self, metrics: Dict[str, Any]):
        """Track cost metrics."""
        cost_entry = {
            'timestamp': time.time(),
            'cost_per_hour': metrics.get('cost_per_hour', 0.0),
            'gpu_utilization': metrics.get('gpu_utilization', 0.0),
            'performance_score': metrics.get('performance_score', 0.0)
        }
        
        self.cost_history.append(cost_entry)
        
        # Keep only last 24 hours of data
        cutoff_time = time.time() - (24 * 3600)
        self.cost_history = [c for c in self.cost_history if c['timestamp'] > cutoff_time]
        
    def should_shutdown_idle(self, metrics: Dict[str, Any]) -> bool:
        """Check if instance should be shutdown due to idleness."""
        if not self.config.enable_cost_optimization:
            return False
            
        gpu_utilization = metrics.get('gpu_utilization', 0.0)
        
        # Track idle time
        if gpu_utilization < 5.0:  # Consider idle if < 5% utilization
            if self.idle_start_time is None:
                self.idle_start_time = time.time()
            else:
                idle_duration = time.time() - self.idle_start_time
                if idle_duration > (self.config.auto_shutdown_idle_minutes * 60):
                    return True
        else:
            self.idle_start_time = None
            
        return False
        
    def get_cost_analysis(self) -> Dict[str, Any]:
        """Get cost analysis and recommendations."""
        if not self.cost_history:
            return {}
            
        recent_costs = self.cost_history[-100:]  # Last 100 entries
        
        total_cost = sum(c['cost_per_hour'] for c in recent_costs) / len(recent_costs)
        avg_utilization = sum(c['gpu_utilization'] for c in recent_costs) / len(recent_costs)
        avg_performance = sum(c['performance_score'] for c in recent_costs) / len(recent_costs)
        
        # Calculate efficiency metrics
        cost_per_performance = total_cost / max(avg_performance, 1.0)
        utilization_efficiency = avg_utilization / 100.0
        
        recommendations = []
        if avg_utilization < 30:
            recommendations.append("Consider using smaller GPU instance")
        if cost_per_performance > 0.5:
            recommendations.append("Performance per dollar is low")
        if utilization_efficiency < 0.5:
            recommendations.append("GPU underutilized - consider workload optimization")
            
        return {
            'avg_cost_per_hour': total_cost,
            'avg_utilization': avg_utilization,
            'avg_performance': avg_performance,
            'cost_per_performance': cost_per_performance,
            'utilization_efficiency': utilization_efficiency,
            'recommendations': recommendations,
            'projected_monthly_cost': total_cost * 24 * 30
        }


class FlyGPUOptimizer:
    """Main fly.io GPU optimization coordinator."""
    
    def __init__(self, config: Optional[FlyOptimizationConfig] = None):
        """Initialize fly.io GPU optimizer."""
        self.config = config or FlyOptimizationConfig()
        
        # Initialize components
        self.metadata_service = FlyMetadataService()
        self.health_checker = FlyHealthChecker(self.config)
        self.auto_scaler = FlyAutoScaler(self.config)
        self.cost_optimizer = FlyCostOptimizer(self.config)
        
        # Get GPU optimization components
        self.gpu_config_manager = get_gpu_config_manager()
        self.gpu_monitor = get_gpu_monitor()
        self.memory_manager = get_gpu_memory_manager()
        self.fallback_manager = get_fallback_manager()
        
        # Initialize optimized components
        self.batch_processor = None
        self.mixed_precision_manager = None
        
        # Monitoring state
        self.is_optimizing = False
        self.optimization_thread = None
        self.performance_history = []
        
    def start_optimization(self):
        """Start fly.io optimization."""
        if self.is_optimizing:
            return
            
        logger.info("Starting fly.io GPU optimization")
        
        # Initialize GPU monitoring
        if not self.gpu_monitor.is_monitoring:
            self.gpu_monitor.start_monitoring()
            
        # Initialize optimized components
        self._initialize_optimized_components()
        
        # Start optimization loop
        self.is_optimizing = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        
        logger.info("Fly.io GPU optimization started")
        
    def stop_optimization(self):
        """Stop fly.io optimization."""
        if not self.is_optimizing:
            return
            
        logger.info("Stopping fly.io GPU optimization")
        
        self.is_optimizing = False
        
        if self.optimization_thread:
            self.optimization_thread.join(timeout=10.0)
            
        logger.info("Fly.io GPU optimization stopped")
        
    def _initialize_optimized_components(self):
        """Initialize optimized GPU components."""
        try:
            # Initialize batch processor with adaptive strategy
            self.batch_processor = create_batch_processor("adaptive")
            
            # Initialize mixed precision manager
            precision_mode = self.gpu_config_manager.config.precision_mode.value
            self.mixed_precision_manager = create_mixed_precision_manager(precision_mode)
            
            logger.info("Optimized GPU components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimized components: {str(e)}")
            
    def _optimization_loop(self):
        """Main optimization loop."""
        while self.is_optimizing:
            try:
                # Get current health metrics
                health_metrics = self.health_checker.get_instance_health()
                
                # Record performance metrics
                self.performance_history.append(health_metrics)
                if len(self.performance_history) > 1000:
                    self.performance_history.pop(0)
                    
                # Track costs
                self.cost_optimizer.track_costs(health_metrics)
                
                # Check for scaling needs
                if self.auto_scaler.should_scale_up(health_metrics):
                    self.auto_scaler.scale_up()
                elif self.auto_scaler.should_scale_down(health_metrics):
                    self.auto_scaler.scale_down()
                    
                # Check for idle shutdown
                if self.cost_optimizer.should_shutdown_idle(health_metrics):
                    logger.warning("Instance idle for too long, considering shutdown")
                    # In practice, would trigger graceful shutdown
                    
                # Optimize GPU memory if needed
                memory_info = self.memory_manager.get_memory_info()
                if memory_info['overall_utilization'] > 0.8:
                    self.memory_manager.optimize_memory()
                    
                # Log optimization status
                if len(self.performance_history) % 20 == 0:  # Every 10 minutes at 30s intervals
                    self._log_optimization_status(health_metrics)
                    
                # Sleep until next optimization cycle
                time.sleep(30)  # 30 second intervals
                
            except Exception as e:
                logger.error(f"Optimization loop error: {str(e)}")
                time.sleep(30)
                
    def _log_optimization_status(self, health_metrics: Dict[str, Any]):
        """Log current optimization status."""
        cost_analysis = self.cost_optimizer.get_cost_analysis()
        
        status = {
            'gpu_utilization': health_metrics.get('gpu_utilization', 0.0),
            'memory_utilization': health_metrics.get('memory_utilization', 0.0),
            'performance_score': health_metrics.get('performance_score', 0.0),
            'cost_per_hour': health_metrics.get('cost_per_hour', 0.0),
            'current_instances': self.auto_scaler.current_instances,
            'health_status': health_metrics.get('health_status', 'unknown')
        }
        
        logger.info(f"Fly.io optimization status: {status}")
        
        if cost_analysis.get('recommendations'):
            logger.info(f"Cost recommendations: {cost_analysis['recommendations']}")
            
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        if not self.performance_history:
            return {'status': 'no_data'}
            
        # Calculate summary statistics
        recent_metrics = self.performance_history[-100:]  # Last ~50 minutes
        
        avg_gpu_util = np.mean([m.get('gpu_utilization', 0) for m in recent_metrics])
        avg_performance = np.mean([m.get('performance_score', 0) for m in recent_metrics])
        avg_cost = np.mean([m.get('cost_per_hour', 0) for m in recent_metrics])
        
        # Get component reports
        cost_analysis = self.cost_optimizer.get_cost_analysis()
        
        gpu_stats = {}
        if self.batch_processor:
            gpu_stats['batch_processor'] = self.batch_processor.get_performance_stats()
            
        if self.mixed_precision_manager:
            gpu_stats['mixed_precision'] = self.mixed_precision_manager.get_performance_stats()
            
        fallback_status = self.fallback_manager.get_status()
        memory_info = self.memory_manager.get_memory_info()
        
        return {
            'timestamp': time.time(),
            'optimization_active': self.is_optimizing,
            'instance_info': self.metadata_service.get_instance_info(),
            'performance_summary': {
                'avg_gpu_utilization': avg_gpu_util,
                'avg_performance_score': avg_performance,
                'avg_cost_per_hour': avg_cost,
                'current_instances': self.auto_scaler.current_instances
            },
            'cost_analysis': cost_analysis,
            'gpu_components': gpu_stats,
            'fallback_status': fallback_status,
            'memory_info': memory_info,
            'config': {
                'auto_scaling_enabled': self.config.enable_auto_scaling,
                'cost_optimization_enabled': self.config.enable_cost_optimization,
                'performance_tuning_enabled': self.config.enable_performance_tuning
            }
        }
        
    def export_optimization_data(self, filepath: str):
        """Export optimization data for analysis."""
        data = {
            'metadata': {
                'export_time': time.time(),
                'optimization_duration': len(self.performance_history) * 30,  # 30s intervals
                'instance_info': self.metadata_service.get_instance_info()
            },
            'performance_history': self.performance_history,
            'cost_history': self.cost_optimizer.cost_history,
            'optimization_report': self.get_optimization_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        logger.info(f"Optimization data exported to {filepath}")
        
    def tune_for_workload(self, workload_type: str = "trading"):
        """Tune optimization for specific workload."""
        logger.info(f"Tuning for {workload_type} workload")
        
        if workload_type == "trading":
            # Optimize for low latency
            self.config.optimize_for_latency = True
            self.config.scale_up_threshold = 0.6  # Scale earlier for consistent performance
            
        elif workload_type == "backtesting":
            # Optimize for throughput
            self.config.optimize_for_latency = False
            self.config.scale_up_threshold = 0.9  # Scale later to maximize utilization
            
        elif workload_type == "research":
            # Optimize for cost efficiency
            self.config.enable_cost_optimization = True
            self.config.auto_shutdown_idle_minutes = 15
            
        # Apply configuration
        self.auto_scaler.config = self.config
        self.cost_optimizer.config = self.config
        
        logger.info(f"Workload tuning applied for {workload_type}")


# Global optimizer instance
_global_fly_optimizer = None


def get_fly_optimizer() -> FlyGPUOptimizer:
    """Get the global fly.io GPU optimizer."""
    global _global_fly_optimizer
    if _global_fly_optimizer is None:
        _global_fly_optimizer = FlyGPUOptimizer()
    return _global_fly_optimizer


def initialize_flyio_optimization(workload_type: str = "trading") -> Dict[str, Any]:
    """Initialize fly.io GPU optimization."""
    logger.info("Initializing fly.io GPU optimization")
    
    try:
        optimizer = get_fly_optimizer()
        
        # Tune for workload
        optimizer.tune_for_workload(workload_type)
        
        # Start optimization
        optimizer.start_optimization()
        
        # Wait a moment for initial metrics
        time.sleep(5)
        
        # Get initial report
        report = optimizer.get_optimization_report()
        
        return {
            'status': 'success',
            'optimizer': optimizer,
            'initial_report': report
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize fly.io optimization: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e)
        }


if __name__ == "__main__":
    # Test fly.io optimization
    logger.info("Testing fly.io GPU optimization...")
    
    # Initialize optimization
    result = initialize_flyio_optimization("trading")
    
    if result['status'] == 'success':
        optimizer = result['optimizer']
        
        # Let it run for a short test
        logger.info("Running optimization for 60 seconds...")
        time.sleep(60)
        
        # Get report
        report = optimizer.get_optimization_report()
        logger.info(f"Optimization report: {report['performance_summary']}")
        
        # Export data
        optimizer.export_optimization_data("/tmp/flyio_optimization.json")
        
        # Stop optimization
        optimizer.stop_optimization()
        
        print("Fly.io GPU optimization tested successfully!")
    else:
        print(f"Fly.io optimization test failed: {result['error']}")
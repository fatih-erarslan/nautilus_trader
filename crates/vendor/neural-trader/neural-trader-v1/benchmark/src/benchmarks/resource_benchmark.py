"""Resource usage benchmark implementation for AI News Trading platform."""

import gc
import os
import psutil
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ResourceUsageResult:
    """Result container for resource usage measurements."""
    
    component: str
    operation: str
    duration: float
    
    # CPU metrics
    avg_cpu_percent: float
    peak_cpu_percent: float
    cpu_time_user: float
    cpu_time_system: float
    
    # Memory metrics
    avg_memory_mb: float
    peak_memory_mb: float
    memory_allocations: int
    memory_deallocations: int
    
    # I/O metrics
    disk_read_mb: float
    disk_write_mb: float
    network_bytes_sent: int
    network_bytes_recv: int
    
    # System metrics
    context_switches: int
    page_faults: int
    open_file_descriptors: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "component": self.component,
            "operation": self.operation,
            "duration": self.duration,
            "cpu": {
                "avg_cpu_percent": self.avg_cpu_percent,
                "peak_cpu_percent": self.peak_cpu_percent,
                "cpu_time_user": self.cpu_time_user,
                "cpu_time_system": self.cpu_time_system,
            },
            "memory": {
                "avg_memory_mb": self.avg_memory_mb,
                "peak_memory_mb": self.peak_memory_mb,
                "memory_allocations": self.memory_allocations,
                "memory_deallocations": self.memory_deallocations,
            },
            "io": {
                "disk_read_mb": self.disk_read_mb,
                "disk_write_mb": self.disk_write_mb,
                "network_bytes_sent": self.network_bytes_sent,
                "network_bytes_recv": self.network_bytes_recv,
            },
            "system": {
                "context_switches": self.context_switches,
                "page_faults": self.page_faults,
                "open_file_descriptors": self.open_file_descriptors,
            }
        }


class ResourceMonitor:
    """Monitors system resource usage during operations."""
    
    def __init__(self, sample_rate_hz: float = 10.0):
        """Initialize resource monitor."""
        self.sample_rate_hz = sample_rate_hz
        self.sample_interval = 1.0 / sample_rate_hz
        self.monitoring = False
        self.samples = []
        self.monitor_thread = None
        self.process = psutil.Process()
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return aggregated results."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        if not self.samples:
            return self._empty_results()
        
        return self._aggregate_samples()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                sample = self._take_sample()
                self.samples.append(sample)
                time.sleep(self.sample_interval)
            except Exception:
                # Continue monitoring even if individual samples fail
                continue
    
    def _take_sample(self) -> Dict:
        """Take a single resource usage sample."""
        # CPU metrics
        cpu_percent = self.process.cpu_percent()
        cpu_times = self.process.cpu_times()
        
        # Memory metrics
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        # I/O metrics
        try:
            io_counters = self.process.io_counters()
            disk_read = io_counters.read_bytes
            disk_write = io_counters.write_bytes
        except (AttributeError, psutil.AccessDenied):
            disk_read = disk_write = 0
        
        # Network metrics (system-wide)
        try:
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent
            net_recv = net_io.bytes_recv
        except (AttributeError, psutil.AccessDenied):
            net_sent = net_recv = 0
        
        # System metrics
        try:
            num_ctx_switches = self.process.num_ctx_switches()
            num_fds = self.process.num_fds() if hasattr(self.process, 'num_fds') else 0
        except (AttributeError, psutil.AccessDenied):
            num_ctx_switches = psutil.pcputimes(0, 0)
            num_fds = 0
        
        return {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'cpu_user_time': cpu_times.user,
            'cpu_system_time': cpu_times.system,
            'memory_rss': memory_info.rss,
            'memory_vms': memory_info.vms,
            'memory_percent': memory_percent,
            'disk_read': disk_read,
            'disk_write': disk_write,
            'net_sent': net_sent,
            'net_recv': net_recv,
            'ctx_switches_voluntary': getattr(num_ctx_switches, 'voluntary', 0),
            'ctx_switches_involuntary': getattr(num_ctx_switches, 'involuntary', 0),
            'num_fds': num_fds,
        }
    
    def _aggregate_samples(self) -> Dict:
        """Aggregate samples into summary statistics."""
        if not self.samples:
            return self._empty_results()
        
        # Extract metrics from samples
        cpu_percents = [s['cpu_percent'] for s in self.samples]
        memory_rss = [s['memory_rss'] for s in self.samples]
        
        # Calculate deltas for cumulative metrics
        first_sample = self.samples[0]
        last_sample = self.samples[-1]
        
        cpu_user_delta = last_sample['cpu_user_time'] - first_sample['cpu_user_time']
        cpu_system_delta = last_sample['cpu_system_time'] - first_sample['cpu_system_time']
        disk_read_delta = last_sample['disk_read'] - first_sample['disk_read']
        disk_write_delta = last_sample['disk_write'] - first_sample['disk_write']
        net_sent_delta = last_sample['net_sent'] - first_sample['net_sent']
        net_recv_delta = last_sample['net_recv'] - first_sample['net_recv']
        
        ctx_switches_vol_delta = (last_sample['ctx_switches_voluntary'] - 
                                  first_sample['ctx_switches_voluntary'])
        ctx_switches_invol_delta = (last_sample['ctx_switches_involuntary'] - 
                                    first_sample['ctx_switches_involuntary'])
        
        return {
            'avg_cpu_percent': np.mean(cpu_percents),
            'peak_cpu_percent': np.max(cpu_percents),
            'cpu_time_user': cpu_user_delta,
            'cpu_time_system': cpu_system_delta,
            'avg_memory_mb': np.mean(memory_rss) / (1024 * 1024),
            'peak_memory_mb': np.max(memory_rss) / (1024 * 1024),
            'disk_read_mb': disk_read_delta / (1024 * 1024),
            'disk_write_mb': disk_write_delta / (1024 * 1024),
            'network_bytes_sent': net_sent_delta,
            'network_bytes_recv': net_recv_delta,
            'context_switches': ctx_switches_vol_delta + ctx_switches_invol_delta,
            'page_faults': 0,  # Not easily available cross-platform
            'open_file_descriptors': last_sample['num_fds'],
        }
    
    def _empty_results(self) -> Dict:
        """Return empty results when no samples are available."""
        return {
            'avg_cpu_percent': 0.0,
            'peak_cpu_percent': 0.0,
            'cpu_time_user': 0.0,
            'cpu_time_system': 0.0,
            'avg_memory_mb': 0.0,
            'peak_memory_mb': 0.0,
            'disk_read_mb': 0.0,
            'disk_write_mb': 0.0,
            'network_bytes_sent': 0,
            'network_bytes_recv': 0,
            'context_switches': 0,
            'page_faults': 0,
            'open_file_descriptors': 0,
        }


class ResourceBenchmark:
    """Benchmarks resource usage for various operations."""
    
    def __init__(self, config):
        """Initialize resource benchmark."""
        self.config = config
        
        # Handle both ConfigManager and dict configurations
        if hasattr(config, 'config'):
            config_dict = config.config
        elif hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            config_dict = config if isinstance(config, dict) else {}
        
        # Extract configuration values with defaults
        memory_config = config_dict.get("metrics", {}).get("memory", {})
        self.sample_rate = memory_config.get("sample_rate", "10Hz")
        
        # Parse sample rate
        if isinstance(self.sample_rate, str):
            self.sample_rate_hz = float(self.sample_rate.replace("Hz", ""))
        else:
            self.sample_rate_hz = float(self.sample_rate)
    
    def benchmark_signal_generation_resources(
        self,
        strategy_name: str = "momentum",
        num_signals: int = 1000
    ) -> ResourceUsageResult:
        """Benchmark resource usage for signal generation."""
        return self._benchmark_operation_resources(
            component="signal_generator",
            operation="generate_signals",
            func=self._resource_intensive_signal_generation,
            args=(strategy_name, num_signals)
        )
    
    def benchmark_data_processing_resources(
        self,
        data_size: int = 10000
    ) -> ResourceUsageResult:
        """Benchmark resource usage for data processing."""
        return self._benchmark_operation_resources(
            component="data_processor",
            operation="process_large_dataset",
            func=self._resource_intensive_data_processing,
            args=(data_size,)
        )
    
    def benchmark_portfolio_optimization_resources(
        self,
        portfolio_size: int = 500,
        optimization_steps: int = 100
    ) -> ResourceUsageResult:
        """Benchmark resource usage for portfolio optimization."""
        return self._benchmark_operation_resources(
            component="portfolio_optimizer",
            operation="optimize_portfolio",
            func=self._resource_intensive_portfolio_optimization,
            args=(portfolio_size, optimization_steps)
        )
    
    def benchmark_risk_calculation_resources(
        self,
        portfolio_size: int = 1000,
        monte_carlo_iterations: int = 10000
    ) -> ResourceUsageResult:
        """Benchmark resource usage for risk calculations."""
        return self._benchmark_operation_resources(
            component="risk_calculator",
            operation="monte_carlo_risk",
            func=self._resource_intensive_risk_calculation,
            args=(portfolio_size, monte_carlo_iterations)
        )
    
    def _benchmark_operation_resources(
        self,
        component: str,
        operation: str,
        func,
        args: tuple = ()
    ) -> ResourceUsageResult:
        """Generic operation resource usage benchmark."""
        # Initialize monitor
        monitor = ResourceMonitor(self.sample_rate_hz)
        
        # Force garbage collection before starting
        gc.collect()
        
        # Start monitoring
        monitor.start_monitoring()
        start_time = time.time()
        
        try:
            # Execute the operation
            result = func(*args)
        finally:
            # Stop monitoring
            end_time = time.time()
            duration = end_time - start_time
            resource_stats = monitor.stop_monitoring()
        
        # Force garbage collection after
        gc.collect()
        
        return ResourceUsageResult(
            component=component,
            operation=operation,
            duration=duration,
            avg_cpu_percent=resource_stats['avg_cpu_percent'],
            peak_cpu_percent=resource_stats['peak_cpu_percent'],
            cpu_time_user=resource_stats['cpu_time_user'],
            cpu_time_system=resource_stats['cpu_time_system'],
            avg_memory_mb=resource_stats['avg_memory_mb'],
            peak_memory_mb=resource_stats['peak_memory_mb'],
            memory_allocations=0,  # Would need custom memory tracker
            memory_deallocations=0,  # Would need custom memory tracker
            disk_read_mb=resource_stats['disk_read_mb'],
            disk_write_mb=resource_stats['disk_write_mb'],
            network_bytes_sent=resource_stats['network_bytes_sent'],
            network_bytes_recv=resource_stats['network_bytes_recv'],
            context_switches=resource_stats['context_switches'],
            page_faults=resource_stats['page_faults'],
            open_file_descriptors=resource_stats['open_file_descriptors']
        )
    
    def _resource_intensive_signal_generation(self, strategy_name: str, num_signals: int):
        """Simulate resource-intensive signal generation."""
        results = []
        
        for i in range(num_signals):
            # Simulate complex calculations
            price_data = np.random.randn(1000)
            
            if strategy_name == "momentum":
                # Moving averages and momentum calculations
                ma_short = np.convolve(price_data, np.ones(20)/20, mode='valid')
                ma_long = np.convolve(price_data, np.ones(50)/50, mode='valid')
                momentum = ma_short[-1] - ma_long[-1] if len(ma_short) > 0 and len(ma_long) > 0 else 0
                
            elif strategy_name == "mean_reversion":
                # Statistical calculations
                mean = np.mean(price_data)
                std = np.std(price_data)
                z_score = (price_data[-1] - mean) / std if std > 0 else 0
                momentum = -z_score  # Reverse signal
                
            else:
                # Generic calculation
                momentum = np.mean(np.diff(price_data))
            
            # Simulate some memory allocation
            temp_arrays = [np.random.randn(100) for _ in range(10)]
            
            signal = "buy" if momentum > 0 else "sell"
            results.append({
                'signal': signal,
                'confidence': abs(momentum),
                'timestamp': time.time()
            })
            
            # Cleanup temp arrays
            del temp_arrays
        
        return results
    
    def _resource_intensive_data_processing(self, data_size: int):
        """Simulate resource-intensive data processing."""
        # Generate large dataset
        raw_data = np.random.randn(data_size, 10)
        
        # Various processing operations
        # 1. Normalization
        normalized = (raw_data - np.mean(raw_data, axis=0)) / np.std(raw_data, axis=0)
        
        # 2. Feature engineering
        features = []
        for i in range(raw_data.shape[1]):
            # Moving averages
            ma5 = np.convolve(raw_data[:, i], np.ones(5)/5, mode='valid')
            ma20 = np.convolve(raw_data[:, i], np.ones(20)/20, mode='valid')
            
            # Technical indicators
            diff = np.diff(raw_data[:, i])
            momentum = np.convolve(diff, np.ones(10)/10, mode='valid')
            
            features.extend([ma5, ma20, momentum])
        
        # 3. Correlation analysis
        correlation_matrix = np.corrcoef(normalized.T)
        
        # 4. Principal component analysis (simplified)
        covariance_matrix = np.cov(normalized.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # 5. Time series analysis
        fft_result = np.fft.fft(raw_data, axis=0)
        
        return {
            'processed_samples': data_size,
            'features_generated': len(features),
            'correlation_matrix_size': correlation_matrix.shape,
            'principal_components': len(eigenvalues)
        }
    
    def _resource_intensive_portfolio_optimization(self, portfolio_size: int, optimization_steps: int):
        """Simulate resource-intensive portfolio optimization."""
        # Generate returns matrix
        returns = np.random.randn(252, portfolio_size) * 0.02  # Daily returns
        
        # Calculate covariance matrix
        cov_matrix = np.cov(returns.T)
        
        # Expected returns
        expected_returns = np.mean(returns, axis=0)
        
        best_weights = None
        best_sharpe = -np.inf
        
        # Optimization loop (Monte Carlo)
        for step in range(optimization_steps):
            # Generate random weights
            weights = np.random.dirichlet(np.ones(portfolio_size))
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, expected_returns) * 252
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights)) * 252
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Sharpe ratio
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_weights = weights.copy()
            
            # Simulate some additional calculations
            if step % 10 == 0:
                # Risk decomposition
                marginal_contributions = np.dot(cov_matrix, weights) * 252
                component_contributions = weights * marginal_contributions
        
        return {
            'portfolio_size': portfolio_size,
            'optimization_steps': optimization_steps,
            'best_sharpe_ratio': best_sharpe,
            'optimal_weights': best_weights
        }
    
    def _resource_intensive_risk_calculation(self, portfolio_size: int, monte_carlo_iterations: int):
        """Simulate resource-intensive risk calculations."""
        # Portfolio setup
        weights = np.random.dirichlet(np.ones(portfolio_size))
        returns = np.random.randn(252, portfolio_size) * 0.02
        cov_matrix = np.cov(returns.T)
        
        # Monte Carlo simulation for VaR calculation
        simulated_returns = []
        
        for iteration in range(monte_carlo_iterations):
            # Generate random shocks
            shocks = np.random.multivariate_normal(np.zeros(portfolio_size), cov_matrix)
            
            # Calculate portfolio return
            portfolio_return = np.dot(weights, shocks)
            simulated_returns.append(portfolio_return)
        
        simulated_returns = np.array(simulated_returns)
        
        # Risk metrics
        var_95 = np.percentile(simulated_returns, 5)
        var_99 = np.percentile(simulated_returns, 1)
        cvar_95 = np.mean(simulated_returns[simulated_returns <= var_95])
        
        # Stress testing scenarios
        stress_scenarios = []
        for _ in range(100):
            stress_shock = np.random.multivariate_normal(np.zeros(portfolio_size), cov_matrix * 3)
            stress_return = np.dot(weights, stress_shock)
            stress_scenarios.append(stress_return)
        
        worst_stress = np.min(stress_scenarios)
        
        return {
            'portfolio_size': portfolio_size,
            'monte_carlo_iterations': monte_carlo_iterations,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'worst_stress_scenario': worst_stress
        }


class ResourceProfiler:
    """Profiles resource usage across different operation scales."""
    
    def __init__(self, config):
        """Initialize resource profiler."""
        self.config = config
        self.benchmark = ResourceBenchmark(config)
    
    def profile_scaling_behavior(
        self,
        operation: str,
        scale_parameters: List[int]
    ) -> Dict[int, ResourceUsageResult]:
        """Profile how resource usage scales with operation size."""
        results = {}
        
        for scale in scale_parameters:
            if operation == "signal_generation":
                result = self.benchmark.benchmark_signal_generation_resources(
                    num_signals=scale
                )
            elif operation == "data_processing":
                result = self.benchmark.benchmark_data_processing_resources(
                    data_size=scale
                )
            elif operation == "portfolio_optimization":
                result = self.benchmark.benchmark_portfolio_optimization_resources(
                    portfolio_size=scale
                )
            elif operation == "risk_calculation":
                result = self.benchmark.benchmark_risk_calculation_resources(
                    portfolio_size=scale
                )
            else:
                continue
            
            results[scale] = result
        
        return results
    
    def analyze_resource_efficiency(
        self,
        results: Dict[str, ResourceUsageResult]
    ) -> Dict[str, Dict]:
        """Analyze resource usage efficiency."""
        analysis = {
            "efficiency_metrics": {},
            "resource_hotspots": [],
            "optimization_recommendations": []
        }
        
        for name, result in results.items():
            # Calculate efficiency metrics
            ops_per_cpu_sec = 1.0 / (result.cpu_time_user + result.cpu_time_system) if (result.cpu_time_user + result.cpu_time_system) > 0 else 0
            mb_per_operation = result.peak_memory_mb
            
            analysis["efficiency_metrics"][name] = {
                "cpu_efficiency": ops_per_cpu_sec,
                "memory_efficiency": mb_per_operation,
                "duration": result.duration,
                "cpu_utilization": result.avg_cpu_percent,
                "peak_memory": result.peak_memory_mb
            }
            
            # Identify resource hotspots
            if result.peak_cpu_percent > 80:
                analysis["resource_hotspots"].append({
                    "component": name,
                    "resource": "CPU",
                    "usage": result.peak_cpu_percent,
                    "severity": "high"
                })
            
            if result.peak_memory_mb > 1000:  # More than 1GB
                analysis["resource_hotspots"].append({
                    "component": name,
                    "resource": "Memory",
                    "usage": result.peak_memory_mb,
                    "severity": "high"
                })
        
        # Generate optimization recommendations
        high_cpu_components = [
            name for name, metrics in analysis["efficiency_metrics"].items()
            if metrics["cpu_utilization"] > 70
        ]
        
        if high_cpu_components:
            analysis["optimization_recommendations"].append({
                "type": "CPU Optimization",
                "components": high_cpu_components,
                "recommendation": "Consider parallelization or algorithm optimization"
            })
        
        high_memory_components = [
            name for name, metrics in analysis["efficiency_metrics"].items()
            if metrics["peak_memory"] > 500
        ]
        
        if high_memory_components:
            analysis["optimization_recommendations"].append({
                "type": "Memory Optimization",
                "components": high_memory_components,
                "recommendation": "Consider memory-efficient algorithms or streaming processing"
            })
        
        return analysis
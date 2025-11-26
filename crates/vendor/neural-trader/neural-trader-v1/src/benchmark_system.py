"""
Neural Trader Comprehensive Benchmarking System

A comprehensive benchmarking system that tests:
- Latency and throughput performance
- Model accuracy across different configurations
- WASM vs GPU performance comparisons
- Historical backtesting with multiple strategies
- Resource utilization metrics
"""

import time
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import psutil
import threading
import concurrent.futures
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

# Mock imports for demonstration - replace with actual imports
try:
    import torch
    import tensorflow as tf
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    print("GPU libraries not available, using CPU fallback")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    test_duration: int = 60  # seconds
    warmup_iterations: int = 100
    test_iterations: int = 1000
    batch_sizes: List[int] = None
    model_types: List[str] = None
    backends: List[str] = None
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 8, 16, 32, 64, 128]
        if self.model_types is None:
            self.model_types = ['lstm', 'transformer', 'cnn', 'ensemble']
        if self.backends is None:
            self.backends = ['cpu', 'gpu', 'wasm']
        if self.metrics is None:
            self.metrics = ['latency', 'throughput', 'accuracy', 'memory', 'cpu']

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    timestamp: datetime
    test_name: str
    backend: str
    model_type: str
    batch_size: int
    latency_ms: float
    throughput_ops_per_sec: float
    accuracy: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: float
    success_rate: float
    error_count: int
    metadata: Dict[str, Any]

class PerformanceMonitor:
    """Monitors system performance during benchmarks"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 0.1):
        """Start performance monitoring"""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return aggregated metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if not self.metrics:
            return {}
        
        df = pd.DataFrame(self.metrics)
        return {
            'avg_cpu_percent': df['cpu_percent'].mean(),
            'max_cpu_percent': df['cpu_percent'].max(),
            'avg_memory_mb': df['memory_mb'].mean(),
            'max_memory_mb': df['memory_mb'].max(),
            'avg_gpu_percent': df['gpu_percent'].mean() if 'gpu_percent' in df else 0.0,
            'max_gpu_percent': df['gpu_percent'].max() if 'gpu_percent' in df else 0.0
        }
    
    def _monitor_loop(self, interval: float):
        """Monitor performance in background thread"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                
                gpu_percent = 0.0
                if GPU_AVAILABLE:
                    try:
                        # Mock GPU monitoring - replace with actual GPU monitoring
                        gpu_percent = np.random.uniform(0, 100)
                    except:
                        gpu_percent = 0.0
                
                self.metrics.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'gpu_percent': gpu_percent
                })
                
                time.sleep(interval)
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")

class MockModel:
    """Mock neural network model for benchmarking"""
    
    def __init__(self, model_type: str = 'lstm', backend: str = 'cpu'):
        self.model_type = model_type
        self.backend = backend
        self.initialized = False
        self._setup_model()
    
    def _setup_model(self):
        """Setup model based on type and backend"""
        # Simulate model initialization time
        init_time = {
            'lstm': 0.1,
            'transformer': 0.3,
            'cnn': 0.05,
            'ensemble': 0.5
        }.get(self.model_type, 0.1)
        
        if self.backend == 'gpu':
            init_time *= 0.7  # GPU initialization is faster
        elif self.backend == 'wasm':
            init_time *= 1.5  # WASM initialization is slower
        
        time.sleep(init_time)
        self.initialized = True
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Mock prediction method"""
        if not self.initialized:
            raise ValueError("Model not initialized")
        
        batch_size = len(data)
        
        # Simulate computation time based on model complexity
        base_time = {
            'lstm': 0.001,
            'transformer': 0.005,
            'cnn': 0.0005,
            'ensemble': 0.008
        }.get(self.model_type, 0.001)
        
        # Adjust for backend
        if self.backend == 'gpu' and batch_size > 8:
            base_time *= 0.3  # GPU is much faster for larger batches
        elif self.backend == 'wasm':
            base_time *= 0.8  # WASM is slightly faster than CPU
        
        # Scale with batch size (with diminishing returns for GPU)
        if self.backend == 'gpu':
            compute_time = base_time * np.log(batch_size + 1)
        else:
            compute_time = base_time * batch_size
        
        time.sleep(compute_time)
        
        # Return mock predictions with some accuracy simulation
        return np.random.random((batch_size, 1))
    
    def evaluate(self, data: np.ndarray, targets: np.ndarray) -> float:
        """Mock evaluation method"""
        predictions = self.predict(data)
        
        # Simulate accuracy based on model type
        base_accuracy = {
            'lstm': 0.75,
            'transformer': 0.82,
            'cnn': 0.68,
            'ensemble': 0.85
        }.get(self.model_type, 0.70)
        
        # Add some noise to simulate real-world variance
        accuracy = base_accuracy + np.random.normal(0, 0.05)
        return max(0.0, min(1.0, accuracy))

class LatencyBenchmark:
    """Benchmark for measuring model inference latency"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
    
    def run_latency_test(self, model: MockModel, batch_size: int) -> Dict[str, float]:
        """Run latency benchmark for a specific model and batch size"""
        logger.info(f"Running latency test: {model.model_type} ({model.backend}) - batch_size: {batch_size}")
        
        # Generate test data
        test_data = np.random.random((batch_size, 10))  # 10 features
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            model.predict(test_data)
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Run benchmark
        latencies = []
        errors = 0
        
        for i in range(self.config.test_iterations):
            try:
                start_time = time.perf_counter()
                predictions = model.predict(test_data)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
            except Exception as e:
                logger.warning(f"Prediction error in iteration {i}: {e}")
                errors += 1
        
        # Stop monitoring
        perf_metrics = self.monitor.stop_monitoring()
        
        if not latencies:
            return {}
        
        return {
            'avg_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'std_latency_ms': np.std(latencies),
            'error_rate': errors / self.config.test_iterations,
            **perf_metrics
        }

class ThroughputBenchmark:
    """Benchmark for measuring model throughput"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
    
    def run_throughput_test(self, model: MockModel, batch_size: int) -> Dict[str, float]:
        """Run throughput benchmark"""
        logger.info(f"Running throughput test: {model.model_type} ({model.backend}) - batch_size: {batch_size}")
        
        test_data = np.random.random((batch_size, 10))
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            model.predict(test_data)
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Run for specified duration
        start_time = time.time()
        operations = 0
        errors = 0
        
        while time.time() - start_time < self.config.test_duration:
            try:
                model.predict(test_data)
                operations += 1
            except Exception as e:
                logger.warning(f"Throughput test error: {e}")
                errors += 1
        
        elapsed_time = time.time() - start_time
        
        # Stop monitoring
        perf_metrics = self.monitor.stop_monitoring()
        
        throughput_ops_per_sec = operations / elapsed_time
        samples_per_sec = (operations * batch_size) / elapsed_time
        
        return {
            'throughput_ops_per_sec': throughput_ops_per_sec,
            'throughput_samples_per_sec': samples_per_sec,
            'total_operations': operations,
            'elapsed_time_sec': elapsed_time,
            'error_rate': errors / max(operations, 1),
            **perf_metrics
        }

class AccuracyBenchmark:
    """Benchmark for measuring model accuracy"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def run_accuracy_test(self, model: MockModel, test_data: np.ndarray, test_targets: np.ndarray) -> Dict[str, float]:
        """Run accuracy benchmark on test dataset"""
        logger.info(f"Running accuracy test: {model.model_type} ({model.backend})")
        
        # Split into batches for testing
        batch_size = 32
        n_samples = len(test_data)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        accuracies = []
        predictions_all = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_data = test_data[start_idx:end_idx]
            batch_targets = test_targets[start_idx:end_idx]
            
            try:
                accuracy = model.evaluate(batch_data, batch_targets)
                accuracies.append(accuracy)
                
                # Get predictions for additional metrics
                predictions = model.predict(batch_data)
                predictions_all.extend(predictions.flatten())
                
            except Exception as e:
                logger.warning(f"Accuracy test error in batch {i}: {e}")
        
        if not accuracies:
            return {}
        
        # Calculate additional metrics
        predictions_all = np.array(predictions_all[:len(test_targets)])
        
        # Mock additional metrics (replace with real metrics)
        mse = np.mean((predictions_all - test_targets.flatten()) ** 2)
        mae = np.mean(np.abs(predictions_all - test_targets.flatten()))
        
        return {
            'accuracy': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'mse': mse,
            'mae': mae,
            'samples_tested': len(test_targets)
        }

class BacktestingBenchmark:
    """Benchmark for historical backtesting"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def generate_historical_data(self, n_days: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate mock historical market data"""
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        # Generate price data with some realistic patterns
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
        prices = 100 * np.exp(np.cumsum(returns))  # Price series
        
        # Add some features
        features = pd.DataFrame({
            'price': prices,
            'volume': np.random.lognormal(15, 0.5, n_days),
            'volatility': pd.Series(returns).rolling(20).std().fillna(0.02),
            'rsi': np.random.uniform(20, 80, n_days),
            'macd': np.random.normal(0, 0.1, n_days),
            'bb_position': np.random.uniform(0, 1, n_days),
            'returns_5d': pd.Series(returns).rolling(5).sum().fillna(0),
            'returns_20d': pd.Series(returns).rolling(20).sum().fillna(0)
        }, index=dates)
        
        # Target: next day return (shifted)
        targets = pd.DataFrame({
            'next_return': features['price'].pct_change().shift(-1).fillna(0)
        }, index=dates)
        
        return features.fillna(0), targets.fillna(0)
    
    def run_backtest(self, model: MockModel, features: pd.DataFrame, targets: pd.DataFrame) -> Dict[str, float]:
        """Run backtesting simulation"""
        logger.info(f"Running backtest: {model.model_type} ({model.backend})")
        
        # Split data for walk-forward analysis
        train_size = int(len(features) * 0.7)
        
        portfolio_value = 10000  # Starting portfolio
        positions = []
        returns = []
        
        # Walk-forward backtesting
        for i in range(train_size, len(features) - 1):
            try:
                # Use recent data for prediction
                recent_data = features.iloc[i-20:i].values  # Last 20 days
                if len(recent_data) < 20:
                    continue
                
                # Predict next day return
                prediction = model.predict(recent_data[-1:])  # Last day only
                predicted_return = prediction[0][0]
                
                # Simple trading strategy: go long if prediction > threshold
                threshold = 0.001  # 0.1%
                position = 1 if predicted_return > threshold else -1 if predicted_return < -threshold else 0
                
                # Calculate actual return
                actual_return = targets.iloc[i].values[0]
                
                # Calculate portfolio return
                portfolio_return = position * actual_return
                returns.append(portfolio_return)
                positions.append(position)
                
                # Update portfolio value
                portfolio_value *= (1 + portfolio_return)
                
            except Exception as e:
                logger.warning(f"Backtest error at index {i}: {e}")
        
        if not returns:
            return {}
        
        returns = np.array(returns)
        total_return = (portfolio_value / 10000) - 1
        
        # Calculate metrics
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        max_drawdown = self._calculate_max_drawdown(returns)
        win_rate = np.sum(np.array(returns) > 0) / len(returns)
        
        return {
            'total_return': total_return,
            'annual_return': total_return * (252 / len(returns)),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(returns),
            'avg_return_per_trade': np.mean(returns),
            'return_volatility': np.std(returns)
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns series"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

class ComprehensiveBenchmarkSuite:
    """Main benchmark suite that orchestrates all benchmarks"""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.results = []
        
        # Initialize benchmark components
        self.latency_benchmark = LatencyBenchmark(self.config)
        self.throughput_benchmark = ThroughputBenchmark(self.config)
        self.accuracy_benchmark = AccuracyBenchmark(self.config)
        self.backtesting_benchmark = BacktestingBenchmark(self.config)
    
    def run_comprehensive_benchmark(self) -> List[BenchmarkResult]:
        """Run all benchmarks across all configurations"""
        logger.info("Starting comprehensive benchmark suite")
        
        # Generate test data for accuracy testing
        test_features = np.random.random((1000, 10))
        test_targets = np.random.random((1000, 1))
        
        # Generate historical data for backtesting
        historical_features, historical_targets = self.backtesting_benchmark.generate_historical_data()
        
        total_tests = len(self.config.model_types) * len(self.config.backends) * len(self.config.batch_sizes)
        test_count = 0
        
        for model_type in self.config.model_types:
            for backend in self.config.backends:
                # Skip GPU backend if not available
                if backend == 'gpu' and not GPU_AVAILABLE:
                    logger.warning(f"GPU not available, skipping GPU tests for {model_type}")
                    continue
                
                try:
                    # Initialize model
                    model = MockModel(model_type=model_type, backend=backend)
                    
                    # Run accuracy test (once per model/backend combination)
                    accuracy_results = self.accuracy_benchmark.run_accuracy_test(
                        model, test_features, test_targets
                    )
                    
                    # Run backtesting (once per model/backend combination)
                    backtest_results = self.backtesting_benchmark.run_backtest(
                        model, historical_features, historical_targets
                    )
                    
                    # Run latency and throughput tests for each batch size
                    for batch_size in self.config.batch_sizes:
                        test_count += 1
                        logger.info(f"Running test {test_count}/{total_tests}: {model_type} ({backend}) - batch_size: {batch_size}")
                        
                        try:
                            # Run latency test
                            latency_results = self.latency_benchmark.run_latency_test(model, batch_size)
                            
                            # Run throughput test
                            throughput_results = self.throughput_benchmark.run_throughput_test(model, batch_size)
                            
                            # Combine all results
                            result = BenchmarkResult(
                                timestamp=datetime.now(),
                                test_name=f"{model_type}_{backend}_{batch_size}",
                                backend=backend,
                                model_type=model_type,
                                batch_size=batch_size,
                                latency_ms=latency_results.get('avg_latency_ms', 0),
                                throughput_ops_per_sec=throughput_results.get('throughput_ops_per_sec', 0),
                                accuracy=accuracy_results.get('accuracy', 0),
                                memory_usage_mb=max(
                                    latency_results.get('avg_memory_mb', 0),
                                    throughput_results.get('avg_memory_mb', 0)
                                ),
                                cpu_usage_percent=max(
                                    latency_results.get('avg_cpu_percent', 0),
                                    throughput_results.get('avg_cpu_percent', 0)
                                ),
                                gpu_usage_percent=max(
                                    latency_results.get('avg_gpu_percent', 0),
                                    throughput_results.get('avg_gpu_percent', 0)
                                ),
                                success_rate=1 - max(
                                    latency_results.get('error_rate', 0),
                                    throughput_results.get('error_rate', 0)
                                ),
                                error_count=0,  # Will be calculated from error rates
                                metadata={
                                    'latency_details': latency_results,
                                    'throughput_details': throughput_results,
                                    'accuracy_details': accuracy_results,
                                    'backtest_results': backtest_results
                                }
                            )
                            
                            self.results.append(result)
                            
                        except Exception as e:
                            logger.error(f"Test failed for {model_type} ({backend}) batch_size {batch_size}: {e}")
                
                except Exception as e:
                    logger.error(f"Model initialization failed for {model_type} ({backend}): {e}")
        
        logger.info(f"Comprehensive benchmark completed. {len(self.results)} results collected.")
        return self.results
    
    def save_results(self, filepath: str):
        """Save benchmark results to file"""
        results_data = [asdict(result) for result in self.results]
        
        # Convert datetime objects to strings for JSON serialization
        for result in results_data:
            result['timestamp'] = result['timestamp'].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        if not self.results:
            return {}
        
        # Convert results to DataFrame for analysis
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Performance comparison by backend
        backend_comparison = {}
        for backend in self.config.backends:
            backend_data = df[df['backend'] == backend]
            if len(backend_data) > 0:
                backend_comparison[backend] = {
                    'avg_latency_ms': backend_data['latency_ms'].mean(),
                    'avg_throughput': backend_data['throughput_ops_per_sec'].mean(),
                    'avg_accuracy': backend_data['accuracy'].mean(),
                    'avg_memory_mb': backend_data['memory_usage_mb'].mean(),
                    'avg_cpu_percent': backend_data['cpu_usage_percent'].mean(),
                    'avg_gpu_percent': backend_data['gpu_usage_percent'].mean(),
                    'avg_success_rate': backend_data['success_rate'].mean()
                }
        
        # Model comparison
        model_comparison = {}
        for model_type in self.config.model_types:
            model_data = df[df['model_type'] == model_type]
            if len(model_data) > 0:
                model_comparison[model_type] = {
                    'avg_latency_ms': model_data['latency_ms'].mean(),
                    'avg_throughput': model_data['throughput_ops_per_sec'].mean(),
                    'avg_accuracy': model_data['accuracy'].mean(),
                    'best_backend': model_data.loc[model_data['throughput_ops_per_sec'].idxmax(), 'backend']
                }
        
        # Batch size analysis
        batch_analysis = {}
        for batch_size in self.config.batch_sizes:
            batch_data = df[df['batch_size'] == batch_size]
            if len(batch_data) > 0:
                batch_analysis[batch_size] = {
                    'avg_latency_ms': batch_data['latency_ms'].mean(),
                    'avg_throughput': batch_data['throughput_ops_per_sec'].mean(),
                    'efficiency_ratio': batch_data['throughput_ops_per_sec'].mean() / batch_data['latency_ms'].mean()
                }
        
        # WASM vs GPU comparison
        wasm_gpu_comparison = {}
        if 'wasm' in df['backend'].values and 'gpu' in df['backend'].values:
            wasm_data = df[df['backend'] == 'wasm']
            gpu_data = df[df['backend'] == 'gpu']
            
            wasm_gpu_comparison = {
                'latency_improvement_gpu': (wasm_data['latency_ms'].mean() - gpu_data['latency_ms'].mean()) / wasm_data['latency_ms'].mean(),
                'throughput_improvement_gpu': (gpu_data['throughput_ops_per_sec'].mean() - wasm_data['throughput_ops_per_sec'].mean()) / wasm_data['throughput_ops_per_sec'].mean(),
                'memory_overhead_gpu': gpu_data['memory_usage_mb'].mean() - wasm_data['memory_usage_mb'].mean(),
                'power_efficiency_wasm': wasm_data['cpu_usage_percent'].mean() < gpu_data['gpu_usage_percent'].mean()
            }
        
        # Best configurations
        best_configs = {
            'lowest_latency': df.loc[df['latency_ms'].idxmin()][['model_type', 'backend', 'batch_size']].to_dict(),
            'highest_throughput': df.loc[df['throughput_ops_per_sec'].idxmax()][['model_type', 'backend', 'batch_size']].to_dict(),
            'highest_accuracy': df.loc[df['accuracy'].idxmax()][['model_type', 'backend', 'batch_size']].to_dict(),
            'most_efficient': df.loc[(df['throughput_ops_per_sec'] / df['latency_ms']).idxmax()][['model_type', 'backend', 'batch_size']].to_dict()
        }
        
        return {
            'summary': {
                'total_tests': len(self.results),
                'test_timestamp': datetime.now().isoformat(),
                'config': asdict(self.config)
            },
            'backend_comparison': backend_comparison,
            'model_comparison': model_comparison,
            'batch_analysis': batch_analysis,
            'wasm_gpu_comparison': wasm_gpu_comparison,
            'best_configurations': best_configs,
            'recommendations': self._generate_recommendations(df)
        }
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on benchmark results"""
        recommendations = []
        
        # Latency recommendations
        best_latency = df.loc[df['latency_ms'].idxmin()]
        recommendations.append(f"For lowest latency, use {best_latency['model_type']} with {best_latency['backend']} backend")
        
        # Throughput recommendations
        best_throughput = df.loc[df['throughput_ops_per_sec'].idxmax()]
        recommendations.append(f"For highest throughput, use {best_throughput['model_type']} with {best_throughput['backend']} backend")
        
        # Accuracy recommendations
        best_accuracy = df.loc[df['accuracy'].idxmax()]
        recommendations.append(f"For highest accuracy, use {best_accuracy['model_type']} model")
        
        # Resource efficiency
        df['efficiency_score'] = df['throughput_ops_per_sec'] / (df['latency_ms'] * df['memory_usage_mb'])
        best_efficiency = df.loc[df['efficiency_score'].idxmax()]
        recommendations.append(f"For best resource efficiency, use {best_efficiency['model_type']} with {best_efficiency['backend']} backend")
        
        # Backend-specific recommendations
        if 'gpu' in df['backend'].values and 'cpu' in df['backend'].values:
            gpu_data = df[df['backend'] == 'gpu']
            cpu_data = df[df['backend'] == 'cpu']
            
            if gpu_data['throughput_ops_per_sec'].mean() > cpu_data['throughput_ops_per_sec'].mean() * 1.5:
                recommendations.append("GPU backend shows significant performance improvement for high-throughput scenarios")
            
            if cpu_data['latency_ms'].mean() < gpu_data['latency_ms'].mean():
                recommendations.append("CPU backend may be better for low-latency, single-inference scenarios")
        
        return recommendations

# Example usage and testing functions
def run_quick_benchmark():
    """Run a quick benchmark for testing"""
    config = BenchmarkConfig(
        test_duration=10,
        warmup_iterations=10,
        test_iterations=50,
        batch_sizes=[1, 16, 64],
        model_types=['lstm', 'transformer'],
        backends=['cpu', 'wasm'] + (['gpu'] if GPU_AVAILABLE else [])
    )
    
    suite = ComprehensiveBenchmarkSuite(config)
    results = suite.run_comprehensive_benchmark()
    
    # Save results
    suite.save_results('/workspaces/neural-trader/benchmark_results.json')
    
    # Generate report
    report = suite.generate_comparison_report()
    
    with open('/workspaces/neural-trader/benchmark_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return results, report

def run_full_benchmark():
    """Run a comprehensive benchmark suite"""
    config = BenchmarkConfig(
        test_duration=60,
        warmup_iterations=100,
        test_iterations=1000,
    )
    
    suite = ComprehensiveBenchmarkSuite(config)
    results = suite.run_comprehensive_benchmark()
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite.save_results(f'/workspaces/neural-trader/benchmark_results_{timestamp}.json')
    
    # Generate and save report
    report = suite.generate_comparison_report()
    with open(f'/workspaces/neural-trader/benchmark_report_{timestamp}.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Total tests completed: {len(results)}")
    print(f"Results saved to: benchmark_results_{timestamp}.json")
    print(f"Report saved to: benchmark_report_{timestamp}.json")
    
    if results:
        best_latency = min(results, key=lambda x: x.latency_ms)
        best_throughput = max(results, key=lambda x: x.throughput_ops_per_sec)
        best_accuracy = max(results, key=lambda x: x.accuracy)
        
        print(f"\nBest Latency: {best_latency.latency_ms:.2f}ms ({best_latency.model_type} on {best_latency.backend})")
        print(f"Best Throughput: {best_throughput.throughput_ops_per_sec:.2f} ops/sec ({best_throughput.model_type} on {best_throughput.backend})")
        print(f"Best Accuracy: {best_accuracy.accuracy:.3f} ({best_accuracy.model_type} on {best_accuracy.backend})")
    
    print("\nRecommendations:")
    for rec in report.get('recommendations', []):
        print(f"- {rec}")
    
    return results, report

if __name__ == "__main__":
    # Run quick benchmark by default
    print("Running Neural Trader Comprehensive Benchmarking System")
    print("For a quick test, this will run a shortened benchmark suite.")
    print("For full benchmarking, call run_full_benchmark()")
    
    try:
        results, report = run_quick_benchmark()
        print(f"\nQuick benchmark completed successfully! {len(results)} tests run.")
        print("Check benchmark_results.json and benchmark_report.json for detailed results.")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        logger.exception("Benchmark execution failed")